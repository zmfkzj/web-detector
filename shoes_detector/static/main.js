// main.js
const pickBtn = document.querySelector("#pickDir");
const startBtn = document.querySelector("#start");
const modelInput = document.querySelector("#modelFile");
const prog = document.querySelector("#prog");
const statusEl = document.querySelector("#status");
const imageSave = document.querySelector("#image_save");

let dirHandle = null;
let outDir = null;
let worker = null;

let totalFiles = 0;
let doneFiles = 0;

// CSV 누적 버퍼 (동시쓰기 금지. 마지막에 한 번만 파일로 기록)
let csvRows = [];

// 한 번만 flush하도록 하는 플래그
let csvFlushed = false;

async function countFilesRecursively(handle) {
  let count = 0;
  for await (const [, h] of handle.entries()) {
    if (h.kind === "file") count++;
    else if (h.kind === "directory" && h.name !== "box") {
      count += await countFilesRecursively(h);
    }
  }
  return count;
}

async function* walk(handle) {
  for await (const [name, h] of handle.entries()) {
    if (h.kind === "file") {
      yield { name, handle: h };
    } else if (h.kind === "directory" && h.name !== "box") {
      yield* walk(h);
    }
  }
}

pickBtn.onclick = async () => {
  dirHandle = await window.showDirectoryPicker({ mode: "readwrite" });
  outDir = await dirHandle.getDirectoryHandle("output", { create: true });
  statusEl.textContent = "폴더 선택 완료: output/에 저장됩니다.";
  startBtn.disabled = !modelInput.files?.[0];
};

modelInput.onchange = () => {
  startBtn.disabled = !(dirHandle && modelInput.files?.[0]);
};

startBtn.onclick = async () => {
  if (!dirHandle) {
    alert("먼저 폴더를 선택하세요.");
    return;
  }
  const modelFile = modelInput.files?.[0];
  if (!modelFile) {
    alert("모델(.onnx) 파일을 선택하세요.");
    return;
  }

  // 워커 새로 생성
  worker?.terminate();
  worker = new Worker("ort/worker.js");

  // 진행률 초기화
  totalFiles = await countFilesRecursively(dirHandle);
  doneFiles = 0;
  prog.value = 0;
  prog.max = totalFiles || 1;
  statusEl.textContent = `총 ${totalFiles}개 파일 처리 중...`;

  // CSV 버퍼 초기화
  csvRows = [];
  csvFlushed = false;

  // 워커 에러: 죽이지 않고 상태만 표시
  worker.onerror = (e) => {
    console.error("Worker runtime error:", e.message || e);
    statusEl.textContent = "워커 에러 감지";
  };

  // 워커 메시지 핸들러
  worker.onmessage = async (e) => {
    const msg = e.data;

    if (msg.type === "cv_ready") {
      // OpenCV 초기화 알림. 여기선 할 일 없다.
      return;
    }

    if (msg.type === "ready") {
      // 모델 로드 완료 → 처리 시작
      processAllFiles().catch((err) => {
        console.error(err);
        statusEl.textContent = "에러 발생: " + err.message;
      });
      return;
    }

    if (msg.type === "result") {
      // 한 이미지 처리 결과
      const { name, pngBytes, coord } = msg.crops;

      // 1. 크롭 PNG 저장
      if (imageSave.checked) {
        const fileHandle = await outDir.getFileHandle(name, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(pngBytes);
        await writable.close();
      }

      // 2. CSV 라인 메모리에 push
      const { x1, y1, x2, y2 } = coord;
      csvRows.push(`${name},${x1},${y1},${x2},${y2}`);

      // 3. 진행률
      doneFiles++;
      prog.value = doneFiles;
      statusEl.textContent = `처리 중... ${doneFiles}/${totalFiles}`;

      // 4. 마지막 파일이면 바로 finalize 시도
      if (doneFiles === totalFiles) {
        await finalizeCsvOnce();
      }
      return;
    }

    if (msg.type === "error") {
      // 워커가 특정 프레임 처리 중 에러 발생했을 때
      console.error("Worker error msg:", msg.error);
      doneFiles++;
      prog.value = doneFiles;

      if (doneFiles === totalFiles) {
        await finalizeCsvOnce();
      }
      return;
    }

    if (msg.type === "done") {
      // 워커가 finish 후 보내는 완료 신호
      await finalizeCsvOnce();
      return;
    }
  };

  // 모델 바이트 전달
  const modelBytes = await modelFile.arrayBuffer();
  worker.postMessage({ type: "init", modelBytes }, [modelBytes]);
};

// 파일 전체 큐잉
async function processAllFiles() {
  // 주의: 이 CONCURRENCY는 이제 워커 내부 직렬 처리와 무관
  // 여기는 단순히 워커에 작업을 "차례대로" 보내는 역할
  const CONCURRENCY = 2;

  const queue = [];
  for await (const { name, handle } of walk(dirHandle)) {
    queue.push({ name, handle });
  }

  let i = 0;

  async function workerRunner() {
    while (true) {
      const idx = i++;
      if (idx >= queue.length) break;

      const item = queue[idx];
      try {
        const file = await item.handle.getFile();
        const imgBytes = await file.arrayBuffer();
        const base = item.name.replace(/\.[^.]+$/, "");

        // 워커에 전송. ArrayBuffer는 transfer
        worker.postMessage(
          {
            type: "process",
            imageBytes: imgBytes,
            baseName: base,
            imageSave: imageSave.checked,
          },
          [imgBytes]
        );
      } catch (err) {
        console.error("process error", err);
        doneFiles++;
        prog.value = doneFiles;
        if (doneFiles === totalFiles) {
          await finalizeCsvOnce();
        }
      }
    }
  }

  // 러너들 실행
  const runners = Array.from(
    { length: Math.min(CONCURRENCY, queue.length) },
    () => workerRunner()
  );
  await Promise.all(runners);

  // 모든 process 메시지를 던졌으면 finish 알림
  worker.postMessage({ type: "finish" });
}
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
// CSV 한 번만 기록
async function finalizeCsvOnce() {
  if (csvFlushed) return;
  csvFlushed = true;

  try {
    await sleep(1000);
    const txtHandle = await outDir.getFileHandle("crop_bboxes.csv", {
      create: true,
    });
    const txtWritable = await txtHandle.createWritable();
    await txtWritable.write(csvRows.join("\n") + "\n");
    await txtWritable.close();

    statusEl.textContent = `완료! ${doneFiles}/${totalFiles}`;
  } catch (err) {
    console.error("CSV flush error", err);
    statusEl.textContent = "CSV 저장 에러: " + err.message;
  }
}
