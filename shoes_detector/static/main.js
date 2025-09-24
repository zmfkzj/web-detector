// main.js
const pickBtn = document.querySelector("#pickDir");
const startBtn = document.querySelector("#start");
const modelInput = document.querySelector("#modelFile");
const prog = document.querySelector("#prog");
const statusEl = document.querySelector("#status");

let dirHandle = null;
let outDir = null;
let worker = null;
let totalFiles = 0;
let doneFiles = 0;

async function countFilesRecursively(handle) {
  let count = 0;
  for await (const [, h] of handle.entries()) {
    if (h.kind === "file") count++;
    else if (h.kind === "directory") count += await countFilesRecursively(h);
  }
  console.log(count);
  return count;
}

async function* walk(handle) {
  for await (const [name, h] of handle.entries()) {
    if (h.kind === "file") yield { name, handle: h };
    else if (h.kind === "directory") yield* walk(h);
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
  if (!dirHandle) return alert("먼저 폴더를 선택하세요.");
  const modelFile = modelInput.files?.[0];
  if (!modelFile) return alert("모델(.onnx) 파일을 선택하세요.");

  // 워커 생성
  worker?.terminate();
  worker = new Worker("ort/worker.js");
  console.log("#1");

  // 진행률 세팅
  totalFiles = await countFilesRecursively(dirHandle);
  doneFiles = 0;
  prog.value = 0;
  prog.max = totalFiles || 1;
  statusEl.textContent = `총 ${totalFiles}개 파일 처리 중...`;
  console.log("#2");

  // 모델 바이트 worker로 전달
  const modelBytes = await modelFile.arrayBuffer();
  worker.postMessage({ type: "init", modelBytes: modelBytes }, [modelBytes]);
  console.log("#3");

  worker.onerror = async (e) => {
    throw new Error(e);
  };

  // 워커 메시지 처리
  worker.onmessage = async (e) => {
    const msg = e.data;
    if (msg.type === "ready") {
      // 모델 로드 완료 → 처리 시작
      processAllFiles().catch((err) => {
        console.error(err);
        statusEl.textContent = "에러 발생: " + err.message;
      });
    } else if (msg.type === "result") {
      // 단일 이미지 처리 결과 수신 → 저장
      // msg.crops: Array<{ name: string, pngBytes: ArrayBuffer }>

      const { name, pngBytes } = msg.crops;
      const fileHandle = await outDir.getFileHandle(name, { create: true });
      const writable = await fileHandle.createWritable();
      await writable.write(pngBytes);
      await writable.close();

      doneFiles++;
      prog.value = doneFiles;
      statusEl.textContent = `처리 중... ${doneFiles}/${totalFiles}`;
    } else if (msg.type === "error") {
      console.error("Worker error:", msg.error, msg.lineno);
      doneFiles++;
      prog.value = doneFiles;
    } else if (msg.type === "done") {
      statusEl.textContent = `완료! ${doneFiles}/${totalFiles}`;
    }
  };
};

async function processAllFiles() {
  const CONCURRENCY = 2; // 동시 처리 개수 (GPU/CPU 상황봐서 조절)
  const queue = [];
  for await (const { name, handle } of walk(dirHandle)) {
    queue.push({ name, handle });
  }

  let i = 0;
  const runners = Array.from(
    { length: Math.min(CONCURRENCY, queue.length) },
    () => workerRunner()
  );

  async function workerRunner() {
    while (i < queue.length) {
      const idx = i++;
      const item = queue[idx];
      try {
        const file = await item.handle.getFile();
        const imgBytes = await file.arrayBuffer();
        // 파일명은 확장자 제거하여 결과명 prefix로 사용
        const base = item.name.replace(/\.[^.]+$/, "");
        worker.postMessage(
          { type: "process", imageBytes: imgBytes, baseName: base },
          [imgBytes]
        );
      } catch (err) {
        console.error("process error", err);
        doneFiles++;
        prog.value = doneFiles;
      }
    }
  }

  await Promise.all(runners);
  worker.postMessage({ type: "finish" });
}
