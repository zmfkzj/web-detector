// worker.js
importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js"
);
importScripts("opencv.js");

let session = null;
let queue = Promise.resolve();

// ====== 모델/전처리 파라미터 (모델에 맞게 조정하세요) ======
const INPUT_NAME = "input"; // 예: YOLOv5/8 변형의 입력명
const MODEL_SIZE = 320; // 정사각 입력 가정
const MEAN = [123.675, 116.28, 103.53]; // 필요 시 [0.485,0.456,0.406]
const STD = [58.395, 57.12, 57.375]; // 필요 시 [0.229,0.224,0.225]
const SCORE_THRESH = 0.97;

// postMessage helper
const send = (type, payload = {}) => postMessage({ type, ...payload });

onmessage = (e) => {
  const msg = e.data;

  if (msg.type === "init") {
    // 세션은 한 번만 생성
    if (!session) {
      queue = queue
        .then(async () => {
          session = await ort.InferenceSession.create(msg.modelBytes, {
            // executionProviders: ["wasm"],
            executionProviders: [
              {
                name: "webnn",
                deviceType: "gpu",
                powerPreference: "default",
              },
              // "webgpu",
              // "wasm",
            ],
            graphOptimizationLevel: "all",
          });
          postMessage({ type: "ready" });
        })
        .catch((err) =>
          postMessage({ type: "error", error: String(err.stack || err) })
        );
    } else {
      postMessage({ type: "ready" });
    }
    return;
  }

  if (msg.type === "process") {
    // ✨ 항상 큐에 연결 → 이전 run이 끝난 뒤에 실행됨
    queue = queue
      .then(async () => {
        const { imageBytes, baseName } = msg;

        const bmp = await createImageBitmap(new Blob([imageBytes]));
        const imW = bmp.width,
          imH = bmp.height;
        const c = new OffscreenCanvas(imW, imH);
        const ctx = c.getContext("2d");
        ctx.drawImage(bmp, 0, 0, imW, imH);
        bmp.close();
        let imgData = ctx.getImageData(0, 0, imW, imH);

        const { name, pngBytes } = await processOne(imgData, baseName); // 내부에서 await session.run(...)
        postMessage({ type: "result", crops: { name, pngBytes } }, [pngBytes]);
      })
      .catch((err) =>
        postMessage({ type: "error", error: String(err.stack || err) })
      );
    return;
  }

  if (msg.type === "finish") {
    // 큐가 모두 끝난 후 완료 알림
    queue = queue.then(() => postMessage({ type: "done" }));
    return;
  }
};

// 단일 이미지 처리
async function processOne(imgData, baseName) {
  // 디코드
  const { inputTensor, ratio, pad } = preprocessToTensor(imgData, MODEL_SIZE);
  // 추론
  const feeds = { [INPUT_NAME]: inputTensor };
  const outputMap = await session.run(feeds);

  // 출력명은 모델에 따라 다릅니다. 보통 첫 키 사용(단일 출력 가정)
  //     output_names=["classes", "scores", "bboxes"]
  // const classes = outputMap["classes"].data; // Float32Array 등
  let scores = outputMap["scores"].data; // Float32Array 등
  const bboxes = outputMap["bboxes"].data; // Float32Array 등

  // ---- 후처리 (모델 맞게 조정) ----
  const rows = scores.filter((v, i, a) => v != 0).length;
  let minCenterDistance = 2.0;
  let box = { x1: 0, y1: 0, x2: MODEL_SIZE, y2: MODEL_SIZE };
  for (let i = 0; i < rows; i++) {
    const off = i * 4;
    const ncx = bboxes[off + 0],
      ncy = bboxes[off + 1],
      nw = bboxes[off + 2] + 0.05,
      nh = bboxes[off + 3] + 0.05;
    const cx = ncx * MODEL_SIZE,
      cy = ncy * MODEL_SIZE,
      w = nw * MODEL_SIZE,
      h = nh * MODEL_SIZE;
    const conf = scores[i];
    if (conf < SCORE_THRESH) continue;
    // 클래스 확률 중 최대값을 score로 쓰려면 아래처럼:
    // let best = 0, bestId = -1;
    // for (let k=5; k<stride; k++) if (out[off+k] > best) { best = out[off+k]; bestId = k-5; }
    // const score = conf * best;
    const centerDistance = Math.sqrt(
      Math.pow(ncx - 0.5, 2) + Math.pow(ncy - 0.5, 2)
    );
    if (centerDistance < minCenterDistance) {
      minCenterDistance = centerDistance;
      const x1 = cx - w / 2,
        y1 = cy - h / 2,
        x2 = cx + w / 2,
        y2 = cy + h / 2;
      box = { x1, y1, x2, y2 };
    }
  }

  // letterbox → 원본 좌표로 역투영
  const mappedBox = deLetterbox(box, 3840, 2160, MODEL_SIZE, ratio, pad);

  // 크롭 PNG 생성
  const pngBytes = await cropToPNG(
    imgData,
    mappedBox.x1,
    mappedBox.y1,
    mappedBox.x2,
    mappedBox.y2
  );
  const crop = { name: `${baseName}_crop.png`, pngBytes };

  return crop;
}

// === 전처리: letterbox resize + normalize → Float32 CHW 텐서 ===
function preprocessToTensor(imgData, size) {
  const inW = 3840,
    inH = 2160;

  let src = cv.matFromImageData(imgData);

  const r = Math.min(size / inW, size / inH);
  const newW = Math.round(inW * r);
  const newH = Math.round(inH * r);
  const padW = size - newW;
  const padH = size - newH;

  let resize_mat = new cv.Mat();
  let dsize = new cv.Size(newW, newH);
  cv.resize(src, resize_mat, dsize);
  src.delete();

  let gray_mat1 = new cv.Mat();
  cv.cvtColor(resize_mat, gray_mat1, cv.COLOR_RGB2GRAY, 0);
  resize_mat.delete();

  let gray_mat2 = new cv.Mat();
  cv.cvtColor(gray_mat1, gray_mat2, cv.COLOR_GRAY2RGBA, 0);
  gray_mat1.delete();

  let pad_mat = new cv.Mat();
  let s = new cv.Scalar(0, 0, 0, 255);
  const top = Math.round(padH / 2),
    left = Math.round(padW / 2);
  const bottom = padH - top,
    right = padW - left;
  cv.copyMakeBorder(
    gray_mat2,
    pad_mat,
    top,
    bottom,
    left,
    right,
    cv.BORDER_CONSTANT,
    s
  );
  gray_mat2.delete();

  const pad_imgData = new ImageData(
    new Uint8ClampedArray(pad_mat.data),
    pad_mat.cols,
    pad_mat.rows
  );

  pad_mat.delete();

  // ctx.fillStyle = "rgb(0,0,0)"; // YOLO letterbox 기본
  // ctx.fillRect(0, 0, size, size);
  // const dx = Math.floor(padW / 2);
  // const dy = Math.floor(padH / 2);
  // ctx.filter = "grayscale(100%)";
  // ctx.drawImage(bmp, 0, 0, inW, inH, dx, dy, newW, newH);

  // const imgData = ctx.getImageData(0, 0, size, size);
  const data = pad_imgData.data;
  const chw = new Float32Array(3 * size * size);

  let c0 = 0,
    c1 = size * size,
    c2 = 2 * size * size;
  for (let i = 0; i < size * size; i++) {
    let j = i * 4;
    const r = data[j++],
      g = data[j++],
      b = data[j++];
    chw[c0++] = (r - MEAN[0]) / STD[0];
    chw[c1++] = (g - MEAN[1]) / STD[1];
    chw[c2++] = (b - MEAN[2]) / STD[2];
  }

  const tensor = new ort.Tensor("float32", chw, [1, 3, size, size]);

  return { inputTensor: tensor, ratio: r, pad: { y: top, x: left } };
}

// === letterbox 좌표 -> 원본 좌표 ===
function deLetterbox(b, origW, origH, size, ratio, pad) {
  const x1 = clamp(b.x1 - pad.x, 0, size) / ratio;
  const y1 = clamp(b.y1 - pad.y, 0, size) / ratio;
  const x2 = clamp(b.x2 - pad.x, 0, size) / ratio;
  const y2 = clamp(b.y2 - pad.y, 0, size) / ratio;
  // 위에서 size 기준으로 그린 게 아니라 letterbox 캔버스가 size×size라
  // 역투영 후 원본 해상도로 다시 맵핑
  return {
    x1: clamp(x1, 0, origW),
    y1: clamp(y1, 0, origH),
    x2: clamp(x2, 0, origW),
    y2: clamp(y2, 0, origH),
  };
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

// === OffscreenCanvas 크롭 → PNG 바이트(ArrayBuffer) ===
async function cropToPNG(imgData, x1, y1, x2, y2) {
  const imW = 3840,
    imH = 2160;

  let src = cv.matFromImageData(imgData);

  const r = 1920 / imW;
  const w = Math.max(1, Math.round((x2 - x1) * r));
  const h = Math.max(1, Math.round((y2 - y1) * r));

  let resize_mat = new cv.Mat();
  let dsize = new cv.Size(1920, 1080);
  cv.resize(src, resize_mat, dsize);
  src.delete();

  let rect = new cv.Rect(Math.round(x1 * r), Math.round(y1 * r), w, h);
  let crop_mat = new cv.Mat();
  crop_mat = resize_mat.roi(rect);

  let png_mat = new cv.Mat();
  cv.cvtColor(crop_mat, png_mat, cv.COLOR_RGB2RGBA, 0);
  resize_mat.delete();
  crop_mat.delete();

  const png_imgData = new ImageData(
    new Uint8ClampedArray(png_mat.data),
    png_mat.cols,
    png_mat.rows
  );
  png_mat.delete();

  const imgBitmap = await createImageBitmap(png_imgData);

  const c = new OffscreenCanvas(w, h);
  const ctx = c.getContext("2d");
  ctx.drawImage(imgBitmap, 0, 0, w, h);
  const blob = await c.convertToBlob({ type: "image/png" });

  return await blob.arrayBuffer(); // transferable
}
