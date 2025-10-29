// worker.js
importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js"
);
importScripts("opencv.js");

let session = null;
let queue = Promise.resolve();

// 준비 상태 플래그
let cvReady = false;
let sessionReady = false;

// 모델 파라미터
const INPUT_NAME = "input";
const MODEL_SIZE = 320;
const MEAN = [123.675, 116.28, 103.53];
const STD = [58.395, 57.12, 57.375];
const SCORE_THRESH = 0.97;

// OpenCV 재사용 Mat들
let resize_mat_1;
let gray_mat1;
let gray_mat2;
let pad_mat;
let blackScalar;

let resize_mat_full;
let dsize1920x1080;
let crop_mat;
let rgba_mat;

// WebAssembly 튜닝 (wasm EP 쓸 때 효과적)
ort.env.wasm.numThreads = self.navigator?.hardwareConcurrency || 4;
ort.env.wasm.simd = true;
ort.env.logLevel = "warning";

// OpenCV 로드 완료 후 초기화
cv.onRuntimeInitialized = () => {
  resize_mat_1 = new cv.Mat();
  gray_mat1 = new cv.Mat();
  gray_mat2 = new cv.Mat();
  pad_mat = new cv.Mat();
  blackScalar = new cv.Scalar(0, 0, 0, 255);

  resize_mat_full = new cv.Mat();
  dsize1920x1080 = new cv.Size(1920, 1080);
  crop_mat = new cv.Mat();
  rgba_mat = new cv.Mat();

  cvReady = true;
  postMessage({ type: "cv_ready" });
};

// readiness wait helper
function waitUntil(pred) {
  return new Promise((resolve) => {
    function check() {
      if (pred()) resolve();
      else setTimeout(check, 5);
    }
    check();
  });
}

// onmessage 직렬화
onmessage = (e) => {
  const msg = e.data;
  queue = queue
    .then(() => handleMessage(msg))
    .catch((err) => {
      postMessage({
        type: "error",
        error: String(err && err.stack ? err.stack : err),
      });
    });
};

async function handleMessage(msg) {
  if (msg.type === "init") {
    // OpenCV 준비까지 대기
    await waitUntil(() => cvReady);

    if (!sessionReady) {
      // EP 하나만 사용해 혼합 실행 문제 제거
      // 빠른 GPU 추론을 원하면 ["webgpu"]로 시도해도 된다.
      session = await ort.InferenceSession.create(msg.modelBytes, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      sessionReady = true;
    }

    postMessage({ type: "ready" });
    return;
  }

  if (msg.type === "process") {
    // 세션과 OpenCV가 준비될 때까지 대기
    await waitUntil(() => cvReady && sessionReady);

    const { imageBytes, baseName, imageSave } = msg;

    // 디코드
    const bmp = await createImageBitmap(new Blob([imageBytes]));
    const imW = bmp.width;
    const imH = bmp.height;

    const c1 = new OffscreenCanvas(imW, imH);
    const ctx = c1.getContext("2d");
    ctx.drawImage(bmp, 0, 0, imW, imH);
    bmp.close();

    const imgData = ctx.getImageData(0, 0, imW, imH);

    // 이 프레임 처리
    let cropResult;
    try {
      cropResult = await processOne(imgData, baseName, imageSave);
    } catch (err) {
      // 한 프레임 실패. 전체 파이프라인은 계속된다.
      postMessage({
        type: "error",
        error: String(err && err.stack ? err.stack : err),
      });
      return;
    }

    const { name, pngBytes, coord } = cropResult;

    // 결과 전송. pngBytes ArrayBuffer는 transfer
    postMessage({ type: "result", crops: { name, pngBytes, coord } }, [
      pngBytes,
    ]);

    return;
  }

  if (msg.type === "finish") {
    // drain 보장
    await waitUntil(() => cvReady && sessionReady);
    postMessage({ type: "done" });
    return;
  }
}

// 단일 프레임 처리
async function processOne(imgData, baseName, imageSave) {
  const { inputTensor, ratio, pad, srcMat } = preprocessToTensor(
    imgData,
    MODEL_SIZE
  );

  // 추론
  const feeds = { [INPUT_NAME]: inputTensor };
  const outputMap = await session.run(feeds);

  const scores = outputMap["scores"].data;
  const bboxes = outputMap["bboxes"].data;

  // bbox 선택
  let bestBox = pickBox(bboxes, scores);

  // 3840x2160을 가정한다. 만약 해상도가 변할 수 있으면
  // srcMat.cols, srcMat.rows를 써서 동적으로 처리해라.
  const origW = 3840;
  const origH = 2160;

  const mappedBox = deLetterbox(bestBox, origW, origH, MODEL_SIZE, ratio, pad);

  // 크롭 및 PNG 인코딩
  const coord = {
    x1: mappedBox.x1,
    y1: mappedBox.y1,
    x2: mappedBox.x2,
    y2: mappedBox.y2,
  };

  let buf;
  if (imageSave) {
    buf = await cropToPNG(srcMat, mappedBox, origW);
  } else {
    buf = "";
  }

  // srcMat 정리
  srcMat.delete();

  return {
    name: `${baseName}_crop.png`,
    pngBytes: buf,
    coord,
  };
}

// 전처리: letterbox + grayscale + normalize → float32 NCHW
function preprocessToTensor(imgData, size) {
  // 해상도 고정 가정
  const inW = 3840;
  const inH = 2160;

  // ImageData -> Mat
  const srcMat = cv.matFromImageData(imgData);

  // letterbox 비율 및 패딩 계산
  const r = Math.min(size / inW, size / inH);
  const newW = Math.round(inW * r);
  const newH = Math.round(inH * r);
  const padW = size - newW;
  const padH = size - newH;

  const dsizeModel = new cv.Size(newW, newH);
  cv.resize(srcMat, resize_mat_1, dsizeModel);

  cv.cvtColor(resize_mat_1, gray_mat1, cv.COLOR_RGB2GRAY, 0);
  cv.cvtColor(gray_mat1, gray_mat2, cv.COLOR_GRAY2RGBA, 0);

  const top = Math.round(padH / 2);
  const left = Math.round(padW / 2);
  const bottom = padH - top;
  const right = padW - left;

  cv.copyMakeBorder(
    gray_mat2,
    pad_mat,
    top,
    bottom,
    left,
    right,
    cv.BORDER_CONSTANT,
    blackScalar
  );

  // RGBA -> Float32 CHW
  const chw = rgbaToCHWFloat32(pad_mat, size);

  const tensor = new ort.Tensor("float32", chw, [1, 3, size, size]);

  return {
    inputTensor: tensor,
    ratio: r,
    pad: { y: top, x: left },
    srcMat,
  };
}

// RGBA Mat -> Float32 CHW 정규화
function rgbaToCHWFloat32(rgbaMat, size) {
  const data = rgbaMat.data; // Uint8Array RGBA
  const plane = size * size;
  const out = new Float32Array(3 * plane);

  let rOff = 0;
  let gOff = plane;
  let bOff = plane * 2;

  for (let i = 0; i < plane; i++) {
    const j = i * 4;
    const rv = data[j];
    const gv = data[j + 1];
    const bv = data[j + 2];

    out[rOff++] = (rv - MEAN[0]) / STD[0];
    out[gOff++] = (gv - MEAN[1]) / STD[1];
    out[bOff++] = (bv - MEAN[2]) / STD[2];
  }

  return out;
}

// bbox 선택: 중앙에 가까우면서 SCORE_THRESH 이상인 박스
function pickBox(bboxes, scores) {
  let minCenter = 2.0;
  let box = { x1: 0, y1: 0, x2: MODEL_SIZE, y2: MODEL_SIZE };

  const rows = scores.length;
  for (let i = 0; i < rows; i++) {
    const conf = scores[i];
    if (conf < SCORE_THRESH) continue;

    const off = i * 4;
    const ncx = bboxes[off + 0];
    const ncy = bboxes[off + 1];
    const nw = bboxes[off + 2] + 0.05;
    const nh = bboxes[off + 3] + 0.05;

    const cx = ncx * MODEL_SIZE;
    const cy = ncy * MODEL_SIZE;
    const w = nw * MODEL_SIZE;
    const h = nh * MODEL_SIZE;

    const centerDist = (ncx - 0.5) * (ncx - 0.5) + (ncy - 0.5) * (ncy - 0.5);

    if (centerDist < minCenter) {
      minCenter = centerDist;
      box = {
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
      };
    }
  }
  return box;
}

// letterbox 좌표를 원본 좌표로 복원
function deLetterbox(b, origW, origH, size, ratio, pad) {
  function clamp(v, lo, hi) {
    return v < lo ? lo : v > hi ? hi : v;
  }

  // 모델 좌표계에서 유효범위 보정
  const x1c = clamp(b.x1, 0, size);
  const y1c = clamp(b.y1, 0, size);
  const x2c = clamp(b.x2, 0, size);
  const y2c = clamp(b.y2, 0, size);

  // 패딩 제거하고 ratio로 나눔
  const x1 = (x1c - pad.x) / ratio;
  const y1 = (y1c - pad.y) / ratio;
  const x2 = (x2c - pad.x) / ratio;
  const y2 = (y2c - pad.y) / ratio;

  return {
    x1: clamp(x1, 0, origW),
    y1: clamp(y1, 0, origH),
    x2: clamp(x2, 0, origW),
    y2: clamp(y2, 0, origH),
  };
}

// 크롭해서 PNG ArrayBuffer로 변환
async function cropToPNG(srcMat, mappedBox, origW) {
  // 전체 프레임을 1920x1080으로 리사이즈 (재사용 buffer)
  cv.resize(srcMat, resize_mat_full, dsize1920x1080);

  const scale = 1920 / origW;
  const rx = Math.round(mappedBox.x1 * scale);
  const ry = Math.round(mappedBox.y1 * scale);
  const rw = Math.max(1, Math.round((mappedBox.x2 - mappedBox.x1) * scale));
  const rh = Math.max(1, Math.round((mappedBox.y2 - mappedBox.y1) * scale));

  // roi 뷰 만들고 copyTo로 안전하게 복사한 뒤 roiMat 해제
  const rect = new cv.Rect(rx, ry, rw, rh);
  const roiMat = resize_mat_full.roi(rect);
  roiMat.copyTo(crop_mat);
  roiMat.delete();

  // RGBA로 변환
  cv.cvtColor(crop_mat, rgba_mat, cv.COLOR_RGB2RGBA, 0);

  const png_imgData = new ImageData(
    new Uint8ClampedArray(rgba_mat.data),
    rgba_mat.cols,
    rgba_mat.rows
  );

  const bmp = await createImageBitmap(png_imgData);

  const c2 = new OffscreenCanvas(rw, rh);
  const ctx = c2.getContext("2d");
  ctx.drawImage(bmp, 0, 0, rw, rh);

  const blob = await c2.convertToBlob({ type: "image/png" });
  const buf = await blob.arrayBuffer();

  return buf;
}
