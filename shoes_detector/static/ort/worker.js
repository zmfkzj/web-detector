importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.all.min.js"
);
importScripts("opencv.js");

let session = null;
let queue = Promise.resolve();

const INPUT_NAME = "input";
const MODEL_SIZE = 320;
const MEAN = [123.675, 116.28, 103.53];
const STD = [58.395, 57.12, 57.375];
const SCORE_THRESH = 0.97;

// readiness flags
let cvReady = false;
let sessionReady = false;

// global Mats (will init after cvReady)
let resize_mat_1,
  gray_mat1,
  gray_mat2,
  pad_mat,
  blackScalar,
  resize_mat,
  dsize1920x1080,
  crop_mat,
  png_mat;

cv.onRuntimeInitialized = () => {
  resize_mat_1 = new cv.Mat();
  gray_mat1 = new cv.Mat();
  gray_mat2 = new cv.Mat();
  pad_mat = new cv.Mat();
  blackScalar = new cv.Scalar(0, 0, 0, 255);

  resize_mat = new cv.Mat();
  dsize1920x1080 = new cv.Size(1920, 1080);
  crop_mat = new cv.Mat();
  png_mat = new cv.Mat();

  cvReady = true;
  postMessage({ type: "cv_ready" });
};

// helper to wait until condition true
function waitUntil(pred) {
  return new Promise((resolve) => {
    function check() {
      if (pred()) resolve();
      else setTimeout(check, 5);
    }
    check();
  });
}

// worker messages
onmessage = (e) => {
  const msg = e.data;
  queue = queue
    .then(() => handleMessage(msg))
    .catch((err) => {
      postMessage({ type: "error", error: String((err && err.stack) || err) });
    });
};

async function handleMessage(msg) {
  if (msg.type === "init") {
    // ensure cv runtime ready
    await waitUntil(() => cvReady);

    if (!sessionReady) {
      session = await ort.InferenceSession.create(msg.modelBytes, {
        executionProviders: ["webgpu", "wasm"],
        graphOptimizationLevel: "all",
      });
      sessionReady = true;
    }

    postMessage({ type: "ready" });
    return;
  }

  if (msg.type === "process") {
    // wait for cv + session to be ready
    await waitUntil(() => cvReady && sessionReady);

    const { imageBytes, baseName } = msg;

    // decode bytes -> ImageData
    const bmp = await createImageBitmap(new Blob([imageBytes]));
    const imW = bmp.width;
    const imH = bmp.height;
    const c1 = new OffscreenCanvas(imW, imH);
    const ctx = c1.getContext("2d");
    ctx.drawImage(bmp, 0, 0, imW, imH);
    bmp.close();
    const imgData = ctx.getImageData(0, 0, imW, imH);

    // run pipeline
    let cropResult;
    try {
      cropResult = await processOne(imgData, baseName);
    } catch (err) {
      postMessage({ type: "error", error: String((err && err.stack) || err) });
      return;
    }

    const { name, pngBytes, coord } = cropResult;

    // transfer pngBytes buffer
    postMessage({ type: "result", crops: { name, pngBytes, coord } }, [
      pngBytes,
    ]);
    return;
  }

  if (msg.type === "finish") {
    // wait for everything to drain
    await waitUntil(() => cvReady && sessionReady);
    postMessage({ type: "done" });
    return;
  }
}

// single frame inference
async function processOne(imgData, baseName) {
  const { inputTensor, ratio, pad, srcMat } = preprocessToTensor(
    imgData,
    MODEL_SIZE
  );

  // inference
  const feeds = { [INPUT_NAME]: inputTensor };
  const outputMap = await session.run(feeds);

  const scores = outputMap["scores"].data;
  const bboxes = outputMap["bboxes"].data;

  // pick best box near center
  const rows = scores.filter((v) => v !== 0).length;
  let minCenterDistance = 2.0;
  let box = { x1: 0, y1: 0, x2: MODEL_SIZE, y2: MODEL_SIZE };

  for (let i = 0; i < rows; i++) {
    const off = i * 4;
    const ncx = bboxes[off + 0];
    const ncy = bboxes[off + 1];
    const nw = bboxes[off + 2] + 0.05;
    const nh = bboxes[off + 3] + 0.05;

    const cx = ncx * MODEL_SIZE;
    const cy = ncy * MODEL_SIZE;
    const w = nw * MODEL_SIZE;
    const h = nh * MODEL_SIZE;

    const conf = scores[i];
    if (conf < SCORE_THRESH) continue;

    const centerDistance =
      (ncx - 0.5) * (ncx - 0.5) + (ncy - 0.5) * (ncy - 0.5);
    if (centerDistance < minCenterDistance) {
      minCenterDistance = centerDistance;
      box = {
        x1: cx - w / 2,
        y1: cy - h / 2,
        x2: cx + w / 2,
        y2: cy + h / 2,
      };
    }
  }

  // map box from letterboxed 320x320 back to 3840x2160
  // Note: here you hardcoded 3840x2160. If images can vary, replace with srcMat.cols/srcMat.rows.
  const origW = 3840;
  const origH = 2160;
  const mappedBox = deLetterbox(box, origW, origH, MODEL_SIZE, ratio, pad);

  // crop -> pngBytes
  const { buf, coord } = await cropToPNG(srcMat, mappedBox, origW);

  // cleanup the per-frame srcMat
  srcMat.delete();

  return {
    name: `${baseName}_crop.png`,
    pngBytes: buf,
    coord,
  };
}

// preprocess: letterbox -> grayscale -> RGBA -> CHW float32
function preprocessToTensor(imgData, size) {
  // assume fixed source resolution 3840x2160. adjust if needed:
  const inW = 3840;
  const inH = 2160;

  const srcMat = cv.matFromImageData(imgData);

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

  // pad_mat is size x size RGBA
  const chw = rgbaToCHWFloat32(pad_mat, size);

  const tensor = new ort.Tensor("float32", chw, [1, 3, size, size]);

  return { inputTensor: tensor, ratio: r, pad: { y: top, x: left }, srcMat };
}

// convert RGBA Mat -> Float32 CHW normalized
function rgbaToCHWFloat32(rgbaMat, size) {
  const data = rgbaMat.data; // Uint8Array RGBA
  const out = new Float32Array(3 * size * size);

  const plane = size * size;
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

// reverse letterbox coords back to original coords
function deLetterbox(b, origW, origH, size, ratio, pad) {
  const clamp01 = (v) => Math.max(0, Math.min(size, v));

  const x1n = (clamp01(b.x1) - pad.x) / ratio;
  const y1n = (clamp01(b.y1) - pad.y) / ratio;
  const x2n = (clamp01(b.x2) - pad.x) / ratio;
  const y2n = (clamp01(b.y2) - pad.y) / ratio;

  return {
    x1: clamp(x1n, 0, origW),
    y1: clamp(y1n, 0, origH),
    x2: clamp(x2n, 0, origW),
    y2: clamp(y2n, 0, origH),
  };
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

// crop and encode PNG
async function cropToPNG(srcMat, mappedBox, origW) {
  // first downscale full frame to 1920x1080
  cv.resize(srcMat, resize_mat, dsize1920x1080);

  const scale = 1920 / origW;
  const rx = Math.round(mappedBox.x1 * scale);
  const ry = Math.round(mappedBox.y1 * scale);
  const rw = Math.max(1, Math.round((mappedBox.x2 - mappedBox.x1) * scale));
  const rh = Math.max(1, Math.round((mappedBox.y2 - mappedBox.y1) * scale));

  // roi view
  const rect = new cv.Rect(rx, ry, rw, rh);
  const roiTmp = resize_mat.roi(rect);

  // copy roiTmp data into crop_mat, then free roiTmp
  roiTmp.copyTo(crop_mat);
  roiTmp.delete();

  // RGB -> RGBA
  cv.cvtColor(crop_mat, png_mat, cv.COLOR_RGB2RGBA, 0);

  const png_imgData = new ImageData(
    new Uint8ClampedArray(png_mat.data),
    png_mat.cols,
    png_mat.rows
  );

  const bmp = await createImageBitmap(png_imgData);

  const canvas = new OffscreenCanvas(rw, rh);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bmp, 0, 0, rw, rh);
  const blob = await canvas.convertToBlob({ type: "image/png" });

  const buf = await blob.arrayBuffer();
  const coord = {
    x1: mappedBox.x1,
    y1: mappedBox.y1,
    x2: mappedBox.x2,
    y2: mappedBox.y2,
  };
  return { buf, coord };
}
