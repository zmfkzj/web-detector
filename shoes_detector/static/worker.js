// worker.js
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');

let session = null;

// ====== 모델/전처리 파라미터 (모델에 맞게 조정하세요) ======
const INPUT_NAME = 'input';       // 예: YOLOv5/8 변형의 입력명
const MODEL_SIZE = 320;            // 정사각 입력 가정
const MEAN = [123.675, 116.28, 103.53];      // 필요 시 [0.485,0.456,0.406]
const STD = [58.395, 57.12, 57.375];      // 필요 시 [0.229,0.224,0.225]
const SCORE_THRESH = 0.97;

// postMessage helper
const send = (type, payload = {}) => postMessage({ type, ...payload });

onmessage = async (e) => {
    const msg = e.data;
    try {
        if (msg.type === 'init') {
            // 초기화: 세션 생성
            const modelBytes = msg.modelBytes;
            session = await ort.InferenceSession.create(modelBytes, {
                executionProviders: ['webgpu', 'wasm'], // 가능하면 WebGPU 우선
                graphOptimizationLevel: 'all',
            });
            send('ready');
        } else if (msg.type === 'process') {
            if (!session) throw new Error('Session not initialized');
            const { imageBytes, baseName } = msg;
            const crops = await processOne(imageBytes, baseName);
            send('result', { crops });
        } else if (msg.type === 'finish') {
            send('done');
        }
    } catch (err) {
        send('error', { error: String(err?.message || err) });
    }
};

// 단일 이미지 처리
async function processOne(imageBytes, baseName) {
    // 디코드
    const bmp = await createImageBitmap(new Blob([imageBytes]));
    const { inputTensor, ratio, pad } = await preprocessToTensor(bmp, MODEL_SIZE);
    // 추론
    const feeds = { [INPUT_NAME]: inputTensor };
    const outputMap = await session.run(feeds);
    // 출력명은 모델에 따라 다릅니다. 보통 첫 키 사용(단일 출력 가정)
    //     output_names=["classes", "scores", "bboxes"]
    // const classes = outputMap["classes"].data; // Float32Array 등
    const scores = outputMap["scores"].data; // Float32Array 등
    const bboxes = outputMap["bboxes"].data; // Float32Array 등

    // ---- 후처리 (모델 맞게 조정) ----
    const rows = scores.filter((v, i, a) => v != 0).length;
    let minCenterDistance = 1.0;
    let box = {};
    for (let i = 0; i < rows; i++) {
        const off = i * 4;
        const cx = bboxes[off + 0] * MODEL_SIZE, cy = bboxes[off + 1] * MODEL_SIZE, w = bboxes[off + 2] * MODEL_SIZE, h = bboxes[off + 3] * MODEL_SIZE
        const conf = scores[i];
        if (conf < SCORE_THRESH) continue;
        // 클래스 확률 중 최대값을 score로 쓰려면 아래처럼:
        // let best = 0, bestId = -1;
        // for (let k=5; k<stride; k++) if (out[off+k] > best) { best = out[off+k]; bestId = k-5; }
        // const score = conf * best;
        const centerDistance = Math.abs(cx - 0.5);
        if (centerDistance < minCenterDistance) {
            minCenterDistance = centerDistance
            const x1 = cx - w / 2, y1 = cy - h / 2, x2 = cx + w / 2, y2 = cy + h / 2;
            box = { x1, y1, x2, y2 }
        }

    }

    // letterbox → 원본 좌표로 역투영
    const mappedBox = deLetterbox(box, bmp.width, bmp.height, MODEL_SIZE, ratio, pad);

    // 크롭 PNG 생성
    const pngBytes = await cropToJPG(bmp, mappedBox.x1, mappedBox.y1, mappedBox.x2, mappedBox.y2);
    const crop = { name: `${baseName}_crop.jpg`, pngBytes };

    bmp.close?.();
    return crop;
}

// === 전처리: letterbox resize + normalize → Float32 CHW 텐서 ===
async function preprocessToTensor(bmp, size) {
    const inW = bmp.width, inH = bmp.height;
    const r = Math.min(size / inW, size / inH);
    const newW = Math.round(inW * r);
    const newH = Math.round(inH * r);
    const padW = size - newW;
    const padH = size - newH;

    const canvas = new OffscreenCanvas(size, size);
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'rgb(0,0,0)'; // YOLO letterbox 기본
    ctx.fillRect(0, 0, size, size);
    const dx = Math.floor(padW / 2);
    const dy = Math.floor(padH / 2);
    ctx.drawImage(bmp, 0, 0, inW, inH, dx, dy, newW, newH);

    const imgData = ctx.getImageData(0, 0, size, size);
    const data = imgData.data;
    const chw = new Float32Array(3 * size * size);

    let p = 0, c0 = 0, c1 = size * size, c2 = 2 * size * size;
    for (let i = 0; i < size * size; i++) {
        const r = data[p++], g = data[p++], b = data[p++];
        p++; // skip alpha
        chw[c0++] = (r - MEAN[0]) / STD[0];
        chw[c1++] = (g - MEAN[1]) / STD[1];
        chw[c2++] = (b - MEAN[2]) / STD[2];
    }

    const tensor = new ort.Tensor('float32', chw, [1, 3, size, size]);
    return { inputTensor: tensor, ratio: r, pad: { x: dx, y: dy } };
}

// === letterbox 좌표 -> 원본 좌표 ===
function deLetterbox(b, origW, origH, size, ratio, pad) {
    const x1 = clamp((b.x1 - pad.x) / ratio, 0, size);
    const y1 = clamp((b.y1 - pad.y) / ratio, 0, size);
    const x2 = clamp((b.x2 - pad.x) / ratio, 0, size);
    const y2 = clamp((b.y2 - pad.y) / ratio, 0, size);
    // 위에서 size 기준으로 그린 게 아니라 letterbox 캔버스가 size×size라
    // 역투영 후 원본 해상도로 다시 맵핑
    return {
        x1: clamp(x1, 0, origW),
        y1: clamp(y1, 0, origH),
        x2: clamp(x2, 0, origW),
        y2: clamp(y2, 0, origH),
        score: b.score,
    };
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }


// === OffscreenCanvas 크롭 → PNG 바이트(ArrayBuffer) ===
async function cropToJPG(bmp, x1, y1, x2, y2) {
    const w = Math.max(1, Math.round(x2 - x1));
    const h = Math.max(1, Math.round(y2 - y1));
    const c = new OffscreenCanvas(w, h);
    const ctx = c.getContext('2d');
    ctx.drawImage(bmp, Math.round(x1), Math.round(y1), w, h, 0, 0, w, h);
    const blob = await c.convertToBlob({ type: 'image/jpg' });
    return await blob.arrayBuffer(); // transferable
}
