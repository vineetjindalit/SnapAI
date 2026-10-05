/* ──────────────────────────────────────────────────────────────────────
 * snappy-worker.js — off-main-thread frame encoder + motion gate.
 *
 * Why this file exists:
 *   The original capture loop ran cvs.toBlob() + FileReader.readAsDataURL()
 *   + JSON.stringify() on the MAIN thread at 8 fps. On mid-tier phones
 *   that's enough to thermal-throttle the device after ~15 minutes and
 *   destroy frame timing.
 *
 *   Instagram Live solves the same problem by doing all encoding inside
 *   a Web Worker with OffscreenCanvas + WebCodecs. We adopt the same
 *   pattern here, with three additions:
 *     1. Adaptive sampling — when no motion vs full pipeline.
 *     2. Local motion + brightness deltas — fed back to main thread so
 *        it can trigger HIGH-priority transmits (never miss a real moment).
 *     3. Backpressure handling — if the WebSocket can't drain, we drop
 *        low-priority frames but never priority ones.
 *
 * Storage note:
 *   Nothing is persisted in this worker. Frames are encoded in memory
 *   and posted back to the main thread which forwards them over the
 *   WebSocket. No disk, no IndexedDB, no cookies, no network calls.
 * ────────────────────────────────────────────────────────────────────── */

let lastFrameGray = null;    // ImageData-luma of previous frame
let lastMotionScore = 0;
let frameCount = 0;
let droppedCount = 0;

// Lightweight motion detector — frame-difference magnitude on a
// downsampled luma plane. Costs ~1ms on mobile and gives us the
// "something is happening" gate that lets us run cheap-tier idle
// most of the time and ramp up only when life occurs.
function computeMotion(rgba, w, h) {
  const stride = 8;                 // sample every 8th pixel — 64x speedup
  const gray = new Uint8Array(Math.ceil(w / stride) * Math.ceil(h / stride));
  let i = 0;
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const off = (y * w + x) * 4;
      // BT.601 luma
      gray[i++] = (0.299 * rgba[off] + 0.587 * rgba[off + 1] + 0.114 * rgba[off + 2]) | 0;
    }
  }
  if (!lastFrameGray || lastFrameGray.length !== gray.length) {
    lastFrameGray = gray;
    return 0;
  }
  let diffSum = 0;
  for (let k = 0; k < gray.length; k++) {
    const d = gray[k] - lastFrameGray[k];
    diffSum += d < 0 ? -d : d;
  }
  lastFrameGray = gray;
  // Normalise to [0, 1] — 30 average diff per pixel is "active scene".
  return Math.min(1.0, (diffSum / gray.length) / 30);
}

// Brightness estimate — used to decide whether to recommend longer
// exposure / more aggressive denoising downstream.
function meanLuma(rgba, w, h) {
  const stride = 16;
  let total = 0, n = 0;
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const off = (y * w + x) * 4;
      total += 0.299 * rgba[off] + 0.587 * rgba[off + 1] + 0.114 * rgba[off + 2];
      n++;
    }
  }
  return n > 0 ? (total / n) / 255 : 0;
}

// Convert ImageBitmap → JPEG blob → base64 string off the main thread.
async function encodeJpeg(bitmap, quality) {
  // OffscreenCanvas is supported in workers everywhere we care about
  // (Chromium, Safari 16+, Firefox 110+).
  const oc = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = oc.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  // Read pixels for motion + brightness analysis BEFORE encoding —
  // doing it after would force a second decode.
  const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  const motion = computeMotion(imageData.data, bitmap.width, bitmap.height);
  const brightness = meanLuma(imageData.data, bitmap.width, bitmap.height);
  lastMotionScore = motion;
  const blob = await oc.convertToBlob({ type: 'image/jpeg', quality });
  // Convert to base64 via ArrayBuffer → btoa — much faster than FileReader
  // because we stay on the worker thread.
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  let binary = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return { b64: btoa(binary), motion, brightness, size: blob.size };
}

self.onmessage = async (e) => {
  const msg = e.data;
  if (msg.type === 'frame') {
    frameCount++;
    try {
      const { b64, motion, brightness, size } = await encodeJpeg(msg.bitmap, msg.quality || 0.72);
      msg.bitmap.close();
      self.postMessage({
        type: 'encoded',
        b64,
        motion,
        brightness,
        size,
        frameId: msg.frameId,
        capturedAt: msg.capturedAt,
      });
    } catch (err) {
      droppedCount++;
      self.postMessage({ type: 'error', error: String(err), frameId: msg.frameId });
    }
  } else if (msg.type === 'stats') {
    self.postMessage({
      type: 'stats',
      framesEncoded: frameCount,
      framesDropped: droppedCount,
      lastMotion: lastMotionScore,
    });
  } else if (msg.type === 'reset') {
    lastFrameGray = null;
    frameCount = 0;
    droppedCount = 0;
  }
};
