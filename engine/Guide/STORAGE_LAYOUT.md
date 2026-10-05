# SnapAI Storage Layout — Where Every Piece of Data Lives

This document answers the question: **"If not server, where does it go?"**

SnapAI is architected so that nothing leaves the user's machine without
an explicit, user-initiated action (export, share, or sync). This file
is the authoritative map of every storage target the system writes to.

---

## TL;DR — Default Mode (no cloud)

| What | Where | Lifecycle |
|---|---|---|
| Live camera frames (raw) | RAM only — never touch disk | Discarded after processing |
| Encoded JPEG frames (sent to server) | RAM, then WebSocket, then RAM on server | Discarded after processing |
| Captured photos (wide + zoom) | `./captures/<session_id>/*.jpg` on local disk | Manual delete |
| Session metadata | `./backend/data/snappy.db` (SQLite) | Manual delete |
| Per-session albums | `./albums/<session_id>.json` | Manual delete |
| Application logs | `./logs/snappy.log` (rotating) | Auto-rotated |
| Model weights (CLIP, NIMA, YOLO, HSEmotion) | `./backend/models/*.pt`, `*.npz` | Read-only |
| Browser session state | `STATE` JS object in tab memory | Discarded on tab close |
| Browser local storage | **NONE** | n/a |
| Browser IndexedDB | **NONE** | n/a |
| Cookies | **NONE** | n/a |
| External network calls during a session | **NONE** | n/a |

The local server (default `http://localhost:8765`) runs on the same
device that opens the browser. Nothing on this list crosses a network
boundary the user did not explicitly initiate.

---

## Detailed Map

### 1. Camera Stream (browser → server)

- **Source**: `navigator.mediaDevices.getUserMedia()` — direct hardware
  access governed by the browser's permission prompt.
- **In-memory only**: The raw video stream never touches disk. The
  `<video>` element holds a reference to the `MediaStream`; the
  `<canvas>` element samples individual frames at the adaptive FPS.
- **Encoding location**: 2026 — `snappy-worker.js` (a Web Worker)
  encodes frames via `OffscreenCanvas.convertToBlob()`. The encoded
  bytes are converted to base64 and posted back to the main thread.
- **Transport**: WebSocket frame `{type:"frame", frame:<base64 JPEG>}`
  to the local server. **No third-party network endpoint is contacted.**

### 2. Server-Side Frame Processing (in RAM)

- **File**: `backend/api/pipeline.py::process_frame`
- The base64 frame is decoded into a NumPy array, run through the model
  cascade (face detection → gaze → moment → CLIP → ensemble), and
  **discarded** once the response is built.
- Only frames that the ensemble decides to *capture* are written to
  disk. Every other frame is GC'd within microseconds.

### 3. Captured Photo Files (on disk)

- **Location**: `./captures/<session_id>/snap_<NNNN>_<timestamp>.jpg`
- **Format**: JPEG, quality 92, OpenCV `cv2.imwrite`.
- **Companion zoom**: If a meaningful ROI was detected, an upscaled
  `<base>_zoom.jpg` is also saved beside the wide shot.
- **Permissions**: Inherits the user's filesystem permissions — no
  special ACLs, no SUID, no daemon writing as another user.
- **No EXIF GPS injection**: We do not write the device's location into
  the EXIF tags. Camera-supplied EXIF tags are passed through as-is.

### 4. Session + Photo Metadata (SQLite)

- **Location**: `./backend/data/snappy.db`
- **File**: `backend/storage/sqlite_store.py` (or equivalent — see
  `backend/storage/` for the actual layer).
- **Schema** (high level): `sessions`, `photos`, `feedback`,
  `auto_classes`, `prompts`.
- **What's stored**: filenames, scores, timestamps, ensemble
  contributions, feedback (👍/👎), discovered class names.
- **What's NOT stored**: face embeddings, biometric IDs, raw frames,
  cloud sync tokens, third-party credentials.

### 5. Album Manifests

- **Location**: `./albums/<session_id>.json`
- Plain JSON listing the curated subset and ordering.

### 6. Application Logs

- **Location**: `./logs/snappy.log` + rotated `.1`, `.2`, …
- **What's logged**: model load events, capture decisions (no pixel
  data), exceptions, performance warnings.
- **Operator can disable**: set `SNAPPY_LOG_LEVEL=WARNING` to suppress
  per-frame debug lines.

### 7. Browser-Side Persistence

| API | Used by SnapAI? |
|---|---|
| `localStorage` | No |
| `sessionStorage` | No |
| `IndexedDB` | No |
| `Cache Storage` | No (no service worker registered) |
| Cookies | No |

Closing the browser tab leaves no SnapAI artifact behind on the client
device. The only state lives on the local server's filesystem.

### 8. Model Weights (read-only)

- `backend/models/yolov8n-face.pt` — YOLOv8 face detector
- `backend/models/face_landmarker.task` — MediaPipe FaceLandmarker
- `backend/models/clip_class_centroids.npz` — fine-tuned CLIP centroids
- Other models load weights from `~/.cache/huggingface/` or
  `~/.cache/torch/` (HSEmotion, transformers CLIP) — these are
  shared across applications by the underlying frameworks.

---

## Optional Cloud Sync (NOT enabled by default)

If a future build wires up cloud sync, the following invariants are
required by design:

1. **Opt-in, off by default.** The user must explicitly enable sync.
2. **End-to-end encryption.** The client encrypts each photo with a
   key derived from a user passphrase + hardware-backed entropy
   (Secure Enclave / StrongBox). The server stores ciphertext only.
3. **No image data in telemetry.** Performance/error telemetry, if
   added, must contain numeric metrics only — never pixel data,
   filenames, or session IDs.
4. **User-visible network indicator.** A UI element must show
   when, and what, is being uploaded — same pattern as the perf
   overlay (press "p" to view).

None of this is implemented yet. Today the answer to "where does it
go?" is **nowhere except the local disk that hosts the local server**.

---

## Auditing for Yourself

To verify these claims on your own machine:

```bash
# 1. See what files SnapAI writes during a session.
sudo fs_usage -w -f filesys python3 | grep snappy_final
#    (macOS — Linux: use `inotifywatch ./snappy_final/`)

# 2. See what network endpoints the browser talks to.
#    Open DevTools → Network tab. You should see only requests to
#    localhost:8765 and (one-time) model downloads on first run.

# 3. See what the server connects to outbound.
sudo lsof -i -P | grep python3
#    During an active session this should show only the WebSocket
#    listener and any model-download connections to huggingface.co /
#    storage.googleapis.com on first run.
```

If you see any unexpected outbound connection, file an issue — that
would be a privacy regression.
