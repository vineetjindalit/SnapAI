# SnapAI v2.7 — Install & Run

## React frontend (NEW in v2.7)

The polished UI is at [`frontend-react/`](frontend-react/). Two modes
(User + Developer), Tailwind-styled, voice prompts, hot reload.

```bash
# 1. Install Node 18+ (https://nodejs.org/)
node --version            # v18 or higher

# 2. Build the React bundle (~30s first time)
cd frontend-react
npm install
npm run build

# 3. SnapAI backend auto-detects the built app and serves it at /
cd ..
python3 run.py
# Open http://localhost:8765 — you'll see the new React UI.

# Dev workflow with hot reload (run BOTH in separate terminals):
#   Terminal A:   python3 run.py             (backend on :8765)
#   Terminal B:   cd frontend-react && npm run dev   (frontend on :5173)
#   Open http://localhost:5173
```

The legacy single-file UI (`frontend/index.html`) is still available
at `http://localhost:8765/legacy` for reference.

---


End-to-end AI event photographer with a tiered model zoo. Every tier is
optional — each one just raises accuracy. The server boots and runs even
with only the core image pipeline.

## Accuracy targets per tier

| Tier | Models | Pipeline accuracy | Disk |
|---|---|---|---|
| 1 (core) | OpenCV, NumPy, scikit-learn | ~60% | ~50 MB |
| 2 (+MediaPipe) | FaceMesh + iris + blendshapes | ~80% | ~150 MB |
| **3 (+production CV)** | **YOLOv8-face + NIMA + HSEmotion** | **~90%** | ~250 MB |
| 4 (+CLIP) | + real CLIP ViT-B/32 moment detection | **~92%+ on moments** | ~1.2 GB |

Tier 3 is the "production-grade" stack the PDF roadmap calls out. Tier 4
is the moment-detection upgrade.

## 1. Install everything

```bash
cd snappy_final
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

That installs all four tiers. First run may take several minutes as the
models auto-download:

| Model | First-call download | Trigger |
|---|---|---|
| YOLOv8n-face weights | ~6 MB from GitHub | first frame |
| NIMA aesthetic weights | ~95 MB via pyiqa | first frame |
| HSEmotion ONNX | ~5 MB | first frame with a face |
| CLIP ViT-B/32 | ~600 MB from HuggingFace | first frame |

Subsequent runs are instant (cached in `~/.cache/`).

## 2. Install only what you want

If you don't want the heavy CLIP/torch download yet, stop after tier 3:

```bash
pip install opencv-python numpy Pillow scikit-learn scipy mediapipe \
            ultralytics pyiqa hsemotion-onnx onnxruntime
# CLIP off; heuristic VLMDetector still runs
```

Or just core + MediaPipe for a quick demo:

```bash
pip install opencv-python numpy Pillow scikit-learn scipy mediapipe
```

The server will still start. `curl localhost:8765/health` reports which
backends are live:

```json
{
  "status": "ok",
  "version": "v2.3",
  "models": {
    "face":      {"backend": "mediapipe+yolo"},
    "clip":      {"available": true, "error": null},
    "nima":      {"available": true, "error": null},
    "hsemotion": {"available": true, "error": null}
  }
}
```

## 3. Run

```bash
python3 run.py
# → http://localhost:8765
```

## 4. What each model does

### Face: YOLOv8-face + MediaPipe FaceMesh
- **YOLOv8-face** (community fine-tune of YOLOv8 on WIDER FACE, ~6 MB) —
  high-recall detector, used as a crowd booster when MediaPipe finds
  fewer than 3 faces. Catches small/profile faces in group shots.
- **MediaPipe FaceMesh** — 478 landmarks per face including iris, plus
  52 blendshapes used for emotion + smile.
- Falls back to OpenCV Haar cascades if both are missing.

### Shot quality: NIMA (Neural Image Assessment)
- Pretrained on the AVA dataset (~250 K human-rated photos).
- Replaces the Laplacian + heuristic stack — outputs a learned aesthetic
  score in [0, 1].
- Blended into the legacy heuristic at weight 0.45 when available, so
  the system still works without it.

### Emotion: HSEmotion + MediaPipe blendshapes
- **HSEmotion** is an AffectNet-pretrained 8-class FER model in ONNX
  (~5 MB, ~10 ms per face on CPU). State-of-the-art for in-the-wild FER.
- Blended 70/30 with the blendshape-derived emotion (which is free since
  we already have the blendshapes).

### Moment detection: real CLIP ViT-B/32
- Pretrained CLIP from OpenAI via HuggingFace transformers.
- Cosine similarity between frame embedding and 11 prompt embeddings.
- Per-session "kept centroid": the running mean of frames you 👍 — future
  frames close to that centroid are auto-promoted (few-shot personalisation).

### Gaze: MediaPipe iris + geometric estimator
- Iris landmarks (468–477) relative to eye corners give a per-face
  "looking at camera" probability, aggregated to a group gaze ratio.
- L2CS-Net is the next upgrade (PDF Phase 2). Drop a torch L2CS model
  into `gaze_estimator.py` and consume it the same way as NIMA — the
  interface is already there.

## 5. New endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Per-model backend status + version |
| `POST` | `/sessions/{sid}/feedback` | Body `{photo_url, kept, moment}` — record 👍/👎 |
| `GET`  | `/sessions/{sid}/learning` | Per-session keep rates |

## 6a. GPU acceleration

SnapAI auto-detects and uses, in order of preference:
**CUDA (NVIDIA) → MPS (Apple Silicon) → CPU**.

### NVIDIA CUDA (workstation, AWS EC2, Colab)

```bash
# 1. Install matching torch wheel (CUDA 12.1 example)
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 2. Verify
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True  NVIDIA GeForce RTX 4070

# 3. Run — SnapAI picks it up automatically
python3 run.py
curl localhost:8765/health | python3 -m json.tool
# → "compute": {"device":"cuda","cuda_device_name":"NVIDIA…","cuda_mem_gb":12.0}
```

Speedup vs CPU on a ~3060/4070-class card:
- CLIP frame embed: 80 ms → 8 ms
- NIMA aesthetic:   60 ms → 6 ms
- Full per-frame:  ~150 ms → ~25 ms (40 fps possible)

### Apple Silicon (M1/M2/M3)

```bash
pip install torch                     # default wheel includes MPS
python3 -c "import torch; print(torch.backends.mps.is_available())"
# → True
python3 run.py
```

About 3-5× CPU speedup. NIMA via pyiqa works on MPS; CLIP works on MPS.

### Force CPU (or pin device)

```bash
SNAPPY_DEVICE=cpu     python3 run.py
SNAPPY_DEVICE=cuda:1  python3 run.py    # second GPU
```

### AWS EC2 quick recipe (g5.xlarge — A10G, ~$1/hr)

```bash
# Ubuntu 22.04 deep-learning AMI already has CUDA + drivers
git clone <your-repo>; cd snappy_final
pip install -r requirements.txt
SNAPPY_CONTINUOUS_LEARNING=1 python3 run.py --port 8765
# Open security group :8765, point browser at the public IP
```

## 6. Tuning (no code changes)

```bash
SNAPPY_MIN_CAPTURE_INTERVAL=1.5   \
SNAPPY_BEST_SHOT_THRESHOLD=0.60   \
SNAPPY_LOG_LEVEL=DEBUG            \
python3 run.py
```

See `backend/utils/config.py` for the full list.

## 7. Troubleshooting

**"YOLO not loaded" but ultralytics is installed**
First run downloads `yolov8n-face.pt` from
<https://github.com/akanametov/yolov8-face/releases>. If your network
blocks GitHub, manually drop the weights into
`backend/models/yolov8n-face.pt`.

**NIMA download fails**
pyiqa pulls weights from cloud storage. Pre-download manually:
```python
import pyiqa; pyiqa.create_metric("nima")
```

**HSEmotion init error**
Some Linux wheels need `libgl1`. Install: `apt install libgl1`.

**CLIP "init_error: torch not installed"**
On Apple Silicon: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## 8. Architecture at a glance

```
         ┌───────────────────────────────────────────────────┐
Frame ─▶ │  FaceProvider (YOLOv8-face → MediaPipe → Haar)    │
         └─┬──────────────┬──────────────┬───────────────────┘
           │ landmarks    │ boxes        │ blendshapes
           ▼              ▼              ▼
   ┌──────────────┐ ┌─────────────┐ ┌───────────────────────┐
   │ GazeEstimator│ │ ShotQuality │ │ Emotion (blendshape)  │
   │ (iris-based) │ │ + NIMA blend│ │ + HSEmotion FER blend │
   └──────┬───────┘ └──────┬──────┘ └───────────┬───────────┘
          │                │                    │
          └────────────────┼────────────────────┘
                           ▼
                  ┌──────────────────┐
                  │ MomentPredictor  │
                  │ + CLIP ViT-B/32  │
                  │ + per-session    │
                  │   kept centroid  │
                  └────────┬─────────┘
                           ▼
                ┌───────────────────────┐
                │ OnlineLearner (Beta)  │
                │ scaled by 👍/👎 rate │
                └──────────┬────────────┘
                           ▼
                       Capture decision
```
