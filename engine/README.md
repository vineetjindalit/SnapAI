# 📸 SnapAI v2.5 — AI Event Photography

End-to-end AI system that watches your event camera, detects key moments
(cake cutting, ring ceremony, group gaze, etc.), auto-captures the best
photos, **learns from your 👍/👎 feedback**, **discovers new moment classes
on its own**, and accepts **voice or text prompt changes mid-event**.

> **Status**: Phase 1 ≈ 90 % · Phase 2 ≈ 80 % (see roadmap below)
> · Tier-3 model zoo (YOLOv8-face + NIMA + HSEmotion + CLIP) wired in
> · Self-training loop + auto-classifier live
> · GPU auto-detect (CUDA / MPS / CPU)
> · 23-file backend, full unit test suite
> · **NEW (v2.5)**: priority-tier capture system + adaptive FPS +
>   Web Worker encoding + on-device privacy guarantees

## Document map
- ~~INSTALL.md~~ — **referenced but not present in this folder** (dead link)
- ~~TRAINING.md~~ — **referenced but not present in this folder** (dead link; see `TRAINING_GUIDE.md` and `backend/training/` instead)
- [ARCHITECTURE.md](ARCHITECTURE.md) — module layout + data flow
- ~~STORAGE_LAYOUT.md~~ — **referenced but not present in this folder** (dead link)
- this file — phase status checklist + quick start + v2.5 changelog

## What changed in v2.5 (2026)

**Headline:** SnapAI now guarantees that high-value moments — prompt-
matched events like "cake cutting" and subjects deliberately looking
at the camera — are **never** dropped due to thermal throttling,
battery throttling, or cooldown. Lower-priority frames may be sampled
less often; priority moments always run the full pipeline.

| Change | File(s) | Why |
|---|---|---|
| **Priority-tier classifier** | `backend/models/event_engine.py` | Classifies each frame as CRITICAL / HIGH / ELEVATED / NORMAL. Critical = prompt-matched + CLIP-confirmed. High = subjects looking at camera. Threshold is lowered for higher tiers (0.28 / 0.35 / 0.38 / 0.42) so we never skip a real moment. |
| **Cooldown bypass for priority frames** | `backend/api/pipeline.py` | The `session.quality.can_capture()` cooldown previously gated every capture. It now only gates NORMAL / ELEVATED tiers. CRITICAL / HIGH bypass it so cake-cutting and intentional-gaze frames always fire. |
| **Adaptive FPS + thermal proxy** | `frontend/index.html` (`maybeAdjustFps`, `classifyThermal`) | The capture loop drops baseline FPS when the device shows sustained latency growth or dropped frames (thermal proxy) — but only for non-priority frames. Targets <7% battery/hour on mid-tier mobile. |
| **Web Worker frame encoding** | `frontend/snappy-worker.js` (new) | All JPEG encoding moves to a Web Worker with OffscreenCanvas. Main thread stays responsive. Inspired by Instagram Live's long-session capture pipeline. |
| **`requestVideoFrameCallback` loop** | `frontend/index.html` (`startCapture`) | Replaces the fixed `setInterval(8fps)` with a callback aligned to actual decoded video frames. Cheaper, smoother, doesn't oversample. |
| **Local motion gate** | `frontend/snappy-worker.js` (`computeMotion`) | Frame-difference luma compute inside the worker. Server sees `client_motion` and `client_brightness` per frame and can use them as pre-filters. |
| **Battery + thermal + latency overlay** | `frontend/index.html` (perf-overlay div, `updatePerfOverlay`) | Press `p` during a session to see live FPS target, thermal proxy, latency, inflight queue, motion, priority tier, and battery drain. |
| **Backpressure handling** | `frontend/index.html` (`STATE.inflightFrames`, `maxInflight`) | If the WebSocket is congested we drop NORMAL frames before the queue blows. CRITICAL / HIGH frames bypass this drop. |
| **Per-session perf summary** | `frontend/index.html` (`endSession` `console.table`) | At session end the console prints sec, frames sent/dropped, avg latency, battery used, battery-used-per-hour. Use this for the "test on real device" phase. |
| **Storage map document** | `STORAGE_LAYOUT.md` (new) | Lists every place data is written. Confirms there is no cloud / IndexedDB / cookies / external network call during a session. |

### Where data goes (TL;DR)

There is **no remote server**. Everything is local to the machine that
runs `python3 run.py`:

| Data | Location |
|---|---|
| Live camera frames | RAM only — discarded after processing |
| Captured photos | `./captures/<session_id>/*.jpg` |
| Session metadata | `./backend/data/snappy.db` (SQLite) |
| Albums | `./albums/<session_id>.json` |
| Logs | `./logs/snappy.log` |
| Browser persistence | **NONE** (no localStorage / IndexedDB / cookies) |

See [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md) for the full audit including
how to verify these claims with `fs_usage` / `lsof`.

### Performance & Priority — what to test on real hardware

The v2.5 changes are designed around two principles:

1. **Accuracy is non-negotiable** for priority moments. Thermal /
   battery throttling **only** affects NORMAL / ELEVATED tiers.
2. **Everything is measurable.** Press `p` during a session to see
   the live FPS, latency, drop count, battery, and thermal proxy.
   At session end, the console prints a single-row summary table.

**Testing checklist (do this on at least one phone + one laptop):**

- [ ] Run a 30-minute session in a normal indoor setting. Confirm the
      `batteryUsedPerHour` summary stays under 10 % (target: 7 %).
- [ ] Confirm `thermalProxy` stays at `normal` or `warm` — never `hot`
      for sustained periods.
- [ ] Trigger a CRITICAL moment: type "cake cutting" into the prompt
      and present a cake. Confirm `priority: critical` appears in the
      overlay AND the capture fires regardless of cooldown.
- [ ] Trigger a HIGH moment: have subjects deliberately look into the
      camera. Confirm `priority: high` appears AND capture fires.
- [ ] Stress test: run on battery-saver mode at <20 % battery. Confirm
      the FPS auto-drops to 2-3 but priority frames still trigger.
- [ ] On a low-end laptop, run for 2 hours. Latency should stay below
      ~280ms average; drop rate should stay below 20 %.

If any of these fail, the issue is most likely in `maybeAdjustFps()`
(too aggressive throttle) or `_classify_priority_tier()` (priority
detection too strict). Both are pure functions and easy to tune.

### Env-var tuning knobs

```bash
# Priority-tier thresholds (lower = more likely to capture)
SNAPPY_THR_CRITICAL=0.28
SNAPPY_THR_HIGH=0.35
SNAPPY_THR_ELEVATED=0.38
SNAPPY_CAPTURE_THRESHOLD=0.42   # NORMAL tier baseline

# Existing ensemble weights (unchanged from v2.4)
SNAPPY_W_CLIP=0.30   SNAPPY_W_QUALITY=0.18   ...
```

## Quick Start

```bash
# Tier 1 only (heuristic mode, ~60% accuracy)
pip install opencv-python numpy Pillow scikit-learn scipy
cd snappy_final && python3 run.py

# OR: full production stack (~90% accuracy)
pip install -r requirements.txt
python3 run.py
# Browser opens at http://localhost:8765
```

## Roadmap status (vs PDF)

### Phase 1 — Foundation Hardening (Month 1–2) ≈ 90%
- [x] MediaPipe FaceMesh replacing Haar — `face_provider.py`
- [x] Iris-based gaze (replaces iris-centering hack) — `gaze_estimator.py`
- [x] SQLite session persistence — `utils/persistence.py`
- [x] Logging pipeline (rotating file) — `utils/logging_setup.py`
- [x] Unit tests — `tests/`
- [x] Threshold tuning hooks (env vars) — `utils/config.py`
- [ ] Multi-threaded 30 fps frame processing  *(deferred)*

### Phase 2 — AI Model Upgrades (Month 3–5) ≈ 75%
- [x] Real CLIP ViT-B/32 integration — `clip_engine.py`
- [x] CLIP class-centroid fine-tune — `training/finetune_clip.py`
- [x] Dataset bootstrapper (1.5K labelled photos, multi-source) — `training/dataset_bootstrap.py`
- [x] ShotQualityNet (NIMA on AVA) — `nima_aesthetic.py`
- [x] YOLOv8-face — `yolo_face.py`
- [x] HSEmotion (AffectNet FER) — `hsemotion_fer.py`
- [x] Online learning from feedback — `online_learner.py`
- [x] **Auto-classifier (discovers new moment classes)** — `auto_classifier.py`
- [x] **Continuous learning thread** — `training/continuous_learning.py`
- [x] **Voice + text live prompt updates** — `prompt_router.py` + Web Speech UI
- [x] **GPU auto-detect (CUDA / MPS / CPU)** — `gpu/device.py`
- [ ] Confidence calibration  *(deferred)*
- [ ] A/B harness  *(deferred)*

## What Was Fixed (v2)

| Bug | Root Cause | Fix |
|---|---|---|
| `bool_ not JSON serializable` | `np.bool_` ≠ Python `bool` | `safe_types.py` converts all numpy types |
| `float64 not serializable` | Same — all numpy numerics | `safe()` recursive converter |
| No event detection | v1 only did faces | VLMDetector with 11 moment profiles |
| Poor capture triggering | Threshold too high | Multi-trigger: quality + gaze + moment + prediction |
| No frame history | Not implemented | 60-frame rolling gallery shown in UI |
| FastAPI not installable | Network restricted | Pure stdlib asyncio server |

## Architecture

**This diagram describes the old v1 pipeline and is out of date — see
[ARCHITECTURE.md](ARCHITECTURE.md) for the real three-pillar system
(`event_engine.py`'s weighted ensemble + priority tiers, `host_group.py`
personalization, `album_generator.py` curation), which is what the code
in `backend/api/pipeline.py` actually runs.**

## Moment Profiles (Prompt-Driven)

| Prompt keyword | Detected by |
|---|---|
| cake cutting | Bright blobs (candles) + brown/white colors + motion |
| ring ceremony | Gold/silver hue + low motion + face present |
| first dance | Stage lights + moderate motion + faces |
| bouquet toss | Red/green hues + high motion |
| candle blowing | Orange flame blobs + face present |
| group photo | 3+ faces + low motion |
| champagne toast | Golden color + faces |
| confetti burst | High color variance + high motion |
| sports action | Very high motion |
| hug moment | 2+ faces close + low motion |

## API Reference

```
POST /sessions          → Create session
                          Body: {event_name, event_type, prompt}
                          Returns: {session_id, ws_url}

GET  /sessions/{id}/stats   → Live stats
GET  /sessions/{id}/photos  → All captured photos  
GET  /sessions/{id}/gallery → Frame gallery (last 60)
POST /sessions/{id}/album   → Generate curated album
DEL  /sessions/{id}         → End session

WS   /ws/{session_id}       → Real-time frame stream
     Send: {"type":"frame","frame":"<base64>"}
     Recv: {type, frame_id, analysis, gaze, moment, prediction, captured, ...}
```

## Known Limitations & How to Fix

**This table is stale v1 copy and contradicts the v2.5 changelog and roadmap
checklist above it — most rows are already fixed in the current code.**
Verified against the actual backend on 2026-10-01:

| Limitation (as originally written) | Actual current state |
|---|---|
| Haar cascades for faces | **Fixed** — MediaPipe FaceLandmarker (`face_provider.py`), per Phase 1 checklist |
| VLM uses visual heuristics not real CLIP | **Fixed** — real CLIP ViT-B/32 (`clip_engine.py`) + a fused ensemble (`event_engine.py`), per Phase 2 checklist |
| Single-threaded server | **Fixed** — asyncio worker pool (`api/pipeline.py`), not FastAPI |
| No auth/sessions database | **Fixed and then some** — SQLite persistence, plus a full JWT auth system (`backend/auth/`) and Stripe billing (`backend/billing/`) that aren't mentioned anywhere else in these docs |
| No mobile app | **Still true** — `mobile/` exists in the repo but wasn't reviewed as part of this pass |

Real, still-open gaps found in this review instead: `INSTALL.md`, `TRAINING.md`,
and `STORAGE_LAYOUT.md` are linked from the doc map above but don't exist in
this folder.

## File Structure

```
snappy_final/
├── run.py                         ← Start here
├── requirements.txt
├── TRAINING_GUIDE.md
├── backend/
│   ├── utils/safe_types.py        ← Numpy JSON fix (critical)
│   ├── models/
│   │   ├── clip_moment_detector.py ← VLM moment detection (11 moments)
│   │   ├── shot_quality.py         ← Frame quality scoring
│   │   ├── gaze_detector.py        ← Multi-person gaze
│   │   ├── moment_predictor.py     ← Trend analysis + preroll
│   │   └── album_generator.py      ← Dedup + curation
│   └── api/server.py               ← Pure stdlib HTTP+WS server
├── frontend/index.html             ← Complete web app
├── captures/                       ← Auto-captured JPEGs
└── albums/                         ← Generated album manifests
```

## Requirements

**Tier 1 (heuristic mode only, ~60% accuracy) — this short list is genuinely all you need:**
```
opencv-python>=4.8
numpy>=1.24
Pillow>=9.0
scikit-learn>=1.2
scipy>=1.10
```
No PyTorch, no FastAPI, no internet needed for *this* path.

**Full production stack (what the "priority-tier" system above actually runs
on) is NOT dependency-light** — `backend/requirements.txt` pulls in
`torch`, `torchaudio`, `torchvision`, `transformers`, `fastapi`, `mediapipe`,
`ultralytics`, `insightface`, and more (the local VLM tagger alone downloads a
multi-GB model on first use). If you ran `pip install -r requirements.txt` per
the Quick Start above, this line does not describe your install — budget disk
space and, for the VLM weights, a one-time internet download.
