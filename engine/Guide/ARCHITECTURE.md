# SnapAI — Architecture (v2.4)

End-to-end AI event photographer: detects ceremonial / candid moments,
auto-captures, learns from feedback, discovers new moment classes from
data. Voice or text prompts can change the watch list mid-event.

## Top-level layout

```
snappy_final/
├── run.py                       # one-click launcher (banner + browser)
├── requirements.txt
├── frontend/
│   └── index.html               # camera + WS client + voice UI + feedback UI
├── backend/
│   ├── api/
│   │   └── server.py            # asyncio HTTP+WS, routes, frame pipeline
│   ├── gpu/
│   │   ├── __init__.py
│   │   └── device.py            # CUDA / MPS / CPU detection (one source)
│   ├── voice/
│   │   ├── __init__.py
│   │   └── transcription.py     # server-side Whisper STT (optional fallback)
│   ├── models/
│   │   ├── face_provider.py     # YOLOv8-face + MediaPipe + Haar zoo
│   │   ├── face_detector.py     # legacy MediaPipe wrapper (kept for compat)
│   │   ├── yolo_face.py         # YOLOv8-face wrapper (Ultralytics)
│   │   ├── shot_quality.py      # heuristic quality + NIMA blend
│   │   ├── nima_aesthetic.py    # NIMA (ResNet on AVA) via pyiqa
│   │   ├── emotion_detector.py  # MediaPipe blendshape emotion
│   │   ├── hsemotion_fer.py     # HSEmotion (AffectNet) ONNX FER
│   │   ├── gaze_estimator.py    # iris-based gaze (consumes face_provider)
│   │   ├── gaze_detector.py     # legacy Haar gaze (kept for compat)
│   │   ├── clip_moment_detector.py  # heuristic VLM (fallback)
│   │   ├── clip_engine.py       # real CLIP ViT-B/32 + kept-centroid + trained centroids
│   │   ├── moment_predictor.py  # 60-frame trend + preroll buffer
│   │   ├── album_generator.py   # post-event album curator
│   │   ├── prompt_router.py     # text/voice → active moment classes
│   │   ├── auto_classifier.py   # discovers new moment clusters via DBSCAN
│   │   └── online_learner.py    # Beta-Bernoulli per-class priors from feedback
│   ├── training/
│   │   ├── dataset_bootstrap.py # multi-source labelled photo downloader
│   │   ├── finetune_clip.py     # CLIP class-centroid trainer
│   │   └── continuous_learning.py  # background re-train thread
│   ├── utils/
│   │   ├── config.py            # env-driven thresholds
│   │   ├── logging_setup.py     # rotating file logger
│   │   ├── persistence.py       # SQLite store: sessions, photos, feedback,
│   │   │                        #   priors, discoveries, prompt_history
│   │   └── safe_types.py        # JSON-safe numpy converter
│   └── data/
│       ├── snappy.db            # auto-created
│       └── training/            # bootstrapped photos, per-class folders
├── tests/
│   ├── conftest.py
│   ├── test_persistence.py
│   ├── test_online_learner.py
│   └── test_prompt_router.py
├── albums/                      # generated event albums
├── captures/                    # per-session JPEG captures
├── logs/                        # rotating server logs
├── INSTALL.md
├── TRAINING.md
└── ARCHITECTURE.md              # this file
```

## Per-frame data flow

```
Browser camera frame (base64 JPEG)
        │
        ▼
WebSocket  ─▶  server.py: process_frame()
        │
        ├──▶ FaceProvider           one MediaPipe call per frame
        │       └─ YOLOv8 booster   (when face_count < 3)
        │       └─ Haar fallback    (if MediaPipe missing)
        │
        ├──▶ ShotQualityAnalyzer    blur, comp, bright, faces, …
        │       └─ NIMA blend       0.45 weight when available
        │
        ├──▶ GazeEstimator          iris-based, multi-face
        │
        ├──▶ Emotion (blendshapes)
        │       └─ HSEmotion blend  0.7 weight when available
        │
        ├──▶ CLIPEngine.score_frame
        │       ├─ text-prompt sims
        │       ├─ trained class centroids (from finetune_clip)
        │       └─ per-session kept centroid (👍 history)
        │
        ├──▶ AutoClassifier.observe
        │       └─ DBSCAN on unknown high-quality frames
        │           (emits new "discovery" event when cluster forms)
        │
        ├──▶ MomentPredictor        60-frame trend + preroll
        │
        ├──▶ OnlineLearner.boost    multiplies confidence by class keep-rate
        │
        ▼
Capture decision (4-trigger OR)
        │
        ├──▶ Save JPEG + persist
        │
        ▼
JSON response → WebSocket
```

## Live prompt flow (voice or text)

```
Browser Web Speech API ─▶ WebSocket {"type":"prompt_update","text":"…"}
                                   │
                                   ▼
                           PromptRouter.update(sid, text)
                                   │
                                   ├─ infer intent (add/remove/replace)
                                   ├─ keyword-match against MOMENT_PROFILES
                                   ├─ update active_classes set
                                   ▼
                           VLMDetector.update_prompt(text)   (heuristic side)
                           STORE.log_prompt(...)             (audit log)
                                   │
                                   ▼
                           WS push {"type":"prompt_changed", transition}
```

The browser also accepts purely text input via the "Live Prompt" card —
same code path, `source: "text"` instead of `source: "voice"`.

## Self-training loop (auto-classifier + continuous learner)

```
                ┌─────────────────────────────────────────────┐
                │ AutoClassifier (per-frame, in-process)      │
                │   - buffers high-quality "unknown" frames   │
                │   - DBSCAN on CLIP embeddings every 10 obs  │
                │   - emits cluster when ≥ MIN_SAMPLES        │
                └──────────────────┬──────────────────────────┘
                                   │
              user names cluster   ▼
       ┌──────────────────────────────────────────────────┐
       │ POST /sessions/<sid>/discoveries/<cid>/name       │
       │     {"name":"sparkler dance"}                     │
       └──────────────────┬───────────────────────────────┘
                          │
                          ▼
                 STORE.name_discovery(...)
                 (centroid sits in DB, ready for continuous learner)

                ┌─────────────────────────────────────────────┐
                │ ContinuousLearner thread (every 10 min)     │
                │   - pulls new 👍 photos since last tick     │
                │   - re-embeds with CLIP, EMA-blends into    │
                │     existing class centroids                │
                │   - appends NAMED discoveries as new classes│
                │   - rewrites clip_class_centroids.npz       │
                │   - tells CLIPEngine to hot-reload          │
                └─────────────────────────────────────────────┘
```

**Net effect**: the model gets better automatically. New moments enter
the taxonomy. No retraining job needed.

Enable: `SNAPPY_CONTINUOUS_LEARNING=1 python3 run.py`

## Persistence (SQLite)

| Table | Rows | Used by |
|---|---|---|
| `sessions` | event metadata + start/end timestamps | restore on boot |
| `photos` | every captured frame | gallery, album |
| `feedback` | append-only 👍 / 👎 events | audit |
| `feedback_priors` | (sid, moment) → kept/trashed counts | OnlineLearner |
| `discoveries` | auto-classifier clusters + names | hot-reload, continuous learner |
| `prompt_history` | every prompt update + intent | audit, replay |

Single file at `backend/data/snappy.db`. Postgres migration is Phase 4.

## Module independence guarantees

Every model wrapper:
1. Has a `singleton.get()` constructor (one instance per server).
2. Lazy-loads its heavy library only when first invoked.
3. Exposes `.available: bool` + `.init_error: Optional[str]`.
4. Falls back to a lower-tier path if its dep is missing.

So you can install in tiers (see [INSTALL.md](INSTALL.md)). The server
boots and runs at any tier. Removing a tier doesn't break anything,
just lowers accuracy.

## How to add a new model

1. Create `backend/models/your_thing.py` with the singleton+lazy pattern.
2. Wire it into [`backend/api/server.py`](backend/api/server.py) `process_frame`
   between the existing models. Output JSON-safe via `safe()`.
3. Surface `available/init_error` in `/health`.
4. Add a test under `tests/test_your_thing.py`.

That's the whole contract.
