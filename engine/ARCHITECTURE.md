# SnapAI — Architecture (birthday MVP)

The system is **three clean pillars**. Each is isolated, each degrades
gracefully (a missing model never breaks a run), and the boundary between them
is sharp so the code stays readable.

```
            LIVE CAMERA / UPLOADED VIDEO  (per frame, on-device)
                              │
            ┌─────────────────┴──────────────────┐
            │   PILLAR 1 — CHEAP REAL-TIME REFLEX │   "WHEN to click"
            │   (api/pipeline.py + models/…)      │
            │   • face / gaze / emotion / audio   │
            │   • quality (NIMA) / peak predictor │
            │   • candle & cake-smash signatures  │
            └─────────────────┬──────────────────┘
                              │ capture candidates
            ┌─────────────────┴──────────────────┐
            │   PILLAR 2 — PERSONALIZATION        │   "WHOSE event"
            │   (models/host_group.py)            │
            │   • infers the HOST group (central+ │
            │     persistent+close faces)         │
            │   • gate: drop stranger-only frames │
            └─────────────────┬──────────────────┘
                              │ saved captures (session.photos)
            ┌─────────────────┴──────────────────┐
            │   PILLAR 3 — ALBUM CURATION         │   "WHAT to keep + how to tag"
            │   (models/album_generator.py)       │
            │   • relevance gate + dedup + story  │
            │   • on-device tags (birthday_fused) │
            │   • OPTIONAL VLM judge (vlm_curator)│
            └────────────────────────────────────┘
```

## Pillar 1 — Cheap real-time reflex  (on-device, every frame)
Decides **when to click** from fast signals — no fine moment-classification needed:
- **Emotion peak** (`models/emotion_detector.py` + HSEmotion): joy / **tears-moved (sad)** / surprise → strong emotion is a **never-miss HIGH** capture. This is the heart of "wholesome" moments (a crying grandma, a belly-laugh).
- **Gaze** (`models/gaze_estimator.py`): group looking at lens → HIGH; one person → ELEVATED.
- **Audio** (`models/audio_event_detector.py`): "Happy Birthday" song / cheer / applause → escalates capture (the ceremonial beats announce themselves).
- **Quality/aesthetic** (`models/shot_quality.py`, NIMA) + **peak predictor** (`models/moment_predictor.py`).
- **Signatures**: candle-blow (`candle_phase.py`: flame + lean) and cake-smash (`cake_smash.py`: cream-on-face).
- Ensemble + priority tiers live in `models/event_engine.py`.

## Pillar 2 — Personalization  (on-device, zero-touch)
`models/host_group.py`, one per `Session`. Infers the **owner's people** from how
faces behave (persistent + central + close = host; brief + peripheral + small =
stranger), tracking by position (cheap). The pipeline **suppresses captures that
contain only strangers** (with a 1–2 shot budget for a stranger gazing at the
lens). No enrollment; `should_capture()` is a no-op until confident (no
cold-start harm). Optional `enroll(x,y)` for a 1-tap "this is the birthday person".

## Pillar 3 — Album curation  ("WHAT to keep, how to tag")
`models/album_generator.py` runs at album time:
- **relevance gate** (drop off-event) + **keep great candids** (high emotion/gaze) + **pixel & semantic dedup** + **birthday-story order**.
- **Tags** come from the trained on-device head `models/birthday_moment_classifier.py` (`birthday_fused.npz`),
  or from `models/vlm_tagger.py` (local Qwen2.5-VL-3B-Instruct) when available — see below.
- **TWO separate, independently-toggled VLMs exist here — don't confuse them:**
  - `models/vlm_tagger.py` — **ON by default** (`SNAPPY_VLM=1`). Runs a **local**
    Qwen2.5-VL-3B-Instruct model (via `transformers`) for photographer-grade moment
    tagging at album time. Nothing leaves the device — weights just download once
    (multi-GB) on first use. Disable with `SNAPPY_VLM=0`. If `transformers`/the
    model/hardware aren't ready, it's a silent no-op and the on-device classifier
    tags instead.
  - `models/vlm_curator.py` — a **separate, cloud** VLM judge that re-tags +
    curates, used **only here, never in the live loop**. **OFF unless** you set
    `SNAPPY_VLM_API_KEY` (+ optional `SNAPPY_VLM_URL`, `SNAPPY_VLM_MODEL`). Off →
    the on-device/local-VLM tags are used. Privacy: it uploads only the few album
    photos, and only when you opt in — this is the only path that ever sends a
    photo off the device.

## Model inventory
| Model | Where | On-device? | Required? |
|---|---|---|---|
| MediaPipe FaceLandmarker (+blendshapes) | face/gaze/emotion | yes | yes |
| HSEmotion FER | emotion | yes | optional (blendshapes fallback) |
| CLIP ViT-B/32 + per-event centroids | scene/relevance | yes | yes |
| NIMA | aesthetic | yes | optional |
| AST | audio events | yes | optional |
| `birthday_fused.npz` (fused LR head) | moment tags | yes | optional (trained from `labels_master.csv`) |
| HostGroup | personalization | yes | yes (pure-python) |
| Qwen2.5-VL-3B-Instruct (`vlm_tagger.py`) | album tagging | **yes, local** | optional — **on by default** (`SNAPPY_VLM=1`) |
| VLM (OpenAI-compatible, `vlm_curator.py`) | album judge | **cloud, opt-in** | **off by default** |

## Degradation guarantees (no assumptions)
- No CLIP / NIMA / AST / FER → those signals contribute 0; capture still works.
- No `birthday_fused.npz` → falls back to CLIP-centroid tags.
- No `SNAPPY_VLM_API_KEY` → album skips the cloud VLM judge, uses local tags.
- `SNAPPY_VLM=0` or local VLM unavailable → album falls back to the on-device classifier.
- HostGroup not yet confident → captures everyone (won't wrongly suppress).

## Data & training (offline, not in the serving path)
`backend/training/` — `fetch_youtube.py` → `prepare_birthday_dataset.py` /
`ingest_labeled.py` → label review (`*.xlsx`/`labels_master.csv`) →
`train_fused.py` → `finalize_model.py` (writes `birthday_fused.npz`).
Dataset: `tests/datasets/Birthday/clips/real` + `tests/datasets/Birthday/Frames`.
