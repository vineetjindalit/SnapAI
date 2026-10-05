# SnapAI Test Clip Dataset

This directory holds the test corpus the priority-tier system is
benchmarked against. The corpus has two halves:

```
tests/datasets/
├── clips/
│   ├── synthetic/       ← 10 deterministic clips (OpenCV-generated)
│   └── real/            ← optional Pexels downloads (CC-licensed)
└── results/
    ├── BENCHMARK_RESULTS.md     ← latest run, human-readable
    ├── BENCHMARK_RESULTS.pdf    ← latest run, distributable PDF
    ├── benchmark_<stamp>.md     ← timestamped history
    └── benchmark_<stamp>.json   ← timestamped raw data
```

## How to regenerate everything

```bash
# 1. Regenerate the 10 synthetic clips (no network needed)
python3 scripts/generate_synthetic_clips.py

# 2. Optionally pull free Pexels clips (needs PEXELS_API_KEY)
PEXELS_API_KEY=your_key python3 scripts/download_test_clips.py

# 3. Run the benchmark
python3 scripts/benchmark_clips.py
```

## What the benchmark validates

Every clip is processed frame-by-frame. For each frame we extract
lightweight signals — motion, brightness, sharpness, a skin-tone face
proxy, a gaze proxy, a NIMA-style aesthetic proxy, and a predictor
trend — and pipe them into the **production** `decide_capture()`
function in `backend/models/event_engine.py`.

This validates the v2.5 priority-tier system (CRITICAL / HIGH /
ELEVATED / NORMAL) end-to-end against real pixels, including:

- the priority-floor debounce (a 0.4-second minimum gap between
  back-to-back priority captures, so a 10-second cake-cutting moment
  doesn't produce 80 photos)
- the cooldown bypass for CRITICAL / HIGH frames
- the NORMAL-tier discard pipeline (dark, blank, no-face scenes
  correctly produce zero captures)

It does **not** invoke CLIP, NIMA, or YOLO. Those need GPU + multi-GB
weights and are out of scope for a CI-friendly priority-tier
benchmark. For full ML-stack validation, run the live server and
upload each clip via the `/sessions/<id>/upload-video` endpoint.

## The 10 synthetic clips

| # | File | What it tests |
|---|---|---|
| 01 | `01_static_face.mp4` | Single face, low motion, good lighting — should produce some captures |
| 02 | `02_group_gaze.mp4` | 3 faces all looking at camera — HIGH/CRITICAL priority |
| 03 | `03_low_light_motion.mp4` | Dim scene + waving — sparse captures, low quality discard |
| 04 | `04_cake_cutting_proxy.mp4` | Brown/white blob + candle blobs + knife motion — CRITICAL with prompt |
| 05 | `05_high_motion_no_face.mp4` | Sports-action proxy, no faces — predictor-driven only |
| 06 | `06_dark_scene.mp4` | Very dark, occasional flicker — discard pipeline check |
| 07 | `07_color_burst.mp4` | Confetti-like color variance — color heuristic check |
| 08 | `08_smile_to_laugh.mp4` | Face transitioning smile → laugh — emotion gradient |
| 09 | `09_two_people_hug.mp4` | Converging faces — emotional moment, prompt-CRITICAL |
| 10 | `10_blank_scene.mp4` | Empty grey frame — **must** produce zero captures (precision check) |

Each clip has a spec entry in `clips/synthetic/_specs.json` that
declares the expected priority tiers and capture-count range. The
benchmark checks observed behavior against these specs and marks
PASS / FAIL per clip.

## Current status

See `results/BENCHMARK_RESULTS.md` for the latest run.

**Last known result:** 10 of 10 spec assertions passing.

## When to add a new clip

Add a clip when you need to lock in a specific behavior — usually
after fixing a bug or adding a feature, so a regression would be
caught by the benchmark. To add:

1. Add a `clip_NN_<name>()` function to `scripts/generate_synthetic_clips.py`
   that writes the clip via `writer(...)`.
2. Append a `CLIP_SPECS` entry declaring expected priorities + capture
   bounds and rationale.
3. Add the function to the `CLIP_FUNCS` list at the bottom.
4. Re-run `scripts/generate_synthetic_clips.py`.
5. Optionally add a prompt mapping in `scripts/benchmark_clips.py`'s
   `PROMPT_FOR_CLIP` dict if the clip should be benchmarked with a
   specific prompt.
6. Re-run `scripts/benchmark_clips.py` and confirm your spec passes.

## Licensing

- **Synthetic clips** are produced from pure OpenCV draw calls. No
  external assets, no copyright. Public domain.
- **Real clips** (`clips/real/`) are downloaded from Pexels under the
  Pexels License (free for any use, no attribution required). The
  per-clip metadata is in `clips/real/_manifest.json`.
