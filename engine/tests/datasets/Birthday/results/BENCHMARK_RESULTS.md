# SnapAI v2.5 — Clip Dataset Benchmark Results

**Generated:** 2026-05-28 20:08:50

**Clip dir:** `/Users/vineetjindal/Downloads/snappy/snappy_final/tests/datasets/clips`


## What this benchmark validates

This run extracts per-frame signals (motion, brightness, sharpness, skin-tone face proxy, gaze proxy, NIMA proxy, predictor trend) from each clip and pipes them into the production `decide_capture()` function. The goal is to validate the v2.5 priority-tier system end-to-end against real pixels.

It does **not** run CLIP / NIMA / YOLO. For full model-stack validation, upload each clip via the live server's video upload endpoint.


## Headline numbers

- **Clips processed:** 10
- **Frames analysed:** 720
- **Total captures fired:** 96
- **Spec assertions passing:** 10 / 10

## Per-clip results

| Clip | Prompt | Frames | Captures | Tiers seen | Avg score | Max score | Spec |
|---|---|---|---|---|---|---|---|
| `01_static_face.mp4` | `—` | 72 | **6** | elevated:72 | 0.585 | 0.586 | ✅ |
| `02_group_gaze.mp4` | `group photo` | 72 | **21** | critical:72 | 0.664 | 0.664 | ✅ |
| `03_low_light_motion.mp4` | `—` | 72 | **15** | high:39, normal:33 | 0.397 | 0.421 | ✅ |
| `04_cake_cutting_proxy.mp4` | `cake cutting` | 72 | **21** | critical:72 | 0.705 | 0.705 | ✅ |
| `05_high_motion_no_face.mp4` | `—` | 72 | **0** | normal:72 | 0.324 | 0.368 | ✅ |
| `06_dark_scene.mp4` | `—` | 72 | **0** | normal:72 | 0.123 | 0.123 | ✅ |
| `07_color_burst.mp4` | `—` | 72 | **6** | elevated:72 | 0.421 | 0.424 | ✅ |
| `08_smile_to_laugh.mp4` | `—` | 72 | **6** | elevated:72 | 0.61 | 0.611 | ✅ |
| `09_two_people_hug.mp4` | `hug moment` | 72 | **21** | critical:72 | 0.666 | 0.669 | ✅ |
| `10_blank_scene.mp4` | `—` | 72 | **0** | normal:72 | 0.242 | 0.242 | ✅ |

## Detailed breakdowns

### `01_static_face.mp4`

**Scenario:** Single static face, gentle motion, good lighting

**Expected:** Should capture some frames via gaze + face signals.

**Expected priority tiers:** high, elevated, normal

**Expected captures:** 1–20


- Frames processed: **72** (of 144 total)
- Captures: **6** (triggered before cooldown: 72)
- Tier distribution: `critical=0`, `high=0`, `elevated=72`, `normal=0`
- Ensemble score: avg `0.585`, max `0.586`
- **Spec verdict:** PASS ✅ (observed 6 captures, expected 1-20)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 2 | **elevated** | 0.585 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 26 | **elevated** | 0.585 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 50 | **elevated** | 0.585 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 74 | **elevated** | 0.585 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 98 | **elevated** | 0.585 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 122 | **elevated** | 0.585 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |

### `02_group_gaze.mp4`

**Scenario:** Three faces all looking at camera — should fire HIGH priority (or CRITICAL when run with a matching prompt)

**Expected:** HIGH priority requires gaze_ratio >= 0.65 and face_count >= 1. Priority floor (~0.4s) keeps capture count reasonable.

**Expected priority tiers:** critical, high

**Expected captures:** 1–30


- Frames processed: **72** (of 144 total)
- Captures: **21** (triggered before cooldown: 72)
- Tier distribution: `critical=72`, `high=0`, `elevated=0`, `normal=0`
- Ensemble score: avg `0.664`, max `0.664`
- **Spec verdict:** PASS ✅ (observed 21 captures, expected 1-30)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 2 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |
| 4 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |
| 12 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 22 | **critical** | 0.664 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 28 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |
| 32 | **critical** | 0.664 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 42 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 52 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |
| 54 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |
| 62 | **critical** | 0.664 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 72 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 78 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |
| 82 | **critical** | 0.664 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 92 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | 🚀 yes |
| 102 | **critical** | 0.663 | 0.28 | `group_photo` | priority:critical, clip:group_photo, quality, aesthetic | — |

*(showing first 15 of 21 captures)*

### `03_low_light_motion.mp4`

**Scenario:** Dim scene with horizontal motion (someone waving in low light)

**Expected:** Quality scores will be low. Capture count depends heavily on whether the dim face is detected. In production the NIMA + blur thresholds would tighten this further; this benchmark uses a lightweight proxy so the count runs a bit higher.

**Expected priority tiers:** normal, elevated, high

**Expected captures:** 0–25


- Frames processed: **72** (of 144 total)
- Captures: **15** (triggered before cooldown: 39)
- Tier distribution: `critical=0`, `high=39`, `elevated=0`, `normal=33`
- Ensemble score: avg `0.397`, max `0.421`
- **Spec verdict:** PASS ✅ (observed 15 captures, expected 0-25)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 6 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |
| 8 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |
| 18 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 28 | **high** | 0.406 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 32 | **high** | 0.405 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |
| 38 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 48 | **high** | 0.41 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 56 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |
| 58 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 72 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 86 | **high** | 0.421 | 0.35 | `watching` | priority:high, aesthetic, group_gaze, group_present | — |
| 88 | **high** | 0.41 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |
| 96 | **high** | 0.405 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | 🚀 yes |
| 138 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |
| 140 | **high** | 0.411 | 0.35 | `watching` | priority:high, aesthetic, group_gaze | — |

### `04_cake_cutting_proxy.mp4`

**Scenario:** Cake-cutting analogue: brown/white blob + bright candle blobs + faces leaning in + downward knife motion. Tests the visual heuristic + CRITICAL priority when prompt is 'cake cutting'.

**Expected:** Benchmark should be run with prompt='cake cutting' to test CRITICAL tier triggering.

**Expected priority tiers:** critical, high, elevated

**Expected captures:** 1–25


- Frames processed: **72** (of 144 total)
- Captures: **21** (triggered before cooldown: 72)
- Tier distribution: `critical=72`, `high=0`, `elevated=0`, `normal=0`
- Ensemble score: avg `0.705`, max `0.705`
- **Spec verdict:** PASS ✅ (observed 21 captures, expected 1-25)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 2 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |
| 4 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |
| 12 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 22 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 28 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |
| 32 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 42 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 52 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |
| 54 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |
| 62 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 72 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 78 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |
| 82 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 92 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | 🚀 yes |
| 102 | **critical** | 0.705 | 0.28 | `cake_cutting` | priority:critical, clip:cake_cutting, quality, aesthetic | — |

*(showing first 15 of 21 captures)*

### `05_high_motion_no_face.mp4`

**Scenario:** High motion, no faces (sports-action proxy)

**Expected:** Predictor 'peak' may trigger ELEVATED. No HIGH (no faces).

**Expected priority tiers:** normal, elevated

**Expected captures:** 0–15


- Frames processed: **72** (of 144 total)
- Captures: **0** (triggered before cooldown: 0)
- Tier distribution: `critical=0`, `high=0`, `elevated=0`, `normal=72`
- Ensemble score: avg `0.324`, max `0.368`
- **Spec verdict:** PASS ✅ (observed 0 captures, expected 0-15)

### `06_dark_scene.mp4`

**Scenario:** Very dark, occasional flicker — tests low-light reject pipeline

**Expected:** Should produce few or zero captures. Validates quality discard.

**Expected priority tiers:** normal

**Expected captures:** 0–5


- Frames processed: **72** (of 144 total)
- Captures: **0** (triggered before cooldown: 0)
- Tier distribution: `critical=0`, `high=0`, `elevated=0`, `normal=72`
- Ensemble score: avg `0.123`, max `0.123`
- **Spec verdict:** PASS ✅ (observed 0 captures, expected 0-5)

### `07_color_burst.mp4`

**Scenario:** Confetti-like falling color particles

**Expected:** High color variance — confetti_burst heuristic may match.

**Expected priority tiers:** normal, elevated

**Expected captures:** 0–15


- Frames processed: **72** (of 144 total)
- Captures: **6** (triggered before cooldown: 72)
- Tier distribution: `critical=0`, `high=0`, `elevated=72`, `normal=0`
- Ensemble score: avg `0.421`, max `0.424`
- **Spec verdict:** PASS ✅ (observed 6 captures, expected 0-15)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 2 | **elevated** | 0.417 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 26 | **elevated** | 0.423 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 50 | **elevated** | 0.422 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 74 | **elevated** | 0.42 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 98 | **elevated** | 0.422 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 122 | **elevated** | 0.423 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |

### `08_smile_to_laugh.mp4`

**Scenario:** Single face transitioning from smile to open laugh

**Expected:** Looking at camera = HIGH priority; emotion rises mid-clip.

**Expected priority tiers:** high, elevated

**Expected captures:** 1–25


- Frames processed: **72** (of 144 total)
- Captures: **6** (triggered before cooldown: 72)
- Tier distribution: `critical=0`, `high=0`, `elevated=72`, `normal=0`
- Ensemble score: avg `0.61`, max `0.611`
- **Spec verdict:** PASS ✅ (observed 6 captures, expected 1-25)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 2 | **elevated** | 0.601 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 26 | **elevated** | 0.611 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 50 | **elevated** | 0.611 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 74 | **elevated** | 0.611 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 98 | **elevated** | 0.61 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |
| 122 | **elevated** | 0.61 | 0.38 | `watching` | priority:elevated, clip:watching, quality, aesthetic | — |

### `09_two_people_hug.mp4`

**Scenario:** Two faces converging into a hug — emotional moment

**Expected:** With a 'hug moment' prompt this fires CRITICAL throttled to the 0.4s priority floor. Without a prompt it stays NORMAL.

**Expected priority tiers:** critical, normal, elevated

**Expected captures:** 0–30


- Frames processed: **72** (of 144 total)
- Captures: **21** (triggered before cooldown: 72)
- Tier distribution: `critical=72`, `high=0`, `elevated=0`, `normal=0`
- Ensemble score: avg `0.666`, max `0.669`
- **Spec verdict:** PASS ✅ (observed 21 captures, expected 0-30)

**Captures:**

| Frame | Tier | Score | Threshold | Class | Reasons | Cooldown bypass |
|---|---|---|---|---|---|---|
| 2 | **critical** | 0.656 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |
| 4 | **critical** | 0.657 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |
| 12 | **critical** | 0.656 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 22 | **critical** | 0.665 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 28 | **critical** | 0.667 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |
| 32 | **critical** | 0.666 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 42 | **critical** | 0.664 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 52 | **critical** | 0.666 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |
| 54 | **critical** | 0.665 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |
| 62 | **critical** | 0.666 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 72 | **critical** | 0.664 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 78 | **critical** | 0.666 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |
| 82 | **critical** | 0.665 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 92 | **critical** | 0.666 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | 🚀 yes |
| 102 | **critical** | 0.668 | 0.28 | `hug_moment` | priority:critical, clip:hug_moment, quality, aesthetic | — |

*(showing first 15 of 21 captures)*

### `10_blank_scene.mp4`

**Scenario:** Empty grey frame, no signals (negative control)

**Expected:** MUST produce zero captures. Anything else is a precision bug.

**Expected priority tiers:** normal

**Expected captures:** 0–0


- Frames processed: **72** (of 144 total)
- Captures: **0** (triggered before cooldown: 0)
- Tier distribution: `critical=0`, `high=0`, `elevated=0`, `normal=72`
- Ensemble score: avg `0.242`, max `0.242`
- **Spec verdict:** PASS ✅ (observed 0 captures, expected 0-0)
