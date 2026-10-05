# SnapAI — Engineering Changelog

Living document. Each entry: what changed, **why**, blast radius, how to verify.
Newest first. Versions follow semver-ish: minor for new capability, patch for
fix/refactor.

---

## v2.8 — Full auth UX + backend hardening

### Why
v2.7 stubbed JWT auth but it wasn't enforceable end-to-end. This drop
closes the loop: real login/register/forgot/reset/verify UX in React,
real WebSocket auth (the biggest security gap before this), token
revocation on logout, rate-limiting on auth endpoints, password
strength validation, anti-enumeration on forgot, and per-IP brute-
force protection.

### Backend hardening
- **`backend/auth/jwt_auth.py`** — tokens now carry a `jti` (JWT ID).
  New `JWTAuth.revoke()` adds the JTI to a denylist via the cache
  module. `verify()` checks the denylist on every call.
- **`backend/auth/rate_limit.py`** — sliding-window per-IP limiter
  using the cache module. Login: 10/min. Register: 5/hr. Forgot: 3/hr.
- **`backend/auth/routes.py`** — added `/auth/logout`, `/auth/refresh`,
  `/auth/forgot`, `/auth/reset`. Strong password policy (8+ chars,
  letters AND digits, common-password blocklist). `/auth/me` includes
  `verified` and `tier`. Same error message for "user not found"
  and "wrong password" — no user enumeration.
- **WebSocket auth** — `_upgrade_ws()` now requires a valid JWT (passed
  via `?token=` query, since browsers can't set Authorization on WS).
  Verifies signature, expiry, denylist, AND that the token's user
  matches the session's `owner_id`. Closes the sid-guessing hijack.
- **`backend/utils/persistence.py`** — added `password_resets` and
  `email_verifications` tables. New methods: `create_password_reset`,
  `consume_password_reset` (atomic), `update_user_password`,
  `mark_email_verified`, `is_email_verified`.
- **`backend/mail/sender.py`** — `send_password_reset()` template.

### Frontend auth UX
New screens, all Tailwind-styled, both modes:

```
src/store/auth.ts             — Zustand auth store w/ persisted token
src/components/AuthShell.tsx  — shared visual frame for auth screens
src/components/AuthGate.tsx   — gates the app, deeplink-aware (?token)
src/components/UserMenu.tsx   — header dropdown: email, tier, sign out
src/pages/Login.tsx           — email/password + forgot link
src/pages/Register.tsx        — w/ live password strength meter (4 bars)
src/pages/ForgotPassword.tsx  — generic success regardless of email
src/pages/ResetPassword.tsx   — confirm-password match validation
src/pages/VerifyEmail.tsx     — handles /verify?token= deeplinks
```

### Routing changes
- `/reset?token=...` deeplink → ResetPassword
- `/verify?token=...` deeplink → VerifyEmail
- All other paths → AuthGate (Login/Register/ForgotPassword if not
  logged in AND `SNAPPY_AUTH=1`, else pass-through)

### Tests
- Full backend auth flow verified: 12 steps (register, login, logout
  + revocation check, refresh + old-token revocation check, forgot
  with anti-enumeration, reset, password change verification, email
  verify). All pass.

### Activation
```bash
SNAPPY_AUTH=1 \
  SNAPPY_PUBLIC_URL=https://your.domain \
  SNAPPY_EMAIL_BACKEND=mailgun \
  SNAPPY_EMAIL_MAILGUN_KEY=key-... \
  SNAPPY_EMAIL_MAILGUN_DOMAIN=mail.your.domain \
  python3 run.py
```

### Migration notes
- SQLite schema auto-migrates. New tables added; existing data preserved.
- No frontend breaking changes when `SNAPPY_AUTH=0` (default for dev).

---

## v2.7 — Phase 3 / 4 production foundations

### Why
Closing the gap from "feature-complete prototype" to "shippable SaaS".
Until now we had auth scaffolding but no enforcement, no S3 abstraction,
no Stripe, no Postgres path, no Terraform. v2.7 makes every one of those
production subsystems real but **opt-in** — flip an env var to turn on,
nothing changes for existing dev users.

### New modules

```
backend/storage/         LocalStorage + S3Storage + factory      (Phase 4)
backend/mail/            Mailgun + SendGrid + console sender     (Phase 3)
backend/auth/            JWT middleware + register/login routes  (Phase 3)
backend/billing/         Stripe Checkout + webhook + tier enforce (Phase 4)
backend/cache/           InMemoryCache + RedisCache              (Phase 4)
backend/utils/error_tracking.py   Sentry init                    (Phase 4)
deploy/terraform/        AWS module: VPC + EC2 + RDS + Redis + S3 (Phase 4)
deploy/k6_loadtest.js    100-VU concurrency test                 (Phase 4)
mobile/                  React Native scaffold + api.ts          (Phase 3)
landing/                 Marketing landing page                  (Phase 5)
YOUR_TASKS.md            Step-by-step user runbook
```

### Key wiring

- `routes.create_session()` now requires Bearer auth when `SNAPPY_AUTH=1`,
  enforces tier quotas via `enforce_session_quota()`, persists `owner_id`.
- `users` table schema migrated to include billing fields (tier,
  stripe_customer_id, stripe_subscription_id, subscription_status).
- New endpoints: `/auth/register`, `/auth/login`, `/auth/me`, `/auth/verify`,
  `/billing/tiers`, `/billing/checkout`, `/billing/webhook`.
- Globals container loads STORAGE, CACHE, BILLING singletons + initialises
  Sentry on import.

### Bugs fixed in this drop
- `backend/email/` package name shadowed Python's stdlib `email` module
  (urllib internally imports `email.parser`). Renamed to `backend/mail/`.

### Tests
- 17 stdlib-only tests pass (persistence, online_learner, event_engine,
  auth round-trip, storage, cache, billing, mail console).
- 0 regressions vs v2.6.

### Activation cheatsheet
```bash
SNAPPY_AUTH=1                                        # enforce JWT
SNAPPY_STORAGE=s3 SNAPPY_S3_BUCKET=...              # S3 photos
SNAPPY_DB_URL=postgresql://...                       # Postgres (instead of SQLite)
SNAPPY_REDIS_URL=redis://...                         # multi-instance
SNAPPY_BILLING=stripe SNAPPY_STRIPE_SECRET=sk_...   # paid plans
SNAPPY_EMAIL_BACKEND=mailgun SNAPPY_EMAIL_MAILGUN_KEY=key-...
SNAPPY_SENTRY_DSN=https://...                        # error tracking
```

---

## v2.6 — Unified EventCaptureEngine + YOLO fix

### Why
Until now each model wrote into a different part of the response, and the
capture decision was a hand-coded `if reasons: capture()` chain. Hard to
reason about, hard to tune. Also the YOLOv8-face download URL had gone
404 (upstream renamed `akanametov/yolov8-face` → `akanametov/yolo-face`
and re-tagged `v0.0.0` → `1.0.0`).

### Changes
- **`backend/models/event_engine.py`** — pure-function `decide_capture()`
  that fuses CLIP, quality, NIMA, emotion, gaze, face count, kept-centroid
  similarity, predictor trend, and learner keep-rate into ONE
  weighted-ensemble score. Per-class weight overrides
  (`group_photo` weights faces, `sports_action` weights predictor, etc).
  Active-class gate so prompts actually constrain captures.
- **`backend/api/pipeline.py`** — replaced the if-chain with a single
  `decide_capture(...)` call. The CaptureDecision goes into the response
  payload as `engine: { final_score, threshold, contributions, weights,
  reasons, triggered }` so the UI can show what fired and why.
- **`backend/models/face_provider.py`** — YOLO is now ALWAYS merged
  with MediaPipe (was: fallback only when MediaPipe found <3 faces).
  Higher recall on small / profile / occluded faces.
- **`backend/models/yolo_face.py`** — updated download URL to
  `akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt`
  (verified live, 6.2 MB). Multi-mirror fallback chain. Non-fatal on
  failure (system still runs on MediaPipe alone).
- **`frontend/index.html`** — live ensemble breakdown panel under
  "Frame Analysis" with per-signal bars + weights + final / threshold.
- **`tests/test_event_engine.py`** — 9 tests covering weight
  normalisation, monotonicity, strong/weak signals, active-class gate,
  per-class weight divergence.

### Tunable from env
```
SNAPPY_W_CLIP=0.30      SNAPPY_W_QUALITY=0.18  SNAPPY_W_AESTHETIC=0.12
SNAPPY_W_EMOTION=0.10   SNAPPY_W_GAZE=0.08     SNAPPY_W_KEPT=0.10
SNAPPY_W_FACE=0.04      SNAPPY_W_PREDICT=0.04  SNAPPY_W_LEARNER=0.04
SNAPPY_CAPTURE_THRESHOLD=0.55
```

### How to verify
```bash
pytest tests/test_event_engine.py -v   # 9 passing
python3 run.py
# In the live HUD, the new "Ensemble signals" panel shows each model's
# contribution as a coloured bar with its weight. The "ensemble X/Y"
# token appears in the moment badge; the row turns green when it triggers.
```

---

## v2.5 — Production refactor + Phase 3 starter (complete)

### Why
After v2.4 the backend hit ~37 files but `server.py` was still a 1,000-line
monolith doing protocol, routing, business logic, and ML orchestration in
one class. Heavy ML calls also blocked the asyncio loop, capping us at
single-digit fps under any GPU model. Phase 1's last open task was
multi-threaded frame processing; now's the time. Phase 2's last open
items (confidence calibration, better gaze) line up too. While we have
the diff open, we lay the groundwork for Phase 3 (auth, mobile-ready API,
Docker) so the next turn can ship features instead of plumbing.

### Changes

#### A. Server.py decomposition
**Before:** one ~1,000-line `server.py` containing
`Session`, `process_frame`, `process_video_async`, WS frame parsing,
HTTP dispatch, and 12 `_api_*` handlers.

**After:**
```
backend/api/
├── server.py          ~180 lines — asyncio.Protocol, HTTP dispatch only
├── globals.py          singletons (FACE_PROVIDER, CLIP, NIMA, …) + SESSIONS dict
├── session.py          Session dataclass
├── ws_protocol.py      ws_accept_key / ws_parse_frame / ws_build_frame
├── pipeline.py         process_frame + ThreadPoolExecutor + video processing
└── routes.py           pure-function HTTP handlers (status, body) tuples
```

**Why this split:**
- `globals.py` → DI container so routes don't import server (no cycles).
- `routes.py` returns `(status, dict)` so they're trivially testable
  without an HTTP transport.
- `pipeline.py` owns the thread pool and the per-frame ML orchestration,
  so the asyncio Protocol doesn't have to know about CLIP / NIMA.

#### B. Multi-threaded frame pipeline
**Before:** `process_frame()` ran on the asyncio loop. CLIP frame embed
(~80 ms CPU) blocked the entire server, capping throughput at ~12 fps.

**After:** `pipeline.process_frame_async()` offloads to a
`ThreadPoolExecutor(max_workers=4)`. Asyncio loop stays free to accept
new WS messages, so we can run multiple cameras in parallel. Pool size
configurable via `SNAPPY_FRAME_WORKERS`.

**Why:** Phase 1 PDF target was 30 fps; we couldn't hit that single-
threaded with the model zoo on. With 4 workers and CLIP/NIMA on CPU
we measure ~25 fps at tier 4, ~45 fps at tier 3. Frame-pipeline state
that needs serialisation (FACE_PROVIDER, CLIP) is already threadsafe via
their own internal locks — no new sync code.

#### C. L2CS-Net gaze
**Before:** iris-corner geometry was good (~12° error). Phase 1 PDF
target was MPIIGaze / GazeNet ~3°.

**After:** `models/l2cs_gaze.py` wraps the `l2cs` pip package. When
available, it produces yaw/pitch per face in radians; we map to
"looking_at_camera" with the same interface as the geometric path so
GazeEstimator drops in transparently. Falls back to iris geometry if
`l2cs` isn't installed.

**Why:** L2CS-Net is the published SOTA gaze regressor. The pip-
installable wrapper makes integration ~80 lines.

#### D. Confidence calibration (Platt scaling per class)
**Before:** raw moment-confidence values were uncalibrated — a "0.55"
for `cake_cutting` didn't mean the same thing as "0.55" for
`first_dance`. Capture decisions used a single threshold across classes.

**After:** `models/calibration.py` fits a logistic regression per moment
class on the (raw_score, kept) pairs the OnlineLearner accumulates. At
inference, raw scores are mapped to true probability of being a "kept"
shot. The capture threshold becomes a single calibrated probability
(default 0.6) that's class-agnostic. Recalibrates every N feedback
events; persists fit params to SQLite.

**Why:** Phase 2 PDF item. Without it, tuning thresholds is per-class
guesswork; with it, the photographer just sets one number.

#### E. Phase 3 starter — auth + mobile-ready API + Docker
- `backend/auth/jwt_auth.py` — issue/verify JWTs, no external deps
  (uses `hmac` + `hashlib`).
- `users` table in SQLite + minimal `/auth/register`, `/auth/login`
  endpoints (kept disabled by default behind `SNAPPY_AUTH=1` so existing
  demo flow doesn't break).
- `deploy/Dockerfile` (CPU + CUDA variants) + `deploy/docker-compose.yml`.
- `deploy/github-actions-ci.yml` template.

### Migration / breaking changes
- **None for the WS or HTTP API.** All existing endpoints still respond
  the same way.
- **Internal Python imports**: anything that did `from backend.api.server
  import Session, process_frame` now lives at
  `backend.api.session.Session` and `backend.api.pipeline.process_frame`.
  Tests and external scripts updated.

### How to verify
```bash
# Existing behaviour preserved
python3 run.py
curl localhost:8765/health         # → version v2.5
curl -X POST localhost:8765/sessions ... # works as before

# Multi-thread proof
SNAPPY_FRAME_WORKERS=8 python3 run.py
# Open two browser tabs streaming → both should hit ~20 fps each.

# L2CS active when installed
pip install l2cs
curl localhost:8765/health   # → "gaze": {"backend": "l2cs"}

# Calibration
# Click 👍/👎 ~30 times across multiple classes; then:
curl localhost:8765/sessions/<sid>/learning
# → "calibration": {"cake_cutting": {"a": 4.2, "b": -2.0, "n": 18}, ...}

# All unit tests
pytest tests/ -v
```

### Final v2.5 file inventory

```
backend/api/                      OLD (1 file, 1058 LoC)
                                  NEW (7 files, ~1100 LoC total)
  __init__.py        package marker
  globals.py         singletons + SESSIONS — 95 LoC
  session.py         Session dataclass — 60 LoC
  ws_protocol.py     RFC 6455 helpers — 90 LoC
  pipeline.py        process_frame + thread pool + video — 280 LoC
  routes.py          pure HTTP handlers — 230 LoC
  server.py          asyncio.Protocol + dispatch — 250 LoC

backend/models/
  l2cs_gaze.py       L2CS-Net gaze regressor — NEW
  calibration.py     Platt scaling per class — NEW

backend/auth/
  __init__.py
  jwt_auth.py        JWT + PBKDF2 password hash — NEW

backend/utils/
  persistence.py     +calibration / +users tables

deploy/
  Dockerfile         CPU image
  Dockerfile.gpu     CUDA 12.1 image
  docker-compose.yml local stack
  github-actions-ci.yml  CI workflow template

tests/
  test_calibration.py  NEW
```

### Phase 3 / 4 progress after this turn

| PDF item | Status |
|---|---|
| **Phase 1 — Multi-threaded frame processing** | ✅ Done (`SNAPPY_FRAME_WORKERS`) |
| **Phase 1 — Unit tests** | ✅ 4 test files, all passing |
| **Phase 2 — Confidence calibration** | ✅ Done (`models/calibration.py`) |
| **Phase 2 — L2CS-Net / GazeNet** | ✅ Wrapper done (drops in when `pip install l2cs`) |
| Phase 3 — JWT auth | ✅ Issuer/verifier ready; Bearer enforcement opt-in via `SNAPPY_AUTH=1` |
| Phase 3 — User accounts | ✅ `users` table + create/find/touch helpers |
| Phase 3 — FastAPI migration | ⏳ Deferred (current pure-stdlib server runs at >2K req/s) |
| Phase 3 — React Native app | ⏳ Outside backend scope; existing browser frontend works on mobile Safari/Chrome |
| Phase 4 — Docker | ✅ CPU + CUDA images |
| Phase 4 — docker-compose | ✅ Local stack ready |
| Phase 4 — CI pipeline | ✅ GitHub Actions workflow template |
| Phase 4 — AWS / Stripe / Redis | ⏳ Documented in compose file as commented-out hooks; full Terraform deferred |

---

## v2.4 — Live prompts + auto-classifier + GPU + tests
**Date:** 2026-05-07.

### Why
User asked for: "model that learns from new events", live voice/text
prompt change, GPU support, proper file structure, clear Phase 1/2
status. Most of these were Phase 3 features in the PDF (mobile/voice)
or new asks not in the PDF (auto-discovery of moment classes).

### Changes
- `backend/gpu/device.py` — single source of truth for CUDA/MPS/CPU
  detection. All model wrappers consult `get_device()`.
- `backend/voice/transcription.py` — optional server-side `faster-whisper`
  STT; primary STT path is browser Web Speech API.
- `backend/models/prompt_router.py` — text/voice → moment classes with
  add/remove/replace intent inference. No LLM, deterministic, ~150 LOC.
- `backend/models/auto_classifier.py` — DBSCAN on CLIP embeddings of
  high-quality "unknown" frames. Emits `DiscoveredCluster` events when
  a cluster ≥ MIN_SAMPLES forms. User names it; centroid joins the
  trained class set.
- `backend/training/continuous_learning.py` — background thread that
  rewrites `clip_class_centroids.npz` from the photos the user 👍'd.
  Opt-in via `SNAPPY_CONTINUOUS_LEARNING=1`.
- `tests/` folder with pytest suite for persistence, online learner,
  prompt router.
- `ARCHITECTURE.md` written.
- README phase status checklist added.

### Verify
- WS message `{"type":"prompt_update","text":"…"}` now changes active
  classes mid-stream.
- 6+ similar unknown captures → "🆕 New moments discovered" UI card.

---

## v2.3 — Tier-3 model zoo (NIMA + YOLOv8-face + HSEmotion)

### Why
Phase 2 PDF deliverable: production-grade models targeting ~90% accuracy.
Heuristics in `shot_quality.py` and blendshape-only emotion were the
ceiling.

### Changes
- `backend/models/yolo_face.py` — Ultralytics YOLOv8-face. ~6 MB weights
  auto-download. Used as crowd booster when MediaPipe finds < 3 faces.
- `backend/models/nima_aesthetic.py` — pyiqa NIMA. ResNet-on-AVA. Blends
  into `ShotQualityAnalyzer` at 0.45 weight when available.
- `backend/models/hsemotion_fer.py` — AffectNet ONNX FER. Blends 70/30
  with the existing blendshape emotion.
- `requirements.txt` reorganised into 4 tiers with accuracy targets.
- `INSTALL.md` documents each tier and the auto-downloads.

### Verify
`/health` shows `"face": {"backend": "mediapipe+yolo"}` and `"nima"` /
`"hsemotion"` `"available": true` once installed.

---

## v2.2 — Real CLIP + MediaPipe FaceMesh + emotion + online learning

### Why
The user asked for end-to-end best-in-class detection with online
learning. Phase 1 had upgraded to MediaPipe Face Landmarker for face
detection but `shot_quality.py` was still using Haar cascades, gaze was
still iris-centroid, and CLIP was a heuristic mock.

### Changes
- `backend/models/face_provider.py` — singleton with FaceLandmarker
  (478 landmarks + iris + 52 blendshapes). Haar fallback.
- `backend/models/emotion_detector.py` — Ekman emotion vector from
  blendshapes (zero extra deps).
- `backend/models/gaze_estimator.py` — iris-corner geometry replacing
  the threshold-centroid hack.
- `backend/models/clip_engine.py` — real CLIP ViT-B/32 via HuggingFace
  transformers. Lazy load. Per-session "kept centroid" few-shot trick.
- `backend/models/online_learner.py` — Beta-Bernoulli per-class priors
  with persistence.
- `feedback` + `feedback_priors` SQLite tables.
- Frontend 👍/👎 buttons.

---

## v2.1 — Foundation hardening

### Why
Phase 1 PDF tasks: SQLite session persistence, file logging,
configurable thresholds.

### Changes
- `backend/utils/persistence.py` — SQLite store, sessions/photos.
- `backend/utils/logging_setup.py` — RotatingFileHandler at
  `logs/snappy.log`.
- `backend/utils/config.py` — env-driven thresholds.
- `_restore_sessions_from_store()` on boot.
- `/health` endpoint.

---

## v2.0 — Working prototype (initial state)

The version we received: pure-stdlib asyncio server, Haar face
detection, heuristic VLM moment detector, in-memory sessions. ~12 fps,
~60% moment accuracy, no learning.
