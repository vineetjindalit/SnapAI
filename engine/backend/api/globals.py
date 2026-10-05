"""
backend/api/globals.py — singletons + shared mutable state.

The "DI container" of the server. Routes import from here instead of from
server.py, so there are no import cycles. Created once on module import.

WHY a globals module instead of constructor injection?
- The routes are pure functions (status, body) and don't have a class to
  hold instance state.
- The model singletons are heavy and should be created exactly once.
- A single module-level container keeps the wiring obvious — anyone
  reading routes.py sees `from .globals import CLIP, STORE, SESSIONS`
  and knows exactly where state lives.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Ensure the backend package is importable regardless of how server is launched.
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from models.album_generator     import AlbumGenerator
from models.auto_classifier     import AutoClassifier
from models.clip_engine         import CLIPEngine
from models.clip_moment_detector import MOMENT_PROFILES
from models.face_provider       import FaceProvider
from models.hsemotion_fer       import HSEmotionRecognizer
from models.nima_aesthetic      import NIMAScorer
from models.online_learner      import OnlineLearner
from models.prompt_router       import PromptRouter
from utils.config               import CONFIG
from utils.logging_setup        import configure_logging
from utils.persistence          import SessionStore
from voice.transcription        import WhisperTranscriber

# ── Filesystem layout ─────────────────────────────────────────────────────
CAPTURES_DIR = ROOT / "captures"
ALBUMS_DIR   = ROOT / "albums"
FRONTEND_DIR = ROOT / "frontend"
LOGS_DIR     = ROOT / "logs"
DATA_DIR     = ROOT / "backend" / "data"
for d in (CAPTURES_DIR, ALBUMS_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────
configure_logging(LOGS_DIR)
log = logging.getLogger("snappy")


# ── Captures housekeeping ───────────────────────────────────────────────────
def _sweep_empty_capture_dirs() -> None:
    """Delete leftover capture folders that hold no photos.

    Sessions create their capture folder lazily (on the first saved photo), so
    a healthy tree only contains folders with real captures. But a server that
    was killed mid-upload can leave behind a folder that is either empty or
    holds just the temporary `_upload.*` video and no snaps. Those carry no
    photos, so we remove them at boot to keep captures/ tidy.

    Only the immediate `<sid>` / `<sid>_<video>` subfolders are considered, and
    a folder is removed solely when it contains no real capture — never one
    with saved photos.
    """
    try:
        subdirs = [d for d in CAPTURES_DIR.iterdir() if d.is_dir()]
    except OSError:
        return
    removed = 0
    for d in subdirs:
        try:
            entries = list(d.iterdir())
            # A folder counts as "no captures" when it is empty, or holds only
            # the orphaned temp upload from a crashed run.
            only_temp = all(e.is_file() and e.name.startswith("_upload")
                            for e in entries)
            if entries and not only_temp:
                continue
            for e in entries:
                e.unlink()
            d.rmdir()
            removed += 1
        except OSError:
            pass  # in-use or permission issue — skip, never fail boot
    if removed:
        log.info(f"captures housekeeping: removed {removed} empty folder(s)")


_sweep_empty_capture_dirs()

# ── Persistence ───────────────────────────────────────────────────────────
STORE: Optional[SessionStore] = (
    SessionStore(DATA_DIR / "snappy.db") if CONFIG.persistence_enabled else None
)

# ── Model zoo singletons ──────────────────────────────────────────────────
FACE_PROVIDER = FaceProvider.get()
CLIP          = CLIPEngine.get()
# Eagerly warm CLIP at boot unless explicitly disabled. Without this the
# first frame of the first session pays a ~5s latency hit while CLIP loads
# (and /health reports clip.available=false until then, which is misleading).
if os.environ.get("SNAPPY_CLIP_EAGER_WARMUP", "1") != "0":
    try: CLIP.warmup()
    except Exception as ex: log.warning(f"CLIP eager warmup failed: {ex}")
NIMA          = NIMAScorer.get()
HSEMO         = HSEmotionRecognizer.get()
# Eagerly warm the local album-tagging VLM (Qwen2.5-VL) in the BACKGROUND at
# boot, same rationale as CLIP above: album_generator.py's optional
# photographer-grade tagging pass (see album_generator.py ~line 312) used to
# lazy-load this multi-GB model on the first "Generate Album" tap of the
# day, which could blow well past the few-minutes UX target for culling a
# batch of photos. A background thread means server startup itself isn't
# blocked by the download/load, but by the time a real partner's first
# session finishes (minutes, not milliseconds), it's already warm.
if os.environ.get("SNAPPY_VLM_TAGGER_EAGER_WARMUP", "1") != "0":
    try:
        import threading as _threading
        from models.vlm_tagger import VLMTagger as _VLMTagger
        _threading.Thread(target=_VLMTagger.get, daemon=True).start()
    except Exception as ex:
        log.warning(f"VLM tagger background warmup failed to start: {ex}")
LEARNER       = OnlineLearner(STORE)
PROMPT_ROUTER = PromptRouter(MOMENT_PROFILES, clip_engine=CLIP)
AUTO_CLASS    = AutoClassifier(store=STORE)
WHISPER       = WhisperTranscriber.get()
album_gen     = AlbumGenerator(str(ALBUMS_DIR))

# Audio event detector (AST / AudioSet) — timing signal from sound: cheering,
# applause, singing, laughter. Lazy: NOT warmed at boot (the model is ~350MB
# and only the recorded-video path with an audio track needs it). Loads on the
# first clip that actually has audio. Degrades to no-op if torchaudio/AST or
# the audio stream are missing.
try:
    from models.audio_event_detector import AudioEventDetector
    AUDIO = AudioEventDetector.get()
except Exception as ex:
    log.warning(f"audio detector unavailable: {ex}")
    AUDIO = None

# Optional confidence calibrator (Platt-scale per moment class).
# Lazy-imported to avoid forcing sklearn at boot if unused.
try:
    from models.calibration import ConfidenceCalibrator
    CALIBRATOR: Optional["ConfidenceCalibrator"] = ConfidenceCalibrator(store=STORE)
except Exception as ex:
    log.warning(f"calibration module unavailable: {ex}")
    CALIBRATOR = None

# Optional L2CS gaze upgrade.
try:
    from models.l2cs_gaze import L2CSGaze
    L2CS = L2CSGaze.get()
    if L2CS.available:
        log.info("L2CS-Net gaze active")
except Exception as ex:
    log.debug(f"L2CS gaze not loaded: {ex}")
    L2CS = None

# Optional continuous learning thread (opt-in via env var).
_CONT_LEARNER = None
if os.environ.get("SNAPPY_CONTINUOUS_LEARNING", "0") == "1":
    try:
        from training.continuous_learning import ContinuousLearner
        _CONT_LEARNER = ContinuousLearner(STORE, CLIP, AUTO_CLASS)
        _CONT_LEARNER.start()
        log.info("ContinuousLearner started")
    except Exception as ex:
        log.warning(f"continuous learner failed to start: {ex}")

# Storage / cache / billing / sentry — all opt-in via env vars
try:
    from storage import get_storage
    STORAGE = get_storage()
except Exception as ex:
    log.warning(f"storage unavailable: {ex}")
    STORAGE = None

try:
    from cache import get_cache
    CACHE = get_cache()
except Exception as ex:
    log.warning(f"cache unavailable: {ex}")
    CACHE = None

try:
    from billing import get_billing
    BILLING = get_billing()
    if BILLING.available:
        log.info("Stripe billing active")
except Exception as ex:
    log.warning(f"billing unavailable: {ex}")
    BILLING = None

try:
    from utils.error_tracking import init_sentry
    init_sentry()
except Exception as ex:
    log.debug(f"sentry init skipped: {ex}")

# ── Live state ────────────────────────────────────────────────────────────
# Active sessions, keyed by sid. Mutated by routes (create/end) and by the
# pipeline (frame counts, capture counts).
SESSIONS: Dict[str, "Session"] = {}    # forward-ref Session

# Dev-dashboard telemetry plumbing. A WS client registers a thread-safe push
# fn keyed by sid; the video-upload pipeline uses it to stream per-frame
# frame_results to the dashboard so the model-activity bars animate through an
# uploaded clip (live frames already stream via the WS message loop).
EVENT_LOOP = None                       # set in server.main()
# sid → {client_id: push_callable}. A dict-of-dicts so MULTIPLE dashboards
# (tabs / tools) on the same session each get the stream — last-writer-wins
# would otherwise let one connection steal another's telemetry.
WS_CLIENTS: Dict[str, dict] = {}


def push_to_session(sid: str, msg: dict) -> None:
    """Broadcast a telemetry message to every dashboard WS on `sid`.
    No-op when nobody is watching. Safe to call from a worker thread."""
    clients = WS_CLIENTS.get(sid)
    if clients:
        for fn in list(clients.values()):
            try:
                fn(msg)
            except Exception:
                pass
