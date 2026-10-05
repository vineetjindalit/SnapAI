"""
utils/config.py — Phase 1 (Foundation Hardening): centralized tunable thresholds.

The roadmap calls out "Tune detection thresholds using 5+ real event video recordings"
as a Phase 1 task. Pulling these out of the model files into one place lets us
sweep them on benchmark runs without code edits, and lets ops override via env vars.
"""
import os
from dataclasses import dataclass


def _f(env: str, default: float) -> float:
    try:
        return float(os.environ.get(env, default))
    except (TypeError, ValueError):
        return default


def _i(env: str, default: int) -> int:
    try:
        return int(os.environ.get(env, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SnappyConfig:
    # Capture cadence
    min_capture_interval_s: float = _f("SNAPPY_MIN_CAPTURE_INTERVAL", 2.5)

    # Quality thresholds
    best_shot_threshold: float    = _f("SNAPPY_BEST_SHOT_THRESHOLD", 0.55)
    best_shot_uplift:    float    = _f("SNAPPY_BEST_SHOT_UPLIFT",    1.07)

    # Near-duplicate suppression (api/pipeline.py + models/dedup.py).
    # A new capture is dropped when its perceptual hash is within
    # `dedup_threshold` bits (out of 64) of any of the last `dedup_window`
    # kept captures — so the captures folder holds only distinct moments.
    dedup_enabled:   bool = os.environ.get("SNAPPY_DEDUP", "1") != "0"
    dedup_threshold: int  = _i("SNAPPY_DEDUP_THRESHOLD", 6)
    dedup_window:    int  = _i("SNAPPY_DEDUP_WINDOW",    40)

    # Smart zoom (models/auto_zoom.py). We only save a zoomed-in crop when
    # the subject is FAR — i.e. the tallest detected face occupies less than
    # this fraction of the frame height. Above it the subject already fills
    # the frame, so the wide shot is kept as-is (no wide+zoom duplication).
    zoom_face_fill_max: float = _f("SNAPPY_ZOOM_FACE_FILL_MAX", 0.16)

    # History sizes
    quality_history:  int = _i("SNAPPY_QUALITY_HISTORY",  60)
    gallery_size:     int = _i("SNAPPY_GALLERY_SIZE",     60)
    timeline_size:    int = _i("SNAPPY_TIMELINE_SIZE",    300)

    # Persistence
    persistence_enabled: bool = os.environ.get("SNAPPY_PERSISTENCE", "1") != "0"

    # Logging
    log_level: str = os.environ.get("SNAPPY_LOG_LEVEL", "INFO").upper()
    log_max_bytes:    int = _i("SNAPPY_LOG_MAX_BYTES", 5 * 1024 * 1024)
    log_backup_count: int = _i("SNAPPY_LOG_BACKUPS", 3)


CONFIG = SnappyConfig()
