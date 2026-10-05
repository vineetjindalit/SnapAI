"""
utils/logging_setup.py — Phase 1 (Foundation Hardening): logging pipeline.

Roadmap task: "Build a basic logging and error monitoring pipeline."
Until now we used `logging.basicConfig` to stderr — fine for dev, useless for
post-event debugging. This wires a rotating file handler under logs/snappy.log.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.config import CONFIG


def configure_logging(logs_dir: Path) -> Path:
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "snappy.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, CONFIG.log_level, logging.INFO))

    # Replace any pre-existing handlers (basicConfig may have added one).
    for h in list(root.handlers):
        root.removeHandler(h)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(sh)

    fh = RotatingFileHandler(
        log_path,
        maxBytes=CONFIG.log_max_bytes,
        backupCount=CONFIG.log_backup_count,
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)

    logging.getLogger("snappy").info("Logging configured → %s", log_path)
    return log_path
