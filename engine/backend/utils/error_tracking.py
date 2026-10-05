"""
backend/utils/error_tracking.py — Sentry initialisation (no-op without DSN).

Set SNAPPY_SENTRY_DSN to enable; everything else is automatic via the
Sentry SDK's stdlib + asyncio integrations.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger("snappy.errortracking")

_initialized: bool = False
_dsn: Optional[str] = None


def init_sentry() -> bool:
    global _initialized, _dsn
    if _initialized:
        return True
    dsn = os.environ.get("SNAPPY_SENTRY_DSN", "").strip()
    if not dsn:
        log.debug("SNAPPY_SENTRY_DSN not set — Sentry disabled")
        return False
    try:
        import sentry_sdk
    except ImportError:
        log.warning("Sentry DSN set but sentry-sdk not installed — install: "
                    "pip install sentry-sdk")
        return False
    try:
        sentry_sdk.init(
            dsn=dsn,
            release=os.environ.get("SNAPPY_VERSION", "v2.5"),
            environment=os.environ.get("SNAPPY_ENV", "dev"),
            traces_sample_rate=float(os.environ.get("SNAPPY_SENTRY_TRACES", "0.1")),
            send_default_pii=False,
        )
        _dsn = dsn
        _initialized = True
        log.info(f"Sentry initialised: env={os.environ.get('SNAPPY_ENV','dev')}")
        return True
    except Exception as e:
        log.warning(f"Sentry init failed: {e}")
        return False


def capture_exception(exc: Exception) -> None:
    if not _initialized:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
