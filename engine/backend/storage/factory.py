"""
backend/storage/factory.py — pick LocalStorage or S3Storage from env.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import Storage
from .local_storage import LocalStorage

log = logging.getLogger("snappy.storage")

_singleton: Storage | None = None


def get_storage() -> Storage:
    global _singleton
    if _singleton is not None:
        return _singleton

    backend = os.environ.get("SNAPPY_STORAGE", "local").lower()

    if backend == "s3":
        try:
            from .s3_storage import S3Storage
            bucket = os.environ.get("SNAPPY_S3_BUCKET")
            region = os.environ.get("SNAPPY_S3_REGION", "us-east-1")
            if not bucket:
                raise RuntimeError("SNAPPY_S3_BUCKET not set; falling back to local")
            _singleton = S3Storage(
                bucket=bucket, region=region,
                public_base=os.environ.get("SNAPPY_S3_PUBLIC_BASE"),
                prefix=os.environ.get("SNAPPY_S3_PREFIX", ""),
            )
            return _singleton
        except Exception as e:
            log.warning(f"S3 unavailable, falling back to local: {e}")

    root = Path(__file__).resolve().parent.parent.parent / "captures"
    _singleton = LocalStorage(root)
    return _singleton
