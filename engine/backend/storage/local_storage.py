"""
backend/storage/local_storage.py — local-filesystem Storage backend.

For dev + single-node deployments. Photos go under captures/<sid>/<file>.
URLs returned are relative paths the existing /captures route serves.
"""
from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import Optional

from .base import Storage, StorageObject

log = logging.getLogger("snappy.storage.local")


class LocalStorage:
    name = "local"

    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        log.info(f"LocalStorage rooted at {self._root}")

    def _session_dir(self, sid: str) -> Path:
        d = self._root / sid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put_photo(self, sid: str, filename: str, data: bytes) -> StorageObject:
        path = self._session_dir(sid) / filename
        path.write_bytes(data)
        return StorageObject(
            url=f"/captures/{sid}/{filename}",
            storage_path=str(path),
            bytes=len(data),
        )

    def get_photo(self, sid: str, filename: str) -> Optional[bytes]:
        path = self._session_dir(sid) / filename
        if not path.exists():
            return None
        return path.read_bytes()

    def signed_url(self, sid: str, filename: str,
                   expires_seconds: int = 7 * 24 * 3600) -> str:
        # Local URLs aren't time-limited; we just return the same URL.
        # If you need expiry on local storage, add a token check in the
        # /captures route — out of scope for the abstraction.
        return f"/captures/{sid}/{filename}?ts={int(time.time())}"

    def delete_session(self, sid: str) -> int:
        d = self._root / sid
        if not d.exists():
            return 0
        n = sum(1 for _ in d.iterdir())
        shutil.rmtree(d, ignore_errors=True)
        return n
