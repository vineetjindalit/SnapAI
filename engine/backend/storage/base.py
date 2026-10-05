"""
backend/storage/base.py — Storage interface for captured photos & albums.

WHY abstract this?
  - Local dev should be `local` (writes to ./captures, no AWS needed).
  - Production deployment uses `s3` (durable, scalable, supports CDN).
  - Both must work without code changes — pick via SNAPPY_STORAGE env var.

Two operations matter:
  1. put_photo(sid, filename, bytes) → returns the URL the frontend
     should fetch this from.
  2. signed_url(path) → returns a time-limited URL for sharing albums
     without exposing the storage account publicly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass
class StorageObject:
    """A stored photo / file. The URL is what the frontend uses to fetch."""
    url:         str            # e.g. /captures/abc/snap_0001.jpg or https://cdn.example.com/...
    storage_path: str           # internal — local path or S3 key
    bytes:       int


class Storage(Protocol):
    """Anything implementing this works as a Snappy storage backend."""
    name: str

    def put_photo(self, sid: str, filename: str, data: bytes) -> StorageObject: ...
    def get_photo(self, sid: str, filename: str) -> Optional[bytes]: ...
    def signed_url(self, sid: str, filename: str,
                   expires_seconds: int = 7 * 24 * 3600) -> str: ...
    def delete_session(self, sid: str) -> int: ...
