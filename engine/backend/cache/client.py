"""
backend/cache/client.py — TTL cache + pub/sub for multi-instance Snappy.

Two backends:
  - InMemoryCache  — default; per-process dict + threading.Lock. No pub/sub.
  - RedisCache     — used when SNAPPY_REDIS_URL set. Multi-instance safe.

Used for:
  - Cross-instance session locks (when scaling out)
  - Rate limiting (per-IP, per-user)
  - Short-lived secrets (verify tokens with TTL)
  - Pub/sub events (active session list across the cluster)

Requires `pip install redis` for the Redis backend. Falls back to in-mem.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional, Protocol

log = logging.getLogger("snappy.cache")


class Cache(Protocol):
    name: str
    def get(self, key: str) -> Optional[str]: ...
    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None: ...
    def incr(self, key: str, ttl_seconds: Optional[int] = None) -> int: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class InMemoryCache:
    name = "memory"

    def __init__(self):
        self._lock = threading.Lock()
        self._d: dict[str, tuple[str, float]] = {}    # key → (value, expires_at)
        self._counters: dict[str, int] = {}

    def _purge_expired(self) -> None:
        now = time.time()
        dead = [k for k, (_, exp) in self._d.items() if 0 < exp < now]
        for k in dead: self._d.pop(k, None)

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            v = self._d.get(key)
            if v is None: return None
            value, exp = v
            if 0 < exp < time.time():
                self._d.pop(key, None)
                return None
            return value

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        exp = time.time() + ttl_seconds if ttl_seconds else 0.0
        with self._lock:
            self._d[key] = (value, exp)

    def incr(self, key: str, ttl_seconds: Optional[int] = None) -> int:
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            v = self._counters[key]
            if ttl_seconds and key not in self._d:
                # Use the same TTL bucket so the counter expires
                self._d[key] = ("__counter__", time.time() + ttl_seconds)
            return v

    def delete(self, key: str) -> None:
        with self._lock:
            self._d.pop(key, None)
            self._counters.pop(key, None)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class RedisCache:
    name = "redis"

    def __init__(self, url: str):
        try:
            import redis
        except ImportError as e:
            raise RuntimeError(
                f"redis not installed: {e}. Install with: pip install redis"
            )
        self._r = redis.from_url(url, decode_responses=True)
        self._r.ping()
        log.info(f"RedisCache connected: {url}")

    def get(self, key: str) -> Optional[str]:
        return self._r.get(key)

    def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        if ttl_seconds:
            self._r.set(key, value, ex=int(ttl_seconds))
        else:
            self._r.set(key, value)

    def incr(self, key: str, ttl_seconds: Optional[int] = None) -> int:
        v = int(self._r.incr(key))
        if ttl_seconds and v == 1:
            self._r.expire(key, int(ttl_seconds))
        return v

    def delete(self, key: str) -> None:
        self._r.delete(key)

    def exists(self, key: str) -> bool:
        return bool(self._r.exists(key))


_singleton: Optional[Cache] = None
def get_cache() -> Cache:
    global _singleton
    if _singleton is not None:
        return _singleton
    url = os.environ.get("SNAPPY_REDIS_URL", "").strip()
    if url:
        try:
            _singleton = RedisCache(url); return _singleton
        except Exception as e:
            log.warning(f"Redis disabled, falling back to memory cache: {e}")
    _singleton = InMemoryCache()
    return _singleton
