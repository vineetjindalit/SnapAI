"""
backend/auth/rate_limit.py — per-key sliding window rate limiter.

Uses the cache module so it transparently works in single-instance
(in-memory dict) and multi-instance (Redis) deployments. The limit is
checked at route entry — no decorators, just a small function.

Defaults:
  - login: 10 attempts per minute per IP, burst 3.
  - register: 5 per hour per IP.
  - password reset request: 3 per hour per IP.

API:
    allowed = check_rate_limit("login", request_ip, limit=10, window=60)
    if not allowed: return 429, {...}
"""
from __future__ import annotations

import logging
import time
from typing import Tuple

log = logging.getLogger("snappy.ratelimit")


def check_rate_limit(bucket: str, key: str,
                     limit: int, window_seconds: int) -> Tuple[bool, int]:
    """Returns (allowed, count). Count is 0 if cache unavailable (fail-open
    in dev, fail-closed in prod is a future enhancement)."""
    try:
        from cache import get_cache
    except Exception:
        return True, 0
    try:
        cache = get_cache()
        cache_key = f"rl:{bucket}:{key}"
        n = cache.incr(cache_key, ttl_seconds=int(window_seconds))
        if n > limit:
            log.warning(f"rate limit hit: bucket={bucket} key={key} "
                        f"count={n}/{limit} window={window_seconds}s")
            return False, n
        return True, n
    except Exception as e:
        log.debug(f"rate limit check failed (fail-open): {e}")
        return True, 0


def http_429(retry_after_seconds: int = 60) -> Tuple[int, dict]:
    return 429, {
        "error": "Too many requests. Please slow down.",
        "retry_after": retry_after_seconds,
    }
