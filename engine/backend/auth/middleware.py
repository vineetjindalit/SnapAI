"""
backend/auth/middleware.py — request-time JWT verification helpers.

Used by routes that need the caller's identity. Pattern:

    user = require_user(headers)
    if isinstance(user, ErrorResult):
        return user.status, user.body
    # ...else `user` is a dict with the token claims

Toggling AUTH:
  SNAPPY_AUTH=0 (default) → require_user always returns AnonymousUser.
  SNAPPY_AUTH=1           → enforced; missing/invalid token → 401.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from .jwt_auth import JWTAuth

_AUTH_ENABLED = os.environ.get("SNAPPY_AUTH", "0") == "1"
_jwt_singleton: Optional[JWTAuth] = None


def _jwt() -> JWTAuth:
    global _jwt_singleton
    if _jwt_singleton is None:
        _jwt_singleton = JWTAuth()
    return _jwt_singleton


@dataclass
class ErrorResult:
    status: int
    body: Dict[str, Any]


@dataclass
class User:
    user_id: int
    email:   str
    role:    str = "user"
    anonymous: bool = False


_ANON = User(user_id=0, email="anonymous@local", role="anon", anonymous=True)


def require_user(headers: Dict[str, str]) -> Union[User, ErrorResult]:
    """Return User if a valid token is presented OR auth is disabled.
    Otherwise an ErrorResult with status 401."""
    if not _AUTH_ENABLED:
        return _ANON
    auth_header = headers.get("authorization", "")
    token = JWTAuth.from_authorization_header(auth_header)
    if not token:
        return ErrorResult(401, {"error": "Missing Bearer token"})
    ok, payload = _jwt().verify(token)
    if not ok:
        return ErrorResult(401, {"error": payload.get("error", "Invalid token")})
    try:
        return User(
            user_id=int(payload["sub"]),
            email=str(payload.get("email", "")),
            role=str(payload.get("role", "user")),
        )
    except (KeyError, ValueError) as e:
        return ErrorResult(401, {"error": f"Bad token claims: {e}"})


def auth_enabled() -> bool:
    return _AUTH_ENABLED
