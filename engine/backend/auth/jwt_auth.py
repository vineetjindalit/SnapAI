"""
backend/auth/jwt_auth.py — minimal JWT issuer/verifier + bcrypt-style password hashing.

WHY no PyJWT dependency?
  We're already shipping a pure-stdlib server. Adding PyJWT just for
  HMAC-SHA256 signing is overkill — RFC 7519 specifies the format and
  Python stdlib has hmac + base64. ~50 lines, zero deps.

Password hashing uses PBKDF2-SHA256 (also stdlib), 600k iterations
following OWASP 2023 guidance. NOT bcrypt-compatible — if you migrate
to bcrypt later, run users through a one-time hash rotation on login.

Signing key:
  - First-class env var: SNAPPY_JWT_SECRET
  - If unset, generated at first import from os.urandom and cached on
    disk at backend/data/.jwt_secret. This means restarts keep the same
    key (sessions stay valid) but a fresh checkout starts new.
  - For multi-host deployment, set SNAPPY_JWT_SECRET to a shared value.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

log = logging.getLogger("snappy.auth")

DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7   # 7 days
PBKDF2_ITERATIONS   = 600_000


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    s = s + "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s.encode("ascii"))


def _resolve_secret() -> bytes:
    env = os.environ.get("SNAPPY_JWT_SECRET", "").strip()
    if env:
        return env.encode()
    cache = Path(__file__).resolve().parent.parent / "data" / ".jwt_secret"
    if cache.exists():
        return cache.read_bytes()
    cache.parent.mkdir(parents=True, exist_ok=True)
    secret = secrets.token_bytes(32)
    cache.write_bytes(secret)
    try:
        os.chmod(cache, 0o600)
    except Exception:
        pass
    log.info(f"generated new JWT secret → {cache}")
    return secret


# ── Password hashing (PBKDF2-SHA256) ─────────────────────────────────────
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt,
                             PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${_b64url(salt)}${_b64url(dk)}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iters_s, salt_b64, dk_b64 = encoded.split("$")
        if algo != "pbkdf2_sha256":
            return False
        iters = int(iters_s)
        salt = _b64url_decode(salt_b64)
        expected = _b64url_decode(dk_b64)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                        salt, iters)
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False


# ── JWT (HS256) ──────────────────────────────────────────────────────────
class JWTAuth:
    def __init__(self, secret: Optional[bytes] = None,
                 ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._secret = secret or _resolve_secret()
        self._ttl    = int(ttl_seconds)

    def issue(self, subject: str, **extra: object) -> str:
        """Issue a JWT for `subject` (typically user_id). Adds a JWT ID
        (`jti`) so the token can be revoked at logout via the cache."""
        now = int(time.time())
        header  = {"alg": "HS256", "typ": "JWT"}
        payload = {"sub": subject, "iat": now, "exp": now + self._ttl,
                   "jti": secrets.token_urlsafe(16), **extra}
        h_b = _b64url(json.dumps(header,  separators=(",", ":")).encode())
        p_b = _b64url(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{h_b}.{p_b}".encode()
        sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        return f"{h_b}.{p_b}.{_b64url(sig)}"

    def verify(self, token: str, check_revoked: bool = True) -> Tuple[bool, Dict]:
        """Returns (ok, payload_or_error_dict)."""
        try:
            h_b, p_b, s_b = token.split(".")
        except ValueError:
            return False, {"error": "malformed token"}
        signing_input = f"{h_b}.{p_b}".encode()
        expected = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        actual = _b64url_decode(s_b)
        if not hmac.compare_digest(expected, actual):
            return False, {"error": "bad signature"}
        try:
            payload = json.loads(_b64url_decode(p_b))
        except Exception:
            return False, {"error": "bad payload"}
        if int(payload.get("exp", 0)) < int(time.time()):
            return False, {"error": "expired"}
        if check_revoked:
            jti = payload.get("jti")
            if jti and _is_revoked(jti):
                return False, {"error": "revoked"}
        return True, payload

    def revoke(self, token: str) -> bool:
        """Add a token's JTI to the denylist until its natural expiry."""
        ok, payload = self.verify(token, check_revoked=False)
        if not ok: return False
        jti = payload.get("jti")
        if not jti: return False
        ttl = max(60, int(payload.get("exp", 0)) - int(time.time()))
        return _revoke_jti(jti, ttl)

    @staticmethod
    def from_authorization_header(value: str) -> Optional[str]:
        """Extract the token from `Authorization: Bearer <token>`."""
        if not value:
            return None
        parts = value.strip().split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None
        return parts[1]


# ── Token denylist (uses cache module so it's multi-instance safe) ──────
def _is_revoked(jti: str) -> bool:
    try:
        from cache import get_cache
        return get_cache().exists(f"jti_denylist:{jti}")
    except Exception:
        return False


def _revoke_jti(jti: str, ttl_seconds: int) -> bool:
    try:
        from cache import get_cache
        get_cache().set(f"jti_denylist:{jti}", "1", ttl_seconds=ttl_seconds)
        return True
    except Exception as e:
        log.warning(f"could not denylist jti={jti}: {e}")
        return False
