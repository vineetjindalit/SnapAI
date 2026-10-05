"""
backend/auth/routes.py — auth HTTP route handlers.

Endpoints:
  POST /auth/register        {email, password}
  POST /auth/login           {email, password} → {token, user}
  POST /auth/logout          (auth) → revokes the bearer token
  POST /auth/refresh         (auth) → fresh token, reused jti is revoked
  GET  /auth/me              (auth) → user profile
  POST /auth/verify          {token: <verify_token from email>}
  POST /auth/forgot          {email} → 200 (always — anti-enumeration)
  POST /auth/reset           {token, password} → set new password

All write endpoints behind `_with_rate_limit()` to slow brute force.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
from typing import Any, Dict, Optional, Tuple

from mail.sender import (send_password_reset, send_verify_link,
                         send_welcome)

from .jwt_auth   import JWTAuth, hash_password, verify_password
from .middleware import ErrorResult, auth_enabled, require_user
from .rate_limit import check_rate_limit, http_429

log = logging.getLogger("snappy.auth.routes")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_jwt = JWTAuth()


def _public_url(path: str) -> str:
    base = os.environ.get("SNAPPY_PUBLIC_URL",
                          "http://localhost:8765").rstrip("/")
    return f"{base}{path}"


def _bad(status: int, msg: str) -> Tuple[int, Dict[str, Any]]:
    return status, {"error": msg}


def _ok(extra: Optional[Dict[str, Any]] = None,
        status: int = 200) -> Tuple[int, Dict[str, Any]]:
    body: Dict[str, Any] = {"ok": True}
    if extra: body.update(extra)
    return status, body


def _validate_password(pw: str) -> Optional[str]:
    """Return error message if password is weak, else None."""
    if len(pw) < 8:
        return "Password must be at least 8 characters"
    if pw.lower() in {"password", "12345678", "qwerty12", "snappy123"}:
        return "Password is too common; please pick something stronger"
    has_letter = any(c.isalpha() for c in pw)
    has_digit  = any(c.isdigit() for c in pw)
    if not (has_letter and has_digit):
        return "Password must contain both letters and numbers"
    return None


def _client_key(headers: Dict[str, str]) -> str:
    """Best-effort caller identifier for rate limiting."""
    return (headers.get("x-forwarded-for", "").split(",")[0].strip()
            or headers.get("x-real-ip", "")
            or "unknown")


# ── /auth/register ─────────────────────────────────────────────────────────
def register(store, body: bytes,
             headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, Any]]:
    if store is None:
        return _bad(503, "Persistence not configured")

    headers = headers or {}
    allowed, _ = check_rate_limit("register", _client_key(headers),
                                  limit=5, window_seconds=3600)
    if not allowed:
        return http_429(retry_after_seconds=3600)

    try:
        data = json.loads(body or b"{}")
    except Exception as e:
        return _bad(400, f"Bad JSON: {e}")
    email = str(data.get("email", "")).strip().lower()
    pw    = str(data.get("password", ""))

    if not _EMAIL_RE.match(email):
        return _bad(400, "Invalid email")
    err = _validate_password(pw)
    if err: return _bad(400, err)

    uid = store.create_user(email, hash_password(pw))
    if uid is None:
        return _bad(409, "Email already registered")

    # Fire-and-forget verification email
    try:
        verify_token = _jwt.issue(str(uid), email=email, purpose="verify")
        verify_url = _public_url(f"/verify?token={verify_token}")
        send_verify_link(email, verify_url)
    except Exception as e:
        log.warning(f"verify email send failed: {e}")
    try: send_welcome(email)
    except Exception: pass

    log.info(f"user registered: id={uid} email={email}")
    return _ok({"user_id": uid, "email": email,
                "auth_enabled": auth_enabled()}, status=201)


# ── /auth/login ────────────────────────────────────────────────────────────
def login(store, body: bytes,
          headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, Any]]:
    if store is None:
        return _bad(503, "Persistence not configured")

    headers = headers or {}
    allowed, _ = check_rate_limit("login", _client_key(headers),
                                  limit=10, window_seconds=60)
    if not allowed:
        return http_429(retry_after_seconds=60)

    try:
        data = json.loads(body or b"{}")
    except Exception as e:
        return _bad(400, f"Bad JSON: {e}")
    email = str(data.get("email", "")).strip().lower()
    pw    = str(data.get("password", ""))

    user = store.find_user_by_email(email)
    if not user or not verify_password(pw, user["password_hash"]):
        # Same error for both — don't leak whether email exists
        return _bad(401, "Invalid email or password")

    store.touch_user_login(user["id"])
    token = _jwt.issue(str(user["id"]), email=email,
                       role=user.get("role", "user"))
    log.info(f"user login: id={user['id']} email={email}")
    return _ok({
        "token": token,
        "user": {"id": user["id"], "email": email,
                 "role": user.get("role", "user"),
                 "verified": store.is_email_verified(user["id"])},
    })


# ── /auth/logout ───────────────────────────────────────────────────────────
def logout(headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    """Revokes the current bearer token via the cache denylist."""
    auth_h = headers.get("authorization", "")
    token = JWTAuth.from_authorization_header(auth_h)
    if not token:
        # Idempotent: 200 even if no token — client just clears local state
        return _ok({"revoked": False})
    ok = _jwt.revoke(token)
    return _ok({"revoked": bool(ok)})


# ── /auth/refresh ──────────────────────────────────────────────────────────
def refresh(headers: Dict[str, str]) -> Tuple[int, Dict[str, Any]]:
    """Issue a fresh token and revoke the old one. Lets long-lived clients
    rotate JTIs without forcing re-login."""
    user = require_user(headers)
    if isinstance(user, ErrorResult):
        return user.status, user.body
    new_token = _jwt.issue(str(user.user_id), email=user.email, role=user.role)
    # Revoke the old one so it can't be reused
    auth_h = headers.get("authorization", "")
    old = JWTAuth.from_authorization_header(auth_h)
    if old:
        _jwt.revoke(old)
    return _ok({"token": new_token})


# ── /auth/me ───────────────────────────────────────────────────────────────
def me(headers: Dict[str, str], store=None) -> Tuple[int, Dict[str, Any]]:
    user = require_user(headers)
    if isinstance(user, ErrorResult):
        return user.status, user.body
    # `user` is now a User (anonymous when SNAPPY_AUTH=0).
    is_anon = bool(user.anonymous)
    verified = (True if is_anon else
                (bool(store.is_email_verified(user.user_id)) if store else False))
    tier = ("free" if is_anon else
            (store.get_user_tier(user.user_id) if store else "free"))
    return _ok({
        "user": {"id": user.user_id, "email": user.email, "role": user.role,
                 "anonymous": is_anon, "verified": verified, "tier": tier},
        "auth_enabled": auth_enabled(),
    })


# ── /auth/verify ───────────────────────────────────────────────────────────
def verify_email(store, body: bytes) -> Tuple[int, Dict[str, Any]]:
    try:
        data = json.loads(body or b"{}")
    except Exception as e:
        return _bad(400, f"Bad JSON: {e}")
    token = str(data.get("token", ""))
    if not token:
        return _bad(400, "Missing token")
    ok, payload = _jwt.verify(token)
    if not ok:
        return _bad(400, payload.get("error", "Invalid token"))
    if payload.get("purpose") != "verify":
        return _bad(400, "Wrong token purpose")
    uid = int(payload["sub"])
    if store is not None:
        store.mark_email_verified(uid)
    log.info(f"email verified: user_id={uid}")
    return _ok({"user_id": uid, "email": payload.get("email", "")})


# ── /auth/forgot ───────────────────────────────────────────────────────────
def forgot_password(store, body: bytes,
                    headers: Optional[Dict[str, str]] = None
                    ) -> Tuple[int, Dict[str, Any]]:
    """Always returns 200 to prevent email enumeration. Send the email
    out-of-band only if the user exists."""
    headers = headers or {}
    allowed, _ = check_rate_limit("forgot", _client_key(headers),
                                  limit=3, window_seconds=3600)
    if not allowed:
        # Even rate-limit responses look uniform to the client
        return _ok({"sent": True})

    try:
        data = json.loads(body or b"{}")
    except Exception:
        return _ok({"sent": True})
    email = str(data.get("email", "")).strip().lower()
    if not _EMAIL_RE.match(email) or store is None:
        return _ok({"sent": True})

    user = store.find_user_by_email(email)
    if user:
        token = secrets.token_urlsafe(32)
        try:
            store.create_password_reset(user["id"], token, ttl_seconds=3600)
            reset_url = _public_url(f"/reset?token={token}")
            send_password_reset(email, reset_url)
        except Exception as e:
            log.exception(f"forgot password flow failed for {email}: {e}")
    return _ok({"sent": True})


# ── /auth/reset ────────────────────────────────────────────────────────────
def reset_password(store, body: bytes) -> Tuple[int, Dict[str, Any]]:
    if store is None:
        return _bad(503, "Persistence not configured")
    try:
        data = json.loads(body or b"{}")
    except Exception as e:
        return _bad(400, f"Bad JSON: {e}")
    token = str(data.get("token", ""))
    pw    = str(data.get("password", ""))
    if not token:  return _bad(400, "Missing token")
    err = _validate_password(pw)
    if err: return _bad(400, err)

    uid = store.consume_password_reset(token)
    if uid is None:
        return _bad(400, "Token is invalid, expired, or already used")
    store.update_user_password(uid, hash_password(pw))
    log.info(f"password reset: user_id={uid}")
    return _ok({"user_id": uid})
