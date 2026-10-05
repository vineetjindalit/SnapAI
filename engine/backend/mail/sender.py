"""
backend/email/sender.py — transactional email abstraction.

Three backends:
  - mailgun  — needs SNAPPY_EMAIL_MAILGUN_KEY + DOMAIN
  - sendgrid — needs SNAPPY_EMAIL_SENDGRID_KEY
  - console  — prints to stdout (dev default; never sends)

All HTTP clients use stdlib urllib so no new pip dep is required for
the simple POST-with-API-key pattern these vendors use.
"""
from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol

log = logging.getLogger("snappy.email")


@dataclass
class EmailMessage:
    to:        str
    subject:   str
    text:      str
    html:      Optional[str] = None
    from_addr: Optional[str] = None     # falls back to SNAPPY_EMAIL_FROM


class EmailSender(Protocol):
    name: str
    def send(self, msg: EmailMessage) -> bool: ...


class ConsoleSender:
    """Dev-only: prints the email to the log instead of sending."""
    name = "console"

    def send(self, msg: EmailMessage) -> bool:
        log.info("=" * 60)
        log.info(f"[CONSOLE EMAIL] to={msg.to!r} subject={msg.subject!r}")
        log.info(f"[CONSOLE EMAIL] body:\n{msg.text}")
        log.info("=" * 60)
        return True


class MailgunSender:
    name = "mailgun"

    def __init__(self, api_key: str, domain: str):
        self._api_key = api_key
        self._domain  = domain
        self._url     = f"https://api.mailgun.net/v3/{domain}/messages"

    def send(self, msg: EmailMessage) -> bool:
        from_addr = msg.from_addr or os.environ.get(
            "SNAPPY_EMAIL_FROM", f"SnapAI <noreply@{self._domain}>")
        data = urllib.parse.urlencode({
            "from": from_addr, "to": msg.to,
            "subject": msg.subject, "text": msg.text,
            **({"html": msg.html} if msg.html else {}),
        }).encode()
        req = urllib.request.Request(self._url, data=data, method="POST")
        # Basic auth: api key as the password, "api" as username
        import base64
        token = base64.b64encode(f"api:{self._api_key}".encode()).decode()
        req.add_header("Authorization", f"Basic {token}")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                ok = 200 <= r.status < 300
                if not ok:
                    log.warning(f"Mailgun send returned {r.status}")
                return ok
        except Exception as e:
            log.exception(f"Mailgun send failed: {e}")
            return False


class SendGridSender:
    name = "sendgrid"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._url = "https://api.sendgrid.com/v3/mail/send"

    def send(self, msg: EmailMessage) -> bool:
        from_addr = msg.from_addr or os.environ.get(
            "SNAPPY_EMAIL_FROM", "noreply@example.com")
        body = {
            "personalizations": [{"to": [{"email": msg.to}]}],
            "from": {"email": from_addr},
            "subject": msg.subject,
            "content": [{"type": "text/plain", "value": msg.text}] +
                       ([{"type": "text/html", "value": msg.html}] if msg.html else []),
        }
        req = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), method="POST")
        req.add_header("Authorization", f"Bearer {self._api_key}")
        req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                ok = 200 <= r.status < 300
                if not ok:
                    log.warning(f"SendGrid returned {r.status}: {r.read()[:200]!r}")
                return ok
        except Exception as e:
            log.exception(f"SendGrid send failed: {e}")
            return False


_singleton: Optional[EmailSender] = None


def get_email_sender() -> EmailSender:
    global _singleton
    if _singleton is not None:
        return _singleton

    backend = os.environ.get("SNAPPY_EMAIL_BACKEND", "console").lower()

    if backend == "mailgun":
        key    = os.environ.get("SNAPPY_EMAIL_MAILGUN_KEY", "")
        domain = os.environ.get("SNAPPY_EMAIL_MAILGUN_DOMAIN", "")
        if key and domain:
            _singleton = MailgunSender(key, domain); return _singleton
        log.warning("Mailgun selected but key/domain missing → console fallback")
    if backend == "sendgrid":
        key = os.environ.get("SNAPPY_EMAIL_SENDGRID_KEY", "")
        if key:
            _singleton = SendGridSender(key); return _singleton
        log.warning("SendGrid selected but key missing → console fallback")

    _singleton = ConsoleSender()
    return _singleton


# Convenience: ready-made templates
def send_welcome(to_email: str, name: str = "") -> bool:
    msg = EmailMessage(
        to=to_email, subject="Welcome to SnapAI 📸",
        text=f"""Hey {name or 'there'},

Thanks for signing up for SnapAI. Your account is ready —
log in and connect your camera at https://your-snappy-host/

Reply to this email if you hit any snags.

— The SnapAI team""",
    )
    return get_email_sender().send(msg)


def send_verify_link(to_email: str, verify_url: str) -> bool:
    msg = EmailMessage(
        to=to_email, subject="Verify your SnapAI email",
        text=f"Click to verify your email:\n\n{verify_url}\n\nLink expires in 24 hours.",
    )
    return get_email_sender().send(msg)


def send_capture_notification(to_email: str, event_name: str,
                              album_url: str, captures: int) -> bool:
    msg = EmailMessage(
        to=to_email,
        subject=f"📸 {event_name}: {captures} moments captured",
        text=f"SnapAI captured {captures} moments at {event_name}.\n\n"
             f"View your album: {album_url}\n",
    )
    return get_email_sender().send(msg)


def send_password_reset(to_email: str, reset_url: str) -> bool:
    msg = EmailMessage(
        to=to_email, subject="Reset your SnapAI password",
        text=(
            "Someone (hopefully you) asked to reset your SnapAI password.\n\n"
            f"Click to set a new one:\n{reset_url}\n\n"
            "This link expires in 1 hour. If you didn't request this, "
            "ignore this email — your password is unchanged."
        ),
    )
    return get_email_sender().send(msg)
