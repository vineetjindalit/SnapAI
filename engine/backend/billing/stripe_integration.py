"""
backend/billing/stripe_integration.py — Stripe Checkout + webhook + tier limits.

Activate via:
    SNAPPY_BILLING=stripe
    SNAPPY_STRIPE_SECRET=sk_…
    SNAPPY_STRIPE_WEBHOOK_SECRET=whsec_…
    SNAPPY_STRIPE_PRO_PRICE=price_…
    SNAPPY_STRIPE_STUDIO_PRICE=price_…

What this module owns:
  - Mapping (tier_name → Stripe price_id)
  - POST /billing/checkout    creates a Checkout Session for upgrade
  - POST /billing/webhook     handles `customer.subscription.*` events
  - Tier enforcement helper that the routes layer calls before allowing
    /sessions creation, etc.

Disabled by default (`SNAPPY_BILLING` unset → free tier for all users,
no Stripe required for dev).

Requires `pip install stripe`. Falls back to a dummy `BillingClient` if
the lib isn't installed so the rest of the server keeps running.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("snappy.billing")


# ── Pricing tiers (mirrors the PDF) ───────────────────────────────────────
@dataclass(frozen=True)
class Tier:
    name:                 str
    price_usd:            float
    events_per_month:     int      # -1 = unlimited
    captures_per_event:   int      # -1 = unlimited
    simultaneous_cameras: int      # -1 = unlimited
    api_access:           bool
    custom_branding:      bool


# Free-tier limits are ENV-CONFIGURABLE because the same code runs in two very
# different modes: paid product (tight limits) and friend/field testing (no
# limits). Default is UNLIMITED — a 3-events/month cap silently failed session
# creation during testing, which surfaced to the user as "0 captures".
# Set SNAPPY_FREE_EVENTS_PER_MONTH=3 / SNAPPY_FREE_CAPTURES_PER_EVENT=100 to
# restore the paid-product limits.
_FREE_EVENTS   = int(os.environ.get("SNAPPY_FREE_EVENTS_PER_MONTH", "-1"))
_FREE_CAPTURES = int(os.environ.get("SNAPPY_FREE_CAPTURES_PER_EVENT", "-1"))
FREE_TIER = Tier("free",     0.0,  _FREE_EVENTS, _FREE_CAPTURES, 1, False, False)
PRO_TIER  = Tier("pro",      49.0, -1,  -1,  1, False, False)
STUD_TIER = Tier("studio",   149.0,-1,  -1,  5, True,  True)
ENT_TIER  = Tier("enterprise", 0.0, -1, -1,  -1, True, True)

TIERS: Dict[str, Tier] = {t.name: t for t in (FREE_TIER, PRO_TIER, STUD_TIER, ENT_TIER)}


# ── Stripe client wrapper ─────────────────────────────────────────────────
class BillingClient:
    """Thin wrapper around stripe-python. The whole module degrades to a
    no-op when stripe isn't installed or SNAPPY_BILLING != 'stripe'."""

    def __init__(self):
        self._stripe = None
        self._init_error: Optional[str] = None
        self._enabled = os.environ.get("SNAPPY_BILLING", "").lower() == "stripe"
        self.webhook_secret = os.environ.get("SNAPPY_STRIPE_WEBHOOK_SECRET", "")
        self._price_map = {
            "pro":    os.environ.get("SNAPPY_STRIPE_PRO_PRICE", ""),
            "studio": os.environ.get("SNAPPY_STRIPE_STUDIO_PRICE", ""),
        }
        if self._enabled:
            try:
                import stripe
                stripe.api_key = os.environ.get("SNAPPY_STRIPE_SECRET", "")
                if not stripe.api_key:
                    raise RuntimeError("SNAPPY_STRIPE_SECRET not set")
                self._stripe = stripe
                log.info("Stripe billing client ready")
            except Exception as e:
                self._init_error = str(e)
                log.warning(f"Stripe disabled: {e}")
                self._enabled = False

    @property
    def available(self) -> bool:
        return self._enabled and self._stripe is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    # ── Public actions ───────────────────────────────────────────────────
    def create_checkout(self, *, user_id: int, email: str,
                        tier: str, success_url: str, cancel_url: str
                        ) -> Tuple[int, Dict[str, Any]]:
        if not self.available:
            return 503, {"error": "Billing not configured (SNAPPY_BILLING != stripe)"}
        if tier not in self._price_map or not self._price_map[tier]:
            return 400, {"error": f"Unknown or unconfigured tier '{tier}'"}
        try:
            session = self._stripe.checkout.Session.create(
                mode="subscription",
                customer_email=email,
                client_reference_id=str(user_id),
                line_items=[{"price": self._price_map[tier], "quantity": 1}],
                success_url=success_url,
                cancel_url=cancel_url,
                allow_promotion_codes=True,
                subscription_data={"metadata": {"user_id": str(user_id), "tier": tier}},
            )
            return 200, {"checkout_url": session.url, "session_id": session.id}
        except Exception as e:
            log.exception("Stripe checkout create failed")
            return 502, {"error": f"Stripe error: {e}"}

    def parse_webhook(self, body: bytes, sig_header: str
                      ) -> Tuple[bool, Dict[str, Any]]:
        if not self.available:
            return False, {"error": "Billing not configured"}
        if not self.webhook_secret:
            return False, {"error": "SNAPPY_STRIPE_WEBHOOK_SECRET not set"}
        try:
            event = self._stripe.Webhook.construct_event(
                payload=body, sig_header=sig_header,
                secret=self.webhook_secret,
            )
            return True, dict(event)
        except Exception as e:
            log.warning(f"webhook signature verify failed: {e}")
            return False, {"error": str(e)}


# ── Tier enforcement ─────────────────────────────────────────────────────
def enforce_session_quota(store, user_id: int, tier: str) -> Tuple[bool, Optional[str]]:
    """Return (allowed, error_msg). Used by POST /sessions."""
    t = TIERS.get(tier, FREE_TIER)
    if t.events_per_month < 0:
        return True, None
    # Count sessions in the trailing 30 days
    if store is None:
        return True, None
    try:
        cutoff = time.time() - 30 * 86400
        with store._lock:
            row = store._conn.execute(
                "SELECT COUNT(*) AS c FROM sessions WHERE owner_id=? AND created>=?",
                (int(user_id), cutoff),
            ).fetchone()
        used = int(row["c"])
    except Exception as e:
        log.debug(f"quota check failed: {e}")
        return True, None
    if used >= t.events_per_month:
        return False, (f"Tier '{tier}' allows {t.events_per_month} events/month "
                       f"(you've used {used}). Upgrade at /billing/checkout.")
    return True, None


# ── Singleton ─────────────────────────────────────────────────────────────
_singleton: Optional[BillingClient] = None
def get_billing() -> BillingClient:
    global _singleton
    if _singleton is None:
        _singleton = BillingClient()
    return _singleton
