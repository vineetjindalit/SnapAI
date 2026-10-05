"""
models/online_learner.py — learns from user 👍/👎 feedback in real time.

We don't run a training loop on-device. Instead we maintain Bayesian
per-(session, moment_class) priors:

    p(keep | moment) = Beta(α + kept, β + trashed)

When the perception stack proposes a capture for moment X with score s,
we multiply s by the posterior mean (a "trust factor" learned from the
photographer's own thumbs-up/thumbs-down history). If they consistently
keep cake_cutting captures, X='cake_cutting' becomes more likely to fire
again. If they trash 'group_photo' captures, that class's threshold
effectively rises.

State is persisted to SQLite (table: feedback_priors) so it survives
restarts and can be exported per-photographer for cohort analysis later.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

log = logging.getLogger("snappy.online")


@dataclass
class ClassPrior:
    kept: int = 0
    trashed: int = 0
    alpha: float = 2.0   # default optimism
    beta:  float = 2.0

    @property
    def keep_rate(self) -> float:
        a = self.alpha + self.kept
        b = self.beta  + self.trashed
        return float(a / (a + b))

    @property
    def total_feedback(self) -> int:
        return self.kept + self.trashed


class OnlineLearner:
    """In-memory + SQLite-backed feedback learner."""

    def __init__(self, store=None):
        self._store = store
        self._lock = threading.Lock()
        # (sid, moment) → ClassPrior
        self._priors: Dict[Tuple[str, str], ClassPrior] = {}
        if store is not None:
            try:
                store.ensure_feedback_tables()
                for row in store.load_priors():
                    self._priors[(row["sid"], row["moment"])] = ClassPrior(
                        kept=row["kept"], trashed=row["trashed"],
                    )
                log.info(f"OnlineLearner restored {len(self._priors)} priors")
            except Exception as e:
                log.warning(f"prior restore failed: {e}")

    # ── Inference-time hook ───────────────────────────────────────────────────
    def boost(self, sid: str, moment: str, base_score: float) -> float:
        """Multiply the perception score by the learned trust factor.

        We bias gently toward 1.0 when feedback is sparse so brand-new
        sessions still trigger captures normally.
        """
        with self._lock:
            p = self._priors.get((sid, moment))
        if p is None or p.total_feedback < 1:
            return base_score
        # Smooth: factor in [0.5, 1.5] depending on keep rate (0..1).
        factor = 0.5 + p.keep_rate
        return float(base_score * factor)

    # ── Feedback recording ────────────────────────────────────────────────────
    def record(self, sid: str, moment: str, kept: bool) -> ClassPrior:
        with self._lock:
            key = (sid, moment)
            p = self._priors.setdefault(key, ClassPrior())
            if kept: p.kept += 1
            else:    p.trashed += 1
        if self._store is not None:
            try:
                self._store.upsert_prior(sid, moment, p.kept, p.trashed)
            except Exception as e:
                log.warning(f"upsert_prior failed: {e}")
        log.info(f"feedback {sid}/{moment}: kept={p.kept} trashed={p.trashed} "
                 f"keep_rate={p.keep_rate:.2f}")
        return p

    def snapshot(self, sid: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        with self._lock:
            for (s, m), p in self._priors.items():
                if sid and s != sid: continue
                out.setdefault(s, {})[m] = round(p.keep_rate, 3)
        return out
