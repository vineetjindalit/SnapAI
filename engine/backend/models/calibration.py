"""
backend/models/calibration.py — Platt-scale per-class confidence calibration.

WHY
---
The raw moment-confidence score for `cake_cutting` is a different scale
than for `first_dance` (different visual evidence, different prompt
similarity distribution, different priors). One global threshold means
some classes over-fire and others under-fire.

Solution: fit a logistic regression per class on (raw_score, kept) pairs
collected from user feedback. At inference, map raw score → calibrated
probability that this would be a "kept" capture. The capture trigger
becomes a single class-agnostic threshold (default 0.6).

We keep this lightweight (no sklearn at import time):
  - Each class has (a, b) parameters: P(kept | s) = sigmoid(a*s + b).
  - Trained via plain Newton-Raphson on the cross-entropy loss in
    < 50 lines, no external dependency.
  - Refit every N feedback events (default 5) per class.
  - Persisted via SessionStore — a JSON blob in feedback_priors.

NOTES
-----
- Below MIN_OBSERVATIONS feedback samples (default 6), we DO NOT
  calibrate — we return the raw score so brand-new classes still trigger.
- Every observation contributes; no decay. For long-running deployments
  add EMA decay to handle drift (TODO).
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("snappy.calibration")

MIN_OBSERVATIONS  = 6        # need this many keeps+trashes before fitting
REFIT_EVERY       = 5        # refit after every N new feedback events
DEFAULT_THRESHOLD = 0.6      # class-agnostic capture threshold


@dataclass
class _ClassParams:
    a: float = 6.0       # slope (steeper = more confident in raw score)
    b: float = -3.0      # intercept (sigmoid centered at s = 0.5)
    n_obs: int = 0       # total observations
    n_kept: int = 0
    n_since_fit: int = 0
    fitted: bool = False
    history: List[Tuple[float, int]] = field(default_factory=list)  # (raw, kept)

    def to_json(self) -> dict:
        return {
            "a": self.a, "b": self.b,
            "n_obs": self.n_obs, "n_kept": self.n_kept,
            "fitted": self.fitted,
        }


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _fit_platt(scores: List[float], labels: List[int],
               iters: int = 50, lam: float = 1e-4) -> Tuple[float, float]:
    """Fit P(y=1|s) = sigmoid(a*s + b) via Newton-Raphson with L2.
    Returns (a, b)."""
    a, b = 1.0, 0.0
    n = len(scores)
    if n == 0:
        return a, b
    for _ in range(iters):
        # gradient + Hessian (2x2)
        gA = gB = 0.0
        hAA = hAB = hBB = 0.0
        for s, y in zip(scores, labels):
            p = _sigmoid(a * s + b)
            err = p - y
            gA += err * s
            gB += err
            w = p * (1 - p)
            hAA += w * s * s
            hAB += w * s
            hBB += w
        gA += lam * a
        gB += lam * b
        hAA += lam
        hBB += lam
        det = hAA * hBB - hAB * hAB
        if abs(det) < 1e-12:
            break
        # invert 2x2
        ihAA = hBB / det
        ihAB = -hAB / det
        ihBB = hAA / det
        da = ihAA * gA + ihAB * gB
        db = ihAB * gA + ihBB * gB
        a -= da
        b -= db
        if abs(da) < 1e-6 and abs(db) < 1e-6:
            break
    return a, b


class ConfidenceCalibrator:
    """One per server. Per-class Platt scaling, persisted to SQLite."""

    def __init__(self, store=None,
                 min_obs: int = MIN_OBSERVATIONS,
                 refit_every: int = REFIT_EVERY):
        self._lock = threading.Lock()
        self._store = store
        self._min_obs = int(min_obs)
        self._refit_every = int(refit_every)
        self._params: Dict[str, _ClassParams] = {}
        self._restore_from_store()

    # ── Inference ─────────────────────────────────────────────────────────
    def calibrate(self, moment_class: str, raw_score: float) -> float:
        """Return a probability in [0, 1]. If not yet fitted, returns
        clipped raw_score so the capture decision still has signal."""
        with self._lock:
            p = self._params.get(moment_class)
        if p is None or not p.fitted:
            return float(max(0.0, min(1.0, raw_score)))
        z = p.a * float(raw_score) + p.b
        return float(_sigmoid(z))

    # ── Feedback intake ───────────────────────────────────────────────────
    def observe(self, moment_class: str, raw_score: float, kept: bool) -> None:
        with self._lock:
            p = self._params.setdefault(moment_class, _ClassParams())
            p.history.append((float(raw_score), 1 if kept else 0))
            p.n_obs   += 1
            p.n_kept  += 1 if kept else 0
            p.n_since_fit += 1
            should_refit = (p.n_obs >= self._min_obs and
                            p.n_since_fit >= self._refit_every)
        if should_refit:
            self._refit(moment_class)

    def _refit(self, moment_class: str) -> None:
        with self._lock:
            p = self._params.get(moment_class)
            if p is None:
                return
            scores = [s for s, _ in p.history]
            labels = [y for _, y in p.history]
        a, b = _fit_platt(scores, labels)
        with self._lock:
            p.a, p.b = a, b
            p.fitted = True
            p.n_since_fit = 0
        if self._store is not None:
            try:
                self._store.upsert_calibration(moment_class, a, b, p.n_obs, p.n_kept)
            except Exception as ex:
                log.warning(f"persist calibration: {ex}")
        log.info(f"calibration refit '{moment_class}': "
                 f"a={a:.2f} b={b:.2f} n={p.n_obs} kept={p.n_kept}")

    # ── Diagnostics ───────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, dict]:
        with self._lock:
            return {k: v.to_json() for k, v in self._params.items()}

    def _restore_from_store(self) -> None:
        if self._store is None:
            return
        try:
            self._store.ensure_calibration_table()
            rows = self._store.load_calibration()
        except Exception as ex:
            log.debug(f"calibration restore: {ex}")
            return
        for r in rows:
            p = _ClassParams(a=r["a"], b=r["b"],
                             n_obs=r["n_obs"], n_kept=r["n_kept"],
                             fitted=True)
            self._params[r["moment"]] = p
        if rows:
            log.info(f"restored {len(rows)} calibration entries")
