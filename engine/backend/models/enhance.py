"""
backend/models/enhance.py — software image enhancement for captured photos.

Most laptop webcams (and even uploaded video) have:
  - Mid-range brightness (auto-exposure overcompensates in dim rooms)
  - Flat colour (low saturation indoors under tungsten lighting)
  - Soft focus (motion blur on 5-fps WebSocket frames)

This module applies modest, configurable post-processing to make captures
look "kept" instead of "raw webcam screengrab". All effects are designed
to be invisible at small magnitudes — we're augmenting, not editing.

Pipeline (applied in order):
  1. Auto-brightness    — pull histogram mean to 128 (target middle gray)
  2. Auto-contrast      — gentle CLAHE on L channel of LAB color space
  3. Auto-white-balance — gray-world correction
  4. Saturation         — small bump (+15% default) for event vibrancy
  5. Unsharp mask       — modest edge enhancement (+20% default)
  6. (Optional) denoise — bilateral filter for noisy low-light frames

All steps are tunable via env vars and per-call kwargs. Auto mode reads
the frame's histogram and decides reasonable values; manual mode lets
you override (e.g. for a dim wedding hall vs a bright outdoor ceremony).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

log = logging.getLogger("snappy.enhance")


def _envf(key: str, default: float) -> float:
    try: return float(os.environ.get(key, default))
    except Exception: return default


# Default strength values — chosen to be invisible at this magnitude.
# Tune via env vars in production.
DEFAULTS = {
    "brightness_target":  _envf("SNAPPY_ENH_BRIGHT_TARGET", 128.0),
    "brightness_max":     _envf("SNAPPY_ENH_BRIGHT_MAX",    35.0),    # ±max shift
    "clahe_clip":         _envf("SNAPPY_ENH_CLAHE_CLIP",    1.8),     # 1.0 = off
    "saturation_gain":    _envf("SNAPPY_ENH_SAT_GAIN",      1.12),    # 1.0 = off
    "unsharp_amount":     _envf("SNAPPY_ENH_SHARP_AMOUNT",  0.25),    # 0 = off
    "unsharp_radius":     _envf("SNAPPY_ENH_SHARP_RADIUS",  1.4),
    "white_balance_str":  _envf("SNAPPY_ENH_WB_STRENGTH",   0.6),     # 0 = off, 1 = full gray-world
    "denoise_thresh":     _envf("SNAPPY_ENH_DENOISE_THR",   18.0),    # apply if std < this
}


@dataclass
class EnhanceReport:
    """Diagnostics — what we actually did. Surfaced into capture metadata."""
    brightness_delta: float = 0.0
    contrast_applied:  bool  = False
    wb_applied:        bool  = False
    saturation_gain:   float = 1.0
    sharpened:         bool  = False
    denoised:          bool  = False

    def to_dict(self) -> Dict[str, float]:
        return {
            "brightness_delta": round(self.brightness_delta, 1),
            "contrast":         self.contrast_applied,
            "wb":               self.wb_applied,
            "saturation":       round(self.saturation_gain, 2),
            "sharpened":        self.sharpened,
            "denoised":         self.denoised,
        }


def enhance(frame: np.ndarray,
            *,
            auto: bool = True,
            brightness_delta: Optional[float] = None,
            saturation_gain:  Optional[float] = None,
            sharpen:          bool            = True,
            white_balance:    bool            = True,
            denoise:          Optional[bool]  = None,
            ) -> tuple[np.ndarray, EnhanceReport]:
    """Apply the enhancement pipeline.

    `auto=True` derives sensible deltas from histogram statistics. Pass any
    of the other args to override.

    Returns (enhanced_frame, EnhanceReport). The report is JSON-safe and
    gets persisted into the photo metadata so we can see — months later —
    what we did to a specific shot.
    """
    if frame is None or frame.size == 0:
        return frame, EnhanceReport()

    out = frame
    report = EnhanceReport()

    # ── 1. Brightness ──────────────────────────────────────────────────
    if auto and brightness_delta is None:
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
        mean = float(np.mean(gray))
        # Pull toward target_mean. Clip to ±max so we don't go nuts on
        # a black frame and produce a gray nothing.
        delta = (DEFAULTS["brightness_target"] - mean) * 0.55
        delta = float(np.clip(delta, -DEFAULTS["brightness_max"],
                                       DEFAULTS["brightness_max"]))
    else:
        delta = float(brightness_delta or 0.0)
    if abs(delta) > 0.5:
        out = np.clip(out.astype(np.int32) + int(delta), 0, 255).astype(np.uint8)
        report.brightness_delta = delta

    # ── 2. Contrast (CLAHE on L channel of LAB) ────────────────────────
    if DEFAULTS["clahe_clip"] > 1.0 and out.ndim == 3:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=DEFAULTS["clahe_clip"],
                                tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        report.contrast_applied = True

    # ── 3. Auto white balance (gray-world) ─────────────────────────────
    if white_balance and out.ndim == 3 and DEFAULTS["white_balance_str"] > 0:
        # Each channel should average to the same gray. Scale each channel
        # by (mean_overall / mean_channel) then blend with the original at
        # `white_balance_str` (0 = original, 1 = full correction).
        b_mean, g_mean, r_mean = [float(np.mean(out[:, :, c])) for c in range(3)]
        overall = (b_mean + g_mean + r_mean) / 3.0
        if min(b_mean, g_mean, r_mean) > 1.0:    # avoid div-by-zero on dark frames
            scales = np.array([overall / b_mean, overall / g_mean, overall / r_mean],
                              dtype=np.float32)
            # Cap the scaling so we don't introduce magenta tints on very bluish frames
            scales = np.clip(scales, 0.85, 1.18)
            corrected = (out.astype(np.float32) * scales).clip(0, 255)
            s = DEFAULTS["white_balance_str"]
            out = (out.astype(np.float32) * (1 - s) + corrected * s).clip(0, 255).astype(np.uint8)
            report.wb_applied = True

    # ── 4. Saturation ──────────────────────────────────────────────────
    sat_gain = saturation_gain if saturation_gain is not None else DEFAULTS["saturation_gain"]
    if abs(sat_gain - 1.0) > 0.01 and out.ndim == 3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_gain, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        report.saturation_gain = sat_gain

    # ── 5. Unsharp mask ────────────────────────────────────────────────
    if sharpen and DEFAULTS["unsharp_amount"] > 0:
        amount = DEFAULTS["unsharp_amount"]
        radius = DEFAULTS["unsharp_radius"]
        blur = cv2.GaussianBlur(out, (0, 0), radius)
        out = cv2.addWeighted(out, 1.0 + amount, blur, -amount, 0)
        out = np.clip(out, 0, 255).astype(np.uint8)
        report.sharpened = True

    # ── 6. Denoise (only when frame is genuinely noisy) ────────────────
    if denoise is None:
        # Auto-decide: if the frame's brightness std is very low, it's
        # likely dim+noisy and would benefit from a gentle denoise.
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out
        std  = float(np.std(gray))
        denoise = std < DEFAULTS["denoise_thresh"]
    if denoise and out.ndim == 3:
        out = cv2.bilateralFilter(out, d=5, sigmaColor=35, sigmaSpace=35)
        report.denoised = True

    return out, report
