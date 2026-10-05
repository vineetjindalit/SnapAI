"""
backend/models/cake_smash.py — detect the cake-SMASH action by geometry.

CLIP matches the cake-smash *scene* loosely ("someone smashing their face into
a cake"), so it both (a) fires on ordinary cake shots and (b) MISSES the real
smash when it reads it as cake_cutting. A real smash is a specific geometry,
which is what the user described:

    "cake smashing is only when the person's face is INSIDE the cake or going to
     be on his face — the distance of the cake and face is minimal or zero …
     some cake on face should be there."

We can't bound the cake without a dedicated detector, so we use two robust,
cheap proxies and require them together:

  • FACE AT THE CAKE — the lowest face sits low in the frame (cake/table level)
    and is reasonably close (not a far bystander). This is the "minimal cake-to-
    face distance" cue.
  • CAKE ON THE FACE — a high fraction of the face region is frosting-coloured
    (bright, low-saturation white/cream, or very bright pastel icing) rather
    than skin. This is the "cake on face" cue.

The instant BEFORE contact (face plunging toward the cake) is caught by a
descent cue when a recent face-position history is available.

Pure functions; thresholds are env-overridable for field tuning.
"""
from __future__ import annotations

import os
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

# Face "at the cake": bottom edge low in the frame AND face not tiny/far.
_SMASH_FACE_BOT = float(os.environ.get("SNAPPY_SMASH_FACE_BOT", "0.66"))
_SMASH_FACE_H   = float(os.environ.get("SNAPPY_SMASH_FACE_H",   "0.16"))
# Frosting coverage of the (padded) face region that means "cake ON the face".
_SMASH_FROST_ON   = float(os.environ.get("SNAPPY_SMASH_FROST_ON",   "0.22"))
# Lower frosting bar accepted only together with a clear plunge (about-to-smash).
_SMASH_FROST_NEAR = float(os.environ.get("SNAPPY_SMASH_FROST_NEAR", "0.10"))
# Descent (about-to-smash): lowest face moved this far DOWN and ended this low.
_SMASH_DESC_DELTA = float(os.environ.get("SNAPPY_SMASH_DESC_DELTA", "0.12"))
_SMASH_DESC_BOT   = float(os.environ.get("SNAPPY_SMASH_DESC_BOT",   "0.72"))

CAKE_SMASH = "cake_smashing"


def _frosting_fraction(frame: np.ndarray, box, W: int, H: int,
                       pad: float = 0.30) -> float:
    """Fraction of the (padded) face region that looks like cake frosting.

    Frosting/cream is bright with low saturation (white/cream) or very bright
    pastel icing. Skin sits at moderate saturation, so a clean face scores low
    while a cake-covered face scores high.
    """
    x, y, fw, fh = box
    ex, ey = int(fw * pad), int(fh * pad)
    x0, y0 = max(0, x - ex), max(0, y - ey)
    x1, y1 = min(W, x + fw + ex), min(H, y + fh + ey)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _h, S, V = cv2.split(hsv)
    white  = (V >= 185) & (S <= 55)      # white / cream frosting
    pastel = (V >= 205) & (S <= 110)     # bright pastel icing (pink/blue/yellow)
    return float((white | pastel).mean())


def classify_cake_smash(
    frame: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
    face_y_history: Optional[Deque[float]] = None,
) -> Tuple[bool, dict]:
    """Return (is_smash, detail). Call only on cake-ish scenes.

    is_smash is True when a face is at the cake WITH cake on it (the contact
    frame), or when a face is plunging into the cake region (the just-before
    frame) — both are the moments the user wants captured as cake_smashing.
    """
    if frame is None or not face_boxes:
        return False, {"reason": "no_face"}
    H, W = frame.shape[:2]
    if H <= 0 or W <= 0:
        return False, {"reason": "bad_frame"}

    # The lowest face is the one at the cake.
    box = max(face_boxes, key=lambda b: (b[1] + b[3]))
    x, y, fw, fh = box
    bot = (y + fh) / float(H)
    hfrac = fh / float(H)
    low_close = (bot >= _SMASH_FACE_BOT and hfrac >= _SMASH_FACE_H)
    frost = _frosting_fraction(frame, box, W, H)

    # Descent — needs a recent history of the lowest-face bottom; only reliable
    # for ≤2 faces (in a crowd the "lowest face" jumps between people).
    descending = False
    low_bottom = max((b[1] + b[3]) / float(H) for b in face_boxes)
    if face_y_history and len(face_boxes) <= 2:
        base = min(face_y_history)
        descending = (low_bottom >= _SMASH_DESC_BOT
                      and (low_bottom - base) >= _SMASH_DESC_DELTA)

    on_face = low_close and frost >= _SMASH_FROST_ON
    about_to = (low_close and descending and frost >= _SMASH_FROST_NEAR) \
        or (descending and low_bottom >= _SMASH_DESC_BOT)
    smash = bool(on_face or about_to)
    return smash, {
        "bot": round(bot, 2), "hfrac": round(hfrac, 2), "frost": round(frost, 3),
        "low_close": low_close, "descending": descending,
        "on_face": on_face, "about_to": about_to,
    }
