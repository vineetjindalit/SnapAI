"""
backend/models/candle_phase.py — distinguish the candle ACTION from the candle SCENE.

CLIP (and the colour heuristic) match the whole candle *scene* — a lit cake
with people around it — so they label the entire "gathering / singing" phase as
`candle_blowing`. But the user only wants `candle_blowing` for the actual blow
(someone leaning in and blowing the candles out). Everyone simply standing in
front of the cake is a different moment: `cake_with_candles`.

A single frame of "people at a lit cake" and "the instant of the blow" look
almost identical to CLIP, so we decide the sub-phase from geometry + time
instead of appearance:

  • LEANING IN — the blower bends down toward the candles, so their face becomes
    large (close) AND sits low in the frame (over the cake). In a gathering shot
    everyone stands back, so faces are small and high. This is the primary cue.

  • FLAME EXTINCTION — across a short window the candle flames drop from "lit"
    to "out". That transition IS the blow. Works even for a far blow where no
    single face is large. (Temporal — needs the per-session flame history.)

If neither cue fires, the candle scene is the gathering → `cake_with_candles`.

Pure functions; the caller owns the flame-history buffer (on the Session).
"""
from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple
import os

import cv2
import numpy as np

# ── Tunables (env-overridable) ──────────────────────────────────────────────
# A face counts as "leaning into the cake" when it is at least this tall
# (fraction of frame height → close to camera) AND its bottom edge is at least
# this far down the frame (over the cake region). Calibrated on real clips:
# gathering shots top out ~0.18 face height; a leaning blower is 0.22-0.45.
# This catches CLOSE blowers; far/solo blowers are caught by descent (below).
_LEAN_FACE_MIN_H   = float(os.environ.get("SNAPPY_CANDLE_LEAN_FACE_H",   "0.22"))
_LEAN_FACE_MIN_BOT = float(os.environ.get("SNAPPY_CANDLE_LEAN_FACE_BOT", "0.62"))

# Descent cue (framing-invariant): the blower bends DOWN toward the cake, so the
# lowest face's bottom edge moves further down the frame over a couple seconds.
# Works even when the face is small/far (the absolute-size lean rule misses it).
# Only used for ≤2 faces — in a crowd the "lowest face" jumps between people and
# the signal is unreliable, so groups fall back to lean + flame-out.
_DESCENT_DELTA      = float(os.environ.get("SNAPPY_CANDLE_DESCENT_DELTA", "0.10"))
_DESCENT_MIN_BOTTOM = float(os.environ.get("SNAPPY_CANDLE_DESCENT_MIN_BOT", "0.45"))
_DESCENT_MAX_FACES  = int(os.environ.get("SNAPPY_CANDLE_DESCENT_MAX_FACES", "2"))

# Flame-extinction: a drop to <= EXTINCT_FRAC of the recent peak flame count
# (over the history window) is read as "candles just blown out".
_FLAME_HISTORY      = int(os.environ.get("SNAPPY_CANDLE_FLAME_HISTORY", "6"))
_FLAME_EXTINCT_FRAC = float(os.environ.get("SNAPPY_CANDLE_EXTINCT_FRAC", "0.35"))
_FLAME_MIN_PEAK     = int(os.environ.get("SNAPPY_CANDLE_MIN_PEAK", "3"))

PHASE_BLOWING   = "candle_blowing"
PHASE_GATHERING = "cake_with_candles"


def new_flame_history() -> Deque[int]:
    """Create the per-session flame-count buffer the caller passes back in."""
    return deque(maxlen=_FLAME_HISTORY)


def new_face_y_history() -> Deque[float]:
    """Create the per-session lowest-face-bottom buffer (for descent detection)."""
    return deque(maxlen=_FLAME_HISTORY)


def face_descending(history: Deque[float], current_bottom: float,
                    face_count: int) -> bool:
    """True if the lowest face has moved DOWN toward the cake — the blow lean.

    Compares the current lowest-face bottom against the highest point (min) it
    held over the recent window. A clear downward move that ends low in the
    frame is the bend-to-blow motion. Skipped for crowds (unreliable there).
    """
    if not history or face_count > _DESCENT_MAX_FACES:
        return False
    base = min(history)                       # highest the face sat recently
    return (current_bottom >= _DESCENT_MIN_BOTTOM
            and (current_bottom - base) >= _DESCENT_DELTA)


def count_warm_flames(frame: np.ndarray) -> int:
    """Count small warm/white-hot blobs in the cake region (candle flames).

    Restricted to the lower-centre of the frame (where a cake sits) and to
    warm-or-hot bright pixels, so it ignores window light and bright clothing
    as much as a cheap detector can. Absolute value is noisy; it's the RELATIVE
    change over time (extinction) we rely on.
    """
    if frame is None or frame.size == 0:
        return 0
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.45):int(h * 0.97), int(w * 0.20):int(w * 0.80)]
    if roi.size == 0:
        return 0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    warm = (H >= 8) & (H <= 40) & (V >= 200) & (S >= 60)   # orange/yellow flame
    hot  = (V >= 245) & (S <= 70)                          # white-hot core
    mask = (warm | hot).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in cnts if 2 < cv2.contourArea(c) < 250)


def face_leaning_in(face_boxes: List[Tuple[int, int, int, int]],
                    frame_h: int, frame_w: int) -> bool:
    """True if any face is large (close) AND low in the frame (over the cake)."""
    if not face_boxes or frame_h <= 0:
        return False
    for (x, y, fw, fh) in face_boxes:
        h_frac   = fh / float(frame_h)
        bot_frac = (y + fh) / float(frame_h)
        if h_frac >= _LEAN_FACE_MIN_H and bot_frac >= _LEAN_FACE_MIN_BOT:
            return True
    return False


def flames_extinguished(history: Deque[int], current: int) -> bool:
    """True if the flame count just dropped sharply from a lit peak → the blow."""
    if not history:
        return False
    peak = max(history)
    if peak < _FLAME_MIN_PEAK:
        return False                      # candles were never clearly lit
    return current <= peak * _FLAME_EXTINCT_FRAC


def classify_candle_phase(
    frame: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
    history: Optional[Deque[int]] = None,
    face_y_history: Optional[Deque[float]] = None,
) -> str:
    """Return PHASE_BLOWING for the actual blow, else PHASE_GATHERING.

    Call this only once the scene is already known to be a candle scene
    (CLIP/VLM said candle_blowing). It refines that into the action vs the
    gathering using three cues, ANY of which signals a real blow:
      • leaning in   — a close face low in the frame (group/near blows)
      • descent      — the (small/far) face bending down toward the cake
      • flame-out    — the candles going dark across the recent window

    `history` / `face_y_history` are the per-session deques from
    new_flame_history() / new_face_y_history(); this function appends to them.
    """
    h, w = frame.shape[:2]
    flames = count_warm_flames(frame)
    extinguished = flames_extinguished(history, flames) if history is not None else False
    if history is not None:
        history.append(flames)

    descending = False
    if face_boxes:
        low_bottom = max((b[1] + b[3]) / float(h) for b in face_boxes)
        if face_y_history is not None:
            descending = face_descending(face_y_history, low_bottom, len(face_boxes))
            face_y_history.append(low_bottom)

    if extinguished or descending or face_leaning_in(face_boxes, h, w):
        return PHASE_BLOWING
    return PHASE_GATHERING
