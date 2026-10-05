"""
models/moment_predictor.py — Score trend analysis + preroll buffer.
"""
import numpy as np, time
import cv2
from collections import deque
from dataclasses import dataclass
from typing import Optional, List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_types import clamp, safe

# Motion is a scalar — we don't need full-resolution diffs. Downsampling
# to this fixed size makes the predictor resilient to within-session
# resolution changes (live camera → video upload at different size).
_MOTION_DIFF_SIZE = (64, 64)

@dataclass
class Prediction:
    trend:        str    # rising | falling | peak | stable
    peak_in_ms:   Optional[int]
    confidence:   float
    preroll:      bool

    def to_dict(self):
        return safe(dict(trend=self.trend, peak_in_ms=self.peak_in_ms,
                         confidence=self.confidence, preroll=self.preroll))

class PrerollBuffer:
    def __init__(self, n=25):
        self._frames = deque(maxlen=n)
        self._scores = deque(maxlen=n)

    def add(self, frame, score: float):
        import copy
        self._frames.append(frame.copy())
        self._scores.append(float(score))

    def best(self):
        if not self._frames: return None, 0.0
        idx = int(np.argmax(list(self._scores)))
        return list(self._frames)[idx], float(list(self._scores)[idx])

class MomentPredictor:
    def __init__(self, fps=15.0):
        self.fps = fps
        self._scores  = deque(maxlen=90)
        self._motion  = deque(maxlen=90)
        self._last_cap = 0.0
        self.MIN_INTERVAL = 3.0
        self._prev = None
        self.preroll = PrerollBuffer(25)

    def update(self, score: float, frame=None):
        self._scores.append(float(score))
        if frame is None:
            self._motion.append(0.0)
            return

        # Downsample to a fixed 64x64 grayscale for motion diff — shape-
        # invariant against the frame resolution we happen to have. This
        # makes mid-session resolution changes (camera → video upload at a
        # different size) safe: the stored `_prev` is always the same shape.
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, _MOTION_DIFF_SIZE)
        except Exception:
            self._motion.append(0.0)
            self.preroll.add(frame, score)
            return

        if self._prev is not None and self._prev.shape == small.shape:
            diff = float(np.mean(np.abs(small.astype(np.float32)
                                        - self._prev.astype(np.float32))))
            self._motion.append(clamp(diff / 60.0))
        else:
            self._motion.append(0.0)

        self._prev = small
        self.preroll.add(frame, score)

    def current_motion(self) -> float:
        """Most recent frame-to-frame motion delta, 0..1-ish (unclamped above
        1 for very large jumps). 0.0 before the first two frames. Public
        accessor so callers (e.g. Custom Mode's motion detector) don't reach
        into the private _motion deque directly."""
        return float(self._motion[-1]) if self._motion else 0.0

    def _trend(self) -> str:
        s = list(self._scores)
        if len(s) < 6: return "stable"
        recent, older = np.mean(s[-4:]), np.mean(s[-10:-4]) if len(s)>=10 else np.mean(s[:4])
        delta = float(recent - older)
        if delta > 0.06: return "rising"
        if delta < -0.06: return "falling"
        if float(recent) > 0.68: return "peak"
        return "stable"

    def predict(self) -> Prediction:
        s = list(self._scores)
        if len(s) < 8:
            return Prediction("stable", None, 0.0, False)
        trend = self._trend()
        vel = float(np.mean(np.gradient(s[-12:]))) if len(s)>=12 else 0.0
        now = time.time()
        cooldown = (now - self._last_cap) < self.MIN_INTERVAL

        if trend == "peak":
            return Prediction("peak", 0, 0.88 if not cooldown else 0.1, not cooldown)
        if trend == "rising" and vel > 0:
            frames_to = max(1, int((0.82-float(s[-1]))/vel))
            ms = int(frames_to*(1000/self.fps))
            conf = clamp(vel*4) if not cooldown else 0.1
            return Prediction("rising", ms, conf, ms < 1200 and not cooldown)
        mot_spike = len(self._motion)>10 and float(self._motion[-1]) > float(np.mean(list(self._motion)))*1.6
        if mot_spike and not cooldown:
            return Prediction("rising", 600, 0.45, True)
        return Prediction(trend, None, 0.0, False)

    def confirm_capture(self): self._last_cap = float(time.time())
