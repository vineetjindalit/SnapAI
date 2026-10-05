"""
models/shot_quality.py  — Frame quality scorer, all numpy types sanitised.
"""
import cv2, numpy as np, time
from dataclasses import dataclass
from typing import List
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_types import clamp, safe
from utils.config import CONFIG

WEIGHTS = dict(blur=0.25, face=0.18, face_q=0.14, comp=0.14, bright=0.11, emotion=0.18)
# When NIMA is available, the heuristic total gets blended with NIMA at this weight.
NIMA_WEIGHT = 0.45

@dataclass
class FrameScore:
    timestamp:   float; blur:   float; faces:  int
    face_q:      float; comp:   float; bright: float
    emotion:     float; total:  float
    nima:        float = -1.0   # -1 = unavailable

    def to_dict(self):
        return safe(dict(
            timestamp=self.timestamp, blur=self.blur, faces=self.faces,
            face_q=self.face_q, comp=self.comp, bright=self.bright,
            emotion=self.emotion, total=self.total, nima=self.nima))

class ShotQualityAnalyzer:
    def __init__(self):
        hc = cv2.data.haarcascades
        self.fc = cv2.CascadeClassifier(hc + "haarcascade_frontalface_default.xml")
        self.sc = cv2.CascadeClassifier(hc + "haarcascade_smile.xml")
        self.ec = cv2.CascadeClassifier(hc + "haarcascade_eye.xml")
        self._hist: List[FrameScore] = []
        # -inf so the FIRST can_capture() always passes regardless of the time
        # base used — 0.0 relied on wall-clock (time.time()) always being a
        # huge Unix epoch number; media-time (an upload's own clock) starts
        # near 0 and would otherwise wrongly fail that first check.
        self._last_cap = float("-inf")
        self.MIN_INTERVAL = CONFIG.min_capture_interval_s
        self.MAX_HIST    = CONFIG.quality_history

    def _blur(self, g):
        return clamp(float(cv2.Laplacian(g, cv2.CV_64F).var()) / 500.0)

    def _bright(self, g):
        m = float(np.mean(g))
        if 70 <= m <= 185: return 1.0
        return clamp(m / 70.0) if m < 70 else clamp((255 - m) / 70.0)

    def _comp(self, faces, shape):
        if not faces: return 0.5
        h, w = shape[:2]
        tx, ty = [w/3, 2*w/3], [h/3, 2*h/3]
        best = 0.0
        for (x, y, fw, fh) in faces:
            cx, cy = x+fw/2, y+fh/2
            d = min(((cx-ax)**2+(cy-ay)**2)**.5 for ax in tx for ay in ty)
            best = max(best, clamp(1.0 - d/((w**2+h**2)**.5)))
        return float(best)

    def _face_q(self, g, faces):
        if not faces: return 0.0
        total = float(g.shape[0]*g.shape[1])
        return float(np.mean([
            (clamp(float(cv2.Laplacian(g[y:y+h,x:x+w],cv2.CV_64F).var())/200)+
             clamp((w*h)/total*10))/2
            for (x,y,w,h) in faces if g[y:y+h,x:x+w].size > 0
        ])) if faces else 0.0

    def _emotion(self, g, faces):
        if not faces: return 0.5
        H, W = g.shape[:2]
        scores = []
        for (x,y,w,h) in faces:
            # Clamp the box into the image — a MediaPipe/YOLO box remapped by the
            # HD-proxy scale factor can land partly out of bounds.
            x0, y0 = max(0, int(x)), max(0, int(y))
            x1, y1 = min(W, int(x)+int(w)), min(H, int(y)+int(h))
            roi = g[y0:y1, x0:x1]
            # Haar cascades ASSERT-CRASH (getScaleData) on any ROI smaller than
            # their detection window — a small/distant face did exactly that and
            # took down the whole frame pipeline. Require a safe floor, and never
            # let a cascade error escape into the pipeline.
            if roi.shape[0] < 28 or roi.shape[1] < 28:
                continue
            try:
                sm = self.sc.detectMultiScale(roi, 1.7, 22, minSize=(20,20))
                ey = self.ec.detectMultiScale(roi, 1.1, 5,  minSize=(12,12))
            except cv2.error:
                continue
            scores.append(clamp(len(sm)*0.5)*0.6 + clamp(len(ey)*0.5)*0.4)
        return float(np.mean(scores)) if scores else 0.5

    def analyze(self, frame: np.ndarray, face_boxes=None, nima_score=None) -> FrameScore:
        """Score a frame. If face_boxes is provided (e.g. from MediaPipe), skip
        the Haar pass — saves CPU and gives more accurate face counts.
        face_boxes format: list of (x, y, w, h) tuples."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if face_boxes is not None:
            faces = [tuple(int(v) for v in b) for b in face_boxes]
        else:
            raw   = self.fc.detectMultiScale(g, 1.1, 5, minSize=(28,28))
            faces = list(raw) if len(raw) > 0 else []

        b, br, c = self._blur(g), self._bright(g), self._comp(faces, frame.shape)
        fq, em   = self._face_q(g, faces), self._emotion(g, faces)
        fn       = clamp(len(faces)/5.0)

        heuristic_total = (b*WEIGHTS["blur"] + fn*WEIGHTS["face"] + fq*WEIGHTS["face_q"] +
                            c*WEIGHTS["comp"] + br*WEIGHTS["bright"] + em*WEIGHTS["emotion"])

        if nima_score is not None and 0.0 <= nima_score <= 1.0:
            total = (1.0 - NIMA_WEIGHT) * heuristic_total + NIMA_WEIGHT * float(nima_score)
            nima_val = float(nima_score)
        else:
            total = heuristic_total
            nima_val = -1.0

        sc = FrameScore(float(time.time()), b, len(faces), fq, c, br, em, clamp(total), nima_val)
        self._hist.append(sc)
        if len(self._hist) > self.MAX_HIST: self._hist.pop(0)
        return sc

    def is_best_shot(self, sc: FrameScore, thr=None) -> bool:
        thr = CONFIG.best_shot_threshold if thr is None else thr
        if sc.total < thr: return False
        if len(self._hist) < 6: return sc.total > thr
        avg = float(np.mean([s.total for s in self._hist[-12:]]))
        return sc.total > avg * CONFIG.best_shot_uplift

    def can_capture(self, now: float = None):
        # `now` lets callers pass the VIDEO'S OWN media-time for an uploaded
        # clip instead of wall-clock. Wall-clock cooldown made upload results
        # non-deterministic: the same video re-processed took a different
        # amount of REAL time (server load, warm vs cold model caches), so a
        # different number of frames fell inside/outside the cooldown window
        # each run — identical content, different capture counts. Live capture
        # has no substitute for wall-clock (it *is* real time), so `now`
        # defaults to it.
        t = time.time() if now is None else float(now)
        return (t - self._last_cap) >= self.MIN_INTERVAL

    def record_capture(self, now: float = None):
        self._last_cap = float(time.time() if now is None else now)
    def recent_avg(self): return float(np.mean([s.total for s in self._hist[-10:]])) if self._hist else 0.0
