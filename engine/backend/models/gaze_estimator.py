"""
models/gaze_estimator.py — group-gaze using MediaPipe iris landmarks.

Replaces the Haar-cascade + threshold-centroid approach in gaze_detector.py.
Uses the per-face `looking_at_camera` score that face_provider already
computes from iris position relative to eye-corner geometry.

If MediaPipe is unavailable, FaceProvider transparently fell back to Haar
boxes with looking_at_camera=0.5 — we still produce a result (just lower
confidence) so the rest of the pipeline keeps running.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .face_provider import FaceFrame
from utils.safe_types import safe, clamp


@dataclass
class GazeResult:
    total_faces:      int
    faces_looking:    int
    gaze_ratio:       float
    is_group_looking: bool
    confidence:       float
    backend:          str

    def to_dict(self) -> dict:
        return safe({
            "total_faces": self.total_faces,
            "faces_looking": self.faces_looking,
            "gaze_ratio": self.gaze_ratio,
            "is_group_looking": self.is_group_looking,
            "confidence": self.confidence,
            "backend": self.backend,
        })


class GazeEstimator:
    def __init__(self, threshold: float = 0.55, smooth_window: int = 8):
        self.threshold = threshold
        self.smooth_window = smooth_window
        self._history: List[float] = []

    def detect(self, face_frame: FaceFrame) -> GazeResult:
        faces = face_frame.faces
        if not faces:
            self._history.append(0.0)
            self._history = self._history[-self.smooth_window:]
            return GazeResult(0, 0, 0.0, False, 0.0, face_frame.backend)

        # Threshold per-face on looking_at_camera
        per = [f.looking_at_camera for f in faces]
        looking = sum(1 for p in per if p > 0.55)
        raw_ratio = float(looking) / float(len(faces))
        self._history.append(raw_ratio)
        self._history = self._history[-self.smooth_window:]
        smoothed = sum(self._history) / len(self._history)

        avg_conf = sum(per) / len(per) if per else 0.0
        return GazeResult(
            total_faces      = int(len(faces)),
            faces_looking    = int(looking),
            gaze_ratio       = float(clamp(smoothed)),
            is_group_looking = bool(smoothed >= self.threshold),
            confidence       = float(clamp(avg_conf)),
            backend          = face_frame.backend,
        )
