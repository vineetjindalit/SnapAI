"""
backend/models/l2cs_gaze.py — L2CS-Net gaze regression.

Roadmap: "MPIIGaze / GazeNet — precise gaze estimation". L2CS-Net is the
2022 SOTA (~3-4° angular error on MPIIGaze, vs ~12° for our iris-corner
geometric path).

Requires: `pip install l2cs`. The package auto-downloads model weights
on first use. Falls back transparently if not installed — the geometric
GazeEstimator keeps working unchanged.

How it integrates:
  - face_provider populates each FaceInfo with a `looking_at_camera` score
    from iris geometry (always available).
  - When L2CS is available, this module overrides that score with the
    real regressor output before the GazeEstimator aggregates per-face
    scores into a group ratio.

API:
    L2CSGaze.get().refine_gaze_inplace(frame, face_infos)
    → mutates face.looking_at_camera in-place.
"""
from __future__ import annotations

import logging
import math
import threading
from typing import List, Optional

import cv2
import numpy as np

log = logging.getLogger("snappy.l2cs")

# Tunable: how forgiving the "looking at camera" classifier is.
# yaw/pitch in radians; default cone is ±20° (about 0.35 rad).
_CONE_HALF_ANGLE = 0.35


class L2CSGaze:
    _instance: Optional["L2CSGaze"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._pipeline = None
        self._init_error: Optional[str] = None
        self._try_load()

    @classmethod
    def get(cls) -> "L2CSGaze":
        with cls._lock:
            if cls._instance is None:
                cls._instance = L2CSGaze()
            return cls._instance

    def _try_load(self) -> None:
        try:
            from l2cs import Pipeline
        except Exception as e:
            self._init_error = (
                f"l2cs not installed ({e}). "
                "Install with: pip install l2cs"
            )
            log.info("L2CS-Net not installed — using iris geometry for gaze")
            return
        try:
            from gpu.device import get_device
            device = get_device()
            log.info(f"Loading L2CS-Net on {device}…")
            try:
                import torch
                self._pipeline = Pipeline(weights="L2CSNet_gaze360.pkl",
                                          arch="ResNet50",
                                          device=torch.device(device))
            except Exception:
                # Some versions take device str, not torch.device
                self._pipeline = Pipeline(weights="L2CSNet_gaze360.pkl",
                                          arch="ResNet50",
                                          device=device)
            log.info("L2CS-Net ready")
        except Exception as e:
            self._init_error = f"L2CS load failed: {e}"
            log.warning(self._init_error)
            self._pipeline = None

    @property
    def available(self) -> bool:
        return self._pipeline is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def predict(self, face_bgr: np.ndarray):
        """Return (yaw_rad, pitch_rad) for one face crop, or None on failure."""
        if self._pipeline is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            # The l2cs Pipeline expects a full frame and runs its own face
            # detection by default. We feed it the face crop (it tolerates
            # this) and read out the gaze regressor output directly.
            import torch
            with torch.no_grad():
                result = self._pipeline.step(face_bgr)
            yaw   = float(np.array(result.yaw).flatten()[0])   * math.pi / 180.0
            pitch = float(np.array(result.pitch).flatten()[0]) * math.pi / 180.0
            return yaw, pitch
        except Exception as e:
            log.debug(f"L2CS predict failed: {e}")
            return None

    @staticmethod
    def _looking_score(yaw: float, pitch: float) -> float:
        """Map (yaw, pitch) in radians to a [0, 1] looking-at-camera score.

        Score = 1.0 when the gaze vector is exactly into the camera, falls
        off as a Gaussian-ish curve to 0 outside the cone.
        """
        d = math.hypot(yaw, pitch)
        if d <= _CONE_HALF_ANGLE:
            return 1.0 - (d / _CONE_HALF_ANGLE) * 0.4   # 1.0 → 0.6 inside cone
        falloff = max(0.0, 1.0 - (d - _CONE_HALF_ANGLE) / 0.6)
        return float(0.6 * falloff)

    def refine_gaze_inplace(self, frame: np.ndarray, face_infos: List) -> None:
        """For each FaceInfo, replace .looking_at_camera with the L2CS-derived
        score. Only updates when L2CS predicts successfully."""
        if not self.available:
            return
        h, w = frame.shape[:2]
        for face in face_infos:
            x, y, fw, fh = face.box
            x1 = max(0, x); y1 = max(0, y)
            x2 = min(w, x + fw); y2 = min(h, y + fh)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            yp = self.predict(crop)
            if yp is None:
                continue
            yaw, pitch = yp
            score = self._looking_score(yaw, pitch)
            # Multiply by eye-openness so L2CS doesn't claim a closed eye is looking.
            eye_open = 0.5 * (face.eye_open_l + face.eye_open_r)
            face.looking_at_camera = float(max(0.0, min(1.0, score * (0.4 + 0.6 * eye_open))))
