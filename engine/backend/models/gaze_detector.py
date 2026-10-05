"""
models/gaze_detector.py — Multi-person gaze detection, numpy-safe.
"""
import cv2, numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_types import clamp, safe

@dataclass
class GazeResult:
    total_faces:     int;   faces_looking: int
    gaze_ratio:      float; is_group_looking: bool
    confidence:      float

    def to_dict(self):
        return safe(dict(total_faces=self.total_faces, faces_looking=self.faces_looking,
                         gaze_ratio=self.gaze_ratio, is_group_looking=self.is_group_looking,
                         confidence=self.confidence))

class GazeDetector:
    def __init__(self, threshold=0.55):
        hc = cv2.data.haarcascades
        self.fc = cv2.CascadeClassifier(hc + "haarcascade_frontalface_default.xml")
        self.ec = cv2.CascadeClassifier(hc + "haarcascade_eye.xml")
        self.threshold = threshold
        self._history: List[float] = []

    def _iris_center(self, eye_roi: np.ndarray) -> float:
        if eye_roi.size == 0 or min(eye_roi.shape[:2]) < 8: return 0.5
        g = cv2.resize(eye_roi if len(eye_roi.shape)==2 else cv2.cvtColor(eye_roi,cv2.COLOR_BGR2GRAY),(60,25))
        _, th = cv2.threshold(g, 50, 255, cv2.THRESH_BINARY_INV)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return 0.5
        M = cv2.moments(max(cnts, key=cv2.contourArea))
        if M["m00"] == 0: return 0.5
        cx = float(M["m10"]/M["m00"])
        return clamp(1.0 - abs(cx - 30.0)/30.0)

    def _face_looking(self, g: np.ndarray, face: Tuple) -> Tuple[bool, float]:
        x, y, w, h = (int(v) for v in face)
        roi = g[y:y+h, x:x+w]
        # Face symmetry
        mw = min(w//2, roi.shape[1]-w//2)
        if mw > 2:
            lh = roi[:, :mw].astype(float)
            rh = cv2.flip(roi[:, w//2:w//2+mw], 1).astype(float)
            sym = clamp(1.0 - float(np.mean(np.abs(lh-rh)))/255.0)
        else:
            sym = 0.5
        # Eye iris
        eyes = self.ec.detectMultiScale(roi, 1.1, 5, minSize=(10,10))
        iris = [self._iris_center(roi[ey:ey+eh, ex:ex+ew]) for ex,ey,ew,eh in eyes[:2]]
        eye_score = float(np.mean(iris)) if iris else 0.5
        eye_count = clamp(len(eyes)/2.0)
        conf = clamp(sym*0.38 + eye_score*0.42 + eye_count*0.20)
        return bool(conf > 0.50), float(conf)

    def detect(self, frame: np.ndarray) -> GazeResult:
        g     = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.fc.detectMultiScale(g, 1.1, 4, minSize=(32,32))
        if len(faces) == 0:
            return GazeResult(0, 0, 0.0, False, 0.0)

        looking, conf_sum = 0, 0.0
        for face in faces:
            il, conf = self._face_looking(g, face)
            if il: looking += 1
            conf_sum += conf

        raw = float(looking)/float(len(faces))
        self._history.append(raw)
        if len(self._history) > 14: self._history.pop(0)
        smoothed = float(np.mean(self._history))
        is_group = bool(smoothed >= self.threshold)
        return GazeResult(int(len(faces)), int(looking), float(smoothed),
                          is_group, clamp(conf_sum/len(faces)))
