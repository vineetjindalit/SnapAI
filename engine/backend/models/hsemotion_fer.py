"""
models/hsemotion_fer.py — HSEmotion AffectNet-pretrained FER model (ONNX).

True emotion recognition (vs the blendshape heuristic in emotion_detector.py).
The package ships a small ONNX model trained on AffectNet (8 emotions, ~63%
top-1 which is SOTA for in-the-wild FER).

Requires: `pip install hsemotion-onnx`. Falls back to disabled if missing.

Why both this AND blendshape emotion?
  - HSEmotion is more accurate but works per-face-crop (~10-20 ms/face).
  - Blendshape mapping is free (uses MediaPipe data we already have).
  - We blend: when HSEmotion is available, weight it 0.7; blendshape 0.3.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("snappy.hsemotion")


# Map HSEmotion's 8-class output to our existing 7-class label set
# (+ "neutral" as fallback for "Contempt"/"Other" if encountered).
_LABEL_MAP = {
    "Anger":     "angry",
    "Contempt":  "angry",
    "Disgust":   "disgust",
    "Fear":      "fear",
    "Happiness": "happy",
    "Neutral":   "neutral",
    "Sadness":   "sad",
    "Surprise":  "surprised",
}


class HSEmotionRecognizer:
    _instance: Optional["HSEmotionRecognizer"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._fer = None
        self._init_error: Optional[str] = None
        self._try_load()

    @classmethod
    def get(cls) -> "HSEmotionRecognizer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = HSEmotionRecognizer()
            return cls._instance

    def _try_load(self) -> None:
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer as _R
        except Exception as e:
            self._init_error = (
                f"hsemotion-onnx not installed ({e}). "
                "Install with: pip install hsemotion-onnx onnxruntime"
            )
            log.warning(self._init_error)
            return

        try:
            # enet_b0_8_best_afew is a small fast ONNX, 8 emotion classes
            self._fer = _R(model_name="enet_b0_8_best_afew")
            log.info("HSEmotion FER ready (enet_b0_8_best_afew)")
        except Exception as e:
            self._init_error = f"HSEmotion load failed: {e}"
            log.exception(self._init_error)
            self._fer = None

    @property
    def available(self) -> bool:
        return self._fer is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def predict_face(self, face_bgr: np.ndarray) -> Optional[Tuple[str, Dict[str, float]]]:
        """Returns (dominant_label, per_emotion_probs_dict) for one face crop."""
        if self._fer is None or face_bgr is None or face_bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            label, scores = self._fer.predict_emotions(rgb, logits=False)
            class_names = list(self._fer.idx_to_class.values()) if hasattr(self._fer, "idx_to_class") \
                else list(_LABEL_MAP.keys())
            probs: Dict[str, float] = {}
            for name, p in zip(class_names, np.asarray(scores).flatten()):
                probs[_LABEL_MAP.get(name, "neutral")] = probs.get(_LABEL_MAP.get(name, "neutral"), 0.0) + float(p)
            mapped_label = _LABEL_MAP.get(label, "neutral")
            return mapped_label, probs
        except Exception as e:
            log.warning(f"HSEmotion inference failed: {e}")
            return None

    def predict_batch(self, frame: np.ndarray,
                      face_boxes: List[Tuple[int, int, int, int]]) -> List[Optional[Tuple[str, Dict[str, float]]]]:
        out = []
        h, w = frame.shape[:2]
        for (x, y, fw, fh) in face_boxes:
            x = max(0, x); y = max(0, y)
            x2 = min(w, x + fw); y2 = min(h, y + fh)
            if x2 <= x or y2 <= y:
                out.append(None); continue
            out.append(self.predict_face(frame[y:y2, x:x2]))
        return out
