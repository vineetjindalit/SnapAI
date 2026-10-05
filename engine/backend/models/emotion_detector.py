"""
models/emotion_detector.py — emotion detection from MediaPipe blendshapes.

The FaceLandmarker we already run produces 52 blendshapes per face. We map
them to the 7 standard emotions (Ekman) using a small linear combination
that works well in practice and needs zero extra ML weights / installs.

Optional: if `onnxruntime` and a FER ONNX model are available, we'd swap
this out — left as TODO with stub.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .face_provider import FaceInfo


# Blendshape → emotion contribution. Values are weights, summed then softmaxed.
_EMOTION_RULES: Dict[str, Dict[str, float]] = {
    "happy":     {"mouthSmileLeft": 1.0, "mouthSmileRight": 1.0,
                  "cheekSquintLeft": 0.5, "cheekSquintRight": 0.5},
    "surprised": {"eyeWideLeft": 1.0, "eyeWideRight": 1.0,
                  "jawOpen": 0.7, "browInnerUp": 0.6},
    "sad":       {"mouthFrownLeft": 1.0, "mouthFrownRight": 1.0,
                  "browDownLeft": 0.4, "browDownRight": 0.4,
                  "mouthLowerDownLeft": 0.3, "mouthLowerDownRight": 0.3},
    "angry":     {"browDownLeft": 1.0, "browDownRight": 1.0,
                  "noseSneerLeft": 0.6, "noseSneerRight": 0.6,
                  "mouthPressLeft": 0.5, "mouthPressRight": 0.5},
    "disgust":   {"noseSneerLeft": 1.0, "noseSneerRight": 1.0,
                  "mouthUpperUpLeft": 0.5, "mouthUpperUpRight": 0.5},
    "fear":      {"eyeWideLeft": 0.7, "eyeWideRight": 0.7,
                  "browInnerUp": 0.6, "mouthStretchLeft": 0.5,
                  "mouthStretchRight": 0.5},
    "neutral":   {},   # fallback — high when nothing else fires
}


@dataclass
class EmotionResult:
    dominant: str = "neutral"
    score:    float = 0.0
    per_emotion: Dict[str, float] = field(default_factory=dict)
    happy_ratio: float = 0.0
    avg_smile: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dominant": self.dominant,
            "score": float(self.score),
            "per_emotion": {k: float(v) for k, v in self.per_emotion.items()},
            "happy_ratio": float(self.happy_ratio),
            "avg_smile": float(self.avg_smile),
        }


def _score_face(face: FaceInfo) -> Dict[str, float]:
    bs = face.blendshapes
    if not bs:
        return {"neutral": 1.0}
    out: Dict[str, float] = {}
    for emo, weights in _EMOTION_RULES.items():
        s = 0.0
        for k, w in weights.items():
            s += w * float(bs.get(k, 0.0))
        out[emo] = s
    # Neutral kicks in when every other emotion is weak
    other_max = max((v for k, v in out.items() if k != "neutral"), default=0.0)
    out["neutral"] = max(0.0, 0.6 - other_max)
    # Normalise
    total = sum(out.values()) or 1.0
    return {k: v / total for k, v in out.items()}


def detect_emotions(faces: List[FaceInfo]) -> EmotionResult:
    if not faces:
        return EmotionResult()

    per_face = [_score_face(f) for f in faces]
    keys = sorted({k for d in per_face for k in d})
    avg = {k: float(sum(d.get(k, 0.0) for d in per_face) / len(per_face)) for k in keys}
    dom = max(avg, key=avg.get)
    smiles = [f.smile() for f in faces]
    avg_smile = float(sum(smiles) / len(smiles)) if smiles else 0.0

    return EmotionResult(
        dominant=dom,
        score=avg.get(dom, 0.0),
        per_emotion=avg,
        happy_ratio=avg.get("happy", 0.0),
        avg_smile=avg_smile,
    )
