"""
models/birthday_moment_classifier.py — the trained fused birthday-moment head,
run at inference. Reuses the scene embedding + faces the pipeline already
computed; adds the action-crop CLIP embedding + the behavioural cues (pout,
jaw, frosting, surprise, flame) exactly as in training, then predicts the moment.

Loaded lazily from models/event_classifiers/birthday_fused.npz (W,b,mu,sd,classes).
Returns None when unavailable so callers degrade to the existing CLIP behaviour.
"""
from __future__ import annotations
import os
import numpy as np

_NPZ = os.path.join(os.path.dirname(__file__), "event_classifiers", "birthday_fused.npz")


class BirthdayMomentClassifier:
    _inst = None

    def __init__(self):
        self.ok = False
        try:
            d = np.load(_NPZ, allow_pickle=True)
            self.W = d["W"].astype(np.float32); self.b = d["b"].astype(np.float32)
            self.mu = d["mu"].astype(np.float32); self.sd = d["sd"].astype(np.float32)
            self.classes = [str(c) for c in d["classes"]]
            self.cv = float(d["cv_acc"]) if "cv_acc" in d else 0.0
            self.ok = True
        except Exception:
            self.ok = False

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def predict(self, frame, scene_emb, faces, clip):
        """(label, prob, all_probs) or None. scene_emb = the full-frame CLIP
        embedding the pipeline already computed; faces = FACE_PROVIDER faces."""
        if not self.ok or scene_emb is None or frame is None or clip is None:
            return None
        try:
            from models.emotion_detector import detect_emotions
            from models.candle_phase import count_warm_flames
            from models.cake_smash import _frosting_fraction
            H, W = frame.shape[:2]
            big = max(faces, key=lambda f: f.box[2] * f.box[3]) if faces else None
            if big is not None:
                x, y, w, h = big.box; pad = int(0.45 * max(w, h))
                crop = frame[max(0, y-pad):min(H, y+h+pad), max(0, x-pad):min(W, x+w+pad)]
            else:
                crop = frame[H//4:3*H//4, W//4:3*W//4]
            sc = clip.score_frame(crop, event="birthday") if crop.size else None
            action = getattr(sc, "_image_emb", None) if sc is not None else None
            scene = np.asarray(scene_emb, np.float32)
            action = np.asarray(action, np.float32) if action is not None else scene
            clipv = np.concatenate([scene, action]); clipv /= (np.linalg.norm(clipv) + 1e-8)

            flame = float(count_warm_flames(frame)); fc = float(len(faces))
            pucker = funnel = jaw = brow = surprise = smile = frost = hfrac = bottom = 0.0
            if big is not None:
                bs = big.blendshapes or {}
                pucker = bs.get("mouthPucker", 0.); funnel = bs.get("mouthFunnel", 0.)
                jaw = bs.get("jawOpen", 0.)
                brow = max(bs.get("browInnerUp", 0.), bs.get("browOuterUpLeft", 0.),
                           bs.get("browOuterUpRight", 0.))
                smile = 0.5 * (bs.get("mouthSmileLeft", 0.) + bs.get("mouthSmileRight", 0.))
                hfrac = h / H; bottom = (y + h) / H
                frost = _frosting_fraction(frame, big.box, W, H)
                try: surprise = detect_emotions([big]).per_emotion.get("surprise", 0.)
                except Exception: pass
            pout = max(pucker, funnel); lit = 1.0 if flame >= 1 else 0.0
            hand = np.array([pucker, funnel, jaw, brow, surprise, smile, frost,
                             hfrac, bottom, flame, flame, fc, fc,
                             pout * lit, pout * hfrac], np.float32)

            feat = np.concatenate([clipv, hand]).astype(np.float32)
            fs = (feat - self.mu) / self.sd
            logits = fs @ self.W.T + self.b
            e = np.exp(logits - logits.max()); probs = e / e.sum()
            i = int(np.argmax(probs))
            return self.classes[i], float(probs[i]), {c: float(p) for c, p in zip(self.classes, probs)}
        except Exception:
            return None
