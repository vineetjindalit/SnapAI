"""
models/face_identity.py — WHO is in the frame (face identity), via ArcFace.

This is used ONLY to tell the host's people apart from strangers — never for
the moment / emotion / capture-timing logic. It returns a 512-d identity
embedding per detected face (same person → high cosine, different → low).

Backed by InsightFace (buffalo_l: RetinaFace detect + ArcFace recognise). If
InsightFace or its model isn't present, it degrades to .ok = False and the
pipeline runs exactly as before (personalization simply off). Set
SNAPPY_PERSONALIZATION=0 to disable.
"""
from __future__ import annotations
import os
import numpy as np

import logging
log = logging.getLogger("snappy.face_identity")


class FaceIdentifier:
    _inst = None

    def __init__(self):
        self.ok = False
        self._app = None
        if os.environ.get("SNAPPY_PERSONALIZATION", "1") != "1":
            log.info("face identity disabled (SNAPPY_PERSONALIZATION=0)")
            return
        try:
            from insightface.app import FaceAnalysis
            # recognition + detection only; CPU is fine (we run it sparingly)
            app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                               providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            self._app = app
            self.ok = True
            log.info("FaceIdentifier ready (ArcFace buffalo_l, CPU)")
        except Exception as e:
            self.ok = False
            log.info(f"face identity unavailable ({e}) — personalization off")

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def faces(self, frame) -> list:
        """Return one dict per face: box(xywh), emb(512 unit), score, area
        fraction, centrality (1=centre). [] when unavailable / no faces."""
        if not self.ok or frame is None:
            return []
        try:
            H, W = frame.shape[:2]
            out = []
            for f in self._app.get(frame):
                x1, y1, x2, y2 = f.bbox
                w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
                cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                out.append(dict(
                    box=(int(x1), int(y1), int(w), int(h)),
                    emb=np.asarray(f.normed_embedding, np.float32),
                    score=float(getattr(f, "det_score", 1.0)),
                    area=float(w * h) / float(W * H),
                    centrality=float(1.0 - min(1.0, abs(cx - 0.5) + abs(cy - 0.5))),
                ))
            return out
        except Exception as e:
            log.debug(f"face identity inference failed: {e}")
            return []
