"""
models/yolo_face.py — YOLOv8-face detector.

Roadmap target: "MediaPipe / YOLOv8-face — ~97% accuracy, all angles".

YOLOv8-face is a community fine-tune of YOLOv8 on the WIDER FACE dataset.
Weights file (~6 MB) auto-downloads on first use to backend/models/.

Requires: `pip install ultralytics`. Without it, this provider reports
unavailable and FaceProvider falls through to MediaPipe.
"""
from __future__ import annotations

import logging
import threading
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger("snappy.yolo_face")

_WEIGHTS_PATH = Path(__file__).with_name("yolov8n-face.pt")
# Multiple fallback URLs — the akanametov repo got renamed in 2024+ and
# old release tags 404. We try a chain; first hit wins.
_WEIGHTS_URLS = [
    # Primary: akanametov/yolo-face release 1.0.0 (verified live 2026-05).
    "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
    # Fallback 1: HuggingFace mirror, ~6 MB, verified live.
    "https://huggingface.co/arnabdhar/YOLOv8-Face-Detection/resolve/main/model.pt",
    # Fallback 2: legacy URLs in case the user is behind a corporate proxy
    # that still has the old releases cached.
    "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt",
]


class YOLOFaceDetector:
    _instance: Optional["YOLOFaceDetector"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._init_error: Optional[str] = None
        self._try_load()

    @classmethod
    def get(cls) -> "YOLOFaceDetector":
        with cls._lock:
            if cls._instance is None:
                cls._instance = YOLOFaceDetector()
            return cls._instance

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as e:
            self._init_error = (
                f"ultralytics not installed ({e}). "
                "Install with: pip install ultralytics"
            )
            log.warning(self._init_error)
            return

        try:
            if not _WEIGHTS_PATH.exists():
                if not self._download_weights():
                    self._init_error = (
                        "YOLOv8-face weights not found. All download mirrors "
                        "failed. Manual fix: download yolov8n-face.pt and place "
                        f"it at {_WEIGHTS_PATH}. The system falls back to "
                        "MediaPipe FaceMesh in the meantime — fully functional."
                    )
                    log.warning(self._init_error)
                    return
            self._model = YOLO(str(_WEIGHTS_PATH))
            self._model.to("cpu")
            log.info("YOLOv8-face ready (CPU)")
        except Exception as e:
            self._init_error = f"YOLOv8-face load failed: {e}"
            log.warning(self._init_error)
            self._model = None

    def _download_weights(self) -> bool:
        """Try each mirror in order. Returns True on success."""
        for url in _WEIGHTS_URLS:
            try:
                log.info(f"Trying YOLOv8-face mirror: {url}")
                urllib.request.urlretrieve(url, _WEIGHTS_PATH)
                if _WEIGHTS_PATH.exists() and _WEIGHTS_PATH.stat().st_size > 1_000_000:
                    log.info(f"Downloaded → {_WEIGHTS_PATH} "
                             f"({_WEIGHTS_PATH.stat().st_size // 1024} KB)")
                    return True
            except Exception as e:
                log.info(f"  mirror failed ({e.__class__.__name__}): {e}")
                # Clean up partial download
                try: _WEIGHTS_PATH.unlink(missing_ok=True)
                except Exception: pass
        return False

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def detect(self, frame: np.ndarray,
               conf: float = 0.4) -> List[Tuple[int, int, int, int]]:
        """Returns list of (x, y, w, h) face boxes in pixel coords."""
        if self._model is None:
            return []
        try:
            results = self._model.predict(frame, conf=conf, verbose=False)
            boxes: List[Tuple[int, int, int, int]] = []
            if not results:
                return boxes
            for r in results:
                if r.boxes is None:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                for x1, y1, x2, y2 in xyxy:
                    boxes.append((int(x1), int(y1),
                                  max(int(x2 - x1), 1), max(int(y2 - y1), 1)))
            return boxes
        except Exception as e:
            log.warning(f"YOLO inference failed: {e}")
            return []
