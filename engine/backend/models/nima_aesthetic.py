"""
models/nima_aesthetic.py — NIMA learned aesthetic scoring.

Roadmap target: "ResNet-18 on AVA dataset — Learned aesthetic scoring".

NIMA (Neural Image Assessment, Talebi & Milanfar 2018) is the canonical
aesthetic-quality model, trained on the AVA dataset (~250K human-rated
photos). pyiqa ships a pretrained NIMA backbone we can call directly.

Requires: `pip install pyiqa`. Falls back to disabled if missing — the
existing Laplacian/heuristic shot quality keeps scoring frames.

Output: float in [0, 1] approximating "human aesthetic preference".
We blend this into ShotQualityAnalyzer with weight ~0.35 when available,
which is the single biggest accuracy gain in the entire pipeline.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

log = logging.getLogger("snappy.nima")


class NIMAScorer:
    _instance: Optional["NIMAScorer"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._metric = None
        self._init_error: Optional[str] = None
        self._device = "cpu"
        self._try_load()

    @classmethod
    def get(cls) -> "NIMAScorer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = NIMAScorer()
            return cls._instance

    def _try_load(self) -> None:
        try:
            import torch
            import pyiqa
        except Exception as e:
            self._init_error = (
                f"pyiqa not installed ({e}). "
                "Install with: pip install pyiqa torch"
            )
            log.warning(self._init_error)
            return

        try:
            from gpu.device import get_device
            self._device = get_device()
            dev_for_pyiqa = self._device if self._device.startswith(("cuda", "mps")) else "cpu"
            log.info(f"Loading NIMA aesthetic model on {dev_for_pyiqa} "
                     "(first call downloads ~95 MB)...")
            t0 = time.time()
            self._metric = pyiqa.create_metric("nima", as_loss=False, device=dev_for_pyiqa)
            self._metric.eval()
            log.info(f"NIMA ready on {self._device} in {time.time()-t0:.1f}s")
        except Exception as e:
            self._init_error = f"NIMA load failed: {e}"
            log.exception(self._init_error)
            self._metric = None

    @property
    def available(self) -> bool:
        return self._metric is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def score(self, bgr_frame: np.ndarray) -> Optional[float]:
        """Returns aesthetic score normalised to [0,1] (NIMA outputs 1-10)."""
        if self._metric is None:
            return None
        try:
            import torch
            from PIL import Image
            rgb = bgr_frame[:, :, ::-1]
            pil = Image.fromarray(rgb)
            with torch.no_grad():
                raw = self._metric(pil)
                if isinstance(raw, torch.Tensor):
                    raw = raw.detach().cpu().item()
            # NIMA outputs roughly 1-10; map to [0,1] and clip.
            normed = max(0.0, min(1.0, (float(raw) - 3.0) / 5.0))
            return normed
        except Exception as e:
            log.warning(f"NIMA inference failed: {e}")
            return None
