"""
backend/gpu/device.py — single source of truth for which device every model runs on.

Detection order (highest performance first):
  1. CUDA   — NVIDIA GPUs (workstations, AWS EC2 g5/p3, Colab)
  2. MPS    — Apple Silicon (M1/M2/M3 Macs, ~3-5x CPU speed for CLIP)
  3. CPU    — fallback

Override via env var:
    SNAPPY_DEVICE=cpu python3 run.py        # force CPU
    SNAPPY_DEVICE=cuda:0 python3 run.py     # specific GPU

All model wrappers (clip_engine, nima_aesthetic, hsemotion_fer) consult
get_device() so changing the env var is the only thing you need.
"""
from __future__ import annotations

import logging
import os
from typing import Dict

log = logging.getLogger("snappy.gpu")


def _detect() -> str:
    override = os.environ.get("SNAPPY_DEVICE", "").strip().lower()
    if override:
        return override
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps") \
                and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


DEVICE: str = _detect()


def get_device() -> str:
    """Returns 'cuda', 'cuda:0', 'mps', or 'cpu'."""
    return DEVICE


def device_summary() -> Dict[str, object]:
    info: Dict[str, object] = {"device": DEVICE}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_mem_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["mps_available"] = True
    except Exception as e:
        info["torch_error"] = str(e)
    return info


# Log at import time so server boot logs show what we detected.
log.info(f"Compute device → {DEVICE}")
