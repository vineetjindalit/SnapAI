"""
utils/safe_types.py
Every value going into WebSocket JSON MUST pass through safe().
This permanently fixes: numpy bool_, float64, float32, int32, ndarray serialization errors.
"""
import json
import numpy as np
from typing import Any


def safe(v: Any) -> Any:
    """Recursively convert numpy types → native Python."""
    if isinstance(v, dict):
        return {k: safe(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(safe(i) for i in v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return 0.0 if (f != f) else f          # guard NaN
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def clamp(v, lo=0.0, hi=1.0) -> float:
    return float(max(lo, min(hi, float(v))))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)
