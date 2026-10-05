"""
training/pre_label_frames.py — pre-fill Birthday_frames/labels.csv with the
model's current best guess so a human only has to CORRECT labels, not create
them from scratch.

For each frame it reproduces the pipeline's labelling decision (CLIP is the
authority; candle flame-count + leaning split candle_blowing vs
cake_with_candles; off-prompt scenes fall back to general_peak) and writes:

    label       ← the model's prediction (edit this to correct it)
    model_pred  ← same prediction, kept read-only for reference
    confidence  ← CLIP score of the winning class (low = review first)

The original CSV is backed up to labels.csv.bak first.

Usage:  python -m training.pre_label_frames
"""
from __future__ import annotations

import csv
import os
import shutil
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.clip_engine   import CLIPEngine
from models.face_provider import FaceProvider
from models.candle_phase  import count_warm_flames, face_leaning_in

ROOT = Path(__file__).resolve().parent.parent.parent


def _find_manifest() -> Path | None:
    """Locate Birthday_frames/labels.csv even if the folder has been moved.

    Checks the original location first, then searches the project tree. Frames
    are resolved relative to the CSV's own directory (via the video/frame_file
    columns), so the manifest works wherever it lives.
    """
    env = os.environ.get("SNAPPY_FRAMES_CSV")
    if env and Path(env).exists():
        return Path(env)
    direct = ROOT / "Birthday_frames" / "labels.csv"
    if direct.exists():
        return direct
    hits = sorted(ROOT.glob("**/Birthday_frames/labels.csv"))
    return hits[0] if hits else None

# Active birthday classes (mirrors category_requirements._BIRTHDAY_CLASSES).
ACTIVE = {"cake_cutting", "candle_blowing", "cake_with_candles", "group_photo",
          "hug_moment", "cake_feeding", "clapping_scene"}
OFF_PROMPT_MARGIN = 0.004     # matches pipeline
FLAME_MIN = 15                # matches pipeline candle presence guard


def predict(frame, clip, face) -> tuple:
    """Return (label, confidence) mirroring the pipeline's per-frame logic."""
    scores = clip.score_frame(frame, sid=None)
    pp = (scores.per_prompt or {}) if scores and scores.available else {}
    if not pp:
        return "general_peak", 0.0

    act = {k: v for k, v in pp.items() if k in ACTIVE}
    off = {k: v for k, v in pp.items() if k not in ACTIVE}
    a_best, a_val = (max(act.items(), key=lambda kv: kv[1]) if act else ("general_peak", 0.0))
    o_best, o_val = (max(off.items(), key=lambda kv: kv[1]) if off else (None, 0.0))

    # Off-prompt scene → generic (per-frame approximation of the scene vote).
    if o_best is not None and (o_val - a_val) >= OFF_PROMPT_MARGIN:
        return "general_peak", float(a_val)

    label = a_best

    # Candle presence + sub-phase (no temporal history for a still frame).
    if label in ("candle_blowing", "cake_with_candles"):
        n = count_warm_flames(frame)
        if n < FLAME_MIN:
            return "general_peak", float(a_val)
        h, w = frame.shape[:2]
        ff = face.detect(frame)
        boxes = [f.box for f in ff.faces] if ff.available else []
        label = "candle_blowing" if face_leaning_in(boxes, h, w) else "cake_with_candles"
    elif label == "cake_cutting":
        if count_warm_flames(frame) >= FLAME_MIN:
            label = "cake_with_candles"

    return label, float(a_val)


def main() -> None:
    CSV_PATH = _find_manifest()
    if CSV_PATH is None:
        print("No Birthday_frames/labels.csv found — run extract_birthday_frames first.")
        return
    frames_dir = CSV_PATH.parent          # frames live under here, per video
    print(f"Manifest: {CSV_PATH}")
    shutil.copy2(CSV_PATH, CSV_PATH.with_suffix(".csv.bak"))

    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))

    clip = CLIPEngine.get(); clip.warmup()
    face = FaceProvider.get()
    print(f"Pre-labelling {len(rows)} frames (CLIP on {clip._device if hasattr(clip,'_device') else '?'})...")

    from collections import Counter
    dist = Counter()
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        # Resolve relative to the CSV (move-proof) via video/frame_file columns.
        fp = frames_dir / r["video"] / r["frame_file"]
        img = cv2.imread(str(fp))
        if img is None:
            r["label"] = ""; r["model_pred"] = ""; r["confidence"] = ""
            continue
        label, conf = predict(img, clip, face)
        r["label"] = label
        r["model_pred"] = label
        r["confidence"] = f"{conf:.3f}"
        dist[label] += 1
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}  ({time.time()-t0:.0f}s)")

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_file", "path",
                                          "t_seconds", "label", "model_pred",
                                          "confidence"])
        w.writeheader(); w.writerows(rows)

    print(f"\nDone in {time.time()-t0:.0f}s. Prediction distribution:")
    for k, v in dist.most_common():
        print(f"  {k:18} {v}")
    print(f"\nManifest updated: {CSV_PATH}  (backup: {CSV_PATH.name}.bak)")


if __name__ == "__main__":
    main()
