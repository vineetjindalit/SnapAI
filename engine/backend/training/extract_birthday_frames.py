"""
training/extract_birthday_frames.py — frame dump for hand-labelling.

Samples every birthday clip at a fixed rate, resizes to a labelling-friendly
size, and writes per-video subfolders plus a single manifest CSV you can open
in any spreadsheet and fill in the `label` column.

    Birthday_frames/
        <video_stem>/
            frame_0000.jpg
            frame_0001.jpg
        ...
        labels.csv          # video, frame_file, path, t_seconds, label

Usage:
    python -m training.extract_birthday_frames [--fps 2.0] [--max-edge 1280]
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent.parent          # snappy_final/
SRC_DIR = ROOT / "tests" / "datasets" / "clips" / "real" / "birthday"
OUT_DIR = ROOT / "Birthday_frames"

# Moment vocabulary the labeller should use (kept in sync with the model's
# active birthday classes) — printed into the CSV header comment for guidance.
LABEL_VOCAB = [
    "cake_with_candles", "candle_blowing", "cake_cutting", "cake_feeding",
    "group_photo", "hug_moment", "clapping_scene", "general_peak", "negative",
]


def _safe(name: str) -> str:
    return re.sub(r"[^\w\-]", "_", name)[:80].strip("_")


def extract(fps_sample: float = 2.0, max_edge: int = 1280) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = sorted(glob.glob(str(SRC_DIR / "*.mp4")))
    if not clips:
        print(f"No clips found in {SRC_DIR}")
        return

    manifest_rows = []
    total_frames = 0
    t_start = time.time()
    print(f"Extracting {len(clips)} clips → {OUT_DIR}  "
          f"(@{fps_sample} fps, ≤{max_edge}px)\n")

    for ci, clip in enumerate(clips, 1):
        stem = _safe(Path(clip).stem)
        out_sub = OUT_DIR / stem
        out_sub.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(clip)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, round(src_fps / fps_sample))
        idx = saved = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % step == 0 and fr is not None and fr.size:
                h, w = fr.shape[:2]
                scale = min(1.0, max_edge / float(max(h, w)))
                if scale < 1.0:
                    fr = cv2.resize(fr, (int(w * scale), int(h * scale)),
                                    interpolation=cv2.INTER_AREA)
                fname = f"frame_{saved:04d}.jpg"
                cv2.imwrite(str(out_sub / fname), fr,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
                manifest_rows.append({
                    "video": stem,
                    "frame_file": fname,
                    "path": str((out_sub / fname).relative_to(ROOT)),
                    "t_seconds": round(idx / src_fps, 2),
                    "label": "",          # ← you fill this in
                })
                saved += 1
            idx += 1
        cap.release()
        total_frames += saved
        print(f"  [{ci:2d}/{len(clips)}] {stem[:50]:50} → {saved:3d} frames")

    # Single manifest CSV for labelling
    csv_path = OUT_DIR / "labels.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "frame_file", "path",
                                          "t_seconds", "label"])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nDone: {total_frames} frames from {len(clips)} clips in "
          f"{time.time()-t_start:.0f}s")
    print(f"Manifest: {csv_path}")
    print(f"Label vocab: {', '.join(LABEL_VOCAB)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=2.0,
                    help="frames sampled per second of video")
    ap.add_argument("--max-edge", type=int, default=1280,
                    help="max long-edge pixels (frames resized down to this)")
    args = ap.parse_args()
    extract(args.fps, args.max_edge)
