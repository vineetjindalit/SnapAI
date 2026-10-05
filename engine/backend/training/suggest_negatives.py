"""
training/suggest_negatives.py — rank the runs most likely to be 'negative'
(boring / transitional / blurry / no-people) so you can label them fast.

Reads your CURRENT labels (xlsx or Numbers), scores each run's representative
frame by blur + face-count + brightness, and writes negative_candidates.csv —
the runs to jump to in the sheet and mark 'negative'. Runs you've already
labelled 'negative' are skipped.

Usage:  python -m training.suggest_negatives [--top 60]
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = Path(__file__).resolve().parent.parent.parent


def _find_xlsx():
    hits = sorted(ROOT.glob("**/Birthday_frames/labels_visual.xlsx"))
    return hits[0] if hits else None


def _runs_rows(path: Path):
    """Yield (video, frames_str, current_label) from the runs sheet."""
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True)
        rows = [tuple(r) for r in wb["LABEL HERE (runs)"].iter_rows(min_row=1, values_only=True)]
    except Exception:
        from numbers_parser import Document
        tmp = Path(tempfile.mkdtemp()) / "labels.numbers"; shutil.copy(path, tmp)
        doc = Document(str(tmp))
        sheet = next(s for s in doc.sheets if "LABEL" in s.name.upper())
        rows = sheet.tables[0].rows(values_only=True)
    for r in rows[1:]:
        if r and len(r) >= 7 and r[1]:
            yield str(r[1]), str(r[2]), (str(r[6]) if r[6] else "")


def _blur(img) -> float:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    xlsx = _find_xlsx()
    if not xlsx:
        print("labels_visual.xlsx not found."); return
    frames_dir = xlsx.parent
    from models.face_provider import FaceProvider
    fp = FaceProvider.get()

    cands = []
    for video, frames, label in _runs_rows(xlsx):
        if label.lower() == "negative":
            continue
        try:
            a, b = (int(x) for x in frames.replace("–", "-").replace("—", "-").split("-"))
        except Exception:
            continue
        mid = (a + b) // 2
        fpth = frames_dir / video / f"frame_{mid:04d}.jpg"
        img = cv2.imread(str(fpth))
        if img is None:
            continue
        faces = fp.detect(img).count if fp else 0
        blur = _blur(img)
        # negativeness: no faces is the strongest cue, then blurriness
        score = (1.0 if faces == 0 else 0.0) * 100 + max(0.0, 120.0 - blur) / 120.0
        cands.append((score, video, frames, label, faces, round(blur, 1)))

    cands.sort(reverse=True)
    out = frames_dir / "negative_candidates.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "video", "frames", "current_label", "faces", "blur",
                    "why"])
        for i, (sc, video, frames, label, faces, blur) in enumerate(cands[:args.top], 1):
            why = ("no faces" if faces == 0 else "") + \
                  (" + blurry" if blur < 80 else "")
            w.writerow([i, video, frames, label, faces, blur, why.strip(" +") or "low-signal"])
    print(f"Scanned {len(cands)} runs. Top {min(args.top,len(cands))} negative "
          f"candidates → {out}")
    print("\nTop 15 to mark 'negative' (video | frames | now → suggest):")
    for sc, video, frames, label, faces, blur in cands[:15]:
        print(f"  {video[:34]:34} {frames:>8}  {label or '∅':16} faces={faces} blur={blur}")


if __name__ == "__main__":
    main()
