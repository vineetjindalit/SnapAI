#!/usr/bin/env python3
"""
scripts/dataset_report.py — summarise collected + labelled dataset.

Reports:
  - Total hours collected, per class
  - Per-source breakdown
  - Per-class label confidence histogram
  - Time to 30-hour goal — how much remains, by class
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "backend" / "data" / "event_videos"
LABELS_DIR = ROOT / "backend" / "data" / "event_labels"
META_PATH  = VIDEOS_DIR / "collection_metadata.jsonl"
LABELS_AGG = LABELS_DIR / "labels.jsonl"


def _h(sec: float) -> str:
    if sec < 60: return f"{sec:.0f}s"
    if sec < 3600: return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-hours", type=float, default=30,
                    help="overall goal in hours (default 30)")
    args = ap.parse_args()

    # --- Collection metadata ---
    per_class_seconds: Dict[str, float] = defaultdict(float)
    per_source_count: Dict[str, int] = defaultdict(int)
    per_source_seconds: Dict[str, float] = defaultdict(float)
    total_clips = 0
    total_size = 0

    if META_PATH.exists():
        for line in META_PATH.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            total_clips += 1
            cls = r.get("class", "?")
            dur = float(r.get("duration_s") or 0)
            src = r.get("source", "?")
            per_class_seconds[cls] += dur
            per_source_count[src] += 1
            per_source_seconds[src] += dur
            total_size += int(r.get("size_bytes") or 0)

    grand_total_sec = sum(per_class_seconds.values())
    target_sec = args.target_hours * 3600

    print("=" * 72)
    print("SnapAI Dataset Report")
    print("=" * 72)
    print(f"  Clips:       {total_clips}")
    print(f"  Total time:  {_h(grand_total_sec)} ({grand_total_sec/3600:.1f}h)")
    print(f"  Total size:  {total_size/1024/1024:.0f} MB")
    print(f"  Goal:        {args.target_hours:.0f}h "
          f"({grand_total_sec/target_sec*100:.0f}% complete)")
    print()

    # Per-class
    target_per_class = target_sec / max(1, len(per_class_seconds))
    print(f"--- Per-class hours ({len(per_class_seconds)} classes) ---")
    print(f"{'class':25s} {'collected':>10s}  {'goal':>8s}  {'%':>5s}  bar")
    for cls in sorted(per_class_seconds, key=lambda c: -per_class_seconds[c]):
        sec = per_class_seconds[cls]
        pct = sec / target_per_class * 100
        bar_len = int(min(40, pct / 2.5))
        bar = "█" * bar_len
        print(f"{cls:25s} {_h(sec):>10s}  {_h(target_per_class):>8s}  "
              f"{pct:>4.0f}%  {bar}")
    print()

    # Per-source
    print("--- Per-source breakdown ---")
    for src, count in sorted(per_source_count.items(), key=lambda x: -x[1]):
        sec = per_source_seconds[src]
        print(f"  {src:15s} {count:>5d} clips · {_h(sec):>10s}")
    print()

    # Labels (if labelled)
    if LABELS_AGG.exists():
        print("--- Labels (auto-labelled) ---")
        n_videos, n_frames, n_moments = 0, 0, 0
        moments_per_class: Dict[str, int] = defaultdict(int)
        conf_hist: Dict[str, List[float]] = defaultdict(list)
        for line in LABELS_AGG.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_videos += 1
            n_frames += int(r.get("frames_labeled", 0))
            for m in r.get("moments", []):
                n_moments += 1
                moments_per_class[m["class"]] += 1
                conf_hist[m["class"]].append(float(m["max_confidence"]))
        print(f"  videos labelled: {n_videos}")
        print(f"  frames analysed: {n_frames}")
        print(f"  moments tagged:  {n_moments}")
        print()
        print(f"  moments per class · avg confidence")
        for cls, n in sorted(moments_per_class.items(), key=lambda x: -x[1]):
            confs = conf_hist[cls]
            avg = sum(confs) / len(confs) if confs else 0
            print(f"    {cls:25s} {n:>5d}  conf={avg:.2f}")
    else:
        print("--- Labels ---")
        print("  Not yet labelled. Run: python3 scripts/auto_label_videos.py")
    print()

    # Goal calculation
    remaining = target_sec - grand_total_sec
    if remaining > 0:
        print(f"--- Remaining to 30h goal ---")
        weakest = sorted(per_class_seconds.items(), key=lambda x: x[1])[:5]
        print(f"  Total needed: {_h(remaining)} more")
        print(f"  Weakest 5 classes (priority targets):")
        for cls, sec in weakest:
            need = target_per_class - sec
            print(f"    {cls:25s} need {_h(need)} more")
        print()
        print("  Re-run collector to fill gaps:")
        cls_list = ",".join(c for c, _ in weakest)
        print(f"    python3 scripts/dataset_collect.py --classes {cls_list} "
              f"--hours {remaining/3600:.0f}")
    else:
        print(f"--- ✓ Goal reached! ---")
        print(f"  You have {grand_total_sec/3600:.1f} hours of footage across "
              f"{len(per_class_seconds)} classes.")
    print("=" * 72)


if __name__ == "__main__":
    main()
