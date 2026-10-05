#!/usr/bin/env python3
"""
scripts/auto_label_videos.py — auto-tag each collected video at moment-level.

Workflow:
  1. Walk backend/data/event_videos/<class>/*.mp4
  2. Sample each video at 1 fps (configurable)
  3. Run each frame through your trained CLIP engine
  4. For each frame, record:
       (timestamp_seconds, top_class, confidence, second_best, second_conf)
  5. Identify "moments" — contiguous runs of high-confidence frames of the
     same class lasting ≥ MIN_MOMENT_SECONDS (default 2s)
  6. Emit one CSV per source video and an aggregate labels.jsonl

The output is consumable by:
  - scripts/labeler.html (for human verification)
  - backend/training/finetune_clip.py (for retraining)

Human review estimate: with auto-pre-labels at ~70% accuracy, a human can
review 30 hours of footage in ~3 hours of focused work (skim + flip
incorrect labels) vs ~30 hours of from-scratch labelling.

Usage
-----
    python3 scripts/auto_label_videos.py
    python3 scripts/auto_label_videos.py --sample-fps 2
    python3 scripts/auto_label_videos.py --classes cake_cutting,ring_ceremony
    python3 scripts/auto_label_videos.py --confidence 0.45 --min-moment 3
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("snapai.label")
ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "backend" / "data" / "event_videos"
LABELS_DIR = ROOT / "backend" / "data" / "event_labels"


def _setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")


def _video_metadata(path: Path) -> Tuple[float, int, int, int]:
    """Returns (duration_s, fps, n_frames, w, h)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0, 0, 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return float(n / fps if fps > 0 else 0.0), fps, n, (w, h)


def _label_one_video(video_path: Path, true_class: str,
                     clip_engine, sample_fps: float, conf_threshold: float,
                     min_moment_seconds: float) -> Dict:
    """Run CLIP on the video, emit per-frame predictions + moment runs."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": "could not open video", "frames": [], "moments": []}

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(src_fps / sample_fps))
    frame_idx = 0
    per_frame: List[Dict] = []
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % step == 0:
            t_sec = frame_idx / src_fps
            try:
                scores = clip_engine.score_frame(frame, sid="label-run")
                if scores and scores.available:
                    top1 = scores.best
                    conf = scores.best_score
                    sorted_pp = sorted(scores.per_prompt.items(),
                                       key=lambda kv: -kv[1])
                    top2 = sorted_pp[1] if len(sorted_pp) > 1 else (None, 0)
                else:
                    top1, conf, top2 = "watching", 0.0, (None, 0)
                per_frame.append({
                    "ts": round(t_sec, 2),
                    "top1": top1, "conf1": round(float(conf), 3),
                    "top2": top2[0], "conf2": round(float(top2[1]), 3),
                })
            except Exception as e:
                LOG.debug(f"  frame {frame_idx} err: {e}")
        frame_idx += 1
    cap.release()

    # Identify moments — contiguous runs of the same top1 above threshold
    moments: List[Dict] = []
    cur_class, cur_start, cur_max_conf = None, None, 0.0
    for f in per_frame:
        cls = f["top1"]
        conf = f["conf1"]
        if cls == "watching" or conf < conf_threshold:
            cls = None
        if cls != cur_class:
            if cur_class is not None and cur_start is not None:
                dur = f["ts"] - cur_start
                if dur >= min_moment_seconds:
                    moments.append({
                        "class": cur_class,
                        "start": round(cur_start, 2),
                        "end":   round(f["ts"], 2),
                        "duration": round(dur, 2),
                        "max_confidence": round(cur_max_conf, 3),
                    })
            cur_class = cls
            cur_start = f["ts"]
            cur_max_conf = conf
        else:
            cur_max_conf = max(cur_max_conf, conf)
    # Tail moment
    if cur_class and cur_start is not None and per_frame:
        last = per_frame[-1]["ts"]
        dur = last - cur_start
        if dur >= min_moment_seconds:
            moments.append({
                "class": cur_class,
                "start": round(cur_start, 2),
                "end":   round(last, 2),
                "duration": round(dur, 2),
                "max_confidence": round(cur_max_conf, 3),
            })

    elapsed = time.time() - t0
    return {
        "file": str(video_path.relative_to(ROOT)),
        "true_class": true_class,
        "frames_labeled": len(per_frame),
        "moments": moments,
        "frames": per_frame,
        "processing_seconds": round(elapsed, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", default=str(VIDEOS_DIR))
    ap.add_argument("--labels-dir", default=str(LABELS_DIR))
    ap.add_argument("--sample-fps", type=float, default=1.0)
    ap.add_argument("--confidence", type=float, default=0.42)
    ap.add_argument("--min-moment", type=float, default=2.0)
    ap.add_argument("--classes", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap on videos to process (0 = no limit)")
    args = ap.parse_args()
    _setup_logging()

    sys.path.insert(0, str(ROOT / "backend"))
    try:
        from models.clip_engine import CLIPEngine
    except Exception as e:
        LOG.error(f"Could not import CLIPEngine: {e}")
        LOG.error("Ensure you've run `pip install -r requirements.txt` "
                  "in the active venv.")
        sys.exit(2)

    clip = CLIPEngine.get()
    if not clip.warmup():
        LOG.error(f"CLIP failed to warm up: {clip.init_error}")
        sys.exit(2)
    LOG.info(f"CLIP ready ({'trained centroids loaded' if clip.has_trained_centroids else 'text prompts only'})")

    vdir = Path(args.videos_dir)
    if not vdir.exists():
        LOG.error(f"Videos dir not found: {vdir}")
        LOG.error("Run scripts/dataset_collect.py first.")
        sys.exit(2)

    ldir = Path(args.labels_dir)
    ldir.mkdir(parents=True, exist_ok=True)

    class_subset = ({c.strip() for c in args.classes.split(",") if c.strip()}
                    if args.classes else None)

    # Walk class folders
    videos: List[Tuple[Path, str]] = []
    for class_dir in sorted(vdir.iterdir()):
        if not class_dir.is_dir(): continue
        cls = class_dir.name
        if class_subset and cls not in class_subset: continue
        for v in sorted(class_dir.glob("*.mp4")):
            videos.append((v, cls))
        for v in sorted(class_dir.glob("*.mkv")):
            videos.append((v, cls))
    if args.limit:
        videos = videos[:args.limit]

    if not videos:
        LOG.error("No videos found to label")
        sys.exit(2)

    LOG.info(f"Labeling {len(videos)} videos at {args.sample_fps} fps, "
             f"conf ≥ {args.confidence}, min moment {args.min_moment}s")
    LOG.info("=" * 60)

    aggregate_path = ldir / "labels.jsonl"
    summary = {"videos": 0, "frames": 0, "moments": 0,
               "moments_per_class": {}, "hours_processed": 0.0}
    t0 = time.time()

    with aggregate_path.open("a") as agg_f:
        for i, (vpath, cls) in enumerate(videos, 1):
            LOG.info(f"[{i}/{len(videos)}] {vpath.relative_to(ROOT)}")
            result = _label_one_video(vpath, cls, clip, args.sample_fps,
                                       args.confidence, args.min_moment)
            if "error" in result:
                LOG.warning(f"  ✗ {result['error']}")
                continue

            # Per-video CSV
            csv_path = ldir / (vpath.stem + ".csv")
            with csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp_s", "top_class", "confidence",
                             "second_class", "second_confidence"])
                for fr in result["frames"]:
                    w.writerow([fr["ts"], fr["top1"], fr["conf1"],
                                fr["top2"] or "", fr["conf2"]])

            # Aggregate JSONL — one line per processed video
            agg_f.write(json.dumps(result) + "\n")
            agg_f.flush()

            n_moments = len(result["moments"])
            LOG.info(f"  ✓ {result['frames_labeled']} frames, "
                     f"{n_moments} moments, {result['processing_seconds']}s")
            for m in result["moments"][:5]:
                LOG.info(f"    {m['start']:.1f}-{m['end']:.1f}s · "
                         f"{m['class']} (conf {m['max_confidence']})")
            if n_moments > 5:
                LOG.info(f"    … and {n_moments - 5} more")

            summary["videos"] += 1
            summary["frames"] += result["frames_labeled"]
            summary["moments"] += n_moments
            for m in result["moments"]:
                summary["moments_per_class"].setdefault(m["class"], 0)
                summary["moments_per_class"][m["class"]] += 1

    elapsed = time.time() - t0
    summary["seconds_elapsed"] = round(elapsed, 1)

    LOG.info("=" * 60)
    LOG.info(f"DONE in {elapsed/60:.1f} min")
    LOG.info(f"  videos:  {summary['videos']}")
    LOG.info(f"  frames:  {summary['frames']}")
    LOG.info(f"  moments: {summary['moments']}")
    for cls, n in sorted(summary["moments_per_class"].items(),
                         key=lambda x: -x[1]):
        LOG.info(f"    {cls:25s} {n}")
    LOG.info("")
    LOG.info(f"Per-video CSVs: {ldir}/")
    LOG.info(f"Aggregate JSONL: {aggregate_path}")
    LOG.info("")
    LOG.info("To human-verify a video's labels:")
    LOG.info(f"  open scripts/labeler.html  # then drag in any .mp4 + .csv")


if __name__ == "__main__":
    main()
