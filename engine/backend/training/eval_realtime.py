"""
backend/training/eval_realtime.py — measure the REAL-TIME model's accuracy.

Runs labeled birthday clips through the exact LIVE capture path
(process_frame(source="live") at 5 fps — same code the camera/replay hits) and
scores the detected moment against the human labels in labels_master.csv.

Reported per class and overall:
  frame-vote acc : confidence-weighted majority of detected_moment over frames
  capture-tag acc: majority moment_type over the photos it actually captured
  zero-capture   : clips where nothing was captured at all (a real failure)

Run:  SNAPPY_VLM=0 .venv/bin/python3 backend/training/eval_realtime.py \
        [--per-class 4] [--max-frames 75] [--json /tmp/eval.json]
"""
from __future__ import annotations
import os, sys, csv, json, time, argparse, tempfile, collections
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2

_HERE  = os.path.dirname(__file__)
_CSV   = os.path.join(_HERE, "labels_master.csv")
_CLIPS = os.path.join(_HERE, "..", "..", "tests", "datasets", "Birthday", "clips", "real")

# canonical label space (ground truth ↔ system labels)
_CANON = {
    "cake_person": "cake",           # ground-truth merge: person posing with cake
    "hug_moment":  "hugging_moments",
}
_EVAL_CLASSES = ["cake_with_candles", "candle_blowing", "cake", "birthday_gifting",
                 "cake_cutting", "cake_feeding", "cake_smashing"]


def canon(lbl: str) -> str:
    return _CANON.get(str(lbl or ""), str(lbl or ""))


def pick_clips(per_class: int):
    rows = list(csv.DictReader(open(_CSV)))
    by = collections.defaultdict(list)
    for r in rows:
        c = canon(r["label"])
        if c in _EVAL_CLASSES and os.path.exists(os.path.join(_CLIPS, r["clip_id"] + ".mp4")):
            by[c].append(r["clip_id"])
    picked = []
    for c in _EVAL_CLASSES:
        for cid in sorted(by[c])[:per_class]:
            picked.append((cid, c))
    return picked


def eval_clip(cid: str, max_frames: int, captures_root: Path):
    from api.session import Session
    from api.pipeline import process_frame
    from models.category_requirements import get_profile
    prof = get_profile("birthday")
    s = Session(f"ev_{cid[:8]}", "E", "birthday", prof.default_prompt, captures_root,
                effective_prompt=prof.default_prompt, required_shots=prof.required_shots)
    cap = cv2.VideoCapture(os.path.join(_CLIPS, cid + ".mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, round(fps / 5.0))              # live cadence: 5 fps
    votes: dict = collections.defaultdict(float)
    i = done = 0
    while done < max_frames:
        ok, fr = cap.read()
        if not ok:
            break
        if i % step == 0:
            res = process_frame(s, "", frame=fr, source="live")
            m = res["moment"]["detected_moment"]; cf = float(res["moment"]["confidence"])
            votes[canon(m)] += max(0.05, cf)
            done += 1
        i += 1
    cap.release()
    frame_pred = max(votes, key=votes.get) if votes else ""
    cap_tags = collections.Counter(canon(p.moment_type) for p in s.photos)
    cap_pred = cap_tags.most_common(1)[0][0] if cap_tags else ""
    return frame_pred, cap_pred, len(s.photos), done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4)
    ap.add_argument("--max-frames", type=int, default=75)
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from api.globals import CLIP
    if not CLIP.available:
        print("CLIP unavailable — aborting."); return
    CLIP.warmup()

    picked = pick_clips(args.per_class)
    print(f"evaluating {len(picked)} clips ({args.per_class}/class) through the LIVE path @5fps…")
    tmp = Path(tempfile.mkdtemp(prefix="snappy_eval_"))
    t0 = time.time()
    results = []
    for k, (cid, truth) in enumerate(picked):
        fp, cp, ncaps, nframes = eval_clip(cid, args.max_frames, tmp)
        ok_f, ok_c = fp == truth, (cp == truth if ncaps else False)
        results.append(dict(cid=cid, truth=truth, frame_pred=fp, cap_pred=cp,
                            captures=ncaps, frames=nframes))
        print(f"  [{k+1:2d}/{len(picked)}] {cid[:26]:26s} truth={truth:18s} "
              f"frame={fp:18s}{'✓' if ok_f else '✗'} "
              f"cap={cp or '—':18s}{'✓' if ok_c else '✗'} n={ncaps}")

    per = collections.defaultdict(lambda: [0, 0, 0])   # class → [n, frame_ok, cap_ok]
    zero = 0
    for r in results:
        per[r["truth"]][0] += 1
        per[r["truth"]][1] += int(r["frame_pred"] == r["truth"])
        per[r["truth"]][2] += int(r["cap_pred"] == r["truth"] and r["captures"] > 0)
        zero += int(r["captures"] == 0)
    n = len(results)
    fa = sum(v[1] for v in per.values()); ca = sum(v[2] for v in per.values())
    print(f"\n=== LIVE-PATH ACCURACY ({n} clips, {time.time()-t0:.0f}s) ===")
    print(f"{'class':20s} {'n':>2s} {'frame':>6s} {'captag':>7s}")
    for c in _EVAL_CLASSES:
        if c in per:
            v = per[c]
            print(f"{c:20s} {v[0]:2d} {v[1]}/{v[0]:<4d} {v[2]}/{v[0]}")
    print(f"{'OVERALL':20s} {n:2d} {fa}/{n} ({fa/n:.0%})  {ca}/{n} ({ca/n:.0%})")
    print(f"zero-capture clips: {zero}/{n}")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)
        print(f"json → {args.json}")


if __name__ == "__main__":
    main()
