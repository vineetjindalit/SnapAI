"""
backend/training/prepare_birthday_dataset.py
─────────────────────────────────────────────
One pass to turn raw birthday clips into a labelled, trainable dataset:

  1. CONSOLIDATE every clip (the old flat Birthday/ + incoming/youtube +
     incoming/other) into  Birthday/clips/<id>.mp4 , de-duped by content hash.
  2. PROBE fps + duration of each (reported, since source fps varies 24–60).
  3. EXTRACT frames at a FIXED TIME RATE (default 2 fps → every 0.5s) so a 60fps
     and a 25fps clip yield the SAME temporal sampling → Birthday/frames/<id>/.
  4. AUTO-LABEL each clip's moment (CLIP over the birthday vocabulary, averaged
     across its frames) — a best guess for you to correct.
  5. WRITE  birthday_labels.xlsx  — one row per clip, with an embedded thumbnail
     and a YOUR_label dropdown, so you fix mistakes without opening videos.

Then: you correct the sheet → train_birthday_from_clips.py reads it.

Run:  .venv/bin/python3 backend/training/prepare_birthday_dataset.py
"""
from __future__ import annotations
import os, sys, glob, hashlib, re, shutil
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2
from api.globals import CLIP

_HERE   = os.path.dirname(__file__)
_BDAY   = os.path.join(_HERE, "..", "..", "tests", "datasets", "Birthday")
_CLIPS  = os.path.join(_BDAY, "clips", "real")
_FRAMES = os.path.join(_BDAY, "Frames")
_INC    = os.path.join(_HERE, "..", "..", "tests", "datasets", "incoming")
_XLSX   = os.path.join(_HERE, "birthday_labels.xlsx")

_SAMPLE_FPS = 2.0      # frames per SECOND of video (fps-independent sampling)
_MAX_FRAMES = 24       # cap per clip
_MIN_FRAMES = 4

# Birthday moment vocabulary (the dropdown + the auto-label space).
VOCAB = ["cake", "cake_with_candles", "candle_blowing", "cake_cutting",
         "cake_feeding", "cake_smashing", "gifting", "group_photo", "party",
         "decorations", "portrait", "surprise", "dancing", "hugging",
         "gazing", "not_birthday"]


def _clean_id(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^\w\-]+", "_", stem)[:48].strip("_") or "clip"


def _content_key(path: str) -> str:
    """Fast dedupe key: size + md5 of the first 1 MB."""
    try:
        sz = os.path.getsize(path)
        with open(path, "rb") as f:
            h = hashlib.md5(f.read(1 << 20)).hexdigest()
        return f"{sz}_{h}"
    except OSError:
        return path


def consolidate() -> list:
    os.makedirs(_CLIPS, exist_ok=True)
    srcs = (glob.glob(os.path.join(_BDAY, "*.mp4"))            # old flat clips
            + glob.glob(os.path.join(_INC, "youtube", "*.mp4"))
            + glob.glob(os.path.join(_INC, "other", "*.mp4"))
            + glob.glob(os.path.join(_CLIPS, "*.mp4")))         # already-consolidated
    seen, out = {}, []
    for s in sorted(srcs):
        k = _content_key(s)
        if k in seen:
            continue
        seen[k] = s
        dest = os.path.join(_CLIPS, _clean_id(s) + ".mp4")
        if os.path.abspath(s) != os.path.abspath(dest) and not os.path.exists(dest):
            try: shutil.copy2(s, dest)
            except OSError: dest = s
        out.append(dest)
    return out


def probe(path):
    c = cv2.VideoCapture(path)
    fps = c.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(c.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    c.release()
    dur = (n / fps) if fps > 0 else 0.0
    return round(fps, 1), round(dur, 1), n


def extract_frames(path, cid, fps, n_total):
    out_dir = os.path.join(_FRAMES, cid)
    os.makedirs(out_dir, exist_ok=True)
    dur = (n_total / fps) if fps > 0 else 0.0
    n_want = int(np.clip(round(dur * _SAMPLE_FPS), _MIN_FRAMES, _MAX_FRAMES)) if dur > 0 else _MIN_FRAMES
    idxs = np.linspace(0, max(0, n_total - 1), n_want).astype(int) if n_total else []
    c = cv2.VideoCapture(path); saved = []
    for j, fi in enumerate(idxs):
        c.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = c.read()
        if not ok: continue
        h, w = fr.shape[:2]
        if w > 1280: fr = cv2.resize(fr, (1280, int(h * 1280 / w)))
        fp = os.path.join(out_dir, f"f{j:02d}.jpg")
        cv2.imwrite(fp, fr, [cv2.IMWRITE_JPEG_QUALITY, 85]); saved.append(fp)
    c.release()
    return saved


def auto_label(frame_paths):
    """Average CLIP birthday-vocab scores over the clip's frames → best moment."""
    if not CLIP.available or not frame_paths:
        return "party", 0.0
    agg = {}
    for fp in frame_paths[::2] or frame_paths:        # every other frame is plenty
        img = cv2.imread(fp)
        if img is None: continue
        sc = CLIP.score_frame(img, event="birthday")
        for k, v in (sc.per_prompt or {}).items():
            agg[k] = agg.get(k, 0.0) + float(v)
    if not agg:
        return "party", 0.0
    # restrict to our vocab-ish classes; ignore generic 'watching'/'negative'
    cand = {k: v for k, v in agg.items() if k not in ("watching", "negative", "")}
    best = max(cand or agg, key=(cand or agg).get)
    n = max(1, len(frame_paths[::2] or frame_paths))
    return best, round(agg[best] / n, 3)


def write_xlsx(rows):
    from openpyxl import Workbook
    from openpyxl.worksheet.datavalidation import DataValidation
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.utils import get_column_letter
    wb = Workbook(); ws = wb.active; ws.title = "labels"
    headers = ["thumb", "clip_id", "duration_s", "fps", "n_frames",
               "my_label", "YOUR_label", "auto_conf", "notes"]
    ws.append(headers)
    ws.freeze_panes = "A2"
    dv = DataValidation(type="list", formula1='"' + ",".join(VOCAB) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 34
    for i, r in enumerate(rows, start=2):
        ws.cell(i, 2, r["clip_id"]); ws.cell(i, 3, r["dur"]); ws.cell(i, 4, r["fps"])
        ws.cell(i, 5, r["n_frames"]); ws.cell(i, 6, r["my_label"])
        ws.cell(i, 8, r["conf"]); ws.cell(i, 9, "")
        dv.add(ws.cell(i, 7))                          # YOUR_label dropdown
        ws.row_dimensions[i].height = 90
        if r.get("thumb") and os.path.exists(r["thumb"]):
            try:
                im = XLImage(r["thumb"]); im.width = 160; im.height = 120
                ws.add_image(im, f"A{i}")
            except Exception:
                pass
    wb.save(_XLSX)


def main():
    if not CLIP.available:
        print("CLIP unavailable."); return
    CLIP.warmup()
    clips = consolidate()
    print(f"consolidated {len(clips)} unique clips → {_CLIPS}")
    os.makedirs(_FRAMES, exist_ok=True)
    rows, fps_hist, total_frames = [], {}, 0
    for k, cp in enumerate(clips):
        cid = _clean_id(cp)
        fps, dur, n = probe(cp)
        fps_hist[fps] = fps_hist.get(fps, 0) + 1
        frames = extract_frames(cp, cid, fps, n)
        total_frames += len(frames)
        lbl, conf = auto_label(frames)
        thumb = frames[len(frames) // 2] if frames else None
        rows.append(dict(clip_id=cid, dur=dur, fps=fps, n_frames=len(frames),
                         my_label=lbl, conf=conf, thumb=thumb))
        if (k + 1) % 10 == 0 or k + 1 == len(clips):
            print(f"  [{k+1}/{len(clips)}] frames so far: {total_frames}")
    write_xlsx(rows)
    print(f"\nFPS distribution: " + ", ".join(f"{f}fps×{c}" for f, c in sorted(fps_hist.items())))
    print(f"TOTAL: {len(clips)} clips, {total_frames} frames → {_FRAMES}")
    print(f"REVIEW SHEET → {_XLSX}  (fix the YOUR_label column, then I train)")
    # auto-label distribution (how the guesses spread across moments)
    from collections import Counter
    print("auto-label spread: " + ", ".join(f"{m}:{c}" for m, c in
          Counter(r['my_label'] for r in rows).most_common()))


if __name__ == "__main__":
    main()
