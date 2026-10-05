"""
ingest_labeled.py — download the user's PRE-SORTED clips (new_clips_labeled.csv:
label,url), extract frames, and build labels_master.csv = the 145 corrected
labels (from birthday_labels.numbers) + these new pre-labelled clips.

Then train_fused.py reads labels_master.csv and retrains on the bigger set.
"""
from __future__ import annotations
import os, sys, csv, re, glob, subprocess
import numpy as np
import cv2

_HERE   = os.path.dirname(__file__)
_BDAY   = os.path.join(_HERE, "..", "..", "tests", "datasets", "Birthday")
_CLIPS  = os.path.join(_BDAY, "clips", "real")
_FRAMES = os.path.join(_BDAY, "Frames")
_CSV    = os.path.join(_HERE, "new_clips_labeled.csv")
_NUMBERS= os.path.join(_HERE, "..", "..", "tests", "datasets", "Frames", "Birthday_frames", "birthday_labels.numbers")
_MASTER = os.path.join(_HERE, "labels_master.csv")

import imageio_ffmpeg
_FFDIR = os.path.join(_HERE, ".ffmpeg_bin")
os.makedirs(_FFDIR, exist_ok=True)
_FFLINK = os.path.join(_FFDIR, "ffmpeg")
if not os.path.exists(_FFLINK):
    try: os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), _FFLINK)
    except OSError: pass


def yid(url):
    m = re.search(r"(?:youtu\.be/|shorts/|v=)([\w\-]{6,})", url)
    return m.group(1) if m else re.sub(r"\W+", "_", url)[-12:]


def download(url, cid):
    import yt_dlp
    dest = os.path.join(_CLIPS, cid + ".mp4")
    if os.path.exists(dest):
        return dest
    opts = {"outtmpl": os.path.join(_CLIPS, cid + ".%(ext)s"),
            "quiet": True, "no_warnings": True, "noprogress": True, "ignoreerrors": True,
            "merge_output_format": "mp4", "ffmpeg_location": _FFDIR,
            "format": "bv*[height<=720][ext=mp4]+ba/b[height<=720][ext=mp4]/b[ext=mp4]/b",
            "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
            "match_filter": yt_dlp.utils.match_filter_func("duration < 600")}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except Exception as e:
        print(f"  x {url}: {e}")
    return dest if os.path.exists(dest) else None


def extract_frames(path, cid):
    out = os.path.join(_FRAMES, cid)
    if os.path.isdir(out) and glob.glob(os.path.join(out, "*.jpg")):
        return                                  # already extracted
    os.makedirs(out, exist_ok=True)
    c = cv2.VideoCapture(path); fps = c.get(5) or 25.0; n = int(c.get(7)) or 0
    dur = n / fps if fps > 0 else 0
    n_want = int(np.clip(round(dur * 2.0), 4, 24)) if dur > 0 else 4
    for j, fi in enumerate(np.linspace(0, max(0, n - 1), n_want).astype(int)):
        c.set(1, int(fi)); ok, fr = c.read()
        if not ok: continue
        h, w = fr.shape[:2]
        if w > 1280: fr = cv2.resize(fr, (1280, int(h * 1280 / w)))
        cv2.imwrite(os.path.join(out, f"f{j:02d}.jpg"), fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    c.release()


def existing_labels():
    """145 corrected labels from the .numbers sheet → {clip_id: final_label}."""
    from numbers_parser import Document
    t = Document(_NUMBERS).sheets[0].tables[0]
    out = {}
    for r in list(t.rows(values_only=True))[1:]:
        cid, my, your = r[1], r[5], r[6]
        if cid:
            out[str(cid)] = (str(your).strip() if (your and str(your).strip()) else str(my))
    return out


def main():
    os.makedirs(_CLIPS, exist_ok=True); os.makedirs(_FRAMES, exist_ok=True)
    master = existing_labels()
    print(f"seeded {len(master)} existing corrected labels")
    rows = list(csv.DictReader(open(_CSV)))
    added = 0
    for i, row in enumerate(rows):
        label, url = row["label"].strip(), row["url"].strip()
        cid = yid(url)
        path = download(url, cid)
        if not path:
            continue
        extract_frames(path, cid)
        master[cid] = label
        added += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] last: {cid} → {label}")
    with open(_MASTER, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["clip_id", "label"])
        for cid, lab in master.items():
            w.writerow([cid, lab])
    import collections
    dist = collections.Counter(master.values())
    print(f"\ningested {added} new pre-labelled clips. master labels = {len(master)} clips")
    print("label distribution:", dict(sorted(dist.items(), key=lambda x: -x[1])))
    print(f"→ {_MASTER}")


if __name__ == "__main__":
    main()
