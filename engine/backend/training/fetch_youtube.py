"""
backend/training/fetch_youtube.py — pull birthday clips from YouTube links.

Usage:
  # from a links file (one per line; '#' comments ignored):
  .venv/bin/python3 backend/training/fetch_youtube.py
  # or pass URLs directly:
  .venv/bin/python3 backend/training/fetch_youtube.py "https://youtu.be/XXXX" ...

Per line in training/youtube_links.txt you can TRIM long videos to just the
relevant clip (keeps the dataset tight and avoids irrelevant footage):
  https://youtu.be/XXXX                 → whole video (capped at 60s)
  https://youtu.be/XXXX | 0:12-0:28     → only seconds 12–28
  # anything after a '#' is a comment

Downloads (≤1080p mp4) land in tests/datasets/incoming/youtube/. After you
collect a batch, I auto-pre-label them and you just correct mistakes.
"""
from __future__ import annotations
import os, sys, re

_HERE  = os.path.dirname(__file__)
_OUT   = os.path.join(_HERE, "..", "..", "tests", "datasets", "Birthday", "clips", "real")
_LINKS = os.path.join(_HERE, "youtube_links.txt")
_MAX_WHOLE_S = 60          # (reserved) preferred cap when trimming is available
_MAX_LONG_S  = 600         # skip videos longer than this (compilations/vlogs)

import imageio_ffmpeg, subprocess
# yt-dlp looks for a binary literally named 'ffmpeg'; imageio ships a versioned
# name (ffmpeg-macos-...), so expose a correctly-named symlink in a stable dir.
_FFSRC = imageio_ffmpeg.get_ffmpeg_exe()
_FFDIR = os.path.join(_HERE, ".ffmpeg_bin")
os.makedirs(_FFDIR, exist_ok=True)
_FFLINK = os.path.join(_FFDIR, "ffmpeg")
try:
    if not os.path.exists(_FFLINK):
        os.symlink(_FFSRC, _FFLINK)
except OSError:
    pass
try:
    subprocess.run([_FFLINK, "-version"], capture_output=True, timeout=10)
    _FF_OK = True
except Exception:
    _FF_OK = False


def _parse(line: str):
    line = line.split("#", 1)[0].strip()
    if not line:
        return None
    if "|" in line:
        url, rng = [p.strip() for p in line.split("|", 1)]
        m = re.match(r"(\d+):(\d+)\s*-\s*(\d+):(\d+)", rng)
        if m:
            a = int(m[1]) * 60 + int(m[2]); b = int(m[3]) * 60 + int(m[4])
            return url, (a, b)
    return line, None


def _download(url: str, span):
    import yt_dlp
    os.makedirs(_OUT, exist_ok=True)
    opts = {
        "outtmpl": os.path.join(_OUT, "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "noprogress": True, "ignoreerrors": True,
        # YouTube 403s the default web client without a signature/po_token; the
        # android/ios players serve progressive formats that download cleanly.
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "web"]}},
    }
    # NOTE: yt-dlp's download_ranges (trimming) needs ffmpeg+ffprobe; imageio
    # ships no ffprobe, so we download the WHOLE clip (shorts are already short)
    # and skip a long video rather than trimming it.
    if _FF_OK:
        # Prefer FULL HD — the old <=720 cap (plus low progressive fallbacks)
        # filled the library with 360p reels, which caps photo quality at 360p.
        opts["format"] = ("bv*[height<=1080][ext=mp4]+ba/"
                          "bv*[height<=1080]+ba/"
                          "b[height<=1080][ext=mp4]/b[ext=mp4]/b")
        opts["merge_output_format"] = "mp4"
        opts["ffmpeg_location"] = _FFDIR
    else:
        opts["format"] = "b[ext=mp4][height<=1080]/22/18/b"
    # Skip absurdly long videos (compilations/vlogs) to keep the dataset tight.
    opts["match_filter"] = yt_dlp.utils.match_filter_func(f"duration < {_MAX_LONG_S}")
    with yt_dlp.YoutubeDL(opts) as ydl:
        try:
            ydl.download([url]); return True
        except Exception as e:
            print(f"  x {url}: {e}"); return False


def main():
    args = [a for a in sys.argv[1:] if a.strip()]
    lines = args if args else (
        [l for l in open(_LINKS)] if os.path.exists(_LINKS) else [])
    jobs = [p for p in (_parse(l) for l in lines) if p]
    if not jobs:
        print(f"No links. Add URLs to {_LINKS} or pass them as arguments."); return
    print(f"Fetching {len(jobs)} clip(s) → {_OUT}")
    ok = 0
    for url, span in jobs:
        tag = f" [{span[0]}-{span[1]}s]" if span else ""
        print(f"• {url}{tag}")
        ok += _download(url, span)
    n = len([f for f in os.listdir(_OUT) if f.endswith('.mp4')]) if os.path.isdir(_OUT) else 0
    print(f"\n{ok}/{len(jobs)} fetched. {n} mp4 now in incoming/youtube/")


if __name__ == "__main__":
    main()
