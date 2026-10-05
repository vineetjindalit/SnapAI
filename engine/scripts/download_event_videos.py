#!/usr/bin/env python3
"""
scripts/download_event_videos.py — auto-download free event videos for training.

Pulls licensed-free event footage from:
  1. Pexels (needs PEXELS_KEY — free at pexels.com/api, takes 2 minutes)
  2. Pixabay (needs PIXABAY_KEY — free at pixabay.com/api/docs)

Both sources allow free commercial use including ML training. The script
records every download's source, license, photographer, and original URL
in `backend/data/eval_videos/videos_metadata.jsonl` so you have full
attribution if you ever ship the trained model commercially.

Resumable. Deduplicated by SHA-256. Skips clips longer than --max-duration
or larger than --max-mb so you don't blow your disk on a single video.

Usage
-----

    # Sign up at pexels.com/api to get a key (instant, no card required)
    export PEXELS_KEY="your_pexels_key"

    # Download ~30 clips across all classes (~1-2 GB total)
    python3 scripts/download_event_videos.py --per-class 5

    # Or focus on one class
    python3 scripts/download_event_videos.py --classes cake_cutting --per-class 15

    # Tighter limits (smaller dataset to start)
    python3 scripts/download_event_videos.py --per-class 3 --max-mb 30 --max-duration 45

What you get
------------
    backend/data/eval_videos/
      cake_cutting_pexels_<id>_<sha8>.mp4
      ring_ceremony_pexels_<id>_<sha8>.mp4
      ...
      videos_metadata.jsonl    # one line per video, full attribution
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger("snappy.video_dl")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "backend" / "data" / "eval_videos"
META_PATH = OUT_DIR / "videos_metadata.jsonl"
USER_AGENT = "SnappyTrainingDataDownloader/1.0"


# ── Class → search-query map ──────────────────────────────────────────────
# Multiple queries per class so we don't get visually-monotonous clips.
# Tuned for what Pexels/Pixabay actually have available.
CLASS_QUERIES: Dict[str, List[str]] = {
    "cake_cutting":    ["wedding cake cutting", "birthday cake cutting",
                        "cake celebration", "cutting cake"],
    "ring_ceremony":   ["wedding rings exchange", "engagement proposal",
                        "ring ceremony"],
    "first_dance":     ["wedding first dance", "couple dancing wedding",
                        "bride groom dance"],
    "bouquet_toss":    ["bouquet toss wedding", "bride throwing flowers"],
    "candle_blowing":  ["birthday candles blow", "blowing out candles",
                        "birthday wish"],
    "group_photo":     ["wedding group photo", "family photo gathering",
                        "graduation group"],
    "champagne_toast": ["champagne toast celebration", "wedding cheers",
                        "raising glasses"],
    "confetti_burst":  ["confetti celebration", "wedding confetti",
                        "party confetti"],
    "sports_action":   ["soccer goal celebration", "basketball celebration",
                        "athlete victory"],
    "hug_moment":      ["wedding hug", "people hugging celebration",
                        "emotional embrace"],
    "general_peak":    ["wedding ceremony moment", "birthday party people",
                        "celebration emotional"],
}


# ── HTTP helpers ──────────────────────────────────────────────────────────
def _http_json(url: str, headers: Optional[Dict[str, str]] = None,
               timeout: int = 20) -> Optional[dict]:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, **(headers or {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        if e.code == 401: LOG.error(f"  401 Unauthorized — check your API key")
        elif e.code == 429: LOG.warning(f"  429 rate-limited — backing off")
        else: LOG.warning(f"  HTTP {e.code} from {url[:80]}")
        return None
    except Exception as e:
        LOG.debug(f"  GET failed: {e}")
        return None


def _http_download(url: str, out_path: Path,
                   max_bytes: int = 100 * 1024 * 1024) -> bool:
    """Stream-download with a size cap. Aborts if file exceeds max_bytes."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            total = 0
            tmp = out_path.with_suffix(".part")
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        LOG.info(f"    aborted (>{max_bytes//1024//1024} MB)")
                        f.close(); tmp.unlink(missing_ok=True)
                        return False
                    f.write(chunk)
            tmp.rename(out_path)
            return True
    except Exception as e:
        LOG.warning(f"  download failed: {e}")
        try: out_path.with_suffix(".part").unlink(missing_ok=True)
        except Exception: pass
        return False


def _sha256_file(path: Path, chunk: int = 64 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data: break
            h.update(data)
    return h.hexdigest()


def _safe(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:40]


# ── Source: Pexels ────────────────────────────────────────────────────────
class PexelsSource:
    name = "pexels"
    license = "Pexels License (free commercial use, attribution appreciated)"

    def __init__(self, key: str):
        self.key = key

    def search(self, query: str, per_page: int = 30,
               min_duration: int = 5, max_duration: int = 120
               ) -> List[Dict]:
        url = ("https://api.pexels.com/videos/search?"
               + urllib.parse.urlencode({
                   "query": query, "per_page": per_page,
                   "orientation": "landscape", "size": "medium",
               }))
        data = _http_json(url, headers={"Authorization": self.key})
        if not data: return []
        out = []
        for v in data.get("videos", []):
            dur = int(v.get("duration", 0))
            if dur < min_duration or dur > max_duration:
                continue
            # Pick the highest-quality MP4 under our size budget — typically "hd" 720p
            files = sorted(v.get("video_files", []),
                           key=lambda f: (f.get("height", 0), f.get("width", 0)))
            chosen = None
            for f in files:
                if f.get("file_type") == "video/mp4" and 480 <= f.get("height", 0) <= 1080:
                    chosen = f
            if chosen is None and files:
                chosen = files[len(files) // 2]   # middle quality fallback
            if not chosen: continue
            out.append({
                "url":          chosen["link"],
                "id":           str(v["id"]),
                "duration":     dur,
                "width":        chosen.get("width"),
                "height":       chosen.get("height"),
                "creator":      v.get("user", {}).get("name", ""),
                "source_url":   v.get("url"),
                "source":       self.name,
                "license":      self.license,
            })
        return out


# ── Source: Pixabay (fallback when Pexels key not available) ─────────────
class PixabaySource:
    name = "pixabay"
    license = "Pixabay Content License (free commercial use)"

    def __init__(self, key: str):
        self.key = key

    def search(self, query: str, per_page: int = 30,
               min_duration: int = 5, max_duration: int = 120
               ) -> List[Dict]:
        url = ("https://pixabay.com/api/videos/?"
               + urllib.parse.urlencode({
                   "key": self.key, "q": query, "per_page": per_page,
                   "video_type": "all", "safesearch": "true",
               }))
        data = _http_json(url)
        if not data: return []
        out = []
        for v in data.get("hits", []):
            dur = int(v.get("duration", 0))
            if dur < min_duration or dur > max_duration:
                continue
            videos = v.get("videos", {})
            chosen = (videos.get("medium") or videos.get("small")
                      or videos.get("large") or videos.get("tiny"))
            if not chosen or not chosen.get("url"):
                continue
            out.append({
                "url":          chosen["url"],
                "id":           str(v["id"]),
                "duration":     dur,
                "width":        chosen.get("width"),
                "height":       chosen.get("height"),
                "creator":      v.get("user", ""),
                "source_url":   v.get("pageURL"),
                "source":       self.name,
                "license":      self.license,
            })
        return out


# ── Resumability ──────────────────────────────────────────────────────────
def _load_seen() -> set:
    """Read existing metadata.jsonl to know which videos we already have."""
    if not META_PATH.exists():
        return set()
    seen = set()
    with META_PATH.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
                seen.add(f"{rec['source']}:{rec['source_id']}")
                seen.add(f"sha:{rec.get('sha', '')}")
            except Exception:
                continue
    return seen


def _append_meta(rec: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Pipeline ──────────────────────────────────────────────────────────────
def download_for_class(cls: str, sources: List, target: int,
                       max_mb: int, max_duration: int,
                       seen: set, pause: float) -> int:
    queries = list(CLASS_QUERIES.get(cls, []))
    random.shuffle(queries)
    got = 0
    for src in sources:
        for q in queries:
            if got >= target: break
            LOG.info(f"  [{cls}] querying {src.name}: {q!r}")
            try:
                hits = src.search(q, per_page=30,
                                  max_duration=max_duration)
            except Exception as e:
                LOG.warning(f"  search failed: {e}"); continue
            time.sleep(pause)
            for h in hits:
                if got >= target: break
                key = f"{h['source']}:{h['id']}"
                if key in seen:
                    continue
                fname = (f"{cls}_{h['source']}_{h['id']}_"
                         f"{_safe(h['creator'])[:12]}.mp4")
                out = OUT_DIR / fname
                if out.exists():
                    seen.add(key); continue

                LOG.info(f"    ↓ {h['source']}#{h['id']} "
                         f"{h.get('width')}×{h.get('height')} "
                         f"{h.get('duration')}s by {h['creator']}")
                ok = _http_download(h["url"], out, max_bytes=max_mb * 1024 * 1024)
                time.sleep(pause * 0.5)
                if not ok:
                    continue
                # Dedup by SHA in case a video was re-uploaded across sources
                sha = _sha256_file(out)
                if f"sha:{sha}" in seen:
                    LOG.info("    duplicate (sha match) — removing")
                    out.unlink(missing_ok=True); continue

                meta = {
                    "filename":    fname,
                    "path":        str(out.relative_to(ROOT)),
                    "class":       cls,
                    "query":       q,
                    "source":      h["source"],
                    "source_id":   h["id"],
                    "source_url":  h.get("source_url"),
                    "license":     h.get("license"),
                    "creator":     h.get("creator"),
                    "duration_s":  h.get("duration"),
                    "width":       h.get("width"),
                    "height":      h.get("height"),
                    "size_bytes":  out.stat().st_size,
                    "sha":         sha,
                    "downloaded":  time.time(),
                }
                _append_meta(meta)
                seen.add(key); seen.add(f"sha:{sha}")
                got += 1
                if got % 5 == 0:
                    LOG.info(f"    [{cls}] {got}/{target}")
    return got


def main():
    ap = argparse.ArgumentParser(
        description="Download free event videos for Snappy training.")
    ap.add_argument("--per-class",    type=int, default=5,
                    help="videos per class (default 5 → ~50 total)")
    ap.add_argument("--classes",      default="",
                    help="comma-separated class subset (default: all)")
    ap.add_argument("--max-mb",       type=int, default=80,
                    help="max size per video (MB, default 80)")
    ap.add_argument("--max-duration", type=int, default=120,
                    help="max video duration (seconds, default 120)")
    ap.add_argument("--out",          default=str(OUT_DIR), help="output dir")
    ap.add_argument("--pause",        type=float, default=0.6,
                    help="seconds between API calls (default 0.6)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    global OUT_DIR, META_PATH
    OUT_DIR = Path(args.out)
    META_PATH = OUT_DIR / "videos_metadata.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wire up sources from env
    sources = []
    if (k := os.environ.get("PEXELS_KEY", "").strip()):
        sources.append(PexelsSource(k))
        LOG.info("Pexels enabled (PEXELS_KEY found)")
    if (k := os.environ.get("PIXABAY_KEY", "").strip()):
        sources.append(PixabaySource(k))
        LOG.info("Pixabay enabled (PIXABAY_KEY found)")
    if not sources:
        LOG.error("No API keys found. Set at least one:")
        LOG.error("  export PEXELS_KEY=...    # free at pexels.com/api")
        LOG.error("  export PIXABAY_KEY=...   # free at pixabay.com/api/docs")
        sys.exit(2)

    classes = [c.strip() for c in args.classes.split(",") if c.strip()] \
              or list(CLASS_QUERIES)
    classes = [c for c in classes if c in CLASS_QUERIES]
    if not classes:
        LOG.error(f"No valid classes. Choose from: {list(CLASS_QUERIES)}")
        sys.exit(2)

    seen = _load_seen()
    LOG.info(f"Resuming with {len(seen)//2} videos already in metadata")
    LOG.info(f"Downloading {args.per_class} clips per class for: {classes}")
    LOG.info(f"Limits: ≤{args.max_mb} MB, ≤{args.max_duration}s per clip")
    LOG.info("=" * 62)

    counts: Dict[str, int] = {}
    t0 = time.time()
    for cls in classes:
        existing = sum(1 for p in OUT_DIR.glob(f"{cls}_*.mp4"))
        need = max(0, args.per_class - existing)
        if need == 0:
            LOG.info(f"[{cls}] already at {existing} — skipping")
            counts[cls] = existing; continue
        LOG.info(f"[{cls}] have {existing}, need {need} more")
        got = download_for_class(cls, sources, need,
                                 args.max_mb, args.max_duration,
                                 seen, args.pause)
        counts[cls] = existing + got
        LOG.info(f"[{cls}] FINAL {counts[cls]}/{args.per_class}")

    LOG.info("=" * 62)
    LOG.info(f"DONE in {(time.time()-t0)/60:.1f} min")
    total_files = sum(1 for p in OUT_DIR.glob("*.mp4"))
    total_bytes = sum(p.stat().st_size for p in OUT_DIR.glob("*.mp4"))
    LOG.info(f"Total: {total_files} videos, "
             f"{total_bytes/1024/1024:.0f} MB on disk")
    LOG.info(f"Output: {OUT_DIR}")
    LOG.info(f"Metadata: {META_PATH}")
    LOG.info("Next: open scripts/labeler.html, drop a video, mark moments.")


if __name__ == "__main__":
    main()
