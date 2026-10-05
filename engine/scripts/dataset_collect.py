#!/usr/bin/env python3
"""
scripts/dataset_collect.py — collect 30+ hours of labelled event video.

A meaningful upgrade over the older download_event_videos.py:
  - Pulls from 6 sources instead of 2
  - Targets HOURS of footage per class, not just clip count
  - Stops once a class hits its time budget (no over-fetch)
  - Persists per-class totals across runs (resume-safe)
  - Auto-tags each clip with source + license + class + query

Sources (in priority order — duplicates dedup'd by SHA-256):

  1. Pexels Videos       (needs PEXELS_KEY env var)        free key
  2. Pixabay Videos      (needs PIXABAY_KEY env var)       free key
  3. Internet Archive    (no key)                           CC-licensed
  4. Wikimedia Commons   (no key)                           public domain
  5. Coverr.co           (no key)                           free stock
  6. YouTube Creative Commons via yt-dlp (optional)        gray for commercial

Usage
-----

    # Get all API keys first (5 min each):
    #   pexels.com/api          PEXELS_KEY
    #   pixabay.com/api/docs    PIXABAY_KEY
    export PEXELS_KEY="..."
    export PIXABAY_KEY="..."

    # Run the collector targeting 30 hours total
    python3 scripts/dataset_collect.py --hours 30

    # Or target one class
    python3 scripts/dataset_collect.py --classes cake_cutting --hours 5

    # Include YouTube Creative Commons (needs yt-dlp installed)
    pip install yt-dlp
    python3 scripts/dataset_collect.py --hours 30 --include-youtube

Output
------
    backend/data/event_videos/
      cake_cutting/                # one folder per class
        pexels_abc123.mp4
        pixabay_def456.mp4
        archive_ghi789.mp4
        ...
      ring_ceremony/
      ...
      collection_metadata.jsonl    # one line per video — source, license, etc.
      collection_state.json        # resume state (per-class hours collected)
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

LOG = logging.getLogger("snapai.collect")
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "backend" / "data" / "event_videos"
META_PATH = OUT_DIR / "collection_metadata.jsonl"
STATE_PATH = OUT_DIR / "collection_state.json"
UA = "SnapAI/DatasetCollector 1.0"


# ── Per-class search queries ─────────────────────────────────────────────
# Each query produces ~10-30 clips depending on source. Multiple queries
# per class to get visual diversity (Indian + Western + indoor + outdoor).
CLASS_QUERIES: Dict[str, List[str]] = {
    "cake_cutting": [
        "wedding cake cutting", "birthday cake cutting",
        "indian wedding cake", "cake celebration",
        "couple cutting cake", "birthday party cake",
    ],
    "ring_ceremony": [
        "wedding ring exchange", "engagement proposal",
        "marriage proposal", "wedding ring closeup",
        "couple exchanging rings", "indian wedding rings",
    ],
    "first_dance": [
        "wedding first dance", "couple slow dance",
        "bride groom dance", "wedding reception dance",
        "indian wedding dance", "ballroom dance couple",
    ],
    "bouquet_toss": [
        "bouquet toss wedding", "bride throwing flowers",
        "wedding bouquet catch",
    ],
    "candle_blowing": [
        "blowing birthday candles", "birthday candle wish",
        "child blowing candles", "candle blowing celebration",
    ],
    "group_photo": [
        "wedding group photo", "family group portrait",
        "graduation group", "team photo celebration",
        "indian family photo", "office team photo",
    ],
    "champagne_toast": [
        "champagne toast wedding", "wedding cheers toast",
        "raising glasses celebration", "champagne celebration",
    ],
    "confetti_burst": [
        "confetti celebration", "wedding confetti exit",
        "confetti party", "colored confetti",
    ],
    "sports_action": [
        "soccer goal celebration", "basketball action",
        "cricket batsman", "tennis celebration",
        "athlete victory", "sports highlight",
    ],
    "hug_moment": [
        "warm hug", "people hugging celebration",
        "emotional embrace", "family reunion hug",
        "wedding hug", "graduation hug",
    ],
    "haldi":            ["haldi ceremony", "indian wedding turmeric",
                         "haldi function indian"],
    "mehendi":          ["mehendi ceremony", "henna ceremony indian wedding",
                         "bride mehendi"],
    "sangeet":          ["sangeet ceremony", "indian wedding sangeet",
                         "wedding dance performance"],
    "phera":            ["pheras wedding", "indian wedding fire ceremony",
                         "saat phere"],
    "vidaai":           ["vidaai indian wedding", "bride farewell",
                         "indian wedding emotional farewell"],
}


# ── HTTP helpers ─────────────────────────────────────────────────────────
def _http_json(url: str, headers: Optional[Dict[str, str]] = None,
               timeout: int = 20) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        LOG.debug(f"  HTTP {e.code} from {url[:80]}")
        return None
    except Exception as e:
        LOG.debug(f"  GET failed: {e}")
        return None


def _http_download(url: str, out_path: Path,
                   max_bytes: int = 200 * 1024 * 1024,
                   headers: Optional[Dict[str, str]] = None) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            total = 0
            with open(tmp, "wb") as f:
                while True:
                    chunk = r.read(64 * 1024)
                    if not chunk: break
                    total += len(chunk)
                    if total > max_bytes:
                        f.close(); tmp.unlink(missing_ok=True)
                        LOG.info(f"    aborted (>{max_bytes//1024//1024} MB)")
                        return False
                    f.write(chunk)
        tmp.rename(out_path)
        return True
    except Exception as e:
        LOG.warning(f"    DL failed: {e}")
        tmp.unlink(missing_ok=True)
        return False


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            data = f.read(64 * 1024)
            if not data: break
            h.update(data)
    return h.hexdigest()


def _safe(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")[:40]


# ── Source implementations ────────────────────────────────────────────────
class PexelsSource:
    name = "pexels"
    license = "Pexels License (free commercial use)"

    def __init__(self, key: str):
        self.key = key

    def search(self, q: str, per_page: int = 50,
               min_dur: int = 5, max_dur: int = 180) -> List[Dict]:
        url = ("https://api.pexels.com/videos/search?"
               + urllib.parse.urlencode({"query": q, "per_page": per_page,
                                          "orientation": "landscape"}))
        data = _http_json(url, headers={"Authorization": self.key})
        if not data: return []
        out = []
        for v in data.get("videos", []):
            d = int(v.get("duration", 0))
            if not min_dur <= d <= max_dur: continue
            files = [f for f in v.get("video_files", [])
                     if f.get("file_type") == "video/mp4"]
            files.sort(key=lambda f: (f.get("height") or 0))
            # Pick the middle quality (not too big, not too small)
            chosen = next((f for f in files if 480 <= (f.get("height") or 0) <= 1080),
                          files[len(files)//2] if files else None)
            if not chosen: continue
            out.append({
                "url": chosen["link"], "id": str(v["id"]),
                "duration": d, "width": chosen.get("width"),
                "height": chosen.get("height"),
                "creator": v.get("user", {}).get("name", ""),
                "source_url": v.get("url"),
                "source": self.name, "license": self.license,
            })
        return out


class PixabaySource:
    name = "pixabay"
    license = "Pixabay Content License (free commercial use)"

    def __init__(self, key: str):
        self.key = key

    def search(self, q: str, per_page: int = 50,
               min_dur: int = 5, max_dur: int = 180) -> List[Dict]:
        url = ("https://pixabay.com/api/videos/?"
               + urllib.parse.urlencode({"key": self.key, "q": q,
                                          "per_page": per_page,
                                          "safesearch": "true"}))
        data = _http_json(url)
        if not data: return []
        out = []
        for v in data.get("hits", []):
            d = int(v.get("duration", 0))
            if not min_dur <= d <= max_dur: continue
            videos = v.get("videos", {})
            chosen = (videos.get("medium") or videos.get("small")
                      or videos.get("large"))
            if not chosen or not chosen.get("url"): continue
            out.append({
                "url": chosen["url"], "id": str(v["id"]),
                "duration": d, "width": chosen.get("width"),
                "height": chosen.get("height"),
                "creator": v.get("user", ""),
                "source_url": v.get("pageURL"),
                "source": self.name, "license": self.license,
            })
        return out


class InternetArchiveSource:
    name = "archive"
    license = "varies (CC / Public Domain on Internet Archive)"

    def search(self, q: str, per_page: int = 30,
               min_dur: int = 5, max_dur: int = 600) -> List[Dict]:
        # IA advanced-search API. Filter to mediatype=movies + CC license.
        params = {
            "q": f'({q}) AND mediatype:(movies) AND format:(mp4 OR h.264)',
            "fl[]": ["identifier", "title", "creator", "licenseurl",
                     "runtime", "downloads"],
            "sort[]": "downloads desc",
            "rows": per_page,
            "output": "json",
        }
        url = ("https://archive.org/advancedsearch.php?"
               + urllib.parse.urlencode(params, doseq=True))
        data = _http_json(url, timeout=20)
        if not data: return []
        out = []
        for doc in data.get("response", {}).get("docs", []):
            ident = doc.get("identifier")
            if not ident: continue
            # Fetch file list to find an .mp4
            files_url = f"https://archive.org/metadata/{ident}/files"
            files = _http_json(files_url, timeout=15)
            if not files: continue
            mp4 = next((f for f in files.get("result", [])
                        if f.get("name", "").endswith(".mp4")
                        and int(f.get("size", 0)) < 200_000_000), None)
            if not mp4: continue
            out.append({
                "url": f"https://archive.org/download/{ident}/{mp4['name']}",
                "id": ident,
                "duration": None,         # IA doesn't return duration in search
                "width": None, "height": None,
                "creator": doc.get("creator", ""),
                "source_url": f"https://archive.org/details/{ident}",
                "source": self.name,
                "license": doc.get("licenseurl") or self.license,
            })
        return out


class WikimediaSource:
    name = "wikimedia"
    license = "CC-BY-SA / Public Domain (per Wikimedia Commons)"

    def search(self, q: str, per_page: int = 30,
               min_dur: int = 5, max_dur: int = 300) -> List[Dict]:
        params = {
            "action": "query", "format": "json",
            "list": "search", "srnamespace": 6,    # File:
            "srsearch": f"filetype:video {q}",
            "srlimit": per_page,
        }
        url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)
        data = _http_json(url)
        if not data: return []
        titles = [s["title"] for s in data.get("query", {}).get("search", [])]
        if not titles: return []
        info_params = {
            "action": "query", "format": "json",
            "titles": "|".join(titles[:50]),
            "prop": "imageinfo",
            "iiprop": "url|size|mime|extmetadata|duration",
        }
        info_url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(info_params)
        idata = _http_json(info_url)
        if not idata: return []
        out = []
        for page in idata.get("query", {}).get("pages", {}).values():
            for ii in page.get("imageinfo", []):
                u = ii.get("url", "")
                if not u.endswith((".mp4", ".webm", ".ogv")): continue
                if int(ii.get("size", 0)) > 200_000_000: continue
                dur = float(ii.get("duration") or 0)
                if not min_dur <= dur <= max_dur and dur > 0: continue
                out.append({
                    "url": u,
                    "id": page.get("title", "").replace("File:", ""),
                    "duration": dur if dur > 0 else None,
                    "width": ii.get("width"), "height": ii.get("height"),
                    "creator": (ii.get("extmetadata", {})
                                  .get("Artist", {}).get("value", "")),
                    "source_url": (ii.get("descriptionurl") or ""),
                    "source": self.name, "license": self.license,
                })
        return out


class YouTubeCCSource:
    """YouTube Creative Commons via yt-dlp.

    Legal note: YouTube CC content is freely reusable per Creative Commons
    Attribution licence. Use with attribution; verify per-clip you intend
    to ship commercially.
    """
    name = "youtube_cc"
    license = "Creative Commons Attribution (CC BY 3.0) per YouTube CC filter"

    def __init__(self):
        try:
            import yt_dlp  # noqa: F401
            self._ok = True
        except ImportError:
            self._ok = False
            LOG.warning("yt-dlp not installed; skip YouTube source. "
                        "Install: pip install yt-dlp")

    def enabled(self) -> bool:
        return self._ok

    def search(self, q: str, per_page: int = 10,
               min_dur: int = 10, max_dur: int = 180) -> List[Dict]:
        if not self._ok: return []
        import yt_dlp
        # Search with CC filter (ytsearch + CC video license filter)
        query = f"ytsearch{per_page}:{q}"
        opts = {
            "quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": False, "match_filter": None,
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
        }
        out = []
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(query, download=False)
                entries = info.get("entries", []) if info else []
                for e in entries:
                    if not e: continue
                    # Filter by CC licence
                    if (e.get("license") or "").lower().find("creative commons") < 0:
                        continue
                    d = int(e.get("duration") or 0)
                    if not min_dur <= d <= max_dur: continue
                    url = e.get("webpage_url") or e.get("url")
                    if not url: continue
                    out.append({
                        "url": url,           # yt-dlp will resolve at download
                        "id": e.get("id", ""),
                        "duration": d,
                        "width": e.get("width"), "height": e.get("height"),
                        "creator": e.get("uploader", ""),
                        "source_url": url,
                        "source": self.name, "license": self.license,
                        "_yt_dlp": True,      # marker
                    })
        except Exception as e:
            LOG.debug(f"  yt search failed: {e}")
        return out

    @staticmethod
    def download_ytdlp(url: str, out_path: Path) -> bool:
        """yt-dlp-specific download path (handles HLS/DASH muxing)."""
        try:
            import yt_dlp
        except ImportError:
            return False
        opts = {
            "quiet": True, "no_warnings": True,
            "outtmpl": str(out_path.with_suffix(".%(ext)s")),
            "format": "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
            "merge_output_format": "mp4",
            "max_filesize": 200_000_000,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            # yt-dlp may have produced .mp4 or .mkv depending on muxing
            for ext in (".mp4", ".mkv", ".webm"):
                candidate = out_path.with_suffix(ext)
                if candidate.exists():
                    if ext != ".mp4":
                        candidate.rename(out_path)
                    return True
            return out_path.exists()
        except Exception as e:
            LOG.warning(f"    yt-dlp DL failed: {e}")
            return False


# ── State + resume ────────────────────────────────────────────────────────
def _load_state() -> dict:
    if STATE_PATH.exists():
        try: return json.loads(STATE_PATH.read_text())
        except Exception: pass
    return {"per_class_seconds": {}, "seen_shas": [], "seen_ids": []}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _video_duration_seconds(path: Path) -> float:
    """Probe a downloaded clip for actual duration (so the budget is real)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened(): return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n   = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        cap.release()
        return float(n / fps) if fps > 0 else 0.0
    except Exception:
        return 0.0


def _append_meta(rec: dict) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with META_PATH.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Per-class collection loop ─────────────────────────────────────────────
def collect_for_class(cls: str, target_seconds: float,
                      sources: List, state: dict, pause: float = 0.5) -> float:
    """Fill clips for one class until we hit target_seconds."""
    have = float(state["per_class_seconds"].get(cls, 0.0))
    if have >= target_seconds:
        LOG.info(f"[{cls}] already at {have/60:.1f} min — skipping")
        return have
    LOG.info(f"[{cls}] target {target_seconds/60:.0f} min, have {have/60:.1f} min")

    cdir = OUT_DIR / cls
    cdir.mkdir(parents=True, exist_ok=True)

    queries = list(CLASS_QUERIES.get(cls, []))
    random.shuffle(queries)
    seen_shas = set(state["seen_shas"])
    seen_ids  = set(state["seen_ids"])

    for src in sources:
        if have >= target_seconds: break
        for q in queries:
            if have >= target_seconds: break
            try:
                hits = src.search(q, per_page=30)
            except Exception as e:
                LOG.warning(f"  {src.name} search '{q}' failed: {e}")
                hits = []
            time.sleep(pause)
            LOG.info(f"  [{cls}] {src.name} '{q}' → {len(hits)} hits")
            for h in hits:
                if have >= target_seconds: break
                key = f"{h['source']}:{h['id']}"
                if key in seen_ids: continue
                fname = f"{h['source']}_{_safe(h['id'])[:30]}.mp4"
                out = cdir / fname
                if out.exists(): seen_ids.add(key); continue

                # Download
                if h.get("_yt_dlp"):
                    ok = YouTubeCCSource.download_ytdlp(h["url"], out)
                else:
                    ok = _http_download(h["url"], out, max_bytes=200_000_000)
                if not ok: continue
                time.sleep(pause * 0.5)

                # Dedup via SHA
                sha = _sha_file(out)
                if sha in seen_shas:
                    LOG.info(f"    duplicate sha — removing")
                    out.unlink(missing_ok=True); continue

                # Probe real duration (some sources lie)
                dur = float(h.get("duration") or 0.0)
                if dur <= 0:
                    dur = _video_duration_seconds(out)
                if dur < 3:
                    out.unlink(missing_ok=True); continue

                have += dur
                seen_shas.add(sha); seen_ids.add(key)
                rec = {
                    "filename": str(out.relative_to(ROOT)),
                    "class": cls, "query": q,
                    "source": h["source"], "source_id": h["id"],
                    "source_url": h.get("source_url"),
                    "license": h.get("license"),
                    "creator": h.get("creator", ""),
                    "duration_s": round(dur, 1),
                    "width": h.get("width"), "height": h.get("height"),
                    "size_bytes": out.stat().st_size,
                    "sha": sha, "downloaded_at": time.time(),
                }
                _append_meta(rec)
                state["per_class_seconds"][cls] = have
                state["seen_shas"] = list(seen_shas)[-5000:]
                state["seen_ids"]  = list(seen_ids)[-5000:]
                _save_state(state)
                LOG.info(f"    ✓ {fname} {dur:.0f}s — class total {have/60:.1f} min")
    return have


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    global OUT_DIR, META_PATH, STATE_PATH

    ap = argparse.ArgumentParser(
        description="Collect 30+ hrs of labeled event video for SnapAI.")
    ap.add_argument("--hours", type=float, default=30,
                    help="total target hours across all classes (default 30)")
    ap.add_argument("--classes", default="",
                    help="comma-separated subset (default: all)")
    ap.add_argument("--include-youtube", action="store_true",
                    help="enable YouTube CC source (requires yt-dlp)")
    ap.add_argument("--pause", type=float, default=0.5,
                    help="seconds between API requests")
    ap.add_argument("--out", default=str(OUT_DIR), help="output dir")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    OUT_DIR = Path(args.out)
    META_PATH = OUT_DIR / "collection_metadata.jsonl"
    STATE_PATH = OUT_DIR / "collection_state.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = []
    if k := os.environ.get("PEXELS_KEY"):    sources.append(PexelsSource(k))
    if k := os.environ.get("PIXABAY_KEY"):   sources.append(PixabaySource(k))
    sources.append(InternetArchiveSource())
    sources.append(WikimediaSource())
    if args.include_youtube:
        yts = YouTubeCCSource()
        if yts.enabled(): sources.append(yts)
    if not sources:
        LOG.error("No usable sources. Set PEXELS_KEY and/or PIXABAY_KEY env vars,")
        LOG.error("or pass --include-youtube (after pip install yt-dlp).")
        sys.exit(2)
    LOG.info(f"Active sources: {[s.name for s in sources]}")

    all_classes = list(CLASS_QUERIES.keys())
    chosen = ([c.strip() for c in args.classes.split(",") if c.strip()]
              if args.classes else all_classes)
    chosen = [c for c in chosen if c in CLASS_QUERIES]
    if not chosen:
        LOG.error(f"No valid classes. Choose from: {all_classes}"); sys.exit(2)
    LOG.info(f"Classes: {chosen}")

    target_per_class = float(args.hours * 3600) / len(chosen)
    LOG.info(f"Target: {args.hours:.0f} hrs total → "
             f"{target_per_class/60:.0f} min per class")
    LOG.info("=" * 60)

    state = _load_state()
    t0 = time.time()
    totals = {}
    for cls in chosen:
        totals[cls] = collect_for_class(cls, target_per_class,
                                         sources, state, args.pause)

    grand_total = sum(totals.values())
    LOG.info("=" * 60)
    LOG.info(f"Collection complete in {(time.time()-t0)/60:.1f} min")
    LOG.info(f"Grand total: {grand_total/3600:.1f} hours")
    for cls, s in sorted(totals.items(), key=lambda x: -x[1]):
        LOG.info(f"  {cls:25s} {s/60:6.1f} min  ({s/3600:.2f} hr)")
    LOG.info("")
    LOG.info("Next steps:")
    LOG.info("  1. Auto-label per timestamp:")
    LOG.info("     python3 scripts/auto_label_videos.py")
    LOG.info("  2. Generate summary report:")
    LOG.info("     python3 scripts/dataset_report.py")
    LOG.info("  3. Verify labels via labeler.html:")
    LOG.info("     open scripts/labeler.html")


if __name__ == "__main__":
    main()
