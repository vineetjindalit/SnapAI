"""
backend/training/dataset_bootstrap.py — download a labelled event-photo dataset.

Roadmap goal: "Collect training dataset: 10,000+ labeled event photos
(YouTube scraping + Getty API)". This script seeds 1–2 K labelled photos
from open APIs (Openverse + Wikimedia Commons by default — no key needed)
so you can fine-tune CLIP without any manual scraping.

Usage:
    python -m backend.training.dataset_bootstrap                   # 1500 imgs
    python -m backend.training.dataset_bootstrap --target 3000
    python -m backend.training.dataset_bootstrap --classes cake_cutting,group_photo
    UNSPLASH_KEY=xxx PEXELS_KEY=yyy python -m backend.training.dataset_bootstrap

Output layout:
    backend/data/training/
      cake_cutting/
        img_0001.jpg
        img_0002.jpg
        ...
      ring_ceremony/
      ...
      metadata.jsonl    # one line per image: {path, class, source, license, query, sha}

The script is RESUMABLE — re-run any time. Already-downloaded files (tracked
by SHA-256 in metadata.jsonl) are skipped, so you can incrementally grow.

Rate-limited and respectful: 0.6 s/request to each API by default.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ── Repo root / out dir ───────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "backend" / "data" / "training"
LOG = logging.getLogger("snappy.bootstrap")


# ── Class → search-query map ──────────────────────────────────────────────
# 5+ queries per class so we don't get a monoculture of stock photos.
CLASS_QUERIES: Dict[str, List[str]] = {
    "cake_cutting": [
        "wedding cake cutting", "birthday cake cutting",
        "couple cutting wedding cake", "cake cutting ceremony",
        "cutting birthday cake",
    ],
    "ring_ceremony": [
        "wedding ring exchange", "engagement ring proposal",
        "ring ceremony wedding", "couple exchanging rings",
        "marriage proposal ring",
    ],
    "first_dance": [
        "first dance wedding", "couple wedding dance",
        "bride groom first dance", "wedding dance floor",
    ],
    "bouquet_toss": [
        "bouquet toss wedding", "bride throwing bouquet",
        "wedding bouquet toss", "bouquet catch wedding",
    ],
    # The ACTION of blowing — pursed lips, leaning in, candles being extinguished.
    "candle_blowing": [
        "blowing out birthday candles", "child blowing birthday candle",
        "person blowing candles cake", "kid blowing out cake candles",
        "blowing candles close up", "woman blowing birthday candle",
        "boy blowing out candles cake",
    ],
    # The GATHERING phase — a lit birthday cake on the table / people around it,
    # nobody actively blowing. This is the class that was MISSING, so CLIP had no
    # centroid to tell "standing at the cake" apart from "the blow".
    "cake_with_candles": [
        "birthday cake with lit candles", "birthday cake candles on table",
        "family around birthday cake candles", "singing happy birthday cake",
        "birthday cake glowing candles", "lit candles birthday cake closeup",
        "children around birthday cake candles",
    ],
    "group_photo": [
        "wedding group photo", "family group portrait",
        "wedding party group photo", "graduation group photo",
        "team group photograph",
    ],
    "champagne_toast": [
        "champagne toast wedding", "wedding cheers toast",
        "people raising glasses celebration", "champagne glasses toast",
    ],
    "confetti_burst": [
        "confetti celebration", "wedding confetti exit",
        "confetti party", "colored confetti throw",
    ],
    "sports_action": [
        "soccer goal celebration", "basketball action shot",
        "athlete jumping action", "cricket batsman action",
        "football celebration",
    ],
    "hug_moment": [
        "people hugging celebration", "warm embrace hug",
        "wedding emotional hug", "family reunion hug",
    ],
    # Negative class: random "non-event" photos to teach the model what to skip
    "negative": [
        "empty banquet hall", "wedding venue setup empty",
        "blurred candid people", "crowd back view",
    ],
}

# ── HTTP helpers ──────────────────────────────────────────────────────────
_USER_AGENT = "SnappyDatasetBootstrapper/1.0 (https://snappy.local; contact admin)"


def _http_json(url: str, timeout: int = 15) -> Optional[dict]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception as e:
        LOG.debug(f"GET {url[:80]}… failed: {e}")
        return None


def _http_bytes(url: str, timeout: int = 30) -> Optional[bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception as e:
        LOG.debug(f"DL {url[:80]}… failed: {e}")
        return None


def _is_decodable_image(data: bytes, min_side: int = 200) -> bool:
    """Verify the bytes are a real, big-enough image. Uses Pillow (already a dep)."""
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        img.verify()
        # Re-open because verify() consumes
        img = Image.open(io.BytesIO(data))
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        w, h = img.size
        return w >= min_side and h >= min_side
    except Exception:
        return False


# ── Sources ───────────────────────────────────────────────────────────────
class Source:
    """Abstract base for an image search backend."""
    name = "abstract"
    rate_limit_s = 0.6

    def enabled(self) -> bool:
        return True

    def search(self, query: str, page_size: int = 25) -> List[Tuple[str, dict]]:
        """Return list of (image_url, metadata_dict)."""
        raise NotImplementedError


class OpenverseSource(Source):
    """Openverse — Creative Commons aggregator. No API key required."""
    name = "openverse"

    def search(self, query: str, page_size: int = 25):
        url = (
            "https://api.openverse.engineering/v1/images/?"
            + urllib.parse.urlencode({
                "q": query, "page_size": page_size,
                "license_type": "all-cc",
                "mature": "false",
            })
        )
        data = _http_json(url) or {}
        out: List[Tuple[str, dict]] = []
        for it in data.get("results", []):
            iurl = it.get("url") or it.get("thumbnail")
            if not iurl: continue
            out.append((iurl, {
                "source": "openverse",
                "license": it.get("license", "cc"),
                "license_url": it.get("license_url"),
                "creator": it.get("creator"),
                "id": it.get("id"),
            }))
        return out


class WikimediaSource(Source):
    """Wikimedia Commons — public-domain / CC images. No API key required."""
    name = "wikimedia"

    def search(self, query: str, page_size: int = 25):
        # 1) text search → file titles
        s_url = (
            "https://commons.wikimedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query", "list": "search",
                "srsearch": f"filetype:bitmap {query}",
                "srnamespace": 6, "srlimit": page_size,
                "format": "json",
            })
        )
        sdata = _http_json(s_url) or {}
        titles = [hit["title"] for hit in sdata.get("query", {}).get("search", [])]
        if not titles:
            return []
        # 2) batch image-info lookup → URLs
        i_url = (
            "https://commons.wikimedia.org/w/api.php?"
            + urllib.parse.urlencode({
                "action": "query",
                "titles": "|".join(titles[:50]),
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": 800,
                "format": "json",
            })
        )
        idata = _http_json(i_url) or {}
        out = []
        pages = idata.get("query", {}).get("pages", {}) or {}
        for _pid, page in pages.items():
            ii = (page.get("imageinfo") or [None])[0]
            if not ii: continue
            iurl = ii.get("thumburl") or ii.get("url")
            if not iurl: continue
            meta = ii.get("extmetadata") or {}
            out.append((iurl, {
                "source": "wikimedia",
                "license": (meta.get("LicenseShortName") or {}).get("value", "unknown"),
                "creator": (meta.get("Artist") or {}).get("value", ""),
                "title": page.get("title"),
            }))
        return out


class UnsplashSource(Source):
    """Unsplash — needs UNSPLASH_KEY env var. 5000 req/hr."""
    name = "unsplash"

    def __init__(self):
        self.key = os.environ.get("UNSPLASH_KEY", "")

    def enabled(self):
        return bool(self.key)

    def search(self, query: str, page_size: int = 25):
        url = (
            "https://api.unsplash.com/search/photos?"
            + urllib.parse.urlencode({"query": query, "per_page": page_size})
            + f"&client_id={self.key}"
        )
        data = _http_json(url) or {}
        return [
            (r["urls"]["regular"], {"source": "unsplash", "license": "unsplash",
                                    "creator": r.get("user", {}).get("name"),
                                    "id": r.get("id")})
            for r in data.get("results", []) if r.get("urls", {}).get("regular")
        ]


class PexelsSource(Source):
    """Pexels — needs PEXELS_KEY env var. 200 req/hr free."""
    name = "pexels"

    def __init__(self):
        self.key = os.environ.get("PEXELS_KEY", "")

    def enabled(self):
        return bool(self.key)

    def search(self, query: str, page_size: int = 25):
        req = urllib.request.Request(
            "https://api.pexels.com/v1/search?"
            + urllib.parse.urlencode({"query": query, "per_page": page_size}),
            headers={"User-Agent": _USER_AGENT, "Authorization": self.key},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            return []
        return [
            (p["src"]["large"], {"source": "pexels", "license": "pexels",
                                 "creator": p.get("photographer"),
                                 "id": p.get("id")})
            for p in data.get("photos", []) if p.get("src", {}).get("large")
        ]


class PixabaySource(Source):
    """Pixabay — needs PIXABAY_KEY env var."""
    name = "pixabay"

    def __init__(self):
        self.key = os.environ.get("PIXABAY_KEY", "")

    def enabled(self):
        return bool(self.key)

    def search(self, query: str, page_size: int = 25):
        url = ("https://pixabay.com/api/?"
               + urllib.parse.urlencode({
                   "key": self.key, "q": query,
                   "image_type": "photo", "per_page": page_size,
                   "safesearch": "true",
               }))
        data = _http_json(url) or {}
        return [
            (h["largeImageURL"], {"source": "pixabay", "license": "pixabay",
                                  "creator": h.get("user"), "id": h.get("id")})
            for h in data.get("hits", []) if h.get("largeImageURL")
        ]


# ── Bootstrap pipeline ────────────────────────────────────────────────────
def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _safe_dirname(s: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")


def _load_seen(meta_path: Path) -> set:
    seen = set()
    if not meta_path.exists():
        return seen
    with meta_path.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["sha"])
            except Exception:
                continue
    return seen


def bootstrap(
    target_per_class: int,
    classes: Optional[List[str]],
    out_dir: Path = DATA_DIR,
    sources: Optional[List[Source]] = None,
    pause: float = 0.6,
) -> Dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "metadata.jsonl"
    seen = _load_seen(meta_path)
    LOG.info(f"Resuming with {len(seen)} files already in metadata.jsonl")

    if sources is None:
        sources = [s for s in (
            OpenverseSource(), WikimediaSource(),
            UnsplashSource(), PexelsSource(), PixabaySource(),
        ) if s.enabled()]
    LOG.info(f"Active sources: {[s.name for s in sources]}")

    chosen_classes = [c for c in (classes or list(CLASS_QUERIES)) if c in CLASS_QUERIES]
    counts: Dict[str, int] = {c: 0 for c in chosen_classes}

    with meta_path.open("a") as meta_f:
        for cls in chosen_classes:
            cdir = out_dir / cls
            cdir.mkdir(exist_ok=True)
            existing = sum(1 for p in cdir.glob("*.jpg") if p.is_file())
            counts[cls] = existing
            need = max(0, target_per_class - existing)
            if need == 0:
                LOG.info(f"[{cls}] already at {existing}, skipping")
                continue
            LOG.info(f"[{cls}] have {existing}, need {need} more")

            queries = CLASS_QUERIES[cls][:]
            random.shuffle(queries)
            done = False
            for src in sources:
                if done: break
                for q in queries:
                    if done: break
                    LOG.info(f"  ↳ {src.name} ⟂ {q!r}")
                    try:
                        results = src.search(q, page_size=25)
                    except Exception as e:
                        LOG.warning(f"    {src.name} search failed: {e}")
                        results = []
                    time.sleep(pause)
                    for iurl, m in results:
                        if counts[cls] >= target_per_class:
                            done = True; break
                        data = _http_bytes(iurl)
                        time.sleep(pause * 0.5)
                        if not data:
                            continue
                        if not _is_decodable_image(data):
                            continue
                        sha = _sha(data)
                        if sha in seen:
                            continue
                        idx = counts[cls] + 1
                        fname = f"img_{idx:04d}_{sha[:8]}.jpg"
                        fpath = cdir / fname
                        try:
                            # Re-encode to JPEG for size + uniformity
                            from PIL import Image
                            img = Image.open(io.BytesIO(data)).convert("RGB")
                            img.thumbnail((1024, 1024))
                            img.save(fpath, "JPEG", quality=88, optimize=True)
                        except Exception as e:
                            LOG.debug(f"save failed: {e}")
                            continue
                        rec = {
                            "path": str(fpath.relative_to(ROOT)),
                            "class": cls, "query": q,
                            "sha": sha,
                            **m,
                        }
                        meta_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        meta_f.flush()
                        seen.add(sha)
                        counts[cls] += 1
                        if counts[cls] % 25 == 0:
                            LOG.info(f"    [{cls}] {counts[cls]}/{target_per_class}")
            LOG.info(f"[{cls}] FINAL {counts[cls]}/{target_per_class}")

    LOG.info("=" * 60)
    LOG.info("Bootstrap complete:")
    for c, n in counts.items():
        LOG.info(f"  {c:>20s}: {n}")
    LOG.info(f"Metadata: {meta_path}")
    return counts


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="Snappy dataset bootstrapper")
    ap.add_argument("--target", type=int, default=150,
                    help="images per class (default 150 → ~1500 total)")
    ap.add_argument("--classes", default="",
                    help="comma-separated class subset (default: all)")
    ap.add_argument("--out", default=str(DATA_DIR), help="output dir")
    ap.add_argument("--pause", type=float, default=0.6,
                    help="seconds between requests (rate limit)")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()] or None
    bootstrap(
        target_per_class=args.target,
        classes=classes,
        out_dir=Path(args.out),
        pause=args.pause,
    )


if __name__ == "__main__":
    main()
