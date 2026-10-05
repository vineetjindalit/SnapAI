#!/usr/bin/env python3
"""
download_test_clips.py — Optional: pull real-world clips from Pexels.

Pexels offers a generous free API for stock video. Sign up at
    https://www.pexels.com/api/
and put your key in env var PEXELS_API_KEY before running this script.

If you don't have an API key (or no network), just skip this script —
the synthetic clips alone are enough for the priority-tier benchmark.
The synthetic clips test specific code paths (CRITICAL / HIGH / dark
discard) deterministically, while real clips test "does the pipeline
work end-to-end on something that looks like a real event."

Search queries are chosen to match the kinds of moments SnapAI is
designed to capture. All results are CC-licensed for any use, no
attribution required (Pexels license).
"""
from __future__ import annotations
import os, sys, json, time
import urllib.request, urllib.parse, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "datasets" / "clips" / "real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("PEXELS_API_KEY")

QUERIES = [
    # (query, target_count)
    ("birthday cake cutting",   2),
    ("group photo smile",       2),
    ("hug family",              1),
    ("kid laughing close up",   1),
    ("wedding kiss",            1),
    ("confetti throw",          1),
    ("toast champagne glasses", 1),
]

# Hard limits — keep clips small for fast iteration
MAX_DURATION_SEC = 12
PREFERRED_HEIGHT = 480     # nearest match to our pipeline's input


def search_pexels(query: str, per_page: int = 5) -> list[dict]:
    if not API_KEY:
        raise RuntimeError("PEXELS_API_KEY env var not set")
    q = urllib.parse.urlencode({"query": query, "per_page": per_page})
    url = f"https://api.pexels.com/videos/search?{q}"
    req = urllib.request.Request(url, headers={"Authorization": API_KEY})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8")).get("videos", [])


def pick_file(video: dict) -> dict | None:
    """Pick the most pipeline-friendly version of a Pexels video object."""
    files = video.get("video_files", [])
    if not files: return None
    # Prefer .mp4 closest to PREFERRED_HEIGHT
    candidates = [f for f in files if (f.get("file_type") or "").endswith("mp4")]
    if not candidates: return None
    candidates.sort(key=lambda f: abs((f.get("height") or 720) - PREFERRED_HEIGHT))
    return candidates[0]


def download(url: str, path: Path) -> int:
    with urllib.request.urlopen(url, timeout=60) as resp, open(path, "wb") as f:
        bytes_written = 0
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk: break
            f.write(chunk); bytes_written += len(chunk)
    return bytes_written


def main():
    if not API_KEY:
        print("PEXELS_API_KEY not set — skipping real-clip download.")
        print("This is fine — synthetic clips cover all priority-tier code paths.")
        print("To enable: get a free key at https://www.pexels.com/api/ and run:")
        print("  PEXELS_API_KEY=your_key python3 scripts/download_test_clips.py")
        sys.exit(0)

    manifest: list[dict] = []
    downloaded = 0
    for query, target in QUERIES:
        try:
            videos = search_pexels(query, per_page=target + 2)
        except urllib.error.HTTPError as e:
            print(f"  ✗ {query!r}: HTTP {e.code} ({e.reason})")
            continue
        except Exception as e:
            print(f"  ✗ {query!r}: {e}")
            continue

        kept = 0
        for v in videos:
            if kept >= target: break
            if v.get("duration", 999) > MAX_DURATION_SEC:
                continue
            picked = pick_file(v)
            if not picked: continue
            safe_q = query.replace(" ", "_")
            name = f"pexels_{v.get('id', 'x')}_{safe_q}.mp4"
            dest = OUT_DIR / name
            if dest.exists():
                print(f"  ⟳ {name} (exists, skipped)")
                kept += 1
                continue
            try:
                size = download(picked["link"], dest)
                manifest.append({
                    "file": name,
                    "query": query,
                    "duration_sec": v.get("duration"),
                    "width": picked.get("width"),
                    "height": picked.get("height"),
                    "credit_url": v.get("url"),
                    "license": "Pexels Free Use",
                    "bytes": size,
                })
                kept += 1
                downloaded += 1
                print(f"  ✓ {name} ({size/1024:.0f} KB)")
                time.sleep(0.5)  # polite to API
            except Exception as e:
                print(f"  ✗ {name}: {e}")

    (OUT_DIR / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. {downloaded} real clips downloaded.")


if __name__ == "__main__":
    main()
