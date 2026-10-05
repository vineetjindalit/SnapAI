#!/usr/bin/env python3
"""
Snappy v2 — One-click launcher.
Run:  python3 run.py
"""
import sys, os, subprocess, time, webbrowser, threading
from pathlib import Path

ROOT = Path(__file__).parent

BANNER = """
╔════════════════════════════════════════════════╗
║  📸  SnapAI v2.8 — AI Event Photography        ║
║                                                ║
║  App     →  http://localhost:8765              ║
║  API     →  http://localhost:8765/sessions     ║
║  Health  →  http://localhost:8765/health       ║
║  Logs    →  ./logs/snappy.log                  ║
║  DB      →  ./backend/data/snappy.db (SQLite)  ║
╚════════════════════════════════════════════════╝
"""

def check_deps():
    missing = []
    for pkg in ["cv2","numpy","PIL","sklearn","scipy"]:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"⚠️  Missing: {missing}")
        print("Run: pip install opencv-python numpy Pillow scikit-learn scipy")
        sys.exit(1)

    optional_missing = []
    for pkg in ["mediapipe"]:
        try: __import__(pkg)
        except ImportError: optional_missing.append(pkg)

    if optional_missing:
        print(f"⚠️  Optional but recommended: {optional_missing}")
        print("Run: pip install mediapipe")

    print("✅ All dependencies present")

def open_browser(port, delay=2.5):
    def _open():
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    threading.Thread(target=_open, daemon=True).start()

if __name__ == "__main__":
    check_deps()
    print(BANNER)
    open_browser(8765)
    server = ROOT / "backend" / "api" / "server.py"
    os.chdir(ROOT)
    os.execv(sys.executable, [sys.executable, str(server)])
