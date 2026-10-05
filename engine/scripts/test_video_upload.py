"""
scripts/test_video_upload.py — diagnose video upload end-to-end.

Run this to confirm video upload is wired correctly:

    python3 scripts/test_video_upload.py                          # uses synth video
    python3 scripts/test_video_upload.py path/to/yourvideo.mp4    # uses your video

It will:
  1. Generate (or load) a small video
  2. Create a session via POST /sessions
  3. Upload the video via POST /sessions/<sid>/upload_video
  4. Poll /sessions/<sid>/video_progress until done or 60s timeout
  5. Print the captures

If anything fails, you get a precise error message. No guessing.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SERVER = os.environ.get("SNAPPY_SERVER", "http://localhost:8765")
TIMEOUT = int(os.environ.get("UPLOAD_TIMEOUT", "120"))


def _post_json(path: str, data: dict) -> dict:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        SERVER + path, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get_json(path: str) -> dict:
    with urllib.request.urlopen(SERVER + path, timeout=10) as r:
        return json.loads(r.read())


def _post_multipart(path: str, file_path: Path, sample_fps: float = 2.0) -> dict:
    boundary = "----snappydiag" + str(int(time.time()))
    parts = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(f'Content-Disposition: form-data; name="video"; filename="{file_path.name}"\r\n'.encode())
    parts.append(b"Content-Type: video/mp4\r\n\r\n")
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="sample_fps"\r\n\r\n')
    parts.append(str(sample_fps).encode())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(
        SERVER + path, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _make_synth_video(out: Path, frames: int = 60, fps: float = 30.0) -> None:
    """Generate a tiny test .mp4 (requires opencv-python)."""
    import cv2
    import numpy as np
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(out), fourcc, fps, (320, 240))
    if not w.isOpened():
        raise RuntimeError("Could not open VideoWriter — install opencv-python with codec support")
    rng = np.random.RandomState(0)
    for i in range(frames):
        # gradient + moving square — makes shot_quality non-trivial
        bg = np.zeros((240, 320, 3), dtype=np.uint8)
        bg[:, :] = (i % 200, (i * 3) % 256, (i * 7) % 256)
        x = (i * 5) % 280
        bg[80:160, x:x + 40] = (255, 255, 255)
        w.write(bg)
    w.release()


def main():
    # 1. Health check
    try:
        h = _get_json("/health")
    except Exception as e:
        print(f"❌ Server not reachable at {SERVER}: {e}")
        print("   Run: python3 run.py")
        sys.exit(2)
    print(f"✅ /health: version={h.get('version')} face={h.get('models',{}).get('face',{}).get('backend')}")

    # 2. Create session
    sess = _post_json("/sessions", {
        "event_name": "Video upload diagnostic",
        "event_type": "general",
        "prompt": "general",
    })
    sid = sess["session_id"]
    print(f"✅ session created: {sid}")

    # 3. Pick a video
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
        if not video.exists():
            print(f"❌ {video} not found"); sys.exit(2)
    else:
        video = Path("/tmp/snappy_synth_test.mp4")
        print(f"   Generating synthetic test video → {video}")
        try:
            _make_synth_video(video)
        except Exception as e:
            print(f"❌ synth video generation failed: {e}")
            print("   Pass a real video path: python3 scripts/test_video_upload.py path/to/video.mp4")
            sys.exit(2)
    print(f"   video: {video} ({video.stat().st_size/1024:.1f} KB)")

    # 4. Upload
    print(f"   uploading…")
    t0 = time.time()
    try:
        resp = _post_multipart(f"/sessions/{sid}/upload_video", video, sample_fps=2.0)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"❌ upload returned {e.code}: {body}")
        sys.exit(2)
    print(f"✅ upload accepted in {time.time()-t0:.1f}s: {resp}")

    # 5. Poll
    deadline = time.time() + TIMEOUT
    last_pct = -1
    while time.time() < deadline:
        prog = _get_json(f"/sessions/{sid}/video_progress")
        pct = prog.get("progress", 0)
        status = prog.get("status", "?")
        if pct != last_pct:
            print(f"   {status}: {prog.get('processed_frames')}/{prog.get('total_frames')} "
                  f"= {pct}%  captures={prog.get('captures', 0)}")
            last_pct = pct
        if status in ("done", "error"):
            break
        time.sleep(1.0)

    final = _get_json(f"/sessions/{sid}/video_progress")
    if final["status"] == "done":
        photos = _get_json(f"/sessions/{sid}/photos")
        print(f"✅ done — {final['captures']} captures in {time.time()-t0:.1f}s")
        for p in photos.get("photos", [])[:5]:
            print(f"   {p['url']}  {p['moment']}  score={p['score']:.2f}")
    else:
        print(f"❌ status={final['status']}, error={final.get('error')}")
        sys.exit(2)


if __name__ == "__main__":
    main()
