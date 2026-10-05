"""
backend/camera/camera_source.py

Universal camera input layer for Snappy.
Abstracts ALL camera types into one interface: get_frame() → np.ndarray

Supported sources:
  - USB webcam / laptop cam     → CameraSource("usb", index=0)
  - DSLR via USB (gphoto2)      → CameraSource("dslr")
  - DSLR via gphoto2 tethered   → CameraSource("dslr_tether")
  - IP Camera / CCTV (RTSP)     → CameraSource("rtsp", url="rtsp://192.168.1.10/stream")
  - Phone IP Webcam app         → CameraSource("ip", url="http://192.168.1.5:8080/video")
  - GoPro (WiFi)                → CameraSource("gopro")
  - Video file (for testing)    → CameraSource("file", url="video.mp4")
  - Browser WebSocket (default) → Not this file — handled by server.py

Usage:
  cam = CameraSource("usb")
  cam.start()
  while True:
      frame = cam.get_frame()
      if frame is not None:
          process_frame(session, frame)
  cam.stop()
"""

import cv2
import numpy as np
import threading
import time
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

log = logging.getLogger("snappy.camera")


@dataclass
class CameraConfig:
    source_type: str          # usb | dslr | dslr_tether | rtsp | ip | gopro | file
    index:       int  = 0    # for USB cam: 0=first, 1=second, etc.
    url:         str  = ""   # for RTSP/IP/file
    width:       int  = 1280
    height:      int  = 720
    fps:         int  = 15   # how many frames/sec to push to Snappy
    username:    str  = ""   # for RTSP cameras with auth
    password:    str  = ""


class CameraSource:
    """
    Thread-safe camera frame source.
    Internal thread reads frames continuously.
    get_frame() returns latest frame instantly (non-blocking).
    """

    def __init__(self, config: CameraConfig):
        self.cfg        = config
        self._cap       = None          # cv2.VideoCapture
        self._frame     = None          # latest frame
        self._lock      = threading.Lock()
        self._running   = False
        self._thread    = None
        self._dslr_dir  = Path("/tmp/snappy_dslr")
        self._dslr_dir.mkdir(exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start the camera. Returns True if successful."""
        try:
            ok = self._init_source()
            if not ok:
                return False
            self._running = True
            self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            log.info(f"Camera started: {self.cfg.source_type}")
            return True
        except Exception as e:
            log.error(f"Camera start failed: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Return latest frame (non-blocking). None if not ready."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        log.info("Camera stopped")

    def is_running(self) -> bool:
        return self._running and self._frame is not None

    # ── Source initialisation ──────────────────────────────────────────────────

    def _init_source(self) -> bool:
        t = self.cfg.source_type

        if t == "usb":
            return self._init_usb()
        elif t in ("dslr", "dslr_tether"):
            return self._init_dslr()
        elif t == "rtsp":
            return self._init_rtsp()
        elif t == "ip":
            return self._init_ip()
        elif t == "gopro":
            return self._init_gopro()
        elif t == "file":
            return self._init_file()
        else:
            log.error(f"Unknown source type: {t}")
            return False

    def _init_usb(self) -> bool:
        """USB webcam or any V4L2 device."""
        self._cap = cv2.VideoCapture(self.cfg.index)
        if not self._cap.isOpened():
            log.error(f"Cannot open USB camera index {self.cfg.index}")
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.cfg.fps)
        log.info(f"USB camera {self.cfg.index} opened: "
                 f"{int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                 f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return True

    def _init_dslr(self) -> bool:
        """
        Canon / Nikon / Sony DSLR via gphoto2 USB tethering.

        Requires: sudo apt install gphoto2
        Or:       pip install gphoto2  (Python bindings)

        Two modes:
          'dslr'        → trigger capture via gphoto2 CLI, pull file
          'dslr_tether' → continuous liveview (not all cameras support it)
        """
        # Check gphoto2 installed
        if subprocess.run(["which", "gphoto2"], capture_output=True).returncode != 0:
            log.error("gphoto2 not found. Install: sudo apt install gphoto2")
            log.error("Falling back to USB camera index 0")
            self.cfg.source_type = "usb"
            return self._init_usb()

        # Check camera detected
        result = subprocess.run(
            ["gphoto2", "--auto-detect"],
            capture_output=True, text=True, timeout=5
        )
        if "usb:" not in result.stdout.lower():
            log.error("No DSLR detected via gphoto2. Check USB connection.")
            return False

        log.info(f"DSLR detected:\n{result.stdout}")

        if self.cfg.source_type == "dslr_tether":
            # Start liveview capture — saves frames to temp dir continuously
            self._dslr_proc = subprocess.Popen([
                "gphoto2", "--capture-movie",
                "--stdout"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            self._cap = cv2.VideoCapture()
            # Read from gphoto2 stdout pipe
            self._dslr_pipe = self._dslr_proc.stdout
            log.info("DSLR liveview started (tethered mode)")
        else:
            # Poll mode: capture single frames on demand
            log.info("DSLR ready (poll capture mode)")
            self._dslr_pipe = None

        return True

    def _init_rtsp(self) -> bool:
        """
        IP Camera / CCTV via RTSP stream.
        URL format: rtsp://username:password@192.168.1.10:554/stream1
        """
        url = self.cfg.url
        if self.cfg.username:
            # Inject credentials into URL
            proto, rest = url.split("://", 1)
            url = f"{proto}://{self.cfg.username}:{self.cfg.password}@{rest}"

        # Use FFMPEG backend for better RTSP compatibility
        self._cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

        # RTSP buffer settings — reduce latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            log.error(f"Cannot open RTSP stream: {self.cfg.url}")
            return False

        log.info(f"RTSP stream opened: {self.cfg.url}")
        return True

    def _init_ip(self) -> bool:
        """
        Phone camera via IP Webcam app (Android) or DroidCam.
        URL: http://192.168.1.5:8080/video
        Install 'IP Webcam' on Android, 'DroidCam' on iPhone.
        """
        self._cap = cv2.VideoCapture(self.cfg.url)
        if not self._cap.isOpened():
            log.error(f"Cannot open IP camera: {self.cfg.url}")
            log.error("Make sure phone and laptop are on same WiFi network")
            return False
        log.info(f"IP camera opened: {self.cfg.url}")
        return True

    def _init_gopro(self) -> bool:
        """
        GoPro via WiFi. Connect laptop to GoPro WiFi hotspot first.
        GoPro Hero 9+: udp://:8554
        GoPro Hero 8-: http://10.5.5.9:8080/live/amba.m3u8
        """
        urls_to_try = [
            "udp://:8554",
            "http://10.5.5.9:8080/live/amba.m3u8",
            "rtsp://10.5.5.9:554/live",
        ]
        if self.cfg.url:
            urls_to_try = [self.cfg.url] + urls_to_try

        for url in urls_to_try:
            log.info(f"Trying GoPro URL: {url}")
            self._cap = cv2.VideoCapture(url)
            if self._cap.isOpened():
                log.info(f"GoPro connected: {url}")
                return True

        log.error("Cannot connect to GoPro. Make sure connected to GoPro WiFi")
        return False

    def _init_file(self) -> bool:
        """Video file for testing — simulates a live camera."""
        if not os.path.exists(self.cfg.url):
            log.error(f"Video file not found: {self.cfg.url}")
            return False
        self._cap = cv2.VideoCapture(self.cfg.url)
        log.info(f"Video file opened: {self.cfg.url}")
        return True

    # ── Capture loop (runs in background thread) ───────────────────────────────

    def _capture_loop(self):
        interval = 1.0 / self.cfg.fps

        while self._running:
            try:
                frame = self._read_one_frame()
                if frame is not None:
                    with self._lock:
                        self._frame = frame
                time.sleep(interval)
            except Exception as e:
                log.error(f"Capture loop error: {e}")
                time.sleep(0.5)

    def _read_one_frame(self) -> Optional[np.ndarray]:
        t = self.cfg.source_type

        if t == "dslr" and not hasattr(self, '_dslr_pipe'):
            return self._dslr_capture_single()

        if t == "dslr_tether" and hasattr(self, '_dslr_pipe') and self._dslr_pipe:
            return self._dslr_read_liveview()

        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret and frame is not None:
                return frame
            elif t == "file":
                # Loop video file
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        return None

    def _dslr_capture_single(self) -> Optional[np.ndarray]:
        """
        Capture one frame from DSLR via gphoto2 CLI.
        Used in poll mode (not liveview).
        ~500ms per capture — only used for final shots, not frame analysis.
        For frame analysis, use a preview command.
        """
        out_path = str(self._dslr_dir / "preview.jpg")
        try:
            # Capture preview (faster than full capture, ~100ms)
            result = subprocess.run([
                "gphoto2",
                "--capture-preview",
                "--filename", out_path,
                "--force-overwrite"
            ], capture_output=True, timeout=3)

            if result.returncode == 0 and os.path.exists(out_path):
                frame = cv2.imread(out_path)
                return frame
        except subprocess.TimeoutExpired:
            log.warning("DSLR preview timeout")
        except Exception as e:
            log.error(f"DSLR capture error: {e}")
        return None

    def _dslr_read_liveview(self) -> Optional[np.ndarray]:
        """Read frame from gphoto2 liveview pipe."""
        try:
            # Read JPEG from pipe (gphoto2 outputs MJPEG stream)
            buf = b""
            while True:
                chunk = self._dslr_pipe.read(4096)
                if not chunk:
                    break
                buf += chunk
                # Look for JPEG end marker
                if b"\xff\xd9" in buf:
                    start = buf.find(b"\xff\xd8")
                    end   = buf.find(b"\xff\xd9") + 2
                    if start >= 0:
                        jpeg = buf[start:end]
                        arr  = np.frombuffer(jpeg, np.uint8)
                        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            log.error(f"Liveview read error: {e}")
        return None

    def trigger_full_capture(self) -> Optional[str]:
        """
        Trigger a FULL RESOLUTION capture on DSLR.
        Returns path to saved file, or None on failure.
        Only available for DSLR source types.
        """
        if self.cfg.source_type not in ("dslr", "dslr_tether"):
            log.warning("trigger_full_capture only for DSLR sources")
            return None

        out = str(self._dslr_dir / f"full_{int(time.time()*1000)}.jpg")
        try:
            result = subprocess.run([
                "gphoto2",
                "--capture-image-and-download",
                "--filename", out,
                "--force-overwrite"
            ], capture_output=True, timeout=10)

            if result.returncode == 0 and os.path.exists(out):
                log.info(f"Full DSLR capture saved: {out}")
                return out
            else:
                log.error(f"DSLR capture failed: {result.stderr.decode()}")
        except subprocess.TimeoutExpired:
            log.error("DSLR full capture timeout (10s)")
        except Exception as e:
            log.error(f"DSLR full capture error: {e}")
        return None


# ── Detector utility ────────────────────────────────────────────────────────────

def detect_available_cameras() -> dict:
    """
    Scan for all available camera sources on this machine.
    Returns dict of available sources.
    """
    found = {}

    # USB cameras
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            found[f"usb_{i}"] = {
                "type": "usb", "index": i,
                "label": f"USB Camera #{i}",
                "resolution": f"{int(cap.get(3))}x{int(cap.get(4))}"
            }
            cap.release()

    # DSLR via gphoto2
    try:
        result = subprocess.run(
            ["gphoto2", "--auto-detect"],
            capture_output=True, text=True, timeout=5
        )
        if "usb:" in result.stdout.lower():
            for line in result.stdout.split("\n"):
                if "usb:" in line.lower():
                    found["dslr_0"] = {
                        "type": "dslr",
                        "label": line.strip() or "DSLR Camera",
                        "note": "Requires gphoto2"
                    }
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # gphoto2 not installed

    return found
