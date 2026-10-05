"""
backend/camera/camera_adapter.py

Universal Camera Adapter for Snappy.
Supports every camera type through a single unified interface.

Camera types supported:
  1. WEBCAM/USB       → cv2.VideoCapture(index)
  2. DSLR via USB     → gphoto2 subprocess (Canon/Nikon/Sony)
  3. RTSP/IP Camera   → cv2.VideoCapture("rtsp://...")
  4. Phone (WiFi)     → MJPEG stream from IP Webcam app
  5. Video file       → cv2.VideoCapture("file.mp4") for testing
  6. Browser (WebRTC) → existing WebSocket path (no change needed)

Usage:
    cam = CameraAdapter.create("webcam", index=0)
    cam = CameraAdapter.create("dslr")
    cam = CameraAdapter.create("rtsp", url="rtsp://192.168.1.10:554/stream")
    cam = CameraAdapter.create("phone", url="http://192.168.1.5:8080/video")
    cam = CameraAdapter.create("file", path="wedding.mp4")

    cam.start()
    while True:
        frame = cam.read()         # returns numpy BGR frame or None
        if frame is not None:
            process_frame(session, encode_frame(frame))
    cam.stop()
"""

import cv2
import numpy as np
import subprocess
import threading
import time
import base64
import os
import sys
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger("snappy.camera")


# ── Base class ─────────────────────────────────────────────────────────────────

class BaseCamera(ABC):
    def __init__(self):
        self._running   = False
        self._frame     = None
        self._lock      = threading.Lock()
        self._thread    = None
        self.fps        = 0.0
        self._fc        = 0
        self._t0        = 0.0
        self.on_frame: Optional[Callable] = None   # callback(frame_bgr)

    def start(self):
        self._running = True
        self._t0      = time.time()
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info(f"[{self.__class__.__name__}] started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        self._release()
        log.info(f"[{self.__class__.__name__}] stopped")

    def read(self) -> Optional[np.ndarray]:
        """Non-blocking read of latest frame."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _store(self, frame: np.ndarray):
        with self._lock:
            self._frame = frame
        self._fc += 1
        elapsed = time.time() - self._t0
        if elapsed > 0:
            self.fps = round(self._fc / elapsed, 1)
        if self.on_frame:
            self.on_frame(frame)

    @abstractmethod
    def _loop(self): ...

    @abstractmethod
    def _release(self): ...

    @abstractmethod
    def get_info(self) -> dict: ...


# ── 1. Webcam / USB Camera ─────────────────────────────────────────────────────

class WebcamCamera(BaseCamera):
    """
    Any camera OpenCV can open: built-in webcam, USB webcam, USB capture card.
    Works with most consumer webcams (Logitech, etc.) out of the box.

    DSLR via HDMI capture card (e.g. Elgato Cam Link):
        The DSLR's HDMI out → capture card → appears as index 0/1/2 here.
        This is the RECOMMENDED way to use a DSLR with Snappy on Windows/Mac.
    """

    def __init__(self, index: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        super().__init__()
        self.index  = index
        self.width  = width
        self.height = height
        self._fps   = fps
        self._cap   = None

    def _loop(self):
        self._cap = cv2.VideoCapture(self.index)
        if not self._cap.isOpened():
            log.error(f"Cannot open camera index {self.index}")
            return

        # Request resolution and FPS
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._fps)
        # Reduce buffer to minimize latency
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info(f"Webcam opened: index={self.index}, resolution={actual_w}x{actual_h}")

        while self._running:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                self._store(frame)
            else:
                time.sleep(0.01)

    def _release(self):
        if self._cap:
            self._cap.release()

    def get_info(self) -> dict:
        return {"type": "webcam", "index": self.index,
                "width": self.width, "height": self.height, "fps": self.fps}


# ── 2. DSLR via gphoto2 (Linux/Mac, USB tethering) ────────────────────────────

class DSLRCamera(BaseCamera):
    """
    Canon / Nikon / Sony DSLR connected via USB tethering using gphoto2.

    SETUP:
        # Linux/Mac:
        sudo apt install gphoto2          # Linux
        brew install gphoto2              # Mac

        # Verify camera is detected:
        gphoto2 --auto-detect

    HOW IT WORKS:
        - Snappy triggers the shutter via gphoto2 command
        - Camera saves RAW+JPEG to memory card
        - gphoto2 downloads the JPEG immediately
        - Frame is fed into the ML pipeline

    LIMITATIONS:
        - Shutter lag: ~200-500ms (mechanical shutter)
        - No live preview (use WebcamCamera via HDMI for live preview)
        - Requires gphoto2 installed

    BEST PRACTICE:
        Use DSLRCamera for actual photo CAPTURE,
        and a separate WebcamCamera (HDMI out) for live preview/analysis.
    """

    def __init__(self, capture_dir: str = "./dslr_captures"):
        super().__init__()
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._cap_count  = 0
        self._preview_proc = None

    def _check_gphoto2(self) -> bool:
        try:
            r = subprocess.run(["gphoto2", "--version"],
                               capture_output=True, timeout=5)
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def detect_camera(self) -> Optional[str]:
        """Returns camera model string if detected."""
        try:
            r = subprocess.run(["gphoto2", "--auto-detect"],
                               capture_output=True, text=True, timeout=10)
            lines = r.stdout.strip().split("\n")
            for line in lines[2:]:
                if line.strip() and "usb" in line.lower():
                    return line.strip()
        except Exception:
            pass
        return None

    def capture_photo(self) -> Optional[np.ndarray]:
        """
        Trigger shutter and download JPEG.
        Returns BGR numpy frame or None on failure.
        """
        fname = f"dslr_{self._cap_count:04d}_{int(time.time())}.jpg"
        fpath = self.capture_dir / fname

        try:
            result = subprocess.run([
                "gphoto2",
                "--capture-image-and-download",
                "--filename", str(fpath),
                "--force-overwrite"
            ], capture_output=True, text=True, timeout=15)

            if result.returncode != 0:
                log.error(f"gphoto2 error: {result.stderr}")
                return None

            if fpath.exists():
                frame = cv2.imread(str(fpath))
                self._cap_count += 1
                log.info(f"DSLR captured: {fname}")
                return frame
        except subprocess.TimeoutExpired:
            log.error("gphoto2 timeout — camera not responding")
        except Exception as e:
            log.error(f"DSLR capture error: {e}")
        return None

    def start_live_preview(self):
        """
        Stream live preview frames from DSLR (if supported by camera model).
        Canon EOS and many Nikon models support this.
        """
        try:
            self._preview_proc = subprocess.Popen([
                "gphoto2", "--capture-movie", "--stdout"
            ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            log.info("DSLR live preview started")
        except Exception as e:
            log.error(f"Live preview error: {e}")

    def _loop(self):
        """For DSLR, loop just keeps the thread alive. Actual capture is triggered externally."""
        if not self._check_gphoto2():
            log.error("gphoto2 not installed. Run: sudo apt install gphoto2")
            return
        camera = self.detect_camera()
        if camera:
            log.info(f"DSLR detected: {camera}")
        else:
            log.warning("No DSLR detected via gphoto2. Check USB connection.")
        while self._running:
            time.sleep(0.5)

    def _release(self):
        if self._preview_proc:
            self._preview_proc.terminate()

    def get_info(self) -> dict:
        cam = self.detect_camera()
        return {"type": "dslr_gphoto2", "detected": cam or "none",
                "capture_dir": str(self.capture_dir)}


# ── 3. RTSP / IP Camera ────────────────────────────────────────────────────────

class RTSPCamera(BaseCamera):
    """
    Network camera via RTSP stream.

    Common RTSP URLs:
        Canon network cameras:  rtsp://192.168.1.100/stream1
        Hikvision:              rtsp://admin:password@192.168.1.64:554/Streaming/Channels/1
        Dahua:                  rtsp://admin:password@192.168.1.108:554/cam/realmonitor?channel=1
        Generic:                rtsp://username:password@ip:port/path

    Also works with:
        - OBS virtual camera stream
        - FFmpeg streams
        - Security cameras
    """

    def __init__(self, url: str, reconnect_sec: float = 5.0):
        super().__init__()
        self.url           = url
        self.reconnect_sec = reconnect_sec
        self._cap          = None

    def _loop(self):
        while self._running:
            try:
                log.info(f"Connecting to RTSP: {self.url}")
                self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self._cap.isOpened():
                    log.warning("RTSP connection failed, retrying...")
                    time.sleep(self.reconnect_sec)
                    continue

                log.info("RTSP connected")
                while self._running:
                    ret, frame = self._cap.read()
                    if not ret:
                        log.warning("RTSP stream lost, reconnecting...")
                        break
                    self._store(frame)

            except Exception as e:
                log.error(f"RTSP error: {e}")
            finally:
                if self._cap:
                    self._cap.release()
            time.sleep(self.reconnect_sec)

    def _release(self):
        if self._cap:
            self._cap.release()

    def get_info(self) -> dict:
        return {"type": "rtsp", "url": self.url, "fps": self.fps}


# ── 4. Phone Camera (WiFi/MJPEG) ──────────────────────────────────────────────

class PhoneCamera(BaseCamera):
    """
    Android/iPhone camera over WiFi using MJPEG stream.

    Android setup (free):
        1. Install "IP Webcam" app (by Pavel Khlebovich)
        2. Open app → Start Server
        3. Note the IP shown (e.g. http://192.168.1.5:8080)
        4. Pass url="http://192.168.1.5:8080/video"

    iPhone setup:
        1. Install "EpocCam" or "DroidCam" app
        2. Follow app instructions to get stream URL

    Alternatively use the browser WebSocket path (already working in current code).
    That's actually better for phones — no extra app needed.
    """

    def __init__(self, url: str):
        super().__init__()
        # Normalize URL
        if not url.endswith("/video"):
            url = url.rstrip("/") + "/video"
        self.url  = url
        self._cap = None

    def _loop(self):
        while self._running:
            try:
                log.info(f"Connecting to phone camera: {self.url}")
                self._cap = cv2.VideoCapture(self.url)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not self._cap.isOpened():
                    log.warning("Phone camera not reachable, retrying in 5s...")
                    time.sleep(5)
                    continue

                log.info("Phone camera connected")
                while self._running:
                    ret, frame = self._cap.read()
                    if not ret:
                        log.warning("Phone stream dropped, reconnecting...")
                        break
                    self._store(frame)

            except Exception as e:
                log.error(f"Phone camera error: {e}")
            finally:
                if self._cap:
                    self._cap.release()
            time.sleep(3)

    def _release(self):
        if self._cap:
            self._cap.release()

    def get_info(self) -> dict:
        return {"type": "phone_mjpeg", "url": self.url, "fps": self.fps}


# ── 5. Video File (for testing) ───────────────────────────────────────────────

class VideoFileCamera(BaseCamera):
    """Feed a recorded video through the pipeline. Perfect for testing."""

    def __init__(self, path: str, loop: bool = True, speed: float = 1.0):
        super().__init__()
        self.path  = path
        self.loop  = loop
        self.speed = speed
        self._cap  = None

    def _loop(self):
        while self._running:
            self._cap = cv2.VideoCapture(self.path)
            if not self._cap.isOpened():
                log.error(f"Cannot open video: {self.path}")
                return

            native_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            delay      = 1.0 / (native_fps * self.speed)

            while self._running:
                ret, frame = self._cap.read()
                if not ret:
                    if self.loop:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        break
                self._store(frame)
                time.sleep(delay)

            self._cap.release()
            if not self.loop:
                break

    def _release(self):
        if self._cap:
            self._cap.release()

    def get_info(self) -> dict:
        return {"type": "video_file", "path": self.path, "fps": self.fps}


# ── Factory ────────────────────────────────────────────────────────────────────

class CameraAdapter:
    """
    Factory that creates the right camera for your setup.

    Examples:
        # Built-in webcam or USB webcam
        cam = CameraAdapter.create("webcam", index=0)

        # DSLR via HDMI capture card (Elgato, AverMedia, etc.) — RECOMMENDED
        cam = CameraAdapter.create("webcam", index=1, width=1920, height=1080)

        # DSLR via USB gphoto2 (Linux/Mac only)
        cam = CameraAdapter.create("dslr")

        # IP / security camera
        cam = CameraAdapter.create("rtsp", url="rtsp://admin:pass@192.168.1.64:554/stream1")

        # Android phone (IP Webcam app)
        cam = CameraAdapter.create("phone", url="http://192.168.1.5:8080")

        # Test with video file
        cam = CameraAdapter.create("file", path="wedding_sample.mp4")
    """

    @staticmethod
    def create(camera_type: str, **kwargs) -> BaseCamera:
        types = {
            "webcam": WebcamCamera,
            "usb":    WebcamCamera,
            "dslr":   DSLRCamera,
            "rtsp":   RTSPCamera,
            "ip":     RTSPCamera,
            "phone":  PhoneCamera,
            "mobile": PhoneCamera,
            "file":   VideoFileCamera,
            "video":  VideoFileCamera,
        }
        cls = types.get(camera_type.lower())
        if not cls:
            raise ValueError(f"Unknown camera type: {camera_type}. "
                             f"Choose from: {list(types.keys())}")
        return cls(**kwargs)

    @staticmethod
    def auto_detect() -> Optional[BaseCamera]:
        """Try to auto-detect an available camera."""
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                log.info(f"Auto-detected camera at index {i}")
                return WebcamCamera(index=i)
        log.warning("No camera auto-detected")
        return None

    @staticmethod
    def list_usb_cameras() -> list:
        """List all available USB/webcam indices."""
        available = []
        for i in range(8):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available.append({"index": i, "resolution": f"{w}x{h}"})
                cap.release()
        return available


# ── Frame encoder ─────────────────────────────────────────────────────────────

def encode_frame(frame: np.ndarray, quality: int = 75, width: int = 640) -> str:
    """
    Encode a BGR numpy frame to base64 JPEG string.
    This is what gets sent to the WebSocket server.
    """
    h, w = frame.shape[:2]
    if w > width:
        scale  = width / w
        frame  = cv2.resize(frame, (width, int(h * scale)))
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


# ── Camera-to-server bridge ────────────────────────────────────────────────────

class CameraServerBridge:
    """
    Connects any BaseCamera to the Snappy WebSocket server.
    Runs on the same machine as the camera (can be different from server machine).

    Usage:
        bridge = CameraServerBridge(
            camera   = CameraAdapter.create("webcam", index=0),
            ws_url   = "ws://localhost:8765/ws/abc123",
            fps      = 10
        )
        bridge.run()   # blocks
    """

    def __init__(self, camera: BaseCamera, ws_url: str,
                 session_id: str, fps: float = 10.0):
        self.camera     = camera
        self.ws_url     = ws_url
        self.session_id = session_id
        self.target_fps = fps
        self._running   = False

    def run(self):
        """Blocking run loop. Ctrl+C to stop."""
        import websocket   # pip install websocket-client
        self.camera.start()
        self._running = True
        interval      = 1.0 / self.target_fps
        last_send     = 0.0

        def on_message(ws, msg):
            try:
                import json
                d = json.loads(msg)
                if d.get("captured"):
                    print(f"📸 CAPTURED [{d.get('capture_reason')}] "
                          f"score={d.get('analysis',{}).get('total_score',0):.2f}")
            except Exception:
                pass

        def on_error(ws, err):
            log.error(f"WS error: {err}")

        def on_close(ws, *args):
            log.info("WS closed")
            self._running = False

        ws = websocket.WebSocketApp(
            self.ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        def sender():
            nonlocal last_send
            while self._running:
                now   = time.time()
                frame = self.camera.read()
                if frame is not None and (now - last_send) >= interval:
                    b64 = encode_frame(frame)
                    try:
                        import json
                        ws.send(json.dumps({"type": "frame", "frame": b64}))
                        last_send = now
                    except Exception as e:
                        log.error(f"Send error: {e}")
                time.sleep(0.005)

        t = threading.Thread(target=sender, daemon=True)
        t.start()

        try:
            ws.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            self.camera.stop()


# ── Standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run this file directly to connect a camera to a running Snappy server.

    Examples:
        python3 camera_adapter.py webcam 0 ws://localhost:8765/ws/SESSION_ID
        python3 camera_adapter.py rtsp rtsp://192.168.1.64:554/stream ws://localhost:8765/ws/SESSION_ID
        python3 camera_adapter.py phone http://192.168.1.5:8080 ws://localhost:8765/ws/SESSION_ID
        python3 camera_adapter.py file wedding.mp4 ws://localhost:8765/ws/SESSION_ID
    """
    import json, urllib.request

    args = sys.argv[1:]
    if len(args) < 3:
        print("Usage: python3 camera_adapter.py <type> <source> <ws_url>")
        print("  type:   webcam | rtsp | phone | file | dslr")
        print("  source: index (webcam), URL (rtsp/phone), path (file), - (dslr)")
        print("  ws_url: ws://localhost:8765/ws/SESSION_ID")
        sys.exit(1)

    cam_type, source, ws_url = args[0], args[1], args[2]

    if cam_type == "webcam":
        cam = CameraAdapter.create("webcam", index=int(source))
    elif cam_type == "dslr":
        cam = CameraAdapter.create("dslr")
    elif cam_type in ("rtsp", "ip"):
        cam = CameraAdapter.create("rtsp", url=source)
    elif cam_type in ("phone", "mobile"):
        cam = CameraAdapter.create("phone", url=source)
    elif cam_type in ("file", "video"):
        cam = CameraAdapter.create("file", path=source)
    else:
        print(f"Unknown type: {cam_type}")
        sys.exit(1)

    sid = ws_url.split("/ws/")[-1]
    print(f"🎥 Camera: {cam_type} → {source}")
    print(f"📡 Server: {ws_url}")
    print(f"🔑 Session: {sid}")
    print("Press Ctrl+C to stop\n")

    bridge = CameraServerBridge(cam, ws_url, sid, fps=10)
    bridge.run()
