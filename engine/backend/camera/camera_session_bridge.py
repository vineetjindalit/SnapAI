"""
backend/camera/camera_session_bridge.py — connects a REAL camera
(webcam/DSLR/RTSP/phone-IP/GoPro/video file) to a LIVE Session running the
CURRENT v2.5 pipeline, instead of the disconnected standalone
camera_runner.py script (which still calls the pre-v2.5 VLMDetector/
GazeDetector path — see the 2026-10-01 codebase review in the project docs).

This module reuses EXACTLY the same api.pipeline.process_frame() every
browser WebSocket frame goes through, so event_engine.py's priority-tier
ensemble, host_group.py personalization, and every other v2.5 model apply
to camera-sourced frames with zero special-casing — a DSLR-fed session gets
the identical capture logic a browser-fed session gets. Frame results are
broadcast to any connected dashboard/browser via api.globals.push_to_session(),
the same out-of-band channel video-upload telemetry already uses, so nothing
new is needed on the wire protocol.

One CameraSessionRunner per session id (sid). Each runs its OWN background
thread pulling frames at the configured fps and feeding them through
process_frame() serially — mirroring how each browser WebSocket already
processes its own frames one at a time. A second camera-backed session gets
its own independent thread, so one camera's pace never blocks another
session.

NOT wired into any route by default — see api/routes.py's create_session /
end_session for the (optional, additive) `camera_source` hookup. A session
created without `camera_source` in the request body behaves exactly as
before; this module changes nothing for the default browser-camera path.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

from .camera_source import CameraConfig, CameraSource

log = logging.getLogger("snappy.camera.bridge")

# sid -> running CameraSessionRunner. Lets routes.py stop a camera cleanly
# when a session ends, or swap cameras on an existing sid.
_RUNNERS: Dict[str, "CameraSessionRunner"] = {}


class CameraSessionRunner:
    """Pulls frames from a CameraSource and feeds them through the live
    v2.5 capture pipeline for one Session, at the camera's configured fps."""

    def __init__(self, sid: str, session, cfg: CameraConfig):
        self.sid = sid
        self.session = session
        self.cam = CameraSource(cfg)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self.last_error: Optional[str] = None
        self.frames_sent = 0
        # DSLR full-res shutter trigger (see _maybe_trigger_fullres below).
        # gphoto2's capture-and-download takes up to ~10s — far longer than
        # the 2.5s min_capture_interval_s between AI capture decisions — so
        # this guards against firing a second physical shutter trigger while
        # one is still in flight, rather than queuing them up and drifting
        # further behind the live event.
        self._fullres_lock = threading.Lock()
        self._fullres_busy = False
        self.fullres_sent = 0
        self.fullres_errors = 0

    def start(self) -> bool:
        if not self.cam.start():
            self.last_error = f"camera failed to start ({self.cam.cfg.source_type})"
            log.error(f"[{self.sid}] {self.last_error}")
            return False
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"camera-{self.sid}")
        self._thread.start()
        log.info(f"[{self.sid}] camera bridge started ({self.cam.cfg.source_type})")
        return True

    def stop(self) -> None:
        self._running = False
        try:
            self.cam.stop()
        except Exception as ex:
            log.warning(f"[{self.sid}] camera stop: {ex}")
        log.info(f"[{self.sid}] camera bridge stopped ({self.frames_sent} frames sent)")

    def _loop(self) -> None:
        # Deferred imports: keeps backend/camera usable standalone (as
        # camera_runner.py already does, with no backend/api dependency)
        # for anyone NOT running it through the live server, and avoids a
        # module-import-order dependency on api.globals having finished
        # building the model zoo before backend/camera itself is imported.
        from api.globals import push_to_session
        from api.pipeline import process_frame

        interval = 1.0 / max(1, self.cam.cfg.fps)
        # Give the capture thread a moment to produce its first frame before
        # we start polling, so a slow-starting DSLR/RTSP source doesn't spam
        # "no frame yet" while gphoto2/ffmpeg are still warming up.
        time.sleep(min(2.0, interval * 3))

        while self._running and getattr(self.session, "active", True):
            t0 = time.time()
            frame = self.cam.get_frame()
            if frame is not None:
                try:
                    result = process_frame(self.session, frame_b64=None,
                                            frame=frame, source="live")
                    push_to_session(self.sid, result)
                    self.frames_sent += 1
                    if result.get("captured"):
                        self._maybe_trigger_fullres()
                except Exception as ex:
                    self.last_error = str(ex)
                    log.warning(f"[{self.sid}] frame processing error: {ex}")
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def _maybe_trigger_fullres(self) -> None:
        """The AI just decided the live-preview frame was capture-worthy.
        For a tethered DSLR, that preview frame is a small liveview JPEG —
        not what we actually want in the album. Fire the camera's REAL
        shutter (gphoto2) to grab the full-resolution shot, then ingest it
        through the exact same enhance/score/reconcile path manual photos
        use (routes.ingest_photo_frame) — so the full-res DSLR shot and the
        AI's own preview-resolution capture of the same moment land in the
        same photo pool, and album_generator.py's existing best-of-cluster
        dedup picks whichever is actually better (almost always the full-res
        one) — no new "which shot wins" logic needed.

        Runs on its own short-lived daemon thread so the ~10s gphoto2 call
        never blocks this runner's live-preview polling loop. Skips firing
        (not queues) if a trigger from an earlier capture is still in
        flight, since min_capture_interval_s (2.5s) between AI capture
        decisions is far shorter than gphoto2's own round trip — queuing
        would just make the DSLR fall further and further behind the event.
        """
        if self.cam.cfg.source_type not in ("dslr", "dslr_tether"):
            return
        with self._fullres_lock:
            if self._fullres_busy:
                log.debug(f"[{self.sid}] DSLR full-res trigger already in "
                          f"flight — skipping this capture")
                return
            self._fullres_busy = True

        def _fire():
            from api.routes import ingest_photo_frame
            try:
                path = self.cam.trigger_full_capture()
                if not path:
                    self.fullres_errors += 1
                    self.last_error = "DSLR full-res trigger failed"
                    return
                import cv2
                full_frame = cv2.imread(path)
                if full_frame is None:
                    self.fullres_errors += 1
                    self.last_error = f"DSLR full-res file unreadable: {path}"
                    return
                status, _ = ingest_photo_frame(
                    self.sid, full_frame, tags=["dslr_fullres"],
                    fname_prefix="dslr")
                if status == 201:
                    self.fullres_sent += 1
                else:
                    self.fullres_errors += 1
                    self.last_error = f"DSLR full-res ingest failed ({status})"
            except Exception as ex:
                self.fullres_errors += 1
                self.last_error = f"DSLR full-res trigger error: {ex}"
                log.warning(f"[{self.sid}] {self.last_error}")
            finally:
                with self._fullres_lock:
                    self._fullres_busy = False

        threading.Thread(target=_fire, daemon=True,
                          name=f"camera-{self.sid}-fullres").start()


def start_camera_for_session(sid: str, session, source_type: str, **kwargs):
    """Attach a real camera to an existing Session and start streaming its
    frames through the live v2.5 pipeline.

    Returns the running CameraSessionRunner, or an error STRING on failure
    (never raises) — callers are HTTP routes that must turn this into a
    JSON response either way.

    source_type: "usb" | "dslr" | "dslr_tether" | "rtsp" | "ip" | "gopro" | "file"
    kwargs: passed straight into CameraConfig — index, url, width, height,
            fps, username, password (see camera_source.py).
    """
    stop_camera_for_session(sid)   # replace any existing camera already on this sid
    try:
        cfg = CameraConfig(source_type=source_type, **kwargs)
    except TypeError as ex:
        return f"bad camera_source config: {ex}"
    runner = CameraSessionRunner(sid, session, cfg)
    if not runner.start():
        return runner.last_error or "camera failed to start"
    _RUNNERS[sid] = runner
    return runner


def stop_camera_for_session(sid: str) -> bool:
    runner = _RUNNERS.pop(sid, None)
    if runner is None:
        return False
    runner.stop()
    return True


def camera_status(sid: str) -> Optional[dict]:
    runner = _RUNNERS.get(sid)
    if runner is None:
        return None
    return {
        "source_type":    runner.cam.cfg.source_type,
        "running":        runner._running,
        "frames_sent":    runner.frames_sent,
        "fullres_sent":   runner.fullres_sent,
        "fullres_errors": runner.fullres_errors,
        "last_error":     runner.last_error,
    }
