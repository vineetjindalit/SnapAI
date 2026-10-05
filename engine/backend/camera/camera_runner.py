"""
backend/camera/camera_runner.py

Runs Snappy pipeline directly from a physical camera (no browser needed).
Use this when running on a dedicated device (Raspberry Pi, laptop at event).

Usage:
  # USB webcam
  python3 camera_runner.py --source usb --index 0

  # DSLR via USB
  python3 camera_runner.py --source dslr

  # IP Camera
  python3 camera_runner.py --source rtsp --url "rtsp://192.168.1.10/stream1"

  # Phone (IP Webcam app)
  python3 camera_runner.py --source ip --url "http://192.168.1.5:8080/video"

  # GoPro
  python3 camera_runner.py --source gopro

  # Test with video file
  python3 camera_runner.py --source file --url "test_video.mp4"

Output:
  - Captured photos saved to captures/<session_id>_<video_name>/
  - Live preview window (optional, --preview flag)
  - Album generated on exit
"""

import argparse, sys, os, time, cv2, json, base64, logging, signal
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from camera.camera_source import CameraSource, CameraConfig, detect_available_cameras
from models.shot_quality         import ShotQualityAnalyzer
from models.gaze_detector        import GazeDetector
from models.clip_moment_detector import VLMDetector
from models.moment_predictor     import MomentPredictor
from models.album_generator      import AlbumGenerator, PhotoEntry
from utils.safe_types            import safe

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("snappy.runner")


def run_camera_session(
    source_type : str  = "usb",
    index       : int  = 0,
    url         : str  = "",
    event_name  : str  = "My Event",
    event_type  : str  = "general",
    prompt      : str  = "group photo, cake cutting",
    show_preview: bool = True,
    fps         : int  = 10,
    dslr_full_on_capture: bool = True,  # trigger full-res DSLR shot when best moment detected
):
    import uuid

    import re as _re
    sid = str(uuid.uuid4())[:8]
    # Include video/source name in the folder so it's easy to identify on disk.
    if source_type == "file" and url:
        stem  = Path(url).stem
        safe  = _re.sub(r"[^\w\-]", "_", stem)[:60].strip("_")
        folder = f"{sid}_{safe}" if safe else sid
    else:
        folder = sid
    cap_dir = ROOT / "captures" / folder
    cap_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Session: {sid} | {event_name} | prompt: {prompt} | dir: {folder}")

    # ── Camera ────────────────────────────────────────────────────────────────
    cfg = CameraConfig(
        source_type=source_type, index=index, url=url,
        width=1280, height=720, fps=fps
    )
    cam = CameraSource(cfg)
    if not cam.start():
        log.error("Failed to start camera. Exiting.")
        sys.exit(1)

    # ── Models ────────────────────────────────────────────────────────────────
    quality   = ShotQualityAnalyzer()
    gaze      = GazeDetector()
    vlm       = VLMDetector(prompt=prompt)
    predictor = MomentPredictor(fps=float(fps))
    album_gen = AlbumGenerator(str(ROOT / "albums"))

    photos    = []
    frame_n   = 0
    cap_count = 0

    # ── Shutdown handler ──────────────────────────────────────────────────────
    running = [True]
    def shutdown(sig, frame):
        log.info("Shutting down…")
        running[0] = False
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    log.info("🎬 Camera running — press Ctrl+C to stop and generate album")

    while running[0]:
        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        frame_n += 1

        # ── Run models ────────────────────────────────────────────────────────
        shot    = quality.analyze(frame)
        g       = gaze.detect(frame)
        moment  = vlm.detect(frame)
        predictor.update(shot.total, frame)
        pred    = predictor.predict()

        # ── Capture decision ──────────────────────────────────────────────────
        should_capture = False
        reasons        = []

        if quality.can_capture():
            if quality.is_best_shot(shot):               reasons.append("best_shot")
            if g.is_group_looking and g.total_faces >= 1: reasons.append("group_gaze")
            if moment.is_capture_moment:                  reasons.append(f"moment:{moment.detected_moment}")
            if pred.trend == "peak":                      reasons.append("predicted_peak")
            should_capture = bool(reasons)

        if should_capture:
            # Get best frame from preroll buffer
            best_frame, best_score = predictor.preroll.best()
            if best_frame is None:
                best_frame, best_score = frame, float(shot.total)

            # Save analysis-resolution capture
            fname = f"snap_{cap_count:04d}_{int(time.time()*1000)}.jpg"
            fpath = str(cap_dir / fname)
            cv2.imwrite(fpath, best_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # For DSLR: ALSO trigger full resolution RAW/JPEG capture
            full_path = None
            if source_type in ("dslr", "dslr_tether") and dslr_full_on_capture:
                full_path = cam.trigger_full_capture()
                if full_path:
                    log.info(f"📸 DSLR full-res capture: {full_path}")

            tags = list(set(reasons + [event_type, moment.detected_moment]))
            pe   = PhotoEntry(
                filepath=fpath,
                url=f"/captures/{sid}/{fname}",
                timestamp=float(time.time()),
                quality_score=float(best_score),
                face_count=int(shot.faces),
                emotion_score=float(shot.emotion),
                gaze_triggered=bool(g.is_group_looking),
                moment_type=str(moment.detected_moment),
                moment_conf=float(moment.confidence),
                tags=tags,
            )
            photos.append(pe)
            cap_count += 1
            quality.record_capture()
            predictor.confirm_capture()

            log.info(
                f"📸 Captured #{cap_count} | score={best_score:.2f} | "
                f"reason={reasons[0]} | moment={moment.detected_moment}"
            )

        # ── Preview window ────────────────────────────────────────────────────
        if show_preview:
            display = frame.copy()
            _draw_overlay(display, shot, g, moment, pred, cap_count)
            cv2.imshow("Snappy — Press Q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                running[0] = False

        # ── Console status every 30 frames ────────────────────────────────────
        if frame_n % 30 == 0:
            log.info(
                f"Frame {frame_n} | score={shot.total:.2f} | "
                f"faces={shot.faces} | gaze={g.gaze_ratio:.0%} | "
                f"moment={moment.detected_moment}({moment.confidence:.0%}) | "
                f"captures={cap_count}"
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────
    cam.stop()
    if show_preview:
        cv2.destroyAllWindows()

    # ── Generate album ────────────────────────────────────────────────────────
    if photos:
        log.info(f"Generating album from {len(photos)} photos…")
        album = album_gen.generate(photos, event_name, event_type, prompt)
        stats = album_gen.stats(album)
        log.info(f"Album saved: {album.total_selected}/{album.total_captured} selected")
        log.info(f"Stats: {json.dumps(stats, indent=2)}")
    else:
        log.info("No photos captured.")

    log.info("Done.")


def _draw_overlay(frame, shot, gaze, moment, pred, cap_count):
    """Draw live debug overlay on preview window."""
    h, w = frame.shape[:2]

    # Score bar
    score_color = (0,200,100) if shot.total > 0.65 else (0,180,220) if shot.total > 0.45 else (60,60,220)
    bar_w = int(shot.total * 200)
    cv2.rectangle(frame, (10, 10), (10+bar_w, 28), score_color, -1)
    cv2.rectangle(frame, (10, 10), (210, 28), (200,200,200), 1)
    cv2.putText(frame, f"Score: {shot.total:.2f}", (14, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

    # Gaze
    gaze_color = (0,200,80) if gaze.is_group_looking else (100,100,200)
    cv2.putText(frame, f"Gaze: {gaze.faces_looking}/{gaze.total_faces} ({gaze.gaze_ratio:.0%})",
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)

    # Moment
    cv2.putText(frame, f"Moment: {moment.detected_moment.replace('_',' ')} {moment.confidence:.0%}",
                (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,180,0), 1)

    # Trend
    cv2.putText(frame, f"Trend: {pred.trend}",
                (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    # Capture count
    cv2.putText(frame, f"Captures: {cap_count}",
                (w-130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,120), 2)

    # Border flash if high score
    if shot.total > 0.70:
        cv2.rectangle(frame, (3,3), (w-3,h-3), (0,220,120), 3)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Snappy Camera Runner")
    parser.add_argument("--source",  default="usb",
                        choices=["usb","dslr","dslr_tether","rtsp","ip","gopro","file"],
                        help="Camera source type")
    parser.add_argument("--index",   type=int, default=0,   help="USB camera index (default 0)")
    parser.add_argument("--url",     default="",            help="URL for RTSP/IP/file sources")
    parser.add_argument("--name",    default="My Event",    help="Event name")
    parser.add_argument("--type",    default="general",
                        choices=["wedding","birthday","party","corporate","sports","general"])
    parser.add_argument("--prompt",  default="group photo, cake cutting",
                        help="Moment detection prompt")
    parser.add_argument("--fps",     type=int, default=10,  help="Analysis FPS (default 10)")
    parser.add_argument("--no-preview", action="store_true", help="Disable preview window")
    parser.add_argument("--detect",  action="store_true",   help="Detect available cameras and exit")

    args = parser.parse_args()

    if args.detect:
        cams = detect_available_cameras()
        print("\n📷 Available cameras:")
        for k, v in cams.items():
            print(f"  {k}: {v}")
        sys.exit(0)

    run_camera_session(
        source_type  = args.source,
        index        = args.index,
        url          = args.url,
        event_name   = args.name,
        event_type   = args.type,
        prompt       = args.prompt,
        show_preview = not args.no_preview,
        fps          = args.fps,
    )
