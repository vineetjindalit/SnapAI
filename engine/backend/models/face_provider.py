"""
models/face_provider.py — single MediaPipe FaceLandmarker, ONE call per frame.

This is the heart of the upgraded perception stack:
  - 478 landmarks per face (incl. iris points 468-477) → precise gaze
  - 52 blendshapes per face (mouthSmileLeft, eyeSquint, browInnerUp, ...) → emotion
  - Bounding boxes derived from landmarks → replaces Haar everywhere

Anyone who needs face info (shot_quality, gaze, emotion) consumes the same
FaceFrame produced once per video frame. Falls back to OpenCV Haar cascade
boxes if MediaPipe isn't installed, so the system never breaks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("snappy.face_provider")

_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")

# MediaPipe iris landmark indices (with refine_landmarks/iris support)
_LEFT_IRIS  = [468, 469, 470, 471, 472]
_RIGHT_IRIS = [473, 474, 475, 476, 477]
_LEFT_EYE_CORNERS  = (33, 133)    # outer, inner
_RIGHT_EYE_CORNERS = (362, 263)   # inner, outer
_LEFT_EYE_TOP_BOT  = (159, 145)
_RIGHT_EYE_TOP_BOT = (386, 374)
_NOSE_TIP = 1
_CHIN     = 152


@dataclass
class FaceInfo:
    box:        Tuple[int, int, int, int]   # (x, y, w, h)
    landmarks:  np.ndarray                  # (478, 3) normalized
    blendshapes: Dict[str, float] = field(default_factory=dict)

    # Derived signals
    iris_l:     Optional[np.ndarray] = None  # (2,) normalized iris center
    iris_r:     Optional[np.ndarray] = None
    eye_open_l: float = 1.0                  # eye-aspect-ratio like
    eye_open_r: float = 1.0
    looking_at_camera: float = 0.5           # 0 = looking away, 1 = direct

    def smile(self) -> float:
        """Derived smile intensity from blendshapes."""
        if not self.blendshapes:
            return 0.0
        l = self.blendshapes.get("mouthSmileLeft", 0.0)
        r = self.blendshapes.get("mouthSmileRight", 0.0)
        return float(min(1.0, (l + r) / 1.4))


@dataclass
class FaceFrame:
    """Result for an entire frame. Always returned, even if no faces."""
    available:  bool
    backend:    str                  # "mediapipe" | "haar" | "none"
    faces:      List[FaceInfo]
    error:      Optional[str] = None

    @property
    def count(self) -> int:
        return len(self.faces)

    @property
    def boxes(self) -> List[Dict[str, int]]:
        return [
            {"x": f.box[0], "y": f.box[1], "w": f.box[2], "h": f.box[3]}
            for f in self.faces
        ]


class FaceProvider:
    """Singleton wrapper over MediaPipe FaceLandmarker (with blendshapes + iris)."""

    _instance: Optional["FaceProvider"] = None
    _lock = Lock()

    def __init__(self):
        self._mp = None
        self._landmarker = None
        self._init_error: Optional[str] = None
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._try_load_mediapipe()
        # Optional YOLOv8-face: if available, used as a high-recall second
        # pass for crowded scenes when MediaPipe finds <3 faces.
        self._yolo = None
        try:
            from .yolo_face import YOLOFaceDetector
            y = YOLOFaceDetector.get()
            if y.available:
                self._yolo = y
                log.info("YOLOv8-face available as crowd booster")
        except Exception as e:
            log.debug(f"YOLOv8-face not loaded: {e}")

    @classmethod
    def get(cls) -> "FaceProvider":
        with cls._lock:
            if cls._instance is None:
                cls._instance = FaceProvider()
            return cls._instance

    def _try_load_mediapipe(self) -> None:
        if not _MODEL_PATH.exists():
            self._init_error = f"face_landmarker.task missing at {_MODEL_PATH}"
            log.warning(self._init_error)
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as e:
            self._init_error = f"mediapipe not installed: {e}"
            log.warning(self._init_error + " — falling back to Haar")
            return

        try:
            base_options = mp_python.BaseOptions(model_asset_path=str(_MODEL_PATH))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=False,
                num_faces=8,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._mp = mp
            log.info("MediaPipe FaceLandmarker ready (blendshapes + iris, num_faces=8)")
        except Exception as e:
            self._init_error = f"FaceLandmarker init failed: {e}"
            log.exception(self._init_error)

    @property
    def backend(self) -> str:
        if self._landmarker is not None and self._yolo is not None:
            return "mediapipe+yolo"
        if self._landmarker is not None:
            return "mediapipe"
        if self._yolo is not None:
            return "yolo"
        return "haar"

    # ── Main entrypoint ───────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> FaceFrame:
        if frame is None or frame.size == 0:
            return FaceFrame(available=False, backend="none", faces=[],
                             error="empty frame")
        if self._landmarker is not None:
            try:
                ff = self._detect_mediapipe(frame)
                # ALWAYS merge YOLO boxes when available — YOLO has higher
                # recall on small / profile / occluded faces, MediaPipe has
                # better landmarks. The merge keeps MediaPipe's landmark-rich
                # entries and only adds YOLO faces that don't overlap an
                # existing MediaPipe one.
                if self._yolo is not None:
                    yolo_boxes = self._yolo.detect(frame)
                    self._merge_yolo_boxes(ff, yolo_boxes)
                return ff
            except Exception as e:
                log.exception("MediaPipe detect failed; falling back")
                if self._yolo is not None:
                    return self._detect_yolo(frame, error=str(e))
                return self._detect_haar(frame, error=str(e))
        if self._yolo is not None:
            return self._detect_yolo(frame)
        return self._detect_haar(frame)

    def _merge_yolo_boxes(self, ff: "FaceFrame", yolo_boxes) -> None:
        """Add YOLO-detected faces that don't overlap existing MediaPipe boxes."""
        def iou(a, b) -> float:
            ax, ay, aw, ah = a; bx, by, bw, bh = b
            x1 = max(ax, bx); y1 = max(ay, by)
            x2 = min(ax+aw, bx+bw); y2 = min(ay+ah, by+bh)
            inter = max(0, x2-x1) * max(0, y2-y1)
            uni   = aw*ah + bw*bh - inter
            return inter / uni if uni > 0 else 0.0
        for yb in yolo_boxes:
            if all(iou(yb, f.box) < 0.3 for f in ff.faces):
                ff.faces.append(FaceInfo(
                    box=yb, landmarks=np.zeros((1, 3), dtype=np.float32),
                    blendshapes={}, eye_open_l=1.0, eye_open_r=1.0,
                    looking_at_camera=0.5,
                ))

    def _detect_yolo(self, frame: np.ndarray, error: Optional[str] = None) -> FaceFrame:
        boxes = self._yolo.detect(frame) if self._yolo else []
        faces = [FaceInfo(box=b, landmarks=np.zeros((1, 3), dtype=np.float32),
                          blendshapes={}, eye_open_l=1.0, eye_open_r=1.0,
                          looking_at_camera=0.5) for b in boxes]
        return FaceFrame(available=True, backend="yolo", faces=faces, error=error)

    # ── MediaPipe path ────────────────────────────────────────────────────────
    def _detect_mediapipe(self, frame: np.ndarray) -> FaceFrame:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)

        faces: List[FaceInfo] = []
        face_lm_lists = result.face_landmarks or []
        blend_lists   = result.face_blendshapes or []

        for i, lm_list in enumerate(face_lm_lists):
            arr = np.array([[lm.x, lm.y, lm.z] for lm in lm_list], dtype=np.float32)
            if arr.shape[0] < 468:
                continue

            xs = np.clip(arr[:, 0], 0.0, 1.0) * w
            ys = np.clip(arr[:, 1], 0.0, 1.0) * h
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()), int(ys.max())
            box = (max(x0, 0), max(y0, 0), max(x1 - x0, 1), max(y1 - y0, 1))

            blend: Dict[str, float] = {}
            if i < len(blend_lists):
                for cat in blend_lists[i]:
                    blend[cat.category_name] = float(cat.score)

            face = FaceInfo(box=box, landmarks=arr, blendshapes=blend)
            self._fill_eye_metrics(face)
            faces.append(face)

        return FaceFrame(available=True, backend="mediapipe", faces=faces)

    def _fill_eye_metrics(self, face: FaceInfo) -> None:
        lm = face.landmarks
        if lm.shape[0] < 478:
            face.iris_l = face.iris_r = None
            face.looking_at_camera = 0.5
            return

        face.iris_l = lm[_LEFT_IRIS].mean(axis=0)[:2]
        face.iris_r = lm[_RIGHT_IRIS].mean(axis=0)[:2]

        # Eye openness (vertical / horizontal)
        def _open(top_idx, bot_idx, l_idx, r_idx) -> float:
            vert = float(np.linalg.norm(lm[top_idx, :2] - lm[bot_idx, :2]))
            horz = float(np.linalg.norm(lm[l_idx, :2] - lm[r_idx, :2])) + 1e-6
            return float(min(1.0, vert / horz / 0.35))

        face.eye_open_l = _open(*_LEFT_EYE_TOP_BOT, *_LEFT_EYE_CORNERS)
        face.eye_open_r = _open(*_RIGHT_EYE_TOP_BOT, *_RIGHT_EYE_CORNERS)

        # "Looking at camera" — iris should be near horizontal/vertical center of eye.
        def _gaze_score(iris_xy, l_corner_idx, r_corner_idx, top_idx, bot_idx) -> float:
            l = lm[l_corner_idx, :2]
            r = lm[r_corner_idx, :2]
            t = lm[top_idx, :2]
            b = lm[bot_idx, :2]
            cx = (l[0] + r[0]) * 0.5
            cy = (t[1] + b[1]) * 0.5
            half_w = abs(r[0] - l[0]) * 0.5 + 1e-6
            half_h = abs(b[1] - t[1]) * 0.5 + 1e-6
            dx = (iris_xy[0] - cx) / half_w
            dy = (iris_xy[1] - cy) / half_h
            # 0 if iris dead-center, → 1 as it drifts
            d = float(min(1.0, (dx * dx + dy * dy) ** 0.5))
            return float(max(0.0, 1.0 - d))

        gl = _gaze_score(face.iris_l, *_LEFT_EYE_CORNERS,  *_LEFT_EYE_TOP_BOT)
        gr = _gaze_score(face.iris_r, *_RIGHT_EYE_CORNERS, *_RIGHT_EYE_TOP_BOT)
        eye_open = (face.eye_open_l + face.eye_open_r) * 0.5
        # Penalise closed eyes — can't tell if they're looking
        face.looking_at_camera = float(max(0.0, min(1.0, (gl + gr) * 0.5 * (0.4 + 0.6 * eye_open))))

    # ── Haar fallback ─────────────────────────────────────────────────────────
    def _detect_haar(self, frame: np.ndarray, error: Optional[str] = None) -> FaceFrame:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(32, 32))
        faces = []
        for (x, y, w, h) in boxes:
            arr = np.zeros((1, 3), dtype=np.float32)
            faces.append(FaceInfo(
                box=(int(x), int(y), int(w), int(h)),
                landmarks=arr,
                blendshapes={},
                eye_open_l=1.0, eye_open_r=1.0,
                looking_at_camera=0.5,
            ))
        return FaceFrame(available=True, backend="haar", faces=faces, error=error)
