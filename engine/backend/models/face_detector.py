"""
MediaPipe face detection/landmark wrapper for backend use.

The original version of this file was copied from a notebook demo, which
meant it recreated the detector on every frame and tried to display results
inside the API process. This version keeps the detector reusable and returns
plain Python data that the rest of the backend can consume.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
_MEDIAPIPE_IMPORT_ERROR: Optional[str] = None
_MEDIAPIPE_MODULE = None
_PYTHON_TASKS = None
_VISION_TASKS = None
_FACE_LANDMARKER = None


def _load_mediapipe_modules() -> Tuple[object, object, object]:
    global _MEDIAPIPE_MODULE, _PYTHON_TASKS, _VISION_TASKS, _MEDIAPIPE_IMPORT_ERROR

    if _MEDIAPIPE_MODULE is not None:
        return _MEDIAPIPE_MODULE, _PYTHON_TASKS, _VISION_TASKS

    if _MEDIAPIPE_IMPORT_ERROR is not None:
        raise RuntimeError(_MEDIAPIPE_IMPORT_ERROR)

    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
    except ImportError as exc:
        _MEDIAPIPE_IMPORT_ERROR = (
            "MediaPipe is not installed. Run `pip install mediapipe` in the "
            "same Python environment that starts the server."
        )
        raise RuntimeError(_MEDIAPIPE_IMPORT_ERROR) from exc

    _MEDIAPIPE_MODULE = mp
    _PYTHON_TASKS = python
    _VISION_TASKS = vision
    return _MEDIAPIPE_MODULE, _PYTHON_TASKS, _VISION_TASKS


def _get_landmarker():
    global _FACE_LANDMARKER

    if _FACE_LANDMARKER is not None:
        return _FACE_LANDMARKER

    mp, python, vision = _load_mediapipe_modules()
    del mp  # Imported for consistency with detect_face; not needed here.

    if not _MODEL_PATH.exists():
        raise RuntimeError(f"MediaPipe model file not found: {_MODEL_PATH}")

    base_options = python.BaseOptions(model_asset_path=str(_MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _FACE_LANDMARKER = vision.FaceLandmarker.create_from_options(options)
    return _FACE_LANDMARKER


def draw_landmarks_on_image(
    rgb_image: np.ndarray,
    face_landmarks_list: List[object],
) -> np.ndarray:
    """Return an RGB copy with MediaPipe mesh overlays drawn."""
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2

    annotated_image = np.copy(rgb_image)

    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend(
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in face_landmarks
        )

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_contours_style(),
        )
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=solutions.drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


def _landmarks_to_box(face_landmarks: List[object], width: int, height: int) -> Dict[str, int]:
    xs = [min(max(float(lm.x), 0.0), 1.0) for lm in face_landmarks]
    ys = [min(max(float(lm.y), 0.0), 1.0) for lm in face_landmarks]

    if not xs or not ys:
        return {"x": 0, "y": 0, "w": 0, "h": 0}

    min_x = int(min(xs) * width)
    min_y = int(min(ys) * height)
    max_x = int(max(xs) * width)
    max_y = int(max(ys) * height)

    return {
        "x": max(min_x, 0),
        "y": max(min_y, 0),
        "w": max(max_x - min_x, 0),
        "h": max(max_y - min_y, 0),
    }


def detect_face(frame: np.ndarray, draw: bool = False) -> Dict[str, object]:
    """
    Detect faces in an OpenCV BGR frame using MediaPipe Face Landmarker.

    Returns a JSON-safe dict:
      {
        "available": bool,
        "face_count": int,
        "boxes": [{"x":..,"y":..,"w":..,"h":..}, ...],
        "error": str | None,
        "annotated_frame": np.ndarray | None,
      }
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return {
            "available": False,
            "face_count": 0,
            "boxes": [],
            "error": "Empty frame passed to detect_face().",
            "annotated_frame": None,
        }

    try:
        mp, _, _ = _load_mediapipe_modules()
        detector = _get_landmarker()
    except Exception as exc:
        return {
            "available": False,
            "face_count": 0,
            "boxes": [],
            "error": str(exc),
            "annotated_frame": None,
        }

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    try:
        detection_result = detector.detect(image)
    except Exception as exc:
        log.exception("MediaPipe face detection failed")
        return {
            "available": False,
            "face_count": 0,
            "boxes": [],
            "error": f"MediaPipe detection failed: {exc}",
            "annotated_frame": None,
        }

    frame_h, frame_w = frame.shape[:2]
    face_landmarks_list = detection_result.face_landmarks or []
    boxes = [
        _landmarks_to_box(face_landmarks, frame_w, frame_h)
        for face_landmarks in face_landmarks_list
    ]

    annotated_frame = None
    if draw and face_landmarks_list:
        annotated_rgb = draw_landmarks_on_image(rgb_frame, face_landmarks_list)
        annotated_frame = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

    return {
        "available": True,
        "face_count": len(boxes),
        "boxes": boxes,
        "error": None,
        "annotated_frame": annotated_frame,
    }
