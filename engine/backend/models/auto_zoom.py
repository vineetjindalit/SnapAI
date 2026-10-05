"""
backend/models/auto_zoom.py — Decide WHETHER and WHERE to zoom a capture.

A photographer only reaches for the zoom when the subject is FAR. So does
this module. `compute_zoom_roi` returns a crop ROI **only when the subject
is small in the frame**; when the subject already fills the frame it returns
None and the caller keeps the wide shot untouched. The caller saves exactly
one photo per capture — never a wide + zoom pair (see api/pipeline.py).

The "is the subject far?" test: take the tallest detected face and compare
its height to the frame height. Below `zoom_face_fill_max` (CONFIG, default
0.16 → 16% of frame height) the subject is far enough to benefit from a
zoom; at or above it the wide shot is already a good close-up.

Once we decide to zoom, this module owns the crop geometry. Given the face
boxes and the detected moment class it returns (x, y, w, h) into the frame.

Per-class padding (applied to the face-cluster box before cropping):
  - cake_cutting / candle_blowing → expand DOWN to include the cake/candles
    that sit below the faces
  - ring_ceremony → expand DOWN slightly (hands are below faces)
  - first_dance / hug_moment → tight around the two-face cluster
  - group_photo → wide around all faces, mild padding
  - fallback → smart center crop preserving aspect

Aspect-ratio preserved (matches the source frame). Min crop guarantees
the upscaled zoom won't be lower-res than 512px on the short edge.

Pure function — no models, no state. Cheap (microseconds).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from utils.config import CONFIG

# (top, bottom, left, right) padding multipliers applied to the face-cluster
# bounding box, expressed as a fraction of the cluster's width/height.
# Tuned by hand against the typical visual composition of each moment.
_CLASS_PAD = {
    "cake_cutting":    {"top": 0.30, "bottom": 1.30, "left": 0.30, "right": 0.30},
    "candle_blowing":  {"top": 0.20, "bottom": 1.10, "left": 0.25, "right": 0.25},
    "ring_ceremony":   {"top": 0.20, "bottom": 0.90, "left": 0.30, "right": 0.30},
    "first_dance":     {"top": 0.30, "bottom": 0.50, "left": 0.40, "right": 0.40},
    "bouquet_toss":    {"top": 0.50, "bottom": 0.50, "left": 0.40, "right": 0.40},
    "group_photo":     {"top": 0.25, "bottom": 0.30, "left": 0.15, "right": 0.15},
    "champagne_toast": {"top": 0.20, "bottom": 0.70, "left": 0.30, "right": 0.30},
    "confetti_burst":  {"top": 0.60, "bottom": 0.40, "left": 0.40, "right": 0.40},
    "sports_action":   {"top": 0.40, "bottom": 0.40, "left": 0.30, "right": 0.30},
    "hug_moment":      {"top": 0.30, "bottom": 0.50, "left": 0.30, "right": 0.30},
    "general_peak":    {"top": 0.25, "bottom": 0.40, "left": 0.25, "right": 0.25},
}
_DEFAULT_PAD = {"top": 0.25, "bottom": 0.40, "left": 0.25, "right": 0.25}

# Cake/candle moments are about the CAKE, not a tight face. For these the auto
# crop frames the person AND the cake together (union with the cake region)
# instead of zooming the face — which is what was cropping the cake out when the
# subject was filmed far (a small face → a 3.9× face crop with no cake).
_CAKE_MOMENTS = {"cake_cutting", "candle_blowing", "cake_with_candles", "cake_feeding"}

# Minimum crop size on the short edge — anything smaller produces visibly
# pixelated upscales. Set to 256 so the final upscaled image is ≥ 512 even
# at 2× upscale. Don't crop smaller than this.
_MIN_SHORT_EDGE = 256


@dataclass
class ZoomROI:
    x: int
    y: int
    w: int
    h: int
    # Multiplicative zoom factor relative to the full frame (informational)
    zoom_factor: float

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h,
                "zoom_factor": round(self.zoom_factor, 2)}


def compute_zoom_roi(
    face_boxes: List[Tuple[int, int, int, int]],
    moment_class: str,
    frame_w: int,
    frame_h: int,
    face_fill_max: Optional[float] = None,
    zoom_hint: str = "auto",
) -> Optional[ZoomROI]:
    """Return a crop box (x, y, w, h) into a (frame_w, frame_h) image, or None.

    None means "don't zoom — keep the wide shot", returned when:
      * the frame is too small to zoom without degrading quality, or
      * no faces are detected (we can't judge subject distance), or
      * the subject already fills the frame (tallest face ≥ face_fill_max of
        the frame height — a close shot doesn't need a zoom), or
      * the computed crop would be ≥ 85% of the frame (a barely-a-zoom).

    `face_fill_max` defaults to CONFIG.zoom_face_fill_max; pass an explicit
    value to keep the function pure/testable.

    zoom_hint overrides
    -------------------
    "auto"         — existing subject-distance logic (default).
    "none"         — always return None (force wide shot, no zoom).
    "force_person" — zoom on tallest face regardless of face_fill_max
                     (used for multi-angle person shot in required shots).
    "cake_region"  — zoom on the cake/object region (bottom-centre of frame).
    "candles"      — zoom on the brightest blob cluster (flame area).
    """
    if zoom_hint == "none":
        return None

    if face_fill_max is None:
        face_fill_max = CONFIG.zoom_face_fill_max

    if min(frame_w, frame_h) <= _MIN_SHORT_EDGE * 2:
        return None   # frame too small to meaningfully zoom

    # ── Special hint modes ────────────────────────────────────────────────────

    if zoom_hint == "cake_region":
        return _cake_region_crop(frame_w, frame_h)

    if zoom_hint == "candles":
        # Zoom on the upper portion of the frame where candle flames sit
        return _candle_region_crop(frame_w, frame_h)

    if zoom_hint == "force_person":
        # Zoom on the tallest face even when it already fills the frame.
        if not face_boxes:
            return None
        tallest = max(face_boxes, key=lambda b: b[3])
        tx, ty, tw, th = tallest
        cx = tx + tw // 2
        cy = ty + th // 2
        # Expand by 80% around the face to include shoulders/body context
        rw = int(tw * 1.8)
        rh = int(th * 1.8)
        x1 = max(0, cx - rw // 2)
        y1 = max(0, cy - rh // 2)
        x2 = min(frame_w, x1 + rw)
        y2 = min(frame_h, y1 + rh)
        if (x2 - x1) < _MIN_SHORT_EDGE or (y2 - y1) < _MIN_SHORT_EDGE:
            return None
        zoom_factor = (frame_w / (x2 - x1) + frame_h / (y2 - y1)) / 2
        if zoom_factor < 1.05:
            return None
        return ZoomROI(x=x1, y=y1, w=x2 - x1, h=y2 - y1, zoom_factor=zoom_factor)

    # ── Standard "auto" path ──────────────────────────────────────────────────

    # 1. Find the bounding cluster of faces. No faces → we can't tell how far
    #    the subject is, so we don't zoom (keep the wide shot).
    #    Exception: cake_cutting / candle_blowing with no visible faces means
    #    the scene is viewed from far and we should zoom in on the cake.
    if not face_boxes:
        if moment_class in {"cake_cutting", "candle_blowing"}:
            return _cake_region_crop(frame_w, frame_h)
        return None

    xs1 = min(b[0]                for b in face_boxes)
    ys1 = min(b[1]                for b in face_boxes)
    xs2 = max(b[0] + b[2]         for b in face_boxes)
    ys2 = max(b[1] + b[3]         for b in face_boxes)
    cluster_w = xs2 - xs1
    cluster_h = ys2 - ys1
    if cluster_w < 30 or cluster_h < 30:
        return None

    if moment_class in _CAKE_MOMENTS:
        # Cake/candle: frame the person AND the cake together. Union the face
        # cluster with the cake region (lower-centre) so the cake is never
        # cropped out, regardless of how far/small the face is. When this union
        # fills the frame (common in a far solo shot) the >85% check below
        # returns None → a clean wide person-with-cake shot.
        x1 = min(xs1, frame_w * 0.12) - cluster_w * 0.10
        x2 = max(xs2, frame_w * 0.88) + cluster_w * 0.10
        y1 = ys1 - cluster_h * 0.30
        y2 = max(ys2, frame_h * 0.98)
    else:
        # 2. "Is the subject far?" gate. Use the TALLEST face: if even the
        #    largest face is small relative to the frame, the subject is far and
        #    a zoom helps; otherwise the subject already fills the frame and we
        #    keep the wide shot (this is what prevents a wide+zoom duplicate on
        #    every capture).
        tallest_face_h = max(b[3] for b in face_boxes)
        if (tallest_face_h / float(frame_h)) >= face_fill_max:
            return None

        # 3. Apply per-class padding
        pad = _CLASS_PAD.get(moment_class, _DEFAULT_PAD)
        x1 = xs1 - cluster_w * pad["left"]
        y1 = ys1 - cluster_h * pad["top"]
        x2 = xs2 + cluster_w * pad["right"]
        y2 = ys2 + cluster_h * pad["bottom"]

    # 4. Force-match the source aspect ratio so the upscaled image
    #    doesn't get distorted on display.
    src_ar = frame_w / frame_h
    bw, bh = x2 - x1, y2 - y1
    if bw / bh > src_ar:
        # Crop is too wide — pad height symmetrically
        target_h = bw / src_ar
        extra    = (target_h - bh) / 2
        y1 -= extra; y2 += extra
    else:
        # Crop is too tall — pad width symmetrically
        target_w = bh * src_ar
        extra    = (target_w - bw) / 2
        x1 -= extra; x2 += extra

    # 5. Clip to frame bounds (slide if needed so we keep the requested size)
    if x1 < 0: x2 -= x1; x1 = 0
    if y1 < 0: y2 -= y1; y1 = 0
    if x2 > frame_w: x1 -= (x2 - frame_w); x2 = frame_w
    if y2 > frame_h: y1 -= (y2 - frame_h); y2 = frame_h
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))

    crop_w = x2 - x1
    crop_h = y2 - y1
    if min(crop_w, crop_h) < _MIN_SHORT_EDGE:
        # The expanded ROI is still tiny — fall back to center crop so we
        # don't ship a pixelated "zoom".
        return _center_crop(frame_w, frame_h, target_zoom=1.5)

    # If the "zoom" would be barely-a-zoom (>= 90% of the frame), skip it.
    full_area  = frame_w * frame_h
    crop_area  = crop_w * crop_h
    if crop_area / full_area > 0.85:
        return None

    zoom_factor = (frame_w / crop_w + frame_h / crop_h) / 2
    return ZoomROI(x=x1, y=y1, w=crop_w, h=crop_h, zoom_factor=zoom_factor)


def _center_crop(frame_w: int, frame_h: int, target_zoom: float = 1.4) -> ZoomROI:
    """Aspect-preserved center crop at the requested zoom factor."""
    z = max(1.05, float(target_zoom))
    crop_w = int(frame_w / z)
    crop_h = int(frame_h / z)
    x1 = (frame_w - crop_w) // 2
    y1 = (frame_h - crop_h) // 2
    return ZoomROI(x=x1, y=y1, w=crop_w, h=crop_h, zoom_factor=z)


def _cake_region_crop(frame_w: int, frame_h: int) -> Optional[ZoomROI]:
    """Zoom on the cake / object region.

    A birthday cake typically sits on a table in the lower-centre of the
    frame.  We crop to the bottom 55% × centre 60% as a simple, camera-
    agnostic approximation.  Returns None when the resulting crop would be
    too small to upscale cleanly.
    """
    # Bottom 55% of frame height, centre 60% of frame width
    y1 = int(frame_h * 0.30)
    y2 = frame_h
    x1 = int(frame_w * 0.20)
    x2 = int(frame_w * 0.80)
    crop_w, crop_h = x2 - x1, y2 - y1
    if min(crop_w, crop_h) < _MIN_SHORT_EDGE:
        return None
    zoom_factor = (frame_w / crop_w + frame_h / crop_h) / 2
    if zoom_factor < 1.05:
        return None
    return ZoomROI(x=x1, y=y1, w=crop_w, h=crop_h, zoom_factor=zoom_factor)


def _candle_region_crop(frame_w: int, frame_h: int) -> Optional[ZoomROI]:
    """Zoom on the candle/flame region.

    Candles sit ON the cake → roughly upper-half of the cake region, which
    is the vertical centre of the frame.  Crops to middle 40% height ×
    centre 60% width.
    """
    y1 = int(frame_h * 0.25)
    y2 = int(frame_h * 0.75)
    x1 = int(frame_w * 0.20)
    x2 = int(frame_w * 0.80)
    crop_w, crop_h = x2 - x1, y2 - y1
    if min(crop_w, crop_h) < _MIN_SHORT_EDGE:
        return None
    zoom_factor = (frame_w / crop_w + frame_h / crop_h) / 2
    if zoom_factor < 1.05:
        return None
    return ZoomROI(x=x1, y=y1, w=crop_w, h=crop_h, zoom_factor=zoom_factor)
