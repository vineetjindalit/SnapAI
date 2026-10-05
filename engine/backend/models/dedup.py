"""
backend/models/dedup.py — Perceptual-hash near-duplicate detection.

Two layers share this module so that "duplicate" means exactly the same
thing everywhere in the system:

  1. Capture path (api/pipeline.py) — a streaming `DuplicateGuard` rejects a
     new capture when it is visually near-identical to one we just kept, so
     the captures FOLDER only ever holds distinct moments. This is what stops
     a 10-second cake-cutting from filling the folder with 20 near-identical
     frames.

  2. Album curation (models/album_generator.py) — groups the surviving photos
     and keeps the best of each near-identical cluster for the final album.

Perceptual hash (average hash / aHash):
  Resize to size×size grayscale, threshold each pixel at the image mean, and
  flatten to a boolean bit array. Two images are "near-identical" when their
  Hamming distance (number of differing bits) is small. aHash is robust to
  JPEG noise, mild exposure shifts and tiny motion — precisely the
  "same moment, adjacent frames" case we want to collapse — while still
  separating genuinely different shots.

Per-class dedup windows (v2 — fixes candle-blow misses)
-------------------------------------------------------
The original implementation stored all capture hashes in a single flat list.
For birthday clips the scene barely changes (same room, same white cake, same
family) so a lit-candle frame at t=2s and a blown-candle frame at t=10s can
have a Hamming distance of 4–6 — under the threshold — causing the blow frame
to be rejected as a duplicate of the earlier context shot.

Fix: hashes are now stored per moment-class.  A candle_blowing frame is only
compared against previous *candle_blowing* captures, not against cake_cutting
ones from 8 seconds earlier.  Window size per class is intentionally smaller
(default 8) to stay focused on the near-term shots of each class.

Required-shot bypass
--------------------
The pipeline passes `bypass_classes` — the set of moment classes whose
required shots are not yet captured.  For those classes, `is_duplicate`
returns False unconditionally: we MUST capture the first genuine instance
of a required moment even when the scene looks similar to a previous frame.

Pure functions + one small stateful guard. No heavy models, microsecond cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import os
import cv2
import numpy as np

from utils.config import CONFIG

# 8×8 hash → 64 bits. The de-facto aHash size: small enough to be cheap and
# tolerant of noise, large enough to discriminate real scene changes.
DEFAULT_HASH_SIZE = 8

# A HELD moment (a long look at the camera, a sustained pose) is the same image
# for many seconds — the per-class hash would otherwise collapse the whole thing
# to ONE capture. But the same-looking scene several seconds later is a fresh
# moment worth another frame. So a hash match OLDER than this gap is NOT treated
# as a duplicate: a long moment yields a few spaced shots, and album curation
# trims any that stay redundant. Tunable via SNAPPY_DEDUP_REFRESH_S.
_REFRESH_GAP_S = float(os.environ.get("SNAPPY_DEDUP_REFRESH_S", "4.0"))

# aHash is GRAYSCALE — it can't tell a blue cake from a pink one (same round
# shape → same hash). So two shots count as duplicates only if they ALSO match
# in colour. This is what stops a montage of different-coloured cakes collapsing
# into one. mean per-channel diff must be <= this to be "same colour".
_COLOR_THR = float(os.environ.get("SNAPPY_DEDUP_COLOR_THR", "12.0"))


def phash(img: np.ndarray, size: int = DEFAULT_HASH_SIZE) -> Optional[np.ndarray]:
    """Average-hash an image → flat boolean array of size*size bits.

    Returns None for an unreadable/empty image so callers can skip it.
    """
    if img is None or getattr(img, "size", 0) == 0:
        return None
    small = cv2.resize(img, (size, size))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return (gray > gray.mean()).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Number of differing bits between two equal-length hashes."""
    return int(np.count_nonzero(a != b))


def is_similar(a: Optional[np.ndarray], b: Optional[np.ndarray],
               threshold: int) -> bool:
    """True when both hashes exist, are the same length, and differ by
    no more than `threshold` bits."""
    if a is None or b is None or a.shape != b.shape:
        return False
    return hamming(a, b) <= threshold


def color_sig(img: np.ndarray, size: int = 6) -> Optional[np.ndarray]:
    """Coarse COLOUR signature of the SUBJECT — a size×size average-BGR grid of
    the CENTRE region (where the cake/subject sits), ignoring the background and
    frame edges. This is what lets us judge 'is it a different cake?' by the cake
    itself, not the surrounding table or on-screen text. Complements the
    grayscale aHash so differently-coloured subjects aren't seen as duplicates."""
    if img is None or getattr(img, "size", 0) == 0:
        return None
    h, w = img.shape[:2]
    cy, cx = h // 5, w // 5                     # crop to the centre ~60%
    center = img[cy:h - cy, cx:w - cx]
    if center.size == 0:
        center = img
    return cv2.resize(center, (size, size)).reshape(-1).astype(np.int16)


def color_close(a: Optional[np.ndarray], b: Optional[np.ndarray],
                thr: float = _COLOR_THR) -> bool:
    """True when two colour signatures match within `thr` mean per-channel diff."""
    if a is None or b is None or a.shape != b.shape:
        return False
    return float(np.abs(a - b).mean()) <= thr


@dataclass
class DuplicateGuard:
    """Streaming near-duplicate suppressor for the capture path.

    Hashes are stored per moment-class so that a blown-candle frame (class
    candle_blowing, t=10s) is never compared against a lit-candle frame
    (class cake_cutting, t=2s) even when the background is identical.

    When `enabled` is False every check returns False (feature switched off
    via SNAPPY_DEDUP=0), so callers need no conditional of their own.
    """
    threshold: int  = CONFIG.dedup_threshold
    window:    int  = CONFIG.dedup_window      # max hashes kept per class
    enabled:   bool = CONFIG.dedup_enabled
    # Per-class rolling windows of hashes.
    _class_hashes: Dict[str, List[np.ndarray]] = field(
        default_factory=dict, repr=False
    )
    # Per-class time of the last KEPT capture — lets a held moment refresh.
    _class_last_t: Dict[str, float] = field(default_factory=dict, repr=False)
    # Per-class colour signatures, paired 1:1 with _class_hashes.
    _class_colors: Dict[str, List[np.ndarray]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls) -> "DuplicateGuard":
        """Build a guard from the central CONFIG (the normal constructor)."""
        return cls(threshold=CONFIG.dedup_threshold,
                   window=CONFIG.dedup_window,
                   enabled=CONFIG.dedup_enabled)

    def is_duplicate(
        self,
        img: np.ndarray,
        moment_class: str = "general_peak",
        bypass_classes: Optional[Set[str]] = None,
        now: Optional[float] = None,
        color_img: Optional[np.ndarray] = None,
    ) -> bool:
        """True if `img` matches a recently-kept capture of the same class.

        Parameters
        ----------
        img           : Frame/crop to test.
        moment_class  : The VLM class assigned to this frame.  Only hashes
                        stored under the same class key are compared.
        bypass_classes: Set of classes whose required shots are not yet
                        captured.  Frames of those classes bypass dedup so
                        we never miss the first genuine required moment.
        """
        if not self.enabled:
            return False
        # Required-shot bypass: if this class still needs to fire a required
        # shot, never call it a duplicate — capture it unconditionally.
        if bypass_classes and moment_class in bypass_classes:
            return False
        h = phash(img)
        if h is None:
            return False
        csig = color_sig(color_img if color_img is not None else img)
        prev_hashes = self._class_hashes.get(moment_class, [])
        prev_colors = self._class_colors.get(moment_class, [])
        # A duplicate must match in BOTH shape (aHash) AND colour — so a blue
        # cake and a pink cake (same shape, different colour) are kept apart.
        match = any(is_similar(h, ph, self.threshold) and color_close(csig, pc)
                    for ph, pc in zip(prev_hashes, prev_colors))
        if not match:
            return False
        # Hash matches a recent kept shot — but if the moment has been HELD long
        # enough (a sustained gaze/pose), let it refresh so a long moment yields
        # a few spaced captures instead of exactly one. Album curation trims any
        # that remain redundant.
        if now is not None and moment_class in self._class_last_t:
            if (now - self._class_last_t[moment_class]) >= _REFRESH_GAP_S:
                return False
        return True

    def remember(self, img: np.ndarray, moment_class: str = "general_peak",
                 now: Optional[float] = None, color_img: Optional[np.ndarray] = None) -> None:
        """Record a kept capture so future candidates dedup against it."""
        h = phash(img)
        if h is None:
            return
        bucket = self._class_hashes.setdefault(moment_class, [])
        cbucket = self._class_colors.setdefault(moment_class, [])
        bucket.append(h); cbucket.append(color_sig(color_img if color_img is not None else img))
        if len(bucket) > self.window:
            bucket.pop(0); cbucket.pop(0)
        if now is not None:
            self._class_last_t[moment_class] = now

    def reset(self) -> None:
        """Forget all remembered hashes (e.g. when starting a new video)."""
        self._class_hashes.clear()
        self._class_colors.clear()
        self._class_last_t.clear()
