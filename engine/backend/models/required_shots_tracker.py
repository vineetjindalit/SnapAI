"""
backend/models/required_shots_tracker.py — Per-session required-shot checklist.

Each session has a RequiredShotsTracker that:

  1. Holds the list of must-capture shots defined by the category.
  2. Marks shots as captured as the pipeline fires them.
  3. Tells the pipeline how much priority boost to give an uncaptured shot
     so the ensemble is less likely to skip it.
  4. Detects when a multi-angle scene is in progress (e.g. cake-cutting
     with 3+ people) and tells the pipeline how many angles to fire.
  5. Provides a summary for the post-album review UI.

Design notes
------------
* Fully in-memory, no I/O.  Reset on new video (same as DuplicateGuard).
* Priority boost is additive on top of the ensemble final_score.  The boost
  decays to zero once a shot is captured so the engine returns to normal
  behaviour.
* "Multi-angle detection": when `cake_cut_scene` has 3+ members AND all
  three are still uncaptured AND the frame shows 3+ faces → return all three
  zoom_hint values so the pipeline saves three photos in one trigger.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models.category_requirements import RequiredShot


# Priority boost added to the engine final_score for uncaptured required shots.
# 0.15 is enough to push a borderline ELEVATED frame over the CRITICAL threshold
# without forcing a capture when the frame is genuinely bad.
_REQUIRED_BOOST = 0.15
# After a required shot is captured the boost drops to zero immediately.
_CAPTURED_BOOST = 0.0

# Multi-angle normally needs a crowd (≥3 faces) — e.g. "5 people clapping while
# one cuts" gives a meaningful person/wide split. But the cake/candle scenes are
# about the CAKE, not the crowd: a solo birthday still deserves a wide
# person-with-cake shot AND a zoomed-in cake/candle close-up. So these classes
# fire their multi-angle set from a single subject.
_LOW_FACE_MULTI_CLASSES = {
    "cake_with_candles", "candle_blowing", "cake_cutting", "cake_feeding",
}
_MULTI_MIN_FACES_DEFAULT = 3
_MULTI_MIN_FACES_CAKE    = 1


@dataclass
class ShotStatus:
    """Live status of one required shot."""
    shot:       RequiredShot
    captured:   bool  = False
    capture_ts: float = 0.0
    capture_url: Optional[str] = None


class RequiredShotsTracker:
    """Track which required shots have been captured for a session.

    Parameters
    ----------
    required_shots : list[RequiredShot]
        The active required shots for this session (from category_requirements
        .merge_prompt — already filtered for user negations).
    """

    def __init__(self, required_shots: List[RequiredShot]) -> None:
        self._shots: Dict[str, ShotStatus] = {
            s.shot_id: ShotStatus(shot=s) for s in required_shots
        }
        # Index: moment_class → shot_ids that depend on it
        self._class_to_shots: Dict[str, List[str]] = {}
        for s in required_shots:
            self._class_to_shots.setdefault(s.moment_class, []).append(s.shot_id)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_uncaptured_classes(self) -> Set[str]:
        """Set of moment classes that still have at least one uncaptured required shot."""
        return {
            st.shot.moment_class
            for st in self._shots.values()
            if not st.captured
        }

    def priority_boost(self, moment_class: str) -> float:
        """Return the additive score boost for this moment class.

        Non-zero only when the class has at least one uncaptured required shot.
        """
        shot_ids = self._class_to_shots.get(moment_class, [])
        if not shot_ids:
            return _CAPTURED_BOOST
        uncaptured = [sid for sid in shot_ids if not self._shots[sid].captured]
        return _REQUIRED_BOOST if uncaptured else _CAPTURED_BOOST

    def needs_multi_angle(self, moment_class: str, face_count: int) -> bool:
        """True when all conditions for a multi-angle burst are met.

        A multi-angle burst fires when:
          • the moment class has ≥ 2 uncaptured multi-angle required shots, and
          • enough faces are visible — ≥3 for crowd scenes, but ≥1 for the
            cake/candle scenes (a solo birthday still wants wide + cake-zoom).
        """
        min_faces = (_MULTI_MIN_FACES_CAKE if moment_class in _LOW_FACE_MULTI_CLASSES
                     else _MULTI_MIN_FACES_DEFAULT)
        if face_count < min_faces:
            return False
        uncaptured_multi = [
            sid for sid, st in self._shots.items()
            if not st.captured
            and st.shot.is_multi_angle
            and st.shot.moment_class == moment_class
        ]
        return len(uncaptured_multi) >= 2

    def get_multi_angle_zoom_hints(self, moment_class: str) -> List[str]:
        """Return zoom_hints for all uncaptured multi-angle shots of this class.

        Called by the pipeline when `needs_multi_angle` is True; iterates over
        these hints to fire one sub-capture per hint.
        """
        hints = []
        for sid, st in self._shots.items():
            if not st.captured and st.shot.is_multi_angle and st.shot.moment_class == moment_class:
                hints.append(st.shot.zoom_hint)
        return hints

    def has_uncaptured(self, moment_class: str) -> bool:
        """True if ANY required shot for this class is not yet captured."""
        return bool([
            sid for sid, st in self._shots.items()
            if not st.captured and st.shot.moment_class == moment_class
        ])

    # ── Capture recording ─────────────────────────────────────────────────────

    def on_captured(
        self,
        moment_class: str,
        zoom_hint_used: str = "auto",
        capture_url: str = "",
        face_count: int = 0,
    ) -> List[str]:
        """Mark the appropriate required shot(s) as captured.

        Matching logic (priority order):
          1. If zoom_hint_used matches a multi-angle shot exactly → mark that one.
          2. Otherwise mark the first uncaptured shot for this moment_class.

        Returns the list of shot_ids that were just marked captured.
        """
        shot_ids = self._class_to_shots.get(moment_class, [])
        if not shot_ids:
            return []

        now = time.time()
        marked: List[str] = []

        # Priority 1: exact zoom_hint match (for multi-angle shots)
        for sid in shot_ids:
            st = self._shots[sid]
            if (not st.captured
                    and st.shot.is_multi_angle
                    and st.shot.zoom_hint == zoom_hint_used):
                st.captured    = True
                st.capture_ts  = now
                st.capture_url = capture_url
                marked.append(sid)

        # Priority 2: first uncaptured non-multi-angle for this class
        if not marked:
            for sid in shot_ids:
                st = self._shots[sid]
                if not st.captured and not st.shot.is_multi_angle:
                    st.captured    = True
                    st.capture_ts  = now
                    st.capture_url = capture_url
                    marked.append(sid)
                    break  # one at a time for non-multi shots

        # Priority 3: any uncaptured shot (catches edge cases)
        if not marked:
            for sid in shot_ids:
                st = self._shots[sid]
                if not st.captured:
                    st.captured    = True
                    st.capture_ts  = now
                    st.capture_url = capture_url
                    marked.append(sid)
                    break

        return marked

    def reset(self) -> None:
        """Forget all captured state (e.g. starting a new video on same session)."""
        for st in self._shots.values():
            st.captured    = False
            st.capture_ts  = 0.0
            st.capture_url = None

    # ── Summary for review UI ─────────────────────────────────────────────────

    def to_summary(self) -> Dict:
        """Return a JSON-safe summary for the post-album review panel."""
        shots_list = []
        for sid, st in self._shots.items():
            shots_list.append({
                "shot_id":     sid,
                "label":       st.shot.label,
                "moment_class": st.shot.moment_class,
                "captured":    st.captured,
                "capture_url": st.capture_url or None,
            })
        total     = len(shots_list)
        captured  = sum(1 for s in shots_list if s["captured"])
        return {
            "total_required": total,
            "captured":       captured,
            "missed":         total - captured,
            "shots":          shots_list,
        }
