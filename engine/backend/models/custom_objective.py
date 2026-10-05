"""
backend/models/custom_objective.py — the live capture goal for Custom Mode.

One CustomObjective represents "what the user just asked SnapAI to watch for"
(free text like "capture me when I pose"), already translated into a visual
detection objective by VLMTagger.parse_capture_intent(). A session holds at
most one at a time — a new prompt REPLACES it (see api/server.py's
_dispatch_custom_prompt).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class CustomObjective:
    raw_text:      str              # exactly what the user said/typed
    clip_prompt:   str               # what CLIP compares live frames against
    watching_for:  str               # short human label, shown in the guide
    guide:         str               # instruction text shown to the user
    detectors:     List[str] = field(default_factory=lambda: ["clip"])
    capture_mode:  str = "one_time"  # "one_time" | "continuous"
    created_ts:    float = 0.0
    fulfilled:     bool = False      # one_time: True once captured, stop watching
    # continuous: cooldown between REPEAT captures only. Must default to
    # -inf, not 0.0 — with 0.0, a fresh objective compared against the clock
    # (which also starts near 0 for a live session, or 0.0 exactly for an
    # upload) looked like a capture had *already* happened at time zero,
    # forcing a mandatory ~2.5s wait before the very first capture could ever
    # fire. Measured: a continuous "lights changing" objective sat completely
    # unresponsive for several seconds of clearly-qualifying frames.
    last_capture_ts: float = float("-inf")
