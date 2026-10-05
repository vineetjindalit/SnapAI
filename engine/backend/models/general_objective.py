"""
backend/models/general_objective.py — General Mode's live capture plan.

Unlike CustomObjective (one user-specified goal at a time), a GeneralObjective
holds a whole LIST of moments the VLM derived from just an event/occasion
NAME (models/vlm_tagger.py's derive_event_moments()) — no pre-built taxonomy
like birthday has, and no per-moment prompting from the user like Custom
Mode's one-objective-at-a-time flow needs. A session holds at most one at a
time (see api/server.py's _dispatch_custom_prompt, which General Mode
reuses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MomentSpec:
    label:         str              # short human name, e.g. "lighting diyas"
    clip_prompt:   str               # what CLIP compares live frames against
    detectors:     List[str] = field(default_factory=lambda: ["clip"])
    # Per-MOMENT cooldown (not per-objective) — different moments in the same
    # event should each be capturable independently; "family gathering" firing
    # doesn't block "lighting diyas" from firing a second later. Same -inf
    # rationale as CustomObjective.last_capture_ts: a fresh moment compared
    # against a clock starting near zero must never look like it already
    # fired at time zero.
    last_capture_ts: float = float("-inf")


@dataclass
class GeneralObjective:
    event_name:    str              # exactly what the user said, e.g. "Diwali"
    event_summary: str              # VLM's one-line description of the occasion
    moments:       List[MomentSpec]
    created_ts:    float = 0.0
