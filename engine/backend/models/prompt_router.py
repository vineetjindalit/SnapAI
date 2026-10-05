"""
backend/models/prompt_router.py — live prompt state machine.

The user can change what to capture mid-event by typing OR speaking. This
module owns ALL prompt logic so the rest of the system just asks
"which classes are active right now?".

Voice flow (live):
    Browser Web Speech API ─▶ WS msg {"type":"voice_transcript","text":...}
                          ─▶ PromptRouter.update(transcript)
                          ─▶ session.vlm.update_prompt(...)

Text flow:
    POST /sessions/<sid>/prompt {"prompt":"also catch confetti"}
    ─▶ PromptRouter.update(...)

Intent inference (no LLM, deterministic):
  - Tokenize transcript, match against keywords in MOMENT_PROFILES
  - Detect "intent verbs":
      add | also  → APPEND classes
      stop | not | except | ignore → REMOVE classes
      switch | only | now → REPLACE classes
      (none of above) → REPLACE if matches found
  - Always keeps "general_peak" as floor

Returns a transition record so the UI can show what changed.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

log = logging.getLogger("snappy.prompt")

# Tokens that change intent. Order matters — first match wins per category.
_ADD_TOKENS    = {"add", "also", "include", "plus", "and"}
_REMOVE_TOKENS = {"stop", "not", "except", "ignore", "skip", "remove"}
_REPLACE_TOKENS = {"switch", "only", "now", "change to", "instead"}


@dataclass
class PromptTransition:
    ts:        float
    text:      str
    intent:    str                 # "add" | "remove" | "replace" | "noop"
    matched:   List[str]
    active:    List[str]
    note:      str = ""

    def to_dict(self) -> dict:
        return {
            "ts": self.ts, "text": self.text, "intent": self.intent,
            "matched": self.matched, "active": self.active, "note": self.note,
        }


@dataclass
class PromptState:
    sid:           str
    raw_prompt:    str = ""
    active_classes: Set[str] = field(default_factory=set)
    history:       List[PromptTransition] = field(default_factory=list)


class PromptRouter:
    """One PromptRouter per server. Stores PromptState per session.

    When clip_engine is supplied, novel phrases that don't match any
    known class get auto-registered as new CLIP prompts on the fly,
    so the user can extend Snappy's vocabulary mid-event:
        "also catch saree pulling" → adds class "saree_pulling" to CLIP.
    """

    def __init__(self, moment_profiles: Dict[str, dict], clip_engine=None):
        self.profiles = moment_profiles
        self.clip = clip_engine
        self.state: Dict[str, PromptState] = {}
        # Track classes we discovered from user prompts at runtime so we
        # don't re-add them on every prompt update.
        self._dynamic_classes: Set[str] = set()

    def init_session(self, sid: str, prompt: str) -> PromptState:
        st = PromptState(sid=sid, raw_prompt=prompt)
        st.active_classes = self._classes_from_text(prompt) | {"general_peak"}
        self.state[sid] = st
        log.info(f"[{sid}] prompt init: {sorted(st.active_classes)}")
        return st

    def get(self, sid: str) -> Optional[PromptState]:
        return self.state.get(sid)

    def end_session(self, sid: str) -> None:
        self.state.pop(sid, None)

    # ── Core update ───────────────────────────────────────────────────────────
    def update(self, sid: str, text: str) -> Optional[PromptTransition]:
        if sid not in self.state:
            log.warning(f"[{sid}] update called before init")
            return None
        text_low = (text or "").strip().lower()
        if not text_low:
            return None

        intent = self._infer_intent(text_low)
        matched = sorted(self._classes_from_text(text_low))

        # No known class matched? Try to coin a new one from the user's words.
        # This is what makes Snappy's vocabulary extensible at runtime:
        # the photographer says "also catch saree pulling" and we add a
        # new CLIP prompt for it immediately.
        if not matched and self.clip is not None:
            coined = self._coin_class(text_low)
            if coined:
                matched = [coined]

        st = self.state[sid]
        before = set(st.active_classes)

        if intent == "add" and matched:
            st.active_classes |= set(matched)
            note = f"added {matched}"
        elif intent == "remove" and matched:
            st.active_classes -= set(matched)
            note = f"removed {matched}"
        elif matched:    # replace
            intent = "replace"
            st.active_classes = set(matched)
            note = f"switched to {matched}"
        else:
            intent = "noop"
            note = "no known moment keywords"

        # Floor: always keep general_peak so we never silently disable everything
        st.active_classes |= {"general_peak"}
        st.raw_prompt = text

        transition = PromptTransition(
            ts=time.time(), text=text, intent=intent, matched=matched,
            active=sorted(st.active_classes), note=note,
        )
        st.history.append(transition)
        if len(st.history) > 50:
            st.history.pop(0)

        if before != st.active_classes:
            log.info(f"[{sid}] prompt {intent}: '{text}' → active={sorted(st.active_classes)}")

        return transition

    # ── Intent + keyword inference ───────────────────────────────────────────
    def _infer_intent(self, text: str) -> str:
        for tok in _REPLACE_TOKENS:
            if tok in text: return "replace"
        for tok in _ADD_TOKENS:
            if re.search(rf"\b{re.escape(tok)}\b", text): return "add"
        for tok in _REMOVE_TOKENS:
            if re.search(rf"\b{re.escape(tok)}\b", text): return "remove"
        return "replace"   # default for bare keywords

    # ── Dynamic class minting ────────────────────────────────────────────
    def _coin_class(self, text: str) -> Optional[str]:
        """Take the user's free-form text, strip filler verbs, and create
        a new CLIP class id + prompt. Returns the class id or None."""
        # Strip intent words so "also catch saree pulling" → "saree pulling"
        all_filler = (_ADD_TOKENS | _REMOVE_TOKENS | _REPLACE_TOKENS |
                      {"catch", "look", "for", "the", "a", "an", "watch"})
        words = [w for w in re.split(r"[^a-z]+", text) if w and w not in all_filler]
        if not words: return None
        cls_id = "_".join(words[:4])    # cap class name length
        if cls_id in self._dynamic_classes or cls_id in self.profiles:
            return cls_id
        # Build a CLIP-friendly prompt
        prompt_text = "a photograph of " + " ".join(words[:6])
        try:
            ok = self.clip.add_prompt(cls_id, prompt_text)
        except Exception as e:
            log.warning(f"add_prompt raised: {e}"); return None
        if not ok:
            return None
        self._dynamic_classes.add(cls_id)
        # Register a minimal profile so heuristic VLM at least keyword-matches
        self.profiles[cls_id] = {
            "keywords": words[:4], "color_cues": [],
            "blob_bright": False, "motion_band": (0.0, 1.0),
            "edge_level": 0.0, "face_needed": 0, "weight": 0.7,
        }
        log.info(f"coined new dynamic class '{cls_id}' from text '{text}'")
        return cls_id

    def _classes_from_text(self, text: str) -> Set[str]:
        text = text.lower()
        out: Set[str] = set()
        for cls, prof in self.profiles.items():
            for kw in prof.get("keywords", []):
                if kw and kw in text:
                    out.add(cls)
                    break
        return out
