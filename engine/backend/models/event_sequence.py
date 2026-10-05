"""
backend/models/event_sequence.py — the model's KNOWLEDGE OF HOW AN EVENT FLOWS.

A real photographer knows a birthday's rhythm before it starts: people arrive,
the surprise, the cake comes out, candles → blow → cut → feed, gifts, group
photo, then dancing and candids. SnapAI should know that too. When the user
picks/writes an event name, we load that event's canonical moment ORDER and use
it two ways:

  1. ALBUM AS A STORY — order the final album by event stage (then by time
     within a stage), so it reads chronologically the way the event unfolded,
     not in random capture order.

  2. CAPTURE PRIOR (foundation) — knowing candle_blowing follows
     cake_with_candles and precedes cake_cutting lets the pipeline disambiguate
     visually-similar moments by WHERE we are in the event, not pixels alone.

Pure-Python, import-free (no torch/cv2) so any layer can use it cheaply.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical moment order per event type. Index = stage number (earlier = lower).
# Birthday is fully specified (the product owner's sequence). Social/emotional
# moments sit after the structured cake sequence — that's where they're grouped
# in the album story; within a stage, photos stay in chronological order.
EVENT_SEQUENCES: Dict[str, List[str]] = {
    "birthday": [
        "pre_preparation",        # setup, decorations (before guests)
        "food_table",             # snacks / food spread (often beside the cake)
        "person_arrival",         # birthday person / guests arrive
        "surprise_celebration",   # the surprise reveal
        "cake",                   # cake brought out (cake alone)
        "cake_person",            # birthday person with the cake
        "cake_with_candles",      # gathered at the lit cake
        "candle_blowing",         # the blow
        "cake_cutting",           # cutting
        "cake_smashing",          # smash (if it happens)
        "cake_feeding",           # feeding each other
        "birthday_gifting",       # gifts
        "group_photo",            # the posed group
        "dancing_moments",
        "hugging_moments",
        "laughing_moments",
        "smiling_moments",
        "crying_moments",
        "gazing_moments",         # someone looking at camera
        "individual_people",      # solo portraits
        "general_peak",           # other nice candids
    ],
    # Lightweight starters for other events (refined later, same as birthday was).
    "wedding": [
        "pre_preparation", "person_arrival", "ring_ceremony", "varmala",
        "vows", "first_kiss", "first_dance", "cake_cutting", "champagne_toast",
        "group_photo", "hugging_moments", "laughing_moments", "smiling_moments",
        "crying_moments", "gazing_moments", "individual_people", "general_peak",
    ],
}

# Free-text event name / type → canonical key.
_ALIASES = {
    "birthday": "birthday", "bday": "birthday", "b'day": "birthday",
    "birthday party": "birthday", "bday party": "birthday",
    "birthdays": "birthday", "happy birthday": "birthday",
    "wedding": "wedding", "marriage": "wedding", "shaadi": "wedding",
    "reception": "wedding", "engagement": "wedding",
}

# A moment whose stage is unknown sorts AFTER all known stages (large index),
# then by timestamp — so we never lose a photo, it just lands at the end.
_UNKNOWN_STAGE = 9_999
UNKNOWN_STAGE = _UNKNOWN_STAGE   # public alias for callers (pipeline prior)


def resolve_event(name_or_type: str) -> str:
    """Map an arbitrary event name/type to a canonical sequence key (or "")."""
    s = (name_or_type or "").strip().lower()
    if not s:
        return ""
    if s in EVENT_SEQUENCES:
        return s
    if s in _ALIASES:
        return _ALIASES[s]
    # Substring match — "sara's birthday 2026" → birthday.
    for key in EVENT_SEQUENCES:
        if key in s:
            return key
    for alias, key in _ALIASES.items():
        if alias in s:
            return key
    return ""


def sequence(event: str) -> List[str]:
    """Canonical ordered moment list for an event (empty if unknown)."""
    return EVENT_SEQUENCES.get(resolve_event(event), [])


def stage_index(event: str, moment: str) -> int:
    """Position of `moment` in the event's sequence; _UNKNOWN_STAGE if absent."""
    seq = sequence(event)
    try:
        return seq.index(moment)
    except ValueError:
        return _UNKNOWN_STAGE


def order_key(event: str, moment: str, timestamp: float) -> Tuple[int, float]:
    """Sort key for arranging photos as the event's story: stage, then time."""
    return (stage_index(event, moment), float(timestamp or 0.0))


def expected_neighbours(event: str, moment: str) -> Tuple[str, str]:
    """(previous, next) canonical moments around `moment` — for capture priors.

    Returns ("", "") when unknown. Lets the pipeline reason about plausibility:
    candle_blowing is plausible right after cake_with_candles, implausible
    before anyone has even arrived.
    """
    seq = sequence(event)
    if moment not in seq:
        return ("", "")
    i = seq.index(moment)
    prev = seq[i - 1] if i > 0 else ""
    nxt = seq[i + 1] if i + 1 < len(seq) else ""
    return (prev, nxt)
