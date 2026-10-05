"""
backend/models/category_requirements.py — Per-category capture intelligence.

Every event category ships with three things:

  1. default_prompt   — moment keywords auto-activated when that category is
                        chosen, even if the user writes nothing in the prompt
                        field.  User's own text is MERGED on top (union), so
                        the minimum requirements survive any customisation.

  2. required_shots   — the minimum distinct photos the album MUST contain.
                        The pipeline boosts priority for these until each is
                        fulfilled, so they are virtually guaranteed to be
                        captured.  A user negation ("no candle") can remove
                        a required shot — the system respects that.

  3. multi_angle_sets — certain scenes need >1 simultaneous angle (e.g.
                        cake-cutting = wide + person-zoom + cake-zoom). The
                        pipeline fires all angles in a single trigger so they
                        stay in sync.

Also provides:
  • parse_negations(prompt)          → set of excluded moment-class names
  • normalize_prompt(text)           → expand informal / vague language into
                                       standard moment vocabulary
  • merge_prompt(user_prompt, type)  → (full_prompt, excluded_classes)
    Returns the final prompt string the session should use, plus a set of
    class names that must never be captured (user explicitly excluded them).

Design notes
------------
* Birthday is fully specified by the product owner.  All other categories
  are initialised with sensible defaults and will be refined progressively.
* This module is pure-Python and import-free (no torch/cv2) so it can be
  used at session-create time without touching the heavy model stack.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


# ── RequiredShot definition ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RequiredShot:
    """One must-capture photo in a category's minimum requirements.

    Attributes
    ----------
    shot_id     : Unique slug used for tracking. Snake_case.
    label       : Human-readable label shown in the review checklist.
    moment_class: The VLM/CLIP class whose trigger satisfies this shot.
    zoom_hint   : "auto"          existing subject-distance logic (default)
                  "none"          force wide shot, no zoom
                  "force_person"  zoom on tallest/central face regardless of size
                  "cake_region"   zoom on cake/object region (bottom-centre)
                  "candles"       zoom on bright flame blobs
    is_multi_angle: True when this shot is one member of a multi-angle set
                    (e.g. the 3 shots of a cake-cutting scene).
    multi_angle_group: Shared key for all shots in the same multi-angle set.
    priority    : Override priority tier for this shot ("critical" / "high").
    """
    shot_id:           str
    label:             str
    moment_class:      str
    zoom_hint:         str  = "auto"
    is_multi_angle:    bool = False
    multi_angle_group: str  = ""
    priority:          str  = "critical"


# ── Per-category profiles ─────────────────────────────────────────────────────

@dataclass
class CategoryProfile:
    """Everything the system needs to know about an event category."""
    # Comma-separated moment keywords auto-seeded into the prompt.
    default_prompt: str
    # Ordered list of required shots for this category.
    required_shots: List[RequiredShot]
    # Moment classes that are always active for this category (regardless of
    # what the user types) unless they explicitly negate them.
    always_active_classes: List[str]


# ── Birthday ──────────────────────────────────────────────────────────────────

_BIRTHDAY_REQUIRED: List[RequiredShot] = [
    # ── Cake shots ─────────────────────────────────────────────────────────
    RequiredShot("cake_wide",         "Cake — wide scene",           "cake_cutting",  "none"),
    RequiredShot("cake_closeup",      "Cake — close-up",             "cake_cutting",  "cake_region"),

    # ── Cake-with-candles (the gathering / singing phase, candles lit) ─────
    # This is "everyone in front of the lit cake" — a DIFFERENT moment from the
    # actual blow. models/candle_phase.py keeps the two apart. Captured as a
    # multi-angle pair so we always get a wide scene AND a zoomed-in close-up
    # of the cake itself, even on a clip that never reaches the blow.
    RequiredShot("cake_with_candles", "Cake with candles lit — wide",
                 "cake_with_candles", "none",
                 is_multi_angle=True, multi_angle_group="candles_scene"),
    RequiredShot("cake_with_candles_zoom", "Cake with candles — close-up",
                 "cake_with_candles", "cake_region",
                 is_multi_angle=True, multi_angle_group="candles_scene"),

    # ── Candle BLOWING multi-angle (the actual blow: person + flames + wide) ─
    # Only fires when someone is really blowing (leaning in / flames going out),
    # so a crowd merely standing at the cake never fills these.
    RequiredShot("candle_blow_person","Person blowing candles — close","candle_blowing", "force_person",
                 is_multi_angle=True, multi_angle_group="candle_scene"),
    RequiredShot("candle_blown",      "Candles being blown — close",  "candle_blowing", "candles",
                 is_multi_angle=True, multi_angle_group="candle_scene"),
    RequiredShot("candle_blow_wide",  "Candle blowing — full scene",  "candle_blowing", "none",
                 is_multi_angle=True, multi_angle_group="candle_scene"),

    # ── Cake-cutting multi-angle (person + knife + wide) ───────────────────
    RequiredShot("cake_cut_person",   "Person cutting cake — close",  "cake_cutting", "force_person",
                 is_multi_angle=True, multi_angle_group="cake_cut_scene"),
    RequiredShot("cake_cut_knife",    "Knife cutting cake — close",   "cake_cutting", "cake_region",
                 is_multi_angle=True, multi_angle_group="cake_cut_scene"),
    RequiredShot("cake_cut_wide",     "Cake cutting — full scene",    "cake_cutting", "none",
                 is_multi_angle=True, multi_angle_group="cake_cut_scene"),

    # ── Detail / social shots ──────────────────────────────────────────────
    RequiredShot("food_table",        "Food / snacks table",           "food_table",   "none"),
    RequiredShot("cake_feeding",      "Person feeding cake to someone", "cake_feeding", "auto"),
    RequiredShot("hug",               "Hug / embrace",                 "hug_moment",   "auto"),
    RequiredShot("group_photo",       "Group photo (any gaze)",        "group_photo",  "none"),
    RequiredShot("gazing",            "Someone looking at the camera",  "gazing_moments", "auto"),
    RequiredShot("emotional_moment",  "Emotional moment (smile/tears)","general_peak", "auto"),
]

_BIRTHDAY_CLASSES = [
    "cake_cutting", "candle_blowing", "cake_with_candles", "group_photo",
    "hug_moment", "cake_feeding", "clapping_scene", "food_table", "gazing_moments",
    "gift_box_reveal",
    "birthday_gifting",
    # A toast is a KEY celebration moment at birthdays too — without this it
    # was wedding-only, and the album dropped a captured birthday toast as
    # "not birthday-relevant".
    "champagne_toast",
]

_BIRTHDAY_PROFILE = CategoryProfile(
    default_prompt=(
        "cake cutting, candle blowing, cake with candles, group photo, hug, "
        "cake feeding, clapping, birthday cake, birthday candle, "
        "opening gifts, birthday gift, surprise box, toast, snacks, food, table spread"
    ),
    required_shots=_BIRTHDAY_REQUIRED,
    always_active_classes=_BIRTHDAY_CLASSES,
)


# ── Wedding ───────────────────────────────────────────────────────────────────

_WEDDING_REQUIRED: List[RequiredShot] = [
    RequiredShot("ring_exchange",    "Ring exchange",              "ring_ceremony",   "force_person"),
    RequiredShot("couple_kiss",      "Couple kiss / first kiss",   "hug_moment",      "force_person"),
    RequiredShot("first_dance",      "First dance",                "first_dance",     "auto"),
    RequiredShot("bouquet_toss",     "Bouquet toss",               "bouquet_toss",    "auto"),
    RequiredShot("wedding_toast",    "Toast / champagne",          "champagne_toast", "auto"),
    RequiredShot("wedding_group",    "Group photo",                "group_photo",     "none"),
    RequiredShot("wedding_emotion",  "Emotional moment",           "general_peak",    "auto"),
]

_WEDDING_PROFILE = CategoryProfile(
    default_prompt=(
        "ring ceremony, first dance, bouquet toss, champagne toast, "
        "kiss, hug, group photo, wedding"
    ),
    required_shots=_WEDDING_REQUIRED,
    always_active_classes=[
        "ring_ceremony", "first_dance", "bouquet_toss",
        "champagne_toast", "hug_moment", "group_photo",
    ],
)


# ── Graduation ────────────────────────────────────────────────────────────────

_GRADUATION_REQUIRED: List[RequiredShot] = [
    RequiredShot("diploma_handshake", "Diploma handshake moment",  "hug_moment",   "force_person"),
    RequiredShot("cap_toss",          "Cap toss",                  "bouquet_toss", "auto"),
    RequiredShot("grad_group",        "Group photo / family",      "group_photo",  "none"),
    RequiredShot("grad_emotion",      "Emotional moment",          "general_peak", "auto"),
    RequiredShot("grad_hug",          "Family hug",                "hug_moment",   "auto"),
]

_GRADUATION_PROFILE = CategoryProfile(
    default_prompt="graduation, diploma, cap toss, group photo, hug, celebration",
    required_shots=_GRADUATION_REQUIRED,
    always_active_classes=["hug_moment", "bouquet_toss", "group_photo"],
)


# ── Corporate / conference ────────────────────────────────────────────────────

_CORPORATE_REQUIRED: List[RequiredShot] = [
    RequiredShot("speaker",           "Speaker / presenter",       "general_peak",  "force_person"),
    RequiredShot("award_handshake",   "Award handshake",           "hug_moment",    "force_person"),
    RequiredShot("audience_applause", "Audience applause",         "clapping_scene","none"),
    RequiredShot("corporate_group",   "Team photo",                "group_photo",   "none"),
]

_CORPORATE_PROFILE = CategoryProfile(
    default_prompt="speaker, award, applause, team photo, presentation, group photo",
    required_shots=_CORPORATE_REQUIRED,
    always_active_classes=["group_photo", "clapping_scene", "hug_moment"],
)


# ── Anniversary ───────────────────────────────────────────────────────────────

_ANNIVERSARY_PROFILE = CategoryProfile(
    default_prompt="cake cutting, toast, hug, couple, group photo, candle",
    required_shots=[
        RequiredShot("anniv_couple",  "Couple moment",     "hug_moment",      "force_person"),
        RequiredShot("anniv_toast",   "Toast",             "champagne_toast", "auto"),
        RequiredShot("anniv_cake",    "Cake",              "cake_cutting",    "none"),
        RequiredShot("anniv_group",   "Group photo",       "group_photo",     "none"),
    ],
    always_active_classes=["hug_moment", "champagne_toast", "cake_cutting", "group_photo"],
)


# ── General (fallback) ────────────────────────────────────────────────────────

_GENERAL_PROFILE = CategoryProfile(
    default_prompt="capture moments, group photo",
    required_shots=[
        RequiredShot("general_group", "Group photo",       "group_photo",  "none"),
        RequiredShot("general_peak",  "Highlight moment",  "general_peak", "auto"),
    ],
    always_active_classes=["group_photo", "general_peak"],
)


# ── Custom (no predefined knowledge — the user defines the objective live) ─────
# Deliberately EMPTY, not a copy of _GENERAL_PROFILE: Custom Mode starts with
# zero event-specific moment knowledge. The real capture objective is set
# later, mid-session, via a free-text/voice prompt (see api/server.py's
# _dispatch_custom_prompt + api/pipeline.py's _evaluate_custom_objective) —
# not from anything category-level. Registered explicitly here (rather than
# silently falling back to "general") so this never accidentally inherits
# whatever "general" happens to contain later.
_CUSTOM_PROFILE = CategoryProfile(
    default_prompt="",
    required_shots=[],
    always_active_classes=[],
)


# ── Smart Event / "General" in the UI (no predefined knowledge — the VLM
# derives the moment list live from just the event's NAME) ────────────────────
# Deliberately EMPTY, same reasoning as Custom Mode above: the real moment
# vocabulary is derived per-event at session start (see models/vlm_tagger.py's
# derive_event_moments() + api/server.py's _dispatch_general_event). Note
# this key is "smart_event", NOT "general" — "general" above is a distinct,
# pre-existing fallback profile (also the silent default for any unrecognized
# event_type via get_profile() below); reusing that string for this feature
# would have hijacked every session that legitimately falls back to it.
_SMART_EVENT_PROFILE = CategoryProfile(
    default_prompt="",
    required_shots=[],
    always_active_classes=[],
)


# ── Registry ──────────────────────────────────────────────────────────────────

CATEGORY_PROFILES: Dict[str, CategoryProfile] = {
    "birthday":    _BIRTHDAY_PROFILE,
    "wedding":     _WEDDING_PROFILE,
    "graduation":  _GRADUATION_PROFILE,
    "corporate":   _CORPORATE_PROFILE,
    "conference":  _CORPORATE_PROFILE,
    "anniversary": _ANNIVERSARY_PROFILE,
    "general":     _GENERAL_PROFILE,
    "custom":      _CUSTOM_PROFILE,
    "smart_event": _SMART_EVENT_PROFILE,
}


def get_profile(event_type: str) -> CategoryProfile:
    """Return the CategoryProfile for an event type, defaulting to general."""
    return CATEGORY_PROFILES.get((event_type or "").lower().strip(),
                                  _GENERAL_PROFILE)


# ── Negative-keyword parser ───────────────────────────────────────────────────

# Words that signal a negation when they precede a subject.
_NEGATION_PREFIXES = (
    "no ", "no-", "don't ", "dont ", "do not ", "do-not ",
    "skip ", "without ", "ignore ", "not ", "exclude ",
    "avoid ", "stop ", "remove ", "cancel ",
)

# Map from plain-English fragment → set of moment classes to exclude.
_NEGATION_CLASS_MAP: Dict[str, Set[str]] = {
    "candle":          {"candle_blowing"},
    "candle blown":    {"candle_blowing"},
    "candle blow":     {"candle_blowing"},
    "blowing":         {"candle_blowing"},
    "cake":            {"cake_cutting", "cake_feeding"},
    "cake cut":        {"cake_cutting"},
    "cutting":         {"cake_cutting"},
    "feeding":         {"cake_feeding"},
    "feed":            {"cake_feeding"},
    "clapping":        {"clapping_scene"},
    "clap":            {"clapping_scene"},
    "applause":        {"clapping_scene"},
    "group photo":     {"group_photo"},
    "group":           {"group_photo"},
    "hug":             {"hug_moment"},
    "embrace":         {"hug_moment"},
    "dance":           {"first_dance"},
    "dancing":         {"first_dance"},
    "ring":            {"ring_ceremony"},
    "bouquet":         {"bouquet_toss"},
    "toast":           {"champagne_toast"},
    "champagne":       {"champagne_toast"},
    "confetti":        {"confetti_burst"},
    "sport":           {"sports_action"},
    "sports":          {"sports_action"},
    "emotional":       {"general_peak"},
    "emotion":         {"general_peak"},
}


def parse_negations(prompt: str) -> Set[str]:
    """Return the set of moment-class names the user explicitly excluded.

    Examples:
        "candle blowing, no clapping" → {"clapping_scene"}
        "skip group photo, don't capture hug" → {"group_photo", "hug_moment"}
        "no cake and no candle" → {"cake_cutting", "cake_feeding", "candle_blowing"}
    """
    excluded: Set[str] = set()
    text = prompt.lower()
    for prefix in _NEGATION_PREFIXES:
        idx = 0
        while True:
            pos = text.find(prefix, idx)
            if pos == -1:
                break
            # Grab the subject fragment after the prefix (up to 4 words)
            rest = text[pos + len(prefix):]
            fragment_words = re.split(r"[,;.\n]+|\band\b", rest)[0].strip().split()
            # Try progressively shorter fragments (longest first for specificity)
            for n in range(min(4, len(fragment_words)), 0, -1):
                fragment = " ".join(fragment_words[:n]).strip()
                if fragment in _NEGATION_CLASS_MAP:
                    excluded |= _NEGATION_CLASS_MAP[fragment]
                    break
            idx = pos + 1
    return excluded


# ── Prompt normaliser ─────────────────────────────────────────────────────────

# Map informal/vague phrases → canonical moment keywords understood by the
# VLM and PromptRouter. Checked as sub-string matches (case-insensitive).
_SYNONYM_TABLE: List[Tuple[str, str]] = [
    # Birthday-specific
    ("birthday girl",      "cake cutting, candle blowing"),
    ("birthday boy",       "cake cutting, candle blowing"),
    ("birthday kid",       "cake cutting, candle blowing"),
    ("birthday person",    "cake cutting, candle blowing"),
    ("blow candles",       "candle blowing"),
    ("blowing candles",    "candle blowing"),
    ("wish",               "candle blowing"),
    ("make a wish",        "candle blowing"),
    ("cutting the cake",   "cake cutting"),
    ("smash cake",         "cake cutting"),
    ("smash the cake",     "cake cutting"),
    ("eat cake",           "cake feeding"),
    ("feed cake",          "cake feeding"),
    ("eating cake",        "cake feeding"),
    ("happy birthday",     "cake cutting, candle blowing, group photo"),
    ("birthday party",     "cake cutting, candle blowing, group photo, hug"),
    ("party",              "group photo, celebration"),
    ("celebrate",          "group photo, cake cutting"),
    ("celebration",        "group photo, general peak"),
    # Social
    ("everyone",           "group photo"),
    ("all together",       "group photo"),
    ("family photo",       "group photo"),
    ("team photo",         "group photo"),
    ("selfie",             "group photo"),
    ("photo together",     "group photo"),
    ("memories",           "group photo, general peak"),
    ("happy moments",      "general peak, group photo"),
    ("fun moments",        "general peak, group photo"),
    ("best moments",       "general peak"),
    ("capture everything", "general peak, group photo"),
    ("everything",         "general peak, group photo"),
    # Wedding
    ("exchange rings",     "ring ceremony"),
    ("say i do",           "ring ceremony"),
    ("vows",               "ring ceremony"),
    ("first kiss",         "hug"),
    ("first dance",        "dance"),
    # Emotions
    ("smile",              "general peak"),
    ("laughing",           "general peak"),
    ("crying",             "general peak"),
    ("tears",              "general peak"),
    ("emotional",          "general peak"),
]


def normalize_prompt(text: str) -> str:
    """Expand informal / vague user text into standard moment vocabulary.

    Idempotent: applying twice gives the same result as once. Preserves any
    text not matched by the synonym table so nothing is lost.

    Examples:
        "birthday girl, also everyone together" →
            "cake cutting, candle blowing, group photo, general peak"
        "smash cake and make a wish" →
            "cake cutting, candle blowing"
    """
    if not text:
        return text
    result = text.lower().strip()
    added: List[str] = []
    for phrase, expansion in _SYNONYM_TABLE:
        if phrase in result:
            # Replace the matched phrase with its expansion in-place
            result = result.replace(phrase, expansion)
            # Guard: don't double-add if the expansion keywords were already there
    # Deduplicate comma-separated tokens while preserving order
    seen: Set[str] = set()
    tokens: List[str] = []
    for tok in re.split(r"[,;]+", result):
        tok = tok.strip()
        if tok and tok not in seen:
            seen.add(tok)
            tokens.append(tok)
    return ", ".join(tokens)


# ── Prompt merger ─────────────────────────────────────────────────────────────

def merge_prompt(
    user_prompt: str,
    event_type: str,
) -> Tuple[str, Set[str], List[RequiredShot]]:
    """Produce the final effective prompt for a session.

    Returns
    -------
    effective_prompt : str
        Union of the category default and the user's normalised prompt, with
        negated items stripped out.  This is the text passed to VLMDetector
        and PromptRouter.
    excluded_classes : set[str]
        Moment classes the user explicitly excluded.  The pipeline will never
        capture frames whose primary moment class is in this set.
    required_shots : list[RequiredShot]
        Category minimum shots, minus any the user negated.
    """
    profile = get_profile(event_type)

    # 1. Normalise user input
    normalised_user = normalize_prompt(user_prompt or "")

    # 2. Parse negations BEFORE merging so they propagate to the defaults too
    excluded = parse_negations(normalised_user)
    # Also check the raw user prompt for negations (in case normalise rewrote them)
    excluded |= parse_negations(user_prompt or "")

    # 3. Build union prompt: category defaults first, user additions appended
    base_tokens: List[str] = [t.strip() for t in profile.default_prompt.split(",") if t.strip()]
    user_tokens: List[str] = [t.strip() for t in normalised_user.split(",") if t.strip()]

    # Remove negated keywords from both lists
    negated_keywords: Set[str] = set()
    for cls in excluded:
        for key, classes in _NEGATION_CLASS_MAP.items():
            if cls in classes:
                negated_keywords.add(key)

    def _keep(tok: str) -> bool:
        tok_l = tok.lower()
        # Drop tokens that ARE a negation phrase themselves
        for prefix in _NEGATION_PREFIXES:
            if tok_l.startswith(prefix):
                return False
        # Drop tokens whose keyword was negated
        for nk in negated_keywords:
            if nk in tok_l:
                return False
        return True

    merged_tokens: List[str] = []
    seen: Set[str] = set()
    for tok in base_tokens + user_tokens:
        if _keep(tok) and tok not in seen:
            seen.add(tok)
            merged_tokens.append(tok)

    effective_prompt = ", ".join(merged_tokens)

    # 4. Filter required shots: remove any whose moment_class was excluded
    active_required = [
        shot for shot in profile.required_shots
        if shot.moment_class not in excluded
    ]

    return effective_prompt, excluded, active_required
