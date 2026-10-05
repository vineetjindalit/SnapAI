"""
backend/models/event_engine.py — UNIFIED event-capture decision engine.

This is the "merged model" that combines every signal in the perception
stack into one explicit ensemble capture decision:

    final_score = w_clip   * clip_calibrated
                + w_quality* shot.total
                + w_emotion* emotion_intensity
                + w_gaze   * gaze.gaze_ratio
                + w_kept   * kept_centroid_sim
                + w_face   * normalised_face_count
                + w_aesth  * nima_score
                + w_pred   * trend_peak_indicator
                + w_learn  * learner_boost_delta

Each weight is class-conditional (cake_cutting cares about face presence
more than confetti_burst). Default weight set lives in DEFAULT_WEIGHTS;
overridable via env vars (SNAPPY_W_CLIP=0.40 etc).

Output is a CaptureDecision with:
    final_score:       0..1 ensemble score
    threshold:         class-aware threshold (calibrated if available)
    triggered:         bool — should we capture this frame?
    reasons:           ordered list of which signals contributed > 0.6
    contributions:     per-signal numeric breakdown (for /health & UI)

Design notes
------------
* Pure function of its inputs. No side effects, no DB writes.
  Capture persistence still happens in pipeline.py.
* Threadsafe — no module-level state mutated per call.
* Falls back gracefully when any input is None (model unavailable).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger("snappy.engine")


# ── Default ensemble weights ──────────────────────────────────────────────
def _f(env: str, default: float) -> float:
    try: return float(os.environ.get(env, default))
    except Exception: return default

DEFAULT_WEIGHTS: Dict[str, float] = {
    "clip":     _f("SNAPPY_W_CLIP",     0.30),  # CLIP class match (calibrated)
    "quality":  _f("SNAPPY_W_QUALITY",  0.18),  # blur+composition+brightness
    "aesthetic":_f("SNAPPY_W_AESTHETIC",0.12),  # NIMA learned aesthetic
    "emotion":  _f("SNAPPY_W_EMOTION",  0.10),  # happy/surprised intensity
    "gaze":     _f("SNAPPY_W_GAZE",     0.08),  # group gaze ratio
    "kept":     _f("SNAPPY_W_KEPT",     0.10),  # similarity to past kept frames
    "face":     _f("SNAPPY_W_FACE",     0.04),  # face count normalised
    "predictor":_f("SNAPPY_W_PREDICT",  0.04),  # rising-to-peak indicator
    "learner":  _f("SNAPPY_W_LEARNER",  0.04),  # online-learner trust delta
}

# Per-class weight overrides — for moments where particular signals matter
# more (e.g. group_photo wants face count + gaze; sports_action wants
# motion/predictor; ring_ceremony wants quality + face symmetry).
PER_CLASS_OVERRIDES: Dict[str, Dict[str, float]] = {
    "group_photo":     {"face": 0.18, "gaze": 0.18, "clip": 0.22, "emotion": 0.06},
    "ring_ceremony":   {"quality": 0.22, "clip": 0.34, "emotion": 0.06},
    "sports_action":   {"predictor": 0.18, "clip": 0.32, "quality": 0.10, "face": 0.02},
    "cake_cutting":    {"clip": 0.34, "quality": 0.20},
    "cake_with_candles": {"clip": 0.30, "face": 0.14, "quality": 0.18, "emotion": 0.08},
    "hug_moment":      {"emotion": 0.18, "face": 0.10, "clip": 0.28},
    "first_dance":     {"aesthetic": 0.20, "clip": 0.30, "quality": 0.16},
    "candle_blowing":  {"clip": 0.36, "emotion": 0.14, "quality": 0.16},
    "champagne_toast": {"face": 0.10, "gaze": 0.12, "clip": 0.30, "emotion": 0.10},
    "confetti_burst":  {"predictor": 0.18, "aesthetic": 0.18, "clip": 0.26, "face": 0.02},
    "bouquet_toss":    {"predictor": 0.16, "clip": 0.32, "quality": 0.14},
}

# Default capture threshold. Calibrator can override per-class.
# Calibrated iteratively against real event videos:
#   0.55 → too strict (0 captures from a 25-sec birthday clip)
#   0.48 → still missed mid-range frames (final ≈ 0.47)
#   0.42 → fires ~3-8 captures per minute on real event footage,
#          which matches photographer intuition. Tighten via env if
#          you start getting too many false positives.
DEFAULT_THRESHOLD = _f("SNAPPY_CAPTURE_THRESHOLD", 0.42)

# ─────────────────────────────────────────────────────────────────────────
# PRIORITY-TIER SYSTEM  (added 2026 — never-miss accuracy guarantee)
# ─────────────────────────────────────────────────────────────────────────
# Some moments must NEVER be missed regardless of thermal/battery/cooldown
# state. We classify every frame into one of four tiers and lower the
# capture threshold for higher tiers. The downstream pipeline also uses
# the tier to decide whether to bypass cooldown and frame-rate throttling.
#
#   CRITICAL — prompt-matched class + CLIP confident
#              (e.g. user typed "cake cutting" AND CLIP says cake_cutting
#               at >= 0.55) → threshold 0.28, bypasses cooldown.
#   HIGH     — subjects intentionally looking at the camera AND face
#              quality is good → threshold 0.35, bypasses cooldown.
#   ELEVATED — strong single signal (NIMA peak, emotion peak, predictor
#              "peak" trend) → threshold 0.38, normal cooldown.
#   NORMAL   — everything else → threshold 0.42, normal cooldown.
#
# Env overrides let ops staff tune without redeploys:
#   SNAPPY_THR_CRITICAL=0.28  SNAPPY_THR_HIGH=0.35
#   SNAPPY_THR_ELEVATED=0.38  SNAPPY_CAPTURE_THRESHOLD=0.42
TIER_CRITICAL = "critical"
TIER_HIGH     = "high"
TIER_ELEVATED = "elevated"
TIER_NORMAL   = "normal"

TIER_THRESHOLDS: Dict[str, float] = {
    TIER_CRITICAL: _f("SNAPPY_THR_CRITICAL", 0.28),
    TIER_HIGH:     _f("SNAPPY_THR_HIGH",     0.35),
    TIER_ELEVATED: _f("SNAPPY_THR_ELEVATED", 0.38),
    TIER_NORMAL:   DEFAULT_THRESHOLD,
}

# Tiers that the pipeline MUST honor even under thermal/battery throttle.
TIERS_BYPASS_THROTTLE = {TIER_CRITICAL, TIER_HIGH}


def _classify_priority_tier(
    *, moment_class: str, clip_per_prompt: Optional[Dict[str, float]],
    moment_confidence: float, gaze_ratio: float, face_count: int,
    emotion, predictor_result, nima_score: Optional[float],
    active_classes: Optional[List[str]], faces_looking: int = 0,
) -> str:
    """Pick the priority tier for this frame BEFORE the ensemble fires.

    The whole point: even if the ensemble score is mid-range, a prompt-
    matched cake-cutting frame should still trigger. We do this by
    lowering the threshold for higher tiers, not by overriding the
    ensemble.
    """
    # Emotional peak — strength of ANY strongly-felt emotion: joy, being MOVED /
    # tears (sad), or surprise. This is the HEART of a birthday — the human
    # reactions (grandma overwhelmed, a belly-laugh, the surprised gasp) — and it
    # matters regardless of the moment label or whether it was in the prompt.
    emo_peak = 0.0
    if emotion is not None and hasattr(emotion, "per_emotion"):
        _pe = emotion.per_emotion
        emo_peak = max(float(_pe.get("happy", 0.0)),
                       float(_pe.get("sad", 0.0)),         # moved / teary / overwhelmed
                       float(_pe.get("surprised", 0.0)))

    # CRITICAL — user-prompted class is CLIP-confirmed.
    if (active_classes and moment_class in active_classes
            and clip_per_prompt is not None
            and clip_per_prompt.get(moment_class, 0.0) >= 0.55):
        return TIER_CRITICAL

    # CRITICAL — moment_confidence is itself very strong AND the class
    # is in the active prompt list. (Catches cases where CLIP isn't
    # available but the heuristic VLM is highly confident.)
    if (active_classes and moment_class in active_classes
            and moment_confidence >= 0.65):
        return TIER_CRITICAL

    # HIGH — a GROUP deliberately posing for the camera. Never miss it, and
    # it earns a cooldown bypass. (Requires most faces looking, not just one.)
    if face_count >= 1 and gaze_ratio >= 0.65:
        return TIER_HIGH

    # HIGH — a STRONG emotional moment on a real face: never miss someone crying
    # with joy, overwhelmed, or bursting out laughing. This is the wholesome
    # heart of the event — captured no matter what "moment" it technically is.
    # (The priority floor + album dedup keep a sustained smile from flooding.)
    if face_count >= 1 and emo_peak >= 0.85:
        return TIER_HIGH

    # ELEVATED — a SINGLE person looking at the camera. Worth capturing, but
    # only at the normal cadence (ELEVATED does NOT bypass the cooldown), so a
    # person idly facing a laptop doesn't flood the gallery with near-dupes.
    if face_count >= 1 and faces_looking >= 1:
        return TIER_ELEVATED

    # ELEVATED — strong single signal even without prompt match.
    if predictor_result is not None and getattr(predictor_result, "trend", "") == "peak":
        return TIER_ELEVATED
    if nima_score is not None and nima_score >= 0.75:
        return TIER_ELEVATED
    # ELEVATED — a moderate emotional moment of ANY valence (joy / moved /
    # surprise) → worth a shot at normal cadence (respects the cooldown).
    if emo_peak >= 0.60:
        return TIER_ELEVATED

    return TIER_NORMAL


@dataclass
class CaptureDecision:
    triggered:     bool
    final_score:   float
    threshold:     float
    moment_class:  str
    reasons:       List[str]                = field(default_factory=list)
    contributions: Dict[str, float]         = field(default_factory=dict)
    weights:       Dict[str, float]         = field(default_factory=dict)
    calibrated_p:  Optional[float]          = None
    # Priority tier + flags used by pipeline.py to decide bypasses.
    priority_tier: str                      = TIER_NORMAL
    bypass_throttle: bool                   = False

    def to_dict(self) -> dict:
        return {
            "triggered":    bool(self.triggered),
            "final_score":  float(self.final_score),
            "threshold":    float(self.threshold),
            "moment_class": str(self.moment_class),
            "reasons":      list(self.reasons),
            "contributions":{k: round(float(v), 3) for k, v in self.contributions.items()},
            "weights":      {k: round(float(w), 3) for k, w in self.weights.items()},
            "calibrated_p": (round(float(self.calibrated_p), 3)
                             if self.calibrated_p is not None else None),
            "priority_tier":   str(self.priority_tier),
            "bypass_throttle": bool(self.bypass_throttle),
        }


def _resolve_weights(moment_class: str) -> Dict[str, float]:
    """Start with the default mix, layer on per-class overrides, renormalise."""
    w = dict(DEFAULT_WEIGHTS)
    overrides = PER_CLASS_OVERRIDES.get(moment_class, {})
    w.update(overrides)
    total = sum(w.values()) or 1.0
    return {k: v / total for k, v in w.items()}


def _normalise_face_count(n: int) -> float:
    """0 faces → 0, 1 → 0.4, 2 → 0.7, 3+ → 1.0 with a soft saturate."""
    if n <= 0: return 0.0
    return float(min(1.0, 0.3 + 0.25 * n))


def _emotion_intensity(emotion) -> float:
    """0..1 emotional intensity — ANY strongly-felt, wholesome emotion makes a
    birthday photo matter: JOY, being MOVED / teary (sad), or SURPRISE. (Anger
    and fear are excluded — we don't want to chase those.) This is what lets a
    crying-with-joy grandma score high enough to fire, not just a big smile."""
    if emotion is None: return 0.0
    pe = emotion.per_emotion
    felt = max(float(pe.get("happy", 0.0)),
               float(pe.get("sad", 0.0)),            # moved / teary / overwhelmed
               0.7 * float(pe.get("surprised", 0.0)))
    return float(min(1.0, felt))


def _trend_indicator(predictor_result) -> float:
    """trend == 'peak' is the strongest signal; 'rising' is partial credit."""
    if predictor_result is None: return 0.0
    t = getattr(predictor_result, "trend", "stable")
    return {"peak": 1.0, "rising": 0.6, "stable": 0.2, "falling": 0.0}.get(t, 0.0)


# ── Main entry point ──────────────────────────────────────────────────────
def decide_capture(
    *,
    moment_class:        str,
    moment_confidence:   float,
    shot_total:          float,
    nima_score:          Optional[float],
    emotion,                                   # EmotionResult-like or None
    gaze_ratio:          float,
    face_count:          int,
    clip_per_prompt:     Optional[Dict[str, float]],
    kept_centroid_sim:   Optional[float],
    predictor_result,                          # PredictionResult or None
    learner_keep_rate:   Optional[float],      # 0..1 if known, else None
    calibrator=None,
    active_classes:      Optional[List[str]] = None,
    threshold:           Optional[float] = None,
    faces_looking:       int = 0,              # # faces deliberately at camera
) -> CaptureDecision:
    """Fuse all model signals into one capture decision.

    Pure function — call this from anywhere. The pipeline calls it once
    per frame; tests can call it with synthetic inputs.
    """
    weights = _resolve_weights(moment_class)

    # Per-signal sub-scores in [0, 1]
    contrib: Dict[str, float] = {}

    # CLIP — use calibrated probability if calibrator exists
    clip_p = float(moment_confidence)
    if calibrator is not None:
        try:
            clip_p = float(calibrator.calibrate(moment_class, moment_confidence))
        except Exception:
            pass
    contrib["clip"] = clip_p

    contrib["quality"]    = float(max(0.0, min(1.0, shot_total)))
    contrib["aesthetic"]  = float(nima_score) if nima_score is not None else 0.0
    contrib["emotion"]    = _emotion_intensity(emotion)
    contrib["gaze"]       = float(max(0.0, min(1.0, gaze_ratio)))
    contrib["kept"]       = float(kept_centroid_sim) if kept_centroid_sim is not None else 0.0
    contrib["face"]       = _normalise_face_count(int(face_count))
    contrib["predictor"]  = _trend_indicator(predictor_result)

    # Online learner: only contributes a delta around 0.5 (neutral). If we
    # haven't seen feedback yet, it stays at 0.5 (no opinion).
    contrib["learner"]    = float(learner_keep_rate) if learner_keep_rate is not None else 0.5

    # Weighted sum
    final = 0.0
    for k, w in weights.items():
        final += w * contrib.get(k, 0.0)
    final = float(max(0.0, min(1.0, final)))

    # Reasons — annotate which signals contributed meaningfully. Used for
    # the UI label and capture_reason, NOT as a strict gate (the weighted
    # ensemble IS the decision). Thresholds lowered to match real-world
    # event-video score distributions:
    #   - clip lands at 0.45-0.55 for matched moments (not 0.65+)
    #   - emotion sits at 0.35-0.55 for happy-but-not-laughing scenes
    #   - face hits 0.85 only with 3+ faces in frame
    reasons: List[str] = []
    if contrib["clip"]      > 0.50: reasons.append(f"clip:{moment_class}")
    if contrib["quality"]   > 0.55: reasons.append("quality")
    if contrib["aesthetic"] > 0.55: reasons.append("aesthetic")
    if contrib["emotion"]   > 0.40: reasons.append("emotion")
    if contrib["gaze"]      > 0.45: reasons.append("group_gaze")
    if contrib["kept"]      > 0.75: reasons.append("matches_past_keepers")
    if contrib["predictor"] > 0.70: reasons.append("predicted_peak")
    if contrib["face"]      > 0.70: reasons.append("group_present")

    # ── Priority-tier classification (lowers threshold for must-capture frames)
    tier = _classify_priority_tier(
        moment_class=moment_class, clip_per_prompt=clip_per_prompt,
        moment_confidence=moment_confidence, gaze_ratio=gaze_ratio,
        face_count=face_count, emotion=emotion,
        predictor_result=predictor_result, nima_score=nima_score,
        active_classes=active_classes, faces_looking=faces_looking,
    )
    # Caller-supplied threshold wins; otherwise tier-based threshold.
    if threshold is None:
        thr = TIER_THRESHOLDS.get(tier, DEFAULT_THRESHOLD)
    else:
        thr = float(threshold)

    # Active-class gate: prefer classes the user requested, but DON'T
    # drop a strong frame just because its class isn't in the prompt.
    # Instead, demote to general_peak so the photographer still gets the
    # moment (and can mark feedback for what kind it really was).
    class_in_prompt = (active_classes is None
                       or moment_class in active_classes
                       or moment_class == "general_peak")

    effective_class = moment_class
    if not class_in_prompt:
        # Demotion: relabel as general_peak but keep all the contributions.
        # This is what makes "the kids hugging at a birthday party" still
        # get captured even when the prompt was just "cake".
        effective_class = "general_peak"
        log.debug(f"demoting '{moment_class}' to general_peak "
                  f"(not in active={active_classes})")

    # Trigger when the ensemble final is at or above threshold. We do NOT
    # require any individual signal to be "strong" — that previously
    # caused zero captures on real birthday videos where every signal sits
    # at 0.4-0.6 (none individually crossing 0.65) but the weighted sum
    # cleared 0.50. Synthesise a generic reason so capture_reason isn't
    # empty when only the ensemble fired.
    triggered = bool(final >= thr)
    if triggered and not reasons:
        reasons.append("ensemble")
    if not class_in_prompt and triggered:
        reasons.insert(0, f"demoted_from:{moment_class}")
        moment_class = effective_class

    # Stamp the priority tier into the reasons so logs + UI can see why.
    if tier != TIER_NORMAL:
        reasons.insert(0, f"priority:{tier}")

    return CaptureDecision(
        triggered=triggered, final_score=final, threshold=thr,
        moment_class=moment_class, reasons=reasons,
        contributions=contrib, weights=weights,
        calibrated_p=clip_p if calibrator is not None else None,
        priority_tier=tier,
        bypass_throttle=(tier in TIERS_BYPASS_THROTTLE),
    )
