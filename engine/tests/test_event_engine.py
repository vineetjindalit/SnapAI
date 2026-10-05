"""EventCaptureEngine: unified capture decision."""
from dataclasses import dataclass
from typing import Dict

from models.event_engine import (
    DEFAULT_THRESHOLD, decide_capture, _resolve_weights,
    _normalise_face_count, _emotion_intensity, _trend_indicator,
)


@dataclass
class _FakeEmotion:
    per_emotion: Dict[str, float]
    dominant:    str = "neutral"
    score:       float = 0.0


@dataclass
class _FakePred:
    trend: str = "stable"


def test_weights_sum_to_one():
    """Per-class overrides shouldn't break normalisation."""
    for cls in ("group_photo", "ring_ceremony", "sports_action",
                "cake_cutting", "general_peak", "unknown_class"):
        w = _resolve_weights(cls)
        assert abs(sum(w.values()) - 1.0) < 1e-6, f"weights for {cls} don't sum to 1"


def test_normalise_face_count_monotonic():
    assert _normalise_face_count(0) == 0.0
    assert _normalise_face_count(1) > 0
    assert _normalise_face_count(3) > _normalise_face_count(1)
    assert _normalise_face_count(20) <= 1.0


def test_emotion_intensity():
    assert _emotion_intensity(None) == 0.0
    e = _FakeEmotion(per_emotion={"happy": 0.8, "neutral": 0.2})
    assert _emotion_intensity(e) > 0.7


def test_trend_indicator():
    assert _trend_indicator(None) == 0.0
    assert _trend_indicator(_FakePred(trend="peak")) == 1.0
    assert _trend_indicator(_FakePred(trend="rising")) > _trend_indicator(_FakePred(trend="stable"))


def test_strong_signals_trigger_capture():
    decision = decide_capture(
        moment_class="cake_cutting",
        moment_confidence=0.85,
        shot_total=0.80,
        nima_score=0.82,
        emotion=_FakeEmotion(per_emotion={"happy": 0.85}),
        gaze_ratio=0.7,
        face_count=4,
        clip_per_prompt={"cake_cutting": 0.85},
        kept_centroid_sim=0.80,
        predictor_result=_FakePred(trend="peak"),
        learner_keep_rate=0.8,
    )
    assert decision.triggered is True
    assert decision.final_score > DEFAULT_THRESHOLD
    assert "clip:cake_cutting" in decision.reasons
    assert "matches_past_keepers" in decision.reasons


def test_weak_signals_dont_trigger():
    decision = decide_capture(
        moment_class="cake_cutting",
        moment_confidence=0.3,
        shot_total=0.35,
        nima_score=0.4,
        emotion=_FakeEmotion(per_emotion={"neutral": 0.9}),
        gaze_ratio=0.2,
        face_count=0,
        clip_per_prompt={"cake_cutting": 0.3},
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="falling"),
        learner_keep_rate=None,
    )
    assert decision.triggered is False
    assert decision.final_score < DEFAULT_THRESHOLD


def test_active_class_gate():
    """Off-prompt class is demoted to general_peak — but capture is gated by
    the actual signal mix.

    2026 NOTE: With the priority-tier system, a frame that has 4 faces all
    looking at the camera (gaze=0.8) is HIGH priority — by design we DO
    capture this even when the originally-detected class isn't in the
    user's prompt list, because intentional gaze is the "never miss"
    signal. The frame still gets demoted to general_peak, which is the
    semantic guarantee the test enforces.
    """
    # Quiet off-prompt frame (no gaze, no faces) — must NOT fire.
    decision = decide_capture(
        moment_class="cake_cutting",
        moment_confidence=0.40, shot_total=0.40, nima_score=0.40,
        emotion=_FakeEmotion(per_emotion={"happy": 0.3}),
        gaze_ratio=0.0, face_count=0,
        clip_per_prompt={"cake_cutting": 0.40},
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="stable"),
        learner_keep_rate=None,
        active_classes=["ring_ceremony"],
    )
    assert decision.triggered is False

    # Loud off-prompt frame (group looking at camera) — under v2.5
    # priority tiers we DO capture this, demoted to general_peak.
    # The "never miss intentional gaze" guarantee overrides the class gate.
    decision_loud = decide_capture(
        moment_class="cake_cutting",
        moment_confidence=0.95, shot_total=0.85, nima_score=0.85,
        emotion=_FakeEmotion(per_emotion={"happy": 0.9}),
        gaze_ratio=0.8, face_count=4,
        clip_per_prompt={"cake_cutting": 0.95},
        kept_centroid_sim=0.85,
        predictor_result=_FakePred(trend="peak"),
        learner_keep_rate=0.8,
        active_classes=["ring_ceremony"],
    )
    assert decision_loud.triggered is True
    assert decision_loud.moment_class == "general_peak"   # demoted
    assert decision_loud.priority_tier in {"high", "critical"}

    # general_peak is always allowed even when not in active list.
    decision2 = decide_capture(
        moment_class="general_peak",
        moment_confidence=0.85, shot_total=0.80, nima_score=0.80,
        emotion=_FakeEmotion(per_emotion={"happy": 0.8}),
        gaze_ratio=0.7, face_count=2,
        clip_per_prompt={"general_peak": 0.85},
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="peak"),
        learner_keep_rate=None,
        active_classes=["ring_ceremony"],
    )
    assert decision2.triggered is True


def test_priority_tier_critical_for_prompt_match():
    """Prompt-matched class with CLIP confirmation → CRITICAL tier."""
    decision = decide_capture(
        moment_class="cake_cutting",
        moment_confidence=0.60, shot_total=0.50, nima_score=0.50,
        emotion=_FakeEmotion(per_emotion={"happy": 0.5}),
        gaze_ratio=0.3, face_count=1,
        clip_per_prompt={"cake_cutting": 0.70},   # CLIP confirms
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="rising"),
        learner_keep_rate=None,
        active_classes=["cake_cutting"],          # user asked for it
    )
    assert decision.priority_tier == "critical"
    assert decision.bypass_throttle is True


def test_priority_tier_high_for_intentional_gaze():
    """Subjects looking at camera (no prompt match) → HIGH tier."""
    decision = decide_capture(
        moment_class="watching",
        moment_confidence=0.40, shot_total=0.55, nima_score=0.50,
        emotion=_FakeEmotion(per_emotion={"happy": 0.5}),
        gaze_ratio=0.75, face_count=3,            # 3 people, looking at cam
        clip_per_prompt=None,
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="stable"),
        learner_keep_rate=None,
        active_classes=["cake_cutting"],
    )
    assert decision.priority_tier == "high"
    assert decision.bypass_throttle is True


def test_priority_tier_normal_for_quiet_frame():
    """No gaze, no prompt match, no signal peaks → NORMAL tier."""
    decision = decide_capture(
        moment_class="watching",
        moment_confidence=0.30, shot_total=0.40, nima_score=0.40,
        emotion=_FakeEmotion(per_emotion={"happy": 0.3}),
        gaze_ratio=0.1, face_count=0,
        clip_per_prompt=None,
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="stable"),
        learner_keep_rate=None,
        active_classes=["cake_cutting"],
    )
    assert decision.priority_tier == "normal"
    assert decision.bypass_throttle is False


def test_per_class_weights_differ():
    """group_photo should weight face count more heavily than ring_ceremony does."""
    w_group = _resolve_weights("group_photo")
    w_ring  = _resolve_weights("ring_ceremony")
    assert w_group["face"] > w_ring["face"]
    assert w_ring["clip"] > w_group["clip"]


def test_contributions_present():
    decision = decide_capture(
        moment_class="cake_cutting",
        moment_confidence=0.5, shot_total=0.5, nima_score=0.5,
        emotion=_FakeEmotion(per_emotion={"neutral": 0.9}),
        gaze_ratio=0.5, face_count=2,
        clip_per_prompt={"cake_cutting": 0.5},
        kept_centroid_sim=None,
        predictor_result=_FakePred(trend="stable"),
        learner_keep_rate=None,
    )
    assert set(decision.contributions.keys()) == {
        "clip", "quality", "aesthetic", "emotion", "gaze",
        "kept", "face", "predictor", "learner",
    }
    assert sum(decision.weights.values()) > 0
