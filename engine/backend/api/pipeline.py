"""
backend/api/pipeline.py — per-frame ML orchestration + worker pool.

Responsibilities:
  - process_frame(session, frame_b64, frame=None): run all models, decide
    whether to capture, build the JSON-safe response. PURE function of
    its inputs aside from session-scoped state.
  - process_frame_async(session, frame_b64): same, dispatched to a
    ThreadPoolExecutor so the asyncio loop never blocks on CLIP/NIMA
    (~80 ms each on CPU).
  - process_video_async(session, path, sample_fps): batch-process an
    uploaded video file through the same pipeline.

Threading model:
  * One ThreadPoolExecutor (size = SNAPPY_FRAME_WORKERS, default 4).
  * The model singletons (FACE_PROVIDER, CLIP, NIMA, HSEMO) hold their
    own internal locks where the underlying lib isn't threadsafe (CLIP/
    transformers is, MediaPipe FaceLandmarker.detect() is — both have
    been verified by reading the upstream source).
  * Per-Session mutable state (frame_count, photos[], gallery[]) is only
    written from the worker that owns that frame, so no cross-frame
    races inside one session.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import numpy as np

from models.album_generator     import PhotoEntry
from models.candle_phase        import (classify_candle_phase, count_warm_flames,
                                        PHASE_BLOWING, PHASE_GATHERING)
from models.cake_smash          import classify_cake_smash, CAKE_SMASH
from models.birthday_moment_classifier import BirthdayMomentClassifier
from models.event_sequence      import (resolve_event as _resolve_event,
                                        stage_index as _stage_index,
                                        UNKNOWN_STAGE as _UNKNOWN_STAGE)
from models.emotion_detector    import detect_emotions
from models.event_engine        import (decide_capture, TIER_NORMAL, TIER_ELEVATED,
                                         TIER_HIGH, TIER_CRITICAL)
from models.auto_zoom           import compute_zoom_roi
from models.dedup               import phash as _phash, hamming as _ham, color_sig as _csig
from models.setup_advisor       import advise as _setup_advise, locate_event as _setup_locate
from models.enhance             import enhance
from models.vlm_tagger           import verify_moment as _vlm_verify_moment
from utils.safe_types           import safe

from api.globals import (
    album_gen, AUDIO, AUTO_CLASS, CALIBRATOR, CAPTURES_DIR, CLIP, FACE_PROVIDER,
    HSEMO, LEARNER, L2CS, NIMA, SESSIONS, STORE, log,
)
from models.audio_event_detector import (audio_capture_boost, celebration_intensity,
                                          audio_priority_tier, birthday_audio_awareness)
from models.audio_extract        import extract_audio_16k_mono

# ── Worker pool ───────────────────────────────────────────────────────────
_FRAME_WORKERS = int(os.environ.get("SNAPPY_FRAME_WORKERS", "4"))
_executor: ThreadPoolExecutor = ThreadPoolExecutor(
    max_workers=_FRAME_WORKERS,
    thread_name_prefix="snappy-frame",
)
log.info(f"frame pipeline ready (workers={_FRAME_WORKERS})")

# When CLIP's best OFF-prompt class outscores the best ACTIVE class by this
# margin consistently across recent frames, the scene's real moment isn't in
# the user's prompt (e.g. a champagne toast at a birthday). Label as
# general_peak instead of inventing a wrong specific label.
# Measured gap: real cake/candle ≤ +0.002, toast/cups +0.005…+0.016.
_OFF_PROMPT_MARGIN = float(os.environ.get("SNAPPY_OFF_PROMPT_MARGIN", "0.004"))
# Gentle confidence nudge for a moment that's sequence-plausible at the current
# point in the event. Small + boost-only — a prior, never a hard rule.
_SEQ_PRIOR_BONUS   = float(os.environ.get("SNAPPY_SEQ_PRIOR_BONUS", "0.06"))
# Moment labels that a deliberate look-at-camera may override with
# gazing_moments — only truly generic frames (a gaze during a specific moment
# like cake_with_candles keeps that richer label but still captures via HIGH tier).
_GAZE_GENERIC = {"general_peak", "watching", ""}

# Detail / establishing shots: birthday OBJECTS worth a photo even with NOBODY
# in frame (the cake, the gift, the decorated table). When CLIP confidently sees
# one of these and there are no faces, we still take a modest-priority shot.
_OBJECT_SCENES = {"cake_with_candles", "cake", "cake_person", "birthday_gifting",
                  "gift_box_reveal",
                  "food_table", "cake_cutting", "cake_feeding"}
_OBJECT_SHOT_CONF = float(os.environ.get("SNAPPY_OBJECT_SHOT_CONF", "0.55"))
# Faceless object shots must beat the best BACKGROUND contrast prompt by this
# margin. Measured: junk (pool table) = +0.015..0.048, real cakes/gifts =
# +0.078..0.098 → 0.06 cleanly separates with a gap on both sides.
_OBJECT_BG_MARGIN = float(os.environ.get("SNAPPY_OBJECT_BG_MARGIN", "0.06"))
# Per-class evidence bars (override the global). birthday_gifting false-fires at
# ~0.06 on a baby swaddled in a patterned blanket (reads as a wrapped gift); a
# "best babies reaction" compilation armed the whole gate this way. Measured:
# baby swaddle peaks at 0.062 and NEVER reaches 0.065; real gift reveals run
# 0.066–0.088. So gifting alone must clear a higher bar to prove an event.
_EVIDENCE_BAR = {"birthday_gifting": 0.065}
def _ev_bar(cls: str) -> float:
    return _EVIDENCE_BAR.get(cls, _OBJECT_BG_MARGIN)
# A frame visually FAR from the last kept capture (Hamming on the 8×8 hash) is a
# genuinely new moment (a gift box → its open contents) — it bypasses the
# cooldown so each distinct stage gets a shot. Per-class dedup still blocks dups.
_NEW_SCENE_DIST = int(os.environ.get("SNAPPY_NEW_SCENE_DIST", "18"))

# Classes that count as EVENT EVIDENCE (arm the capture gate): event-specific
# CONTENT only. People-generic classes (person_arrival, group_photo, hugs,
# smiles…) appear in any video of humans — they may be captured AT an event,
# but they can never PROVE one is happening.
_EVIDENCE_CLASSES = {
    "gift_box_reveal",
    "cake", "cake_person", "cake_with_candles", "birthday_gifting",
    "food_table", "champagne_toast",
    # OBJECT-anchored classes only. NOT included:
    #  • surprise_celebration / pre_preparation — scene-"vibe" classes that
    #    match any emotive people or decorated room (anniversary car reel).
    #  • ACTION classes (candle_blowing, cake_cutting/feeding/smashing) —
    #    they match the MOTION alone: a mom feeding her baby armed evidence
    #    as cake_feeding; a confetti ad as candle_blowing. A real cake action
    #    always shows the CAKE too, so the object classes arm instead.
}
# …or a big COLOUR jump from the last kept shot (a different-coloured cake). Small
# changes (a hand moving in frame) stay below this, so they don't bypass.
_NEW_SCENE_COLOR = float(os.environ.get("SNAPPY_NEW_SCENE_COLOR", "45.0"))


def shutdown_workers() -> None:
    _executor.shutdown(wait=False)


# ── Custom Mode ─────────────────────────────────────────────────────────────
# No predefined event knowledge — the user's own free-text/voice request IS
# the entire moment vocabulary (see models/custom_objective.py, and
# api/server.py's _dispatch_custom_prompt which turns the request into a
# CustomObjective via VLMTagger.parse_capture_intent()).
#
# Deliberately a SEPARATE, self-contained decision path rather than a change
# to the birthday flow below: decide_capture() is heavily birthday-tuned
# (per-class weights, a calibrator trained on birthday feedback) and has no
# meaningful weights for an arbitrary user-described moment. Reuses what's
# already generic and already computed every frame regardless of event type:
# face/emotion/gaze detection, CLIP scoring, session.predictor's rolling
# preroll buffer + motion signal, and session._fullres_ring — see the plan
# this was built from for the full reuse rationale.
# NOT an absolute threshold on the custom class's raw CLIP score — measured
# directly (scratchpad probe, 2026-08-24): a real photo of someone NOT
# matching the description and a "xyzzy nonsense unmatched impossible
# prompt" both scored 0.58-0.63 for ANY text prompt against ANY image, on
# THIS model. That's the same lesson as tonight's bg_* prompt fix for
# birthday mode: CLIP's raw per-prompt similarity sits in a narrow,
# almost content-independent band — only the MARGIN over a generic
# "background/nothing-in-particular" contrast set carries real signal.
# Reuses the SAME bg_* prompts already in clip_engine.py's
# DEFAULT_MOMENT_PROMPTS (they're scene-generic, not birthday-specific).
_CUSTOM_CLIP_MARGIN     = float(os.environ.get("SNAPPY_CUSTOM_CLIP_MARGIN",     "0.03"))
_CUSTOM_MOTION_SPIKE    = float(os.environ.get("SNAPPY_CUSTOM_MOTION_SPIKE",    "0.35"))
_CUSTOM_LIGHT_JUMP      = float(os.environ.get("SNAPPY_CUSTOM_LIGHT_JUMP",      "35.0"))
_CUSTOM_CONTINUOUS_COOLDOWN_S = 2.5   # min gap between repeat captures in "continuous" mode

# General Mode: per-MOMENT cooldown (not per-objective, since one objective
# holds several independent moments) — longer than Custom's continuous
# cooldown because a whole event runs far longer than one held pose, and
# repeatedly re-capturing e.g. "family gathering" every ~3s would flood the
# album with near-duplicates over a multi-hour event.
_GENERAL_MOMENT_COOLDOWN_S = float(os.environ.get("SNAPPY_GENERAL_MOMENT_COOLDOWN", "8.0"))

# These two gates are a CHEAP PRE-FILTER only — _vlm_verify_moment() below is
# what actually decides correctness by looking at the frame. They exist so
# obviously-nothing-happening frames never cost a VLM call, not to be the
# accuracy backstop themselves.
#
# Calibration history, both measured live: raising these to 0.05/0.02 (from
# 0.03/uncapped) fixed the false-positive case (sid cb0c0a14, "Diwali" in a
# bedroom — near-uniform 0.048-0.086 scores saved the same wrong frame 5x).
# But that same bar then starved real content of ever reaching the VLM check
# at all — sid f57adf5a ("diwali", live frames actually shown to the camera)
# produced exactly ONE candidate in the whole session. CLIP's raw similarity
# for culturally-specific compositional moments runs low regardless of
# whether the match is real (this codebase's known CLIP weak spot — see
# _vlm_verify_moment's docstring), so true and false positives cluster in
# the same low band and a threshold alone can't cleanly separate them.
# Lowered back toward Custom Mode's bar; the VLM gate is what now carries
# the correctness burden the stricter numbers used to.
_GENERAL_CLIP_MARGIN = float(os.environ.get("SNAPPY_GENERAL_CLIP_MARGIN", "0.02"))
# Still requires the winner to beat the runner-up by SOMETHING — a totally
# flat, uniform score across every moment (the original false-positive
# signature) still shouldn't reach the VLM at all — but small enough that a
# real, moderately-confident match isn't thrown out before it gets a look.
_GENERAL_DECISIVE_MARGIN = float(os.environ.get("SNAPPY_GENERAL_DECISIVE_MARGIN", "0.008"))


def _save_custom_capture(session, full_frame, tag: str, reasons: list,
                          mode_tag: str = "custom") -> tuple:
    """Simplified save path for Custom Mode (and General Mode, which reuses
    this unchanged via mode_tag="general") — no multi-angle bursts, no
    required-shot bookkeeping (there is no checklist), just: enhance, save,
    thumbnail, PhotoEntry. Mirrors the birthday path's file/URL conventions
    (models/album_generator.PhotoEntry, session.dir naming) so the album
    generator, admin panel, and captures/ static serving all keep working
    unchanged. mode_tag ("custom" or "general") is what album_generator.py's
    relevance scorer exempts from birthday-taxonomy scoring — see its
    _relevance() docstring.

    Returns (photo_url, thumb_b64).
    """
    final_img, report = enhance(full_frame, auto=True)
    session.ensure_dir()
    stamp = int(time.time() * 1000)
    fname = f"snap_{session.capture_count:04d}_{stamp}.jpg"
    fpath = str(session.dir / fname)
    cv2.imwrite(fpath, final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    small = cv2.resize(final_img, (160, 120))
    _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
    thumb_b64 = base64.b64encode(enc.tobytes()).decode()
    photo_url = f"/captures/{session.dir.name}/{fname}"

    pe = PhotoEntry(
        filepath=fpath, url=photo_url, timestamp=float(time.time()),
        quality_score=0.7, face_count=0, emotion_score=0.5,
        gaze_triggered=False, moment_type=str(tag), moment_conf=1.0,
        tags=list(set(reasons + [mode_tag])),
    )
    setattr(pe, "zoom_url", None)
    setattr(pe, "zoom_info", None)
    setattr(pe, "enhance_report", report.to_dict())
    setattr(pe, "color_sig", None)
    session.photos.append(pe)
    session.capture_count += 1

    if STORE is not None:
        try:
            STORE.add_photo(session.sid, {
                "url": pe.url, "filepath": pe.filepath, "ts": pe.timestamp,
                "score": pe.quality_score, "faces": pe.face_count,
                "emotion": pe.emotion_score, "gaze": pe.gaze_triggered,
                "moment": pe.moment_type, "moment_conf": pe.moment_conf,
                "tags": pe.tags,
            })
        except Exception as ex:
            log.warning(f"Persist photo failed: {ex}")

    return photo_url, thumb_b64


def _light_setup(frame, jitter: int) -> dict:
    """Minimal lighting/steadiness coach for Custom + General Mode.

    Birthday sessions get the full models/setup_advisor.advise() coaching via
    SetupCoach, but Custom/General render ConversationalGuide instead and were
    returning "setup": None outright — so those modes gave the user NO
    feedback at all when the room was too dark to photograph, which reads on
    screen as "the camera is just showing a black picture and nothing is
    happening". Same brightness thresholds as setup_advisor's light check so
    the guidance is consistent across modes. Deliberately does NOT run the
    CLIP-based framing scan advise() does — that's birthday-specific and
    costs extra CLIP calls.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    mean_b = float(gray.mean())
    tips, checks, score = [], [], 100
    if mean_b < 60:
        tips.append("Too dark — turn on the room lights or move near a lamp")
        score -= 35
        checks.append({"label": "Lighting", "ok": False,
                       "detail": f"Too dark for clean photos (brightness {mean_b:.0f}/255)"})
    elif mean_b < 85:
        tips.append("A little dim — add light and the photos get much better")
        score -= 15
        checks.append({"label": "Lighting", "ok": False,
                       "detail": f"Dim — more light recommended (brightness {mean_b:.0f}/255)"})
    else:
        checks.append({"label": "Lighting", "ok": True,
                       "detail": f"Good light (brightness {mean_b:.0f}/255)"})
    if jitter >= 14:
        tips.append("Camera is shaky — prop it against something solid")
        score -= 20
        checks.append({"label": "Steady", "ok": False, "detail": "Hand-held / moving"})
    else:
        checks.append({"label": "Steady", "ok": True, "detail": "Held steady"})
    return {"tips": tips[:2], "score": max(0, score), "checklist": checks, "guide": []}


def _display_face_boxes(face_frame, frame, full_frame):
    """Face boxes for the API response, rescaled to full_frame's resolution.

    FACE_PROVIDER.detect() runs on `frame`, the ≤1280-wide analysis proxy
    (see process_frame()'s HD/4K downscale) — so face_frame.boxes come back
    in THAT smaller coordinate space. The frontend overlay maps boxes
    against the camera's full native resolution, so left unscaled, every
    box lands shrunk and shifted toward the top-left by the proxy/full-res
    ratio whenever the camera delivers >1400px-wide frames (any modern
    webcam) — measured live as a face box floating well above the actual
    face. A no-op (returns face_frame.boxes as-is) when no downscale
    happened.
    """
    scale = full_frame.shape[1] / frame.shape[1]
    if scale == 1.0:
        return face_frame.boxes
    return [{"x": b["x"]*scale, "y": b["y"]*scale,
             "w": b["w"]*scale, "h": b["h"]*scale} for b in face_frame.boxes]


def _evaluate_custom_objective(
    session, frame, full_frame, source: str, _clock: float,
    face_frame, face_count: int, nima_score, shot, gaze, emotion, pred,
    clip_scores,
) -> dict:
    """Custom Mode's whole decision + response, given signals process_frame()
    already computed generically (face/emotion/gaze/quality/CLIP/prediction).
    Self-contained: builds its own capture decision AND its own response
    dict, matching the shape frontend/api/types.ts's FrameResult expects, so
    it never has to thread through the birthday-specific machinery below.
    """
    obj = session.custom_objective
    captured = False
    capture_reason = None
    captured_url = None
    thumb_b64 = None
    tag = "custom"

    if obj is not None and not obj.fulfilled:
        class_name = f"custom_{session.sid}"
        pp = (clip_scores.per_prompt or {}) if clip_scores and clip_scores.available else {}
        clip_sim = float(pp.get(class_name, 0.0))
        bg_best  = max((v for k, v in pp.items() if k.startswith("bg_")), default=0.0)
        margin   = clip_sim - bg_best
        tag = (obj.watching_for or "custom").strip().lower().replace(" ", "_")[:40] or "custom"

        # Gate: an objective that names "face" but sees nobody can't be met.
        score = margin
        if "face" in obj.detectors and face_count == 0:
            score -= 0.5
        if "emotion" in obj.detectors:
            score += 0.15 * float(getattr(emotion, "per_emotion", {}).get("happy", 0.0))
        if "gaze" in obj.detectors:
            score += 0.10 * (1.0 if getattr(gaze, "is_group_looking", False) else 0.0)

        clip_triggered = score >= _CUSTOM_CLIP_MARGIN

        # Objective-scored ring — separate from the generic _fullres_ring
        # (which ranks frames by overall photo quality, not by how well they
        # match THIS objective). Reusing the generic ring was the bug behind
        # "smile capture was wrong": the trigger correctly fired on a smiling
        # frame, but the saved photo was whichever recent frame had the best
        # generic sharpness/quality score — frequently a completely different,
        # non-smiling frame from a second earlier. Reset whenever the
        # objective itself changes so a stale frame scored against the
        # PREVIOUS request never wins the new one's selection.
        if getattr(session, "_custom_ring_obj_ts", None) != obj.created_ts:
            session._custom_obj_ring = deque(maxlen=6)
            session._custom_ring_obj_ts = obj.created_ts
        session._custom_obj_ring.append((full_frame, score))

        # Motion/lighting are given their OWN authoritative signal (not just
        # folded into the CLIP score) — a literal brightness jump is a much
        # more reliable "lights turned on" detector than semantic similarity,
        # and a literal motion spike is a much more reliable "something
        # sudden just happened" signal for jump/wave/entering than CLIP,
        # which has no sense of TIME at all. Either path can trigger.
        motion_triggered = ("motion" in obj.detectors
                            and session.predictor.current_motion() >= _CUSTOM_MOTION_SPIKE)

        light_triggered = False
        if "lighting" in obj.detectors:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(gray.mean())
            baseline = getattr(session, "_custom_light_baseline", None)
            if baseline is None:
                session._custom_light_baseline = brightness
            elif brightness - baseline >= _CUSTOM_LIGHT_JUMP:
                light_triggered = True

        triggered = clip_triggered or motion_triggered or light_triggered

        # Continuous mode: still respect a cooldown so one held pose doesn't
        # fire every single frame (~5/sec) — same anti-spam intent as the
        # birthday pipeline's dedup, just time-based instead of visual-hash
        # based (a repeated identical action IS the point in continuous mode).
        cooldown_ok = (obj.capture_mode != "continuous"
                       or (_clock - obj.last_capture_ts) >= _CUSTOM_CONTINUOUS_COOLDOWN_S)

        if triggered and cooldown_ok:
            # Up to _MAX_PENDING_FRAMES frames for this session can be
            # mid-flight across the shared executor at once — `cooldown_ok`/
            # `fulfilled` above were read from a snapshot a CONCURRENT frame
            # may have already advanced by the time we get here. Re-verify
            # freshness inside the lock — same fix, same measured symptom
            # (repeat fires within milliseconds), as General Mode's identical
            # race in _evaluate_general_objective.
            with session._capture_lock:
                cooldown_ok_now = (obj.capture_mode != "continuous"
                                    or (_clock - obj.last_capture_ts) >= _CUSTOM_CONTINUOUS_COOLDOWN_S)
                if cooldown_ok_now and not obj.fulfilled:
                    # Pick the frame that best matched THIS objective, not just
                    # whichever recent frame looked sharpest in general — motion/
                    # lighting triggers have no per-frame "objective score" of their
                    # own (they're pass/fail, not graded), so for those the CURRENT
                    # frame (the one that actually crossed the brightness/motion
                    # threshold) is the right choice; for a CLIP/emotion/gaze
                    # objective, the ring lets us reach back to the actual peak.
                    best_frame, best_score = full_frame, score
                    if clip_triggered:
                        ring = list(getattr(session, "_custom_obj_ring", []) or [])
                        if ring:
                            best_frame, best_score = max(ring, key=lambda t: t[1])
                    reasons = ["custom_objective"]
                    if motion_triggered: reasons.append("motion")
                    if light_triggered:  reasons.append("lighting")
                    obj.last_capture_ts = _clock   # claim BEFORE the save so a
                                                    # racing thread's re-check sees it
                    captured_url, thumb_b64 = _save_custom_capture(session, best_frame, tag, reasons)
                    captured = True
                    capture_reason = reasons[0]
                    log.info(f"[{session.sid}] custom capture: '{obj.watching_for}' "
                             f"(reasons={reasons}, score={best_score:.3f}) → {captured_url}")
                    if obj.capture_mode == "one_time":
                        obj.fulfilled = True
                        session.watch_status = "captured"
                        session.assistant_reply = f"Captured! I got {obj.watching_for}."
                    else:
                        session.assistant_reply = f"Captured {obj.watching_for} again — still watching."

    small_frame = cv2.resize(frame, (160, 120))
    _, enc = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
    gallery_entry = safe({
        "id": session.frame_count, "b64": base64.b64encode(enc.tobytes()).decode(),
        "score": float(shot.total), "faces": face_count,
        "moment": tag, "mconf": 1.0 if captured else 0.0,
        "gaze": bool(getattr(gaze, "is_group_looking", False)),
        "captured": bool(captured), "ts": float(time.time()),
    })
    session.frame_gallery.append(gallery_entry)
    if len(session.frame_gallery) > session.MAX_GALLERY:
        session.frame_gallery.pop(0)
    session.score_timeline.append(float(shot.total))
    if len(session.score_timeline) > session.MAX_TIMELINE:
        session.score_timeline.pop(0)

    return safe({
        "type": "frame_result", "frame_id": int(session.frame_count),
        "analysis": shot.to_dict(),
        "face_detection": {
            "available": bool(face_frame.available), "backend": str(face_frame.backend),
            "face_count": int(face_count),
            "boxes": _display_face_boxes(face_frame, frame, full_frame),
            "error": face_frame.error,
        },
        "gaze": gaze.to_dict(), "moment": {
            "detected_moment": tag, "confidence": 1.0 if captured else 0.0,
            "active_moments": [], "is_capture_moment": bool(captured),
        },
        "emotion": emotion.to_dict(), "prediction": pred.to_dict(),
        "clip": {
            "available": bool(CLIP.available), "backend": "transformers" if CLIP.available else "none",
            "best": None, "best_score": None,
            "infer_ms": (round(clip_scores.inference_ms, 1) if clip_scores else None),
            "kept_centroid_sim": None, "init_error": CLIP.init_error,
        },
        "nima": {"available": bool(NIMA.available),
                 "score": float(nima_score) if nima_score is not None else None},
        "hsemotion": {"available": bool(HSEMO.available)},
        "engine": None, "calibration": None,
        "setup": _light_setup(frame, int(getattr(session, "_last_jitter", 0))),
        "captured": bool(captured), "capture_reason": capture_reason,
        "captured_url": captured_url, "captured_zoom_url": None,
        "captured_zoom_info": None, "captured_thumb": thumb_b64,
        "total_captures": int(session.capture_count),
        "score_timeline": [round(float(s), 3) for s in session.score_timeline[-60:]],
        "gallery_frame": gallery_entry, "active_moments": [],
        "audio": {"events": {}, "boost": 0.0, "voice_tier": "", "birthday_awareness": 0.0},
        "sequence": {"event": "", "stage": None, "cursor": -1},
    })


def _evaluate_general_objective(
    session, frame, full_frame, source: str, _clock: float,
    face_frame, face_count: int, nima_score, shot, gaze, emotion, pred,
    clip_scores,
) -> dict:
    """General Mode's whole decision + response — same shape/contract as
    _evaluate_custom_objective, but scores EVERY VLM-derived moment each
    frame (not just one user-specified objective) and captures whichever
    crosses threshold, tagged with that moment's own label. Unlike Custom
    Mode's default one_time-then-stop, a GeneralObjective never "fulfills" —
    an event has many distinct moments worth catching across its whole
    length, each independently cooled down (see _GENERAL_MOMENT_COOLDOWN_S).
    """
    obj = session.general_objective
    captured = False
    capture_reason = None
    captured_url = None
    thumb_b64 = None
    tag = "general"

    if obj is not None and obj.moments:
        pp = (clip_scores.per_prompt or {}) if clip_scores and clip_scores.available else {}
        bg_best = max((v for k, v in pp.items() if k.startswith("bg_")), default=0.0)

        # Score every derived moment this frame — pick the single best one
        # that's ALSO past its own cooldown, rather than the globally best
        # score regardless of cooldown (a moment on cooldown must not block
        # a different, ready moment from firing the same frame).
        scored = []
        for i, m in enumerate(obj.moments):
            class_name = f"general_{session.sid}_{i}"
            clip_sim = float(pp.get(class_name, 0.0))
            score = clip_sim - bg_best
            if "face" in m.detectors and face_count == 0:
                score -= 0.5
            if "emotion" in m.detectors:
                score += 0.15 * float(getattr(emotion, "per_emotion", {}).get("happy", 0.0))
            if "gaze" in m.detectors:
                score += 0.10 * (1.0 if getattr(gaze, "is_group_looking", False) else 0.0)
            scored.append((m, score))

        # Ring keyed by moment index → (frame, score) list, reset whenever a
        # NEW GeneralObjective replaces this one — same rationale as Custom
        # Mode's ring: pick the frame that best matched the TRIGGERING
        # moment, not just whichever frame looked sharpest in general.
        if getattr(session, "_general_ring_obj_ts", None) != obj.created_ts:
            session._general_ring = [deque(maxlen=6) for _ in obj.moments]
            session._general_ring_obj_ts = obj.created_ts
        for i, (m, score) in enumerate(scored):
            session._general_ring[i].append((full_frame, score))

        # Motion/lighting BOOST a semantically plausible moment; they can no
        # longer qualify one on their own. Previously each added a flat +0.5,
        # which meant any motion spike fired a capture with effectively no
        # semantic check at all — measured on session cb0c0a14: "Gifting
        # Diyas" captured at score 0.574 and "Serving Food" at 0.523 in a
        # bedroom, both entirely from that +0.5 with a ~0.05 CLIP score
        # underneath. Now they only lift a moment that already looks right.
        motion_now = session.predictor.current_motion()
        light_triggered_idx = None
        gray_mean = None
        boosted = list(scored)
        for i, m in enumerate(obj.moments):
            if "motion" in m.detectors and motion_now >= _CUSTOM_MOTION_SPIKE:
                boosted[i] = (m, boosted[i][1] + 0.03)
            if "lighting" in m.detectors:
                if gray_mean is None:
                    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
                baseline = getattr(session, "_general_light_baseline", None)
                if baseline is None:
                    session._general_light_baseline = gray_mean
                elif gray_mean - baseline >= _CUSTOM_LIGHT_JUMP:
                    boosted[i] = (m, boosted[i][1] + 0.03)
                    light_triggered_idx = i

        # Coverage boost: without this, whichever moment is EASIEST for CLIP
        # to match (usually something generic like "a group gathering")
        # wins the ranking every single time it's even plausible, and a
        # session can end with 10 captures of that one moment and zero of
        # the others — not because the others never happened on camera, but
        # because they never got to WIN the per-frame comparison. A small,
        # one-time nudge for a moment that hasn't fired yet is enough to let
        # it win close calls against an already-covered moment, without
        # being large enough to override a genuinely stronger different
        # match. Vanishes automatically once every moment has captured at
        # least once (last_capture_ts is no longer -inf for any of them).
        for i, m in enumerate(obj.moments):
            if m.last_capture_ts == float("-inf"):
                boosted[i] = (boosted[i][0], boosted[i][1] + 0.015)

        # Decisiveness: a real match stands out from its siblings. If every
        # moment scores about the same (the "matches nothing in particular"
        # case), capture nothing rather than picking an arbitrary winner.
        ranked = sorted(range(len(boosted)), key=lambda i: boosted[i][1], reverse=True)
        top_i = ranked[0]
        runner_up = boosted[ranked[1]][1] if len(ranked) > 1 else float("-inf")
        decisive = (boosted[top_i][1] - runner_up) >= _GENERAL_DECISIVE_MARGIN

        eligible = []
        if decisive and boosted[top_i][1] >= _GENERAL_CLIP_MARGIN \
                and (_clock - boosted[top_i][0].last_capture_ts) >= _GENERAL_MOMENT_COOLDOWN_S:
            eligible.append((boosted[top_i][0], boosted[top_i][1], top_i))

        if eligible:
            # Up to _MAX_PENDING_FRAMES frames for this session can be
            # mid-flight across the shared executor at once — `eligible` was
            # built from a last_capture_ts snapshot that a CONCURRENT frame
            # may have already advanced by the time we get here. Re-verify
            # freshness inside the lock (not just take the lock) or every
            # thread that raced past the check above still fires — measured
            # live: without this, one moment fired 4-5 times within ~10ms.
            with session._capture_lock:
                eligible_now = [(m, s, i) for (m, s, i) in eligible
                                if (_clock - m.last_capture_ts) >= _GENERAL_MOMENT_COOLDOWN_S]
                if eligible_now:
                    best_m, best_score, best_i = max(eligible_now, key=lambda t: t[1])
                    ring = list(getattr(session, "_general_ring", [[]])[best_i] or [])
                    best_frame = max(ring, key=lambda t: t[1])[0] if ring else full_frame
                    # Cross-moment visual dedup, under ONE shared key rather
                    # than per-moment. Per-moment cooldowns alone let several
                    # DIFFERENT moments each fire on the same static scene, and
                    # since each moment's ring picks the same peak frame, the
                    # identical JPEG got written N times under N labels —
                    # verified by md5 on session cb0c0a14: one photo saved 5x.
                    # A shared key means "we already have this picture", full
                    # stop, regardless of which moment claims it.
                    if session.dup_guard.is_duplicate(best_frame, "general_all", now=_clock):
                        best_m.last_capture_ts = _clock   # respect the cooldown anyway
                        log.debug(f"[{session.sid}] general capture suppressed as duplicate "
                                  f"('{best_m.label}', score={best_score:.3f})")
                    else:
                        # VLM sanity check on the ONE candidate frame before
                        # committing. General Mode's moment vocabulary is
                        # free-form and often a specific compositional
                        # relationship ("sister tying rakhi on brother's
                        # wrist") — exactly what CLIP is known to score
                        # unreliably (it reads rough scene gist, not WHO did
                        # WHAT to WHOM). Measured live: CLIP fired at 0.08 on
                        # a solo photo of someone holding a paper packet —
                        # indistinguishable from noise. This binary check
                        # (same is_birthday()-style yes/no) only runs on a
                        # frame CLIP already flagged, so it costs one VLM
                        # call per actual capture attempt, not per frame.
                        # None (VLM unavailable) does NOT veto — same
                        # graceful-degradation contract as every other VLM
                        # check in this codebase.
                        verified = _vlm_verify_moment(best_frame, best_m.clip_prompt)
                        if verified is False:
                            best_m.last_capture_ts = _clock   # don't hammer the VLM on
                                                               # the same stuck near-miss
                            log.info(f"[{session.sid}] general capture REJECTED by VLM check: "
                                     f"'{best_m.label}' (event={obj.event_name!r}, "
                                     f"clip_score={best_score:.3f})")
                        else:
                            tag = (best_m.label or "general").strip().lower().replace(" ", "_")[:40] or "general"
                            reasons = ["event_moment"]
                            if light_triggered_idx == best_i: reasons.append("lighting")
                            if "motion" in best_m.detectors and motion_now >= _CUSTOM_MOTION_SPIKE:
                                reasons.append("motion")
                            if verified is True: reasons.append("vlm_verified")
                            best_m.last_capture_ts = _clock   # claim BEFORE the save so a
                                                               # racing thread's re-check sees it
                            session.dup_guard.remember(best_frame, "general_all", now=_clock)
                            captured_url, thumb_b64 = _save_custom_capture(
                                session, best_frame, tag, reasons, mode_tag="general")
                            captured = True
                            capture_reason = reasons[0]
                            log.info(f"[{session.sid}] general capture: '{best_m.label}' "
                                     f"(event={obj.event_name!r}, reasons={reasons}, "
                                     f"score={best_score:.3f}) → {captured_url}")
                            covered = sum(1 for m in obj.moments if m.last_capture_ts != float("-inf"))
                            session.assistant_reply = (
                                f"Captured {best_m.label}! ({covered}/{len(obj.moments)} moments so "
                                f"far) — still watching {obj.event_name}.")

    small_frame = cv2.resize(frame, (160, 120))
    _, enc = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
    gallery_entry = safe({
        "id": session.frame_count, "b64": base64.b64encode(enc.tobytes()).decode(),
        "score": float(shot.total), "faces": face_count,
        "moment": tag, "mconf": 1.0 if captured else 0.0,
        "gaze": bool(getattr(gaze, "is_group_looking", False)),
        "captured": bool(captured), "ts": float(time.time()),
    })
    session.frame_gallery.append(gallery_entry)
    if len(session.frame_gallery) > session.MAX_GALLERY:
        session.frame_gallery.pop(0)
    session.score_timeline.append(float(shot.total))
    if len(session.score_timeline) > session.MAX_TIMELINE:
        session.score_timeline.pop(0)

    return safe({
        "type": "frame_result", "frame_id": int(session.frame_count),
        "analysis": shot.to_dict(),
        "face_detection": {
            "available": bool(face_frame.available), "backend": str(face_frame.backend),
            "face_count": int(face_count),
            "boxes": _display_face_boxes(face_frame, frame, full_frame),
            "error": face_frame.error,
        },
        "gaze": gaze.to_dict(), "moment": {
            "detected_moment": tag, "confidence": 1.0 if captured else 0.0,
            "active_moments": [], "is_capture_moment": bool(captured),
        },
        "emotion": emotion.to_dict(), "prediction": pred.to_dict(),
        "clip": {
            "available": bool(CLIP.available), "backend": "transformers" if CLIP.available else "none",
            "best": None, "best_score": None,
            "infer_ms": (round(clip_scores.inference_ms, 1) if clip_scores else None),
            "kept_centroid_sim": None, "init_error": CLIP.init_error,
        },
        "nima": {"available": bool(NIMA.available),
                 "score": float(nima_score) if nima_score is not None else None},
        "hsemotion": {"available": bool(HSEMO.available)},
        "engine": None, "calibration": None,
        "setup": _light_setup(frame, int(getattr(session, "_last_jitter", 0))),
        "captured": bool(captured), "capture_reason": capture_reason,
        "captured_url": captured_url, "captured_zoom_url": None,
        "captured_zoom_info": None, "captured_thumb": thumb_b64,
        "total_captures": int(session.capture_count),
        "score_timeline": [round(float(s), 3) for s in session.score_timeline[-60:]],
        "gallery_frame": gallery_entry, "active_moments": [],
        "audio": {"events": {}, "boost": 0.0, "voice_tier": "", "birthday_awareness": 0.0},
        "sequence": {"event": "", "stage": None, "cursor": -1},
    })


# ── Sync per-frame processing ─────────────────────────────────────────────
def process_frame(session, frame_b64: str, frame=None, source: str = "live") -> dict:
    """Decode, analyse, maybe capture. Returns JSON-safe dict.

    `session` is an api.session.Session.
    `source` is "live" (camera) or "upload" (recorded video) — used to enforce
    the single-source lock so live and upload captures never mix.
    """
    try:
        if frame is None:
            raw = base64.b64decode(frame_b64)
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"error": "Could not decode frame"}
    except Exception as e:
        return {"error": f"Frame decode: {e}"}

    # ── HD/4K without losing real-time ─────────────────────────────────────
    # The PHOTO must carry every pixel the camera delivered, but the models
    # must stay fast — so all analysis runs on a ≤1280-wide PROXY while the
    # full-resolution frame is kept aside as the photo source (what real
    # camera apps do: analyze small, shoot full).
    full_frame = frame
    _H0, _W0 = frame.shape[:2]
    if _W0 > 1400:
        _ps = 1280.0 / _W0
        frame = cv2.resize(frame, (1280, int(_H0 * _ps)))

    session.frame_count += 1

    # ── Session clock + SCENE-CUT detector ─────────────────────────────────
    # One deterministic clock (media time for uploads, frame-count time live)
    # used by all rate logic. A hard CUT (consecutive-frame hash jump ≥ 28 —
    # measured: cuts 30+, handheld pan/motion 5–20) marks genuinely NEW content;
    # a drifting camera over the same scene does not.
    _clock = (float(getattr(session, "_media_ts", 0.0)) if source == "upload"
              else session.frame_count / 5.0)
    _cur_ph = _phash(frame)
    _prev_ph = getattr(session, "_prev_ph", None)
    _jitter = (_ham(_cur_ph, _prev_ph)
               if (_cur_ph is not None and _prev_ph is not None) else 0)
    if _jitter >= 28:
        session._last_cut_ts = _clock
    session._prev_ph = _cur_ph
    session._last_jitter = int(_jitter)

    # ── Single-source guarantee ───────────────────────────────────────────
    # A session commits to ONE capture source. An upload always takes ownership
    # (the upload handler also clears any prior live photos). A live frame
    # arriving into an upload-owned session may still be analysed for telemetry
    # but is NEVER committed as a capture — so live and upload photos can never
    # land in the same album. This holds regardless of the UI.
    locked = getattr(session, "capture_source", None)
    if source == "upload":
        session.capture_source = "upload"
    elif locked is None:
        session.capture_source = source
    capture_allowed = not (
        getattr(session, "capture_source", None) == "upload" and source != "upload")

    # ── Custom Mode fast-path: skip heavy GPU work while OUR OWN VLM
    # intent-parse call is in flight and there's no armed objective yet ────
    # VLMTagger.parse_capture_intent() (api/server.py's _dispatch_custom_
    # prompt) runs on the same Apple MPS GPU as NIMA/CLIP/HSEmotion/
    # MediaPipe below. MPS serializes GPU work across threads, so a live
    # 5fps frame stream running concurrently with that VLM call doesn't
    # just add a little overhead — measured live: it turned a ~5s VLM call
    # into ~30s. Nothing CAN capture without an armed objective anyway
    # (custom_objective is left untouched, i.e. None on a session's first
    # prompt, while watch_status=="understanding"), so there is nothing to
    # lose by skipping analysis for the few seconds the VLM call takes. If
    # an objective is already armed AND still unfulfilled (a "replace" sent
    # while a continuous objective is actively watching), this is
    # deliberately skipped so an in-progress capture is never missed — but
    # a fulfilled one_time objective (nothing left to trigger) or no
    # objective at all both qualify for the fast path.
    # General Mode shares this exact contention problem for its own VLM step
    # (derive_event_moments(), also on the same executor/GPU) — before its
    # moments list exists, nothing can capture either, so the same skip
    # applies while general_objective is still None.
    _cur_obj = getattr(session, "custom_objective", None)
    _cur_gen_obj = getattr(session, "general_objective", None)
    _etype = getattr(session, "event_type", "")
    _custom_fastpath = (_etype == "custom" and (_cur_obj is None or _cur_obj.fulfilled))
    # NOTE: "smart_event" (not "general") — "general" is already a real,
    # separate event_type in this codebase (models/category_requirements.py's
    # fallback profile, also the silent default for ANY unrecognized event
    # type). Reusing that string here would have hijacked every session that
    # legitimately falls back to it. "smart_event" is General Mode's actual
    # internal identifier; the UI still labels the button "General".
    _general_fastpath = (_etype == "smart_event" and _cur_gen_obj is None)
    if (getattr(session, "watch_status", "") == "understanding"
            and (_custom_fastpath or _general_fastpath)):
        session.frame_count += 1
        small_frame = cv2.resize(frame, (160, 120))
        _, enc = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
        gallery_entry = safe({
            "id": session.frame_count, "b64": base64.b64encode(enc.tobytes()).decode(),
            "score": 0.0, "faces": 0, "moment": "understanding", "mconf": 0.0,
            "gaze": False, "captured": False, "ts": float(time.time()),
        })
        session.frame_gallery.append(gallery_entry)
        if len(session.frame_gallery) > session.MAX_GALLERY:
            session.frame_gallery.pop(0)
        return safe({
            "type": "frame_result", "frame_id": int(session.frame_count),
            "analysis": {"timestamp": float(time.time()), "blur": 0.0, "faces": 0,
                         "face_q": 0.0, "comp": 0.0, "bright": 0.0, "emotion": 0.0,
                         "total": 0.0},
            "face_detection": {"available": False, "backend": "skipped",
                               "face_count": 0, "boxes": [], "error": None},
            "gaze": {"total_faces": 0, "faces_looking": 0, "gaze_ratio": 0.0,
                     "is_group_looking": False, "confidence": 0.0, "backend": "skipped"},
            "moment": {"detected_moment": "understanding", "confidence": 0.0,
                       "active_moments": [], "is_capture_moment": False},
            "emotion": {"dominant": "neutral", "score": 0.0, "per_emotion": {},
                        "happy_ratio": 0.0, "avg_smile": 0.0},
            "prediction": {"trend": "stable", "peak_in_ms": None,
                           "confidence": 0.0, "preroll": False},
            "clip": {"available": bool(CLIP.available), "backend": "skipped",
                     "best": None, "best_score": None, "infer_ms": None,
                     "kept_centroid_sim": None, "init_error": None},
            "nima": {"available": bool(NIMA.available), "score": None},
            "hsemotion": {"available": bool(HSEMO.available)},
            "engine": None, "calibration": None, "setup": None,
            "captured": False, "capture_reason": None,
            "captured_url": None, "captured_zoom_url": None,
            "captured_zoom_info": None, "captured_thumb": None,
            "total_captures": int(session.capture_count),
            "score_timeline": [round(float(s), 3) for s in session.score_timeline[-60:]],
            "gallery_frame": gallery_entry, "active_moments": [],
            "audio": {"events": {}, "boost": 0.0, "voice_tier": "", "birthday_awareness": 0.0},
            "sequence": {"event": "", "stage": None, "cursor": -1},
        })

    # ── Shared face perception ───────────────────────────────────────────
    face_frame = FACE_PROVIDER.detect(frame)
    face_count = face_frame.count

    boxes_xywh = [f.box for f in face_frame.faces] if face_frame.available else None
    nima_score = NIMA.score(frame) if NIMA.available else None

    # Personalization — track faces to learn the HOST group (owner's people) and
    # get this frame's host relevance. Cheap (position tracking), runs every
    # frame, degrades to "allow all" until confident. (models/host_group.py)
    _Hh, _Ww = frame.shape[:2]
    frame_host_score, n_host_faces = session.host_group.update(boxes_xywh, _Ww, _Hh, frame=frame)

    # Per-frame ML
    shot    = session.quality.analyze(frame, face_boxes=boxes_xywh, nima_score=nima_score)
    gaze    = session.gaze.detect(face_frame)
    # Heuristic VLM is run here but used ONLY as a fallback when CLIP is
    # unavailable (no GPU, warmup failed). When CLIP is running, the moment
    # label and confidence are set entirely from CLIP scores below so the
    # heuristic's colour-based guesses never cause mislabelling.
    moment  = session.vlm.detect(frame)
    emotion = detect_emotions(face_frame.faces)
    session.predictor.update(shot.total, frame)
    pred    = session.predictor.predict()

    # Short FULL-RES ring — the photo source. 6 frames (~1.2s @5fps) is enough
    # to pick the best-scored recent frame at capture time without the memory
    # cost of a full-res preroll (25 × 4K would be ~600 MB).
    if not hasattr(session, "_fullres_ring"):
        session._fullres_ring = deque(maxlen=6)
    session._fullres_ring.append((full_frame, float(shot.total)))

    # L2CS gaze override (when installed) — overwrites looking_at_camera
    # values with regressor output.
    if L2CS is not None and L2CS.available and face_frame.faces:
        try:
            L2CS.refine_gaze_inplace(frame, face_frame.faces)
            gaze = session.gaze.detect(face_frame)
        except Exception as ex:
            log.debug(f"L2CS refine skipped: {ex}")

    # HSEmotion blend (real FER) — 70/30 with blendshape fallback
    if HSEMO.available and face_frame.faces:
        try:
            hse = HSEMO.predict_batch(frame, [f.box for f in face_frame.faces])
            valid = [r for r in hse if r is not None]
            if valid:
                agg = {}
                for _, probs in valid:
                    for k, v in probs.items():
                        agg[k] = agg.get(k, 0.0) + v / len(valid)
                blended, keys = {}, set(agg) | set(emotion.per_emotion)
                for k in keys:
                    blended[k] = 0.7 * agg.get(k, 0.0) + 0.3 * emotion.per_emotion.get(k, 0.0)
                tot = sum(blended.values()) or 1.0
                emotion.per_emotion = {k: v / tot for k, v in blended.items()}
                emotion.dominant = max(emotion.per_emotion, key=emotion.per_emotion.get)
                emotion.score = emotion.per_emotion[emotion.dominant]
                emotion.happy_ratio = emotion.per_emotion.get("happy", 0.0)
        except Exception as ex:
            log.debug(f"HSEmotion blend skipped: {ex}")

    session.score_timeline.append(float(shot.total))
    if len(session.score_timeline) > session.MAX_TIMELINE:
        session.score_timeline.pop(0)

    # ── CLIP scoring ─────────────────────────────────────────────────────
    # If CLIP hasn't been warmed up yet (e.g. eager warmup was disabled or
    # the install lazily fetched CLIP on first frame), warm it up FIRST so
    # this very frame can still get CLIP scores. Previously the warmup
    # happened after the score_frame check, meaning the first frame of the
    # first session always lost its CLIP signal.
    if not CLIP.available:
        ok = CLIP.warmup()
        if not ok and session.frame_count == 1:
            # Make the failure visible in the response so the UI can show it.
            log.warning(f"CLIP unavailable on first frame: {CLIP.init_error}")
    clip_scores = (CLIP.score_frame(frame, sid=session.sid,
                                    event=getattr(session, "event_type", ""))
                   if CLIP.available else None)

    # Auto-classifier observation (uses CLIP embedding + per-prompt sims)
    if clip_scores and clip_scores.available and clip_scores.per_prompt:
        try:
            emb = getattr(clip_scores, "_image_emb", None)
            AUTO_CLASS.observe(
                sid=session.sid, shot_total=float(shot.total),
                clip_per_prompt=clip_scores.per_prompt,
                clip_emb=emb, photo_url=None,
            )
        except Exception as ex:
            log.debug(f"auto-classify skipped: {ex}")

    # ── Custom Mode — separate, self-contained path ──────────────────────
    # No predefined moment knowledge for this event type; everything below
    # this point (heuristic-VLM labeling, candle-phase, cake-smash, the
    # birthday-fused head, required-shot boost, event-evidence gate,
    # decide_capture()) is birthday-category machinery a custom session
    # should never touch. Guarded strictly by event_type — no birthday
    # session ever takes this branch.
    if getattr(session, "event_type", "") == "custom":
        return _evaluate_custom_objective(
            session, frame, full_frame, source, _clock,
            face_frame, face_count, nima_score, shot, gaze, emotion, pred,
            clip_scores,
        )

    # ── General Mode — separate, self-contained path ─────────────────────
    # Same reasoning as Custom Mode above: no predefined birthday taxonomy
    # applies here either — the moment vocabulary was derived per-event by
    # the VLM at session start (models/vlm_tagger.py's derive_event_moments()).
    if getattr(session, "event_type", "") == "smart_event":
        return _evaluate_general_objective(
            session, frame, full_frame, source, _clock,
            face_frame, face_count, nima_score, shot, gaze, emotion, pred,
            clip_scores,
        )

    # ── Moment label + confidence — CLIP is the authority ────────────────────
    #
    # Architecture (clean):
    #   • CLIP available  → CLIP scores drive the label and confidence entirely.
    #                        The heuristic VLM result above is IGNORED.
    #   • CLIP unavailable → fall back to the heuristic VLM (offline / no GPU).
    #
    # The heuristic VLM was the source of all the mislabelling we fixed
    # (toast→cake_cutting, gathering→candle_blowing). Removing it from the
    # hot path eliminates those bugs structurally rather than patching them.
    _bg_flat = False   # True when the scene barely beats background (junk)
    if clip_scores and clip_scores.available:
        pp     = clip_scores.per_prompt or {}
        # Active = prompt-derived moments PLUS the classes this event was
        # trained on (cake_person, birthday_gifting, …). The latter is empty
        # for untrained events, so this never affects other events.
        active = set(session.vlm.active_profiles)
        active |= set(CLIP.event_classes(getattr(session, "event_type", "")))
        active.discard("negative")   # internal/hidden, never a visible label
        # Background contrast prompts are never active — they exist so junk
        # scenes (pool table, empty room) WIN them instead of a birthday class,
        # feeding the off-prompt detector below and the object-shot gate.
        active = {a for a in active if not a.startswith("bg_")}

        # ── Active-class scores (what the user's prompt cares about) ──────
        act = {k: v for k, v in pp.items() if k in active}
        # gazing_moments may ONLY come from gaze GEOMETRY (the override below).
        # CLIP text-matching cannot judge gaze DIRECTION — it labels any close-up
        # of faces "looking into the camera" (a couple gazing at EACH OTHER got
        # captured as a gazing moment while the geometry correctly said no).
        act.pop("gazing_moments", None)
        off = {k: v for k, v in pp.items() if k not in active}
        a_best, a_val = (max(act.items(), key=lambda kv: kv[1]) if act else (None, 0.0))
        o_best, o_val = (max(off.items(), key=lambda kv: kv[1]) if off else (None, 0.0))

        # ── Scene off-prompt detection ────────────────────────────────────
        # When CLIP's best match is consistently an OFF-prompt class (e.g.
        # champagne_toast at a birthday party), the real moment isn't in the
        # user's prompt → label as general_peak and don't mislabel it as
        # cake_cutting just because the heuristic saw a round object.
        if o_best is not None:
            session.clip_offprompt_history.append(o_val - a_val)
        hist = session.clip_offprompt_history
        strong = sum(1 for m in hist if m >= _OFF_PROMPT_MARGIN)
        scene_offprompt = (len(hist) >= 1 and strong / len(hist) >= 0.5)

        # ── Set label from CLIP ranking ───────────────────────────────────
        if scene_offprompt:
            # The scene's real moment isn't in the prompt (e.g. a toast/cheer).
            # Label generically so we still capture the nice photo but don't
            # fire a multi-angle cake-cutting burst on a table of cups.
            moment.detected_moment   = "general_peak"
            moment.confidence        = float(a_val) if a_val else 0.5
            moment.is_capture_moment = bool(moment.confidence >= 0.45)
            log.debug(f"[{session.sid}] off-prompt scene (top={o_best} "
                      f"{o_val:.3f}) → general_peak")

        elif a_best:
            # CLIP's top active class IS the label. The candle_phase detector
            # below refines candle_blowing vs cake_with_candles after this.
            # BACKGROUND DEMOTION (birthday): if the winner barely beats the
            # background contrast prompts, the scene isn't really that moment
            # (a pool table "wins" cake at +0.02; real cakes win by +0.08+) —
            # label generically so junk never claims a birthday moment.
            _bgv = max((v for k, v in pp.items() if k.startswith("bg_")), default=0.0)
            # Track when a class last beat background CONVINCINGLY. A real
            # moment spikes ≥ the margin regularly (gifting: 0.06–0.07), so its
            # weak in-between frames (0.03–0.05) keep the label; junk (pool
            # table: never ≥ 0.06) has no strong frame to lean on → demoted.
            if not hasattr(session, "_bg_strong"):
                session._bg_strong = {}
            # Record strong frames for EVERY evidence class, not just the argmax:
            # in an action close-up the TOP class is the ACTION (cake_cutting)
            # while the cake OBJECT is strong-but-second — it must still
            # accumulate evidence, or real cutting/feeding clips arm nothing.
            for _cls in _EVIDENCE_CLASSES:
                if _cls != a_best and pp.get(_cls, 0.0) - _bgv >= _ev_bar(_cls):
                    session._bg_strong.setdefault(_cls, []).append(session.frame_count)
            _hits = session._bg_strong.setdefault(a_best, [])
            if a_val - _bgv >= _ev_bar(a_best):
                _hits.append(session.frame_count)
            _hits[:] = [h for h in _hits if session.frame_count - h <= 12]
            # Trust needs CONSISTENCY: ≥2 strong frames in the window before the
            # class may claim the label AT ALL (strong or weak frame alike). A
            # real moment spikes repeatedly (gifting: most frames ≥0.06), so it
            # unlocks on its 2nd strong frame (~0.2–1s); junk spikes at most
            # once by noise and never unlocks.
            _recent_strong = len(_hits) >= 2
            # gift_box_reveal (giant surprise box): the engine's flat band caps
            # its margins at ~0.03-0.05 — below the global 0.06 bar — but a real
            # box holds a SATURATED plateau (measured ~30 straight frames at
            # ≥0.025) while junk peaks at ~5 scattered frames (pool table). So
            # this one class may arm evidence via near-saturation instead:
            # ≥16 of the last 25 frames at ≥0.030 (pool table stalls at ~2
            # stills ≥0.030; the real box plateau holds ~21 straight).
            if pp.get("gift_box_reveal", 0.0) - _bgv >= 0.030:
                _gb = session._bg_strong.setdefault("_gbr_mild", [])
                if not _gb or _gb[-1] != session.frame_count:
                    _gb.append(session.frame_count)
            _gb = session._bg_strong.get("_gbr_mild", [])
            _gb[:] = [h for h in _gb if session.frame_count - h <= 25]
            if len(_gb) >= 16:
                session._event_evidence = True
            for _cls in _EVIDENCE_CLASSES:
                _lst = session._bg_strong.get(_cls, [])
                _lst[:] = [h for h in _lst if session.frame_count - h <= 25]
                if len(_lst) >= 4:
                    # Evidence = PERSISTENCE, not spikes: real event objects stay
                    # strong in 20-50%% of frames even when occluded by the action
                    # (feeding/smashing close-ups); junk's flat band crosses the
                    # margin only sporadically (~3%%) and never reaches 4-in-25.
                    session._event_evidence = True
                    break
            if (_resolve_event(getattr(session, "event_type", "")) == "birthday"
                    and not _recent_strong):
                moment.detected_moment   = "general_peak"
                # Low confidence: this frame's CLIP evidence is noise — it must
                # not push the capture engine over its threshold by itself.
                moment.confidence        = float(min(a_val, 0.40))
                moment.is_capture_moment = False
                _bg_flat = True          # scene ≈ background: block re-labelling too
            else:
                moment.detected_moment   = a_best
                moment.confidence        = float(min(1.0, a_val))
                moment.is_capture_moment = bool(a_val >= 0.50)

        else:
            # No active class scored — fall through to general_peak.
            moment.detected_moment   = "general_peak"
            moment.confidence        = 0.45
            moment.is_capture_moment = False

        # ── Kept-centroid similarity boost ────────────────────────────────
        # When this frame is close to a centroid the photographer already
        # kept (online learning), give a small confidence boost.
        if (clip_scores.similarity_to_kept_centroid is not None
                and not scene_offprompt):
            sim = clip_scores.similarity_to_kept_centroid
            if sim > 0.78:
                moment.confidence = float(min(1.0, moment.confidence + 0.08))
                moment.is_capture_moment = True

    # CLIP unavailable — heuristic VLM result stands as-is (already set above).
    # This path only runs without a GPU / during CLIP warmup failure.

    # ── Candle sub-phase refinement + presence guard ──────────────────────
    # Only frames with actual candle flames can be cake_with_candles or
    # candle_blowing. The CLIP text prompt "birthday cake with lit candles"
    # also matches party/toast scenes with no cake present — so we gate on
    # a minimum flame count. If no flames → demote to general_peak.
    # For frames that DO have flames, refine the sub-phase: actual blow needs
    # a face leaning in or descending (or flames going out); otherwise it's
    # the gathering phase (cake_with_candles).
    # Always track candle flame count — needed for both candle AND cake_cutting
    # scenes (a candle-heavy frame that CLIP calls cake_cutting is really
    # cake_with_candles; a non-candle frame that CLIP calls cake_with_candles
    # is really general_peak).
    n_flames = count_warm_flames(frame)

    if moment.detected_moment in (PHASE_BLOWING, PHASE_GATHERING):
        # Threshold calibrated on real clips:
        #   real candles (solo):  min=28  real candles (group): min=13
        #   toast/party (warm):   max=12  cake cutting (knife):  spikes high
        # ≥15 keeps all genuine candle scenes and rejects all toast/party frames.
        if n_flames < 15:
            # No real candle signal — CLIP text matched a party scene without
            # actual candles (toast, confetti, etc.). Label generically so it
            # can still be captured as a highlight, won't trigger candle burst.
            log.debug(f"[{session.sid}] candle label {moment.detected_moment} "
                      f"suppressed (flames={n_flames}) → general_peak")
            moment.detected_moment = "general_peak"
        else:
            phase = classify_candle_phase(
                frame, boxes_xywh or [],
                session.candle_flame_history, session.candle_face_y_history)
            if phase != moment.detected_moment:
                log.debug(f"[{session.sid}] candle phase {moment.detected_moment}→{phase}")
                moment.detected_moment = phase

    elif moment.detected_moment == "cake_cutting" and n_flames >= 15:
        # High flame count on a cake_cutting label → scene has lit candles, not
        # a knife. CLIP confused the two (both have "cake" in text). Override to
        # cake_with_candles so the right required shots fire.
        log.debug(f"[{session.sid}] cake_cutting overridden by flame evidence "
                  f"(flames={n_flames}) → cake_with_candles")
        moment.detected_moment = PHASE_GATHERING
        # Now run the sub-phase check — is someone actually blowing?
        phase = classify_candle_phase(
            frame, boxes_xywh or [],
            session.candle_flame_history, session.candle_face_y_history)
        moment.detected_moment = phase

    # ── Cake-smash refinement (geometry) ──────────────────────────────────
    # CLIP matches the smash SCENE loosely, so it fires cake_smashing on plain
    # cake shots AND misses the real smash (reading it as cake_cutting). Decide
    # from GEOMETRY instead: a real smash needs a face AT the cake (low + close)
    # with cake ON the face — or a face plunging into the cake region. This both
    # KILLS false cake_smashing and PROMOTES a genuine smash CLIP under-called.
    # Birthday-only by construction (only cake_* labels reach here), so it can
    # never affect another event's accuracy.
    _CAKE_SCENES = (CAKE_SMASH, "cake_cutting", "cake_feeding")
    if moment.detected_moment in _CAKE_SCENES and (boxes_xywh or []):
        is_smash, smash_detail = classify_cake_smash(
            frame, boxes_xywh or [], session.candle_face_y_history)
        # The ACT (plunge / at-cake) anchors an aftermath window. Frosting stays
        # on the face for the REST of the video — without a window, every later
        # group/gifting scene gets hijacked into 'cake_smashing'.
        if is_smash and smash_detail.get("about_to"):
            session._smash_act_ts = _clock
        _aftermath_ok = (_clock - getattr(session, "_smash_act_ts", -999.0)) <= 20.0
        if is_smash and smash_detail.get("on_face") and not smash_detail.get("about_to") \
                and not _aftermath_ok:
            is_smash = False          # old frosting, new scene — not a smash moment
        if moment.detected_moment == CAKE_SMASH and not is_smash:
            log.debug(f"[{session.sid}] cake_smashing denied by geometry "
                      f"{smash_detail} → cake_cutting")
            moment.detected_moment = "cake_cutting"
        elif moment.detected_moment != CAKE_SMASH and is_smash:
            log.debug(f"[{session.sid}] cake smash confirmed by geometry "
                      f"{smash_detail} → cake_smashing")
            moment.detected_moment   = CAKE_SMASH
            moment.confidence        = float(min(1.0, max(moment.confidence, 0.70)))
            moment.is_capture_moment = True

    # ── Trained birthday-moment head (CLIP + behavioural cues) ────────────
    # The learned fused classifier (models/birthday_fused.npz, trained on the
    # labelled clip library) is the one that actually tells candle_blowing from
    # cake_with_candles, surprise from cake, etc. — using the pout/jaw/frosting/
    # surprise/flame cues the user's notes described. Birthday-only, and only
    # OVERRIDES when it is confident, so it can never hurt other events.
    # Skipped when the scene barely beats BACKGROUND (_bg_flat): the classifier
    # only knows birthday classes, so on junk (a pool table) it hallucinates
    # 'cake' with high confidence and would undo the background demotion.
    if _resolve_event(getattr(session, "event_type", "")) == "birthday" and not _bg_flat:
        _bd = BirthdayMomentClassifier.get()
        if _bd.ok:
            _pred = _bd.predict(frame, getattr(clip_scores, "_image_emb", None),
                                face_frame.faces if face_frame.available else [], CLIP)
            if _pred is not None:
                _lbl, _prob, _ = _pred
                if _prob >= 0.55 and _lbl != "not_birthday" and _lbl != moment.detected_moment:
                    moment.detected_moment = _lbl
                    moment.confidence = float(max(moment.confidence, _prob))

    # ── Sequence-aware prior (per-event, boost-only) ──────────────────────
    # The model knows the event's flow (models/event_sequence). A moment that
    # is plausible at the CURRENT point in the event — at/just after how far
    # we've progressed — gets a gentle confidence nudge, which helps
    # disambiguate look-alikes (candle_blowing right after cake_with_candles
    # beats a stray cake_cutting). Boost-only + per-event: it never penalises,
    # and is a complete no-op for events with no known sequence, so it can
    # never lower another event's accuracy.
    _seq_event = _resolve_event(getattr(session, "event_type", ""))
    _seq_stage = _UNKNOWN_STAGE
    if _seq_event:
        _seq_stage = _stage_index(_seq_event, moment.detected_moment)
        cur = session.event_stage_cursor
        if _seq_stage != _UNKNOWN_STAGE and (cur < 0 or (cur - 2) <= _seq_stage <= (cur + 4)):
            moment.confidence = float(min(1.0, moment.confidence + _SEQ_PRIOR_BONUS))

    # ── Gaze capture trigger (event-agnostic) ─────────────────────────────
    # Someone looking straight at the camera is a portrait-worthy moment in
    # ANY event. When the frame has no STRONGER specific moment, label it
    # gazing_moments so a deliberate look is captured and kept — the dedup
    # guard stops a held gaze from flooding, but every DISTINCT look survives.
    # HIGH priority is granted by decide_capture (faces_looking ≥ 1).
    if (face_count >= 1 and int(gaze.faces_looking) >= 1
            and moment.detected_moment in _GAZE_GENERIC):
        moment.detected_moment = "gazing_moments"
        moment.confidence = float(max(moment.confidence, 0.70))

    # ── Temporal label history (stable capture tags) ──────────────────────
    # Single-frame CLIP flicker can mis-tag a photo (one stray 'cake_feeding'
    # frame inside a gifting scene). Record the FINAL per-frame label; at
    # capture time the photo tag prefers the locally dominant label over a
    # one-frame blip (see the save block). ~1.4s window at 5 fps.
    if not hasattr(session, "_label_hist"):
        session._label_hist = deque(maxlen=7)
    session._label_hist.append((str(moment.detected_moment), float(moment.confidence)))

    # ── Capture decision (unified EventCaptureEngine ensemble) ──────────
    # CHANGED 2026 — priority-tier aware:
    #   We always run decide_capture() so we know the priority tier of
    #   every frame. The cooldown check (session.quality.can_capture)
    #   only gates NORMAL/ELEVATED tier captures. CRITICAL and HIGH
    #   priority captures (prompt-matched moments, intentional camera
    #   gaze) BYPASS the cooldown — they must never be missed.
    captured = False
    capture_reason = None
    captured_url = None
    captured_zoom_url = None
    captured_zoom_info: dict | None = None
    thumb_b64 = None
    decision = None
    audio_events: dict = {}      # filled in the capture block (live/recorded)
    audio_boost = 0.0

    # ── Excluded-class gate ───────────────────────────────────────────────
    # If the user explicitly excluded this moment ("no candle blown" etc.),
    # skip all capture work for this frame entirely.
    moment_class_excluded = (
        hasattr(session, "excluded_classes")
        and moment.detected_moment in session.excluded_classes
    )

    # ── Emotional-intelligence always-capture ─────────────────────────────
    # Regardless of category/prompt, strong positive emotions are always
    # worth capturing: smiling people, surprised faces, happy tears.
    # We detect this here and elevate the frame so the ensemble fires even
    # if the moment class isn't in the active prompt.
    _emotion_happy    = getattr(emotion, "per_emotion", {}).get("happy",     0.0)
    _emotion_surprise = getattr(emotion, "per_emotion", {}).get("surprised", 0.0)
    _emotion_strong   = (_emotion_happy > 0.70 or _emotion_surprise > 0.70)
    if _emotion_strong and face_count >= 1 and not moment_class_excluded:
        # Boost moment confidence so ensemble sees this as a peak
        moment.confidence = float(min(1.0, moment.confidence + 0.20))
        if "emotion_peak" not in (moment.active_moments or []):
            if not hasattr(moment, "_emotion_boosted"):
                moment._emotion_boosted = True

    if session.active and not moment_class_excluded:
        # Look up current learner trust for this class (used as one signal
        # in the ensemble; OnlineLearner.boost() returns base * factor, so
        # we re-derive the rate by sampling at base=1.0).
        learner_keep_rate = LEARNER.boost(session.sid, moment.detected_moment, 1.0)
        if learner_keep_rate is not None and learner_keep_rate <= 0.0:
            learner_keep_rate = None

        # Pull CLIP per-prompt + kept-centroid sim (None if CLIP off)
        clip_per_prompt = (clip_scores.per_prompt
                           if clip_scores and clip_scores.available else None)
        kept_sim        = (clip_scores.similarity_to_kept_centroid
                           if clip_scores else None)

        # ── Required-shot priority boost ──────────────────────────────────
        # If this moment class still has uncaptured required shots, add a
        # score boost so marginal frames still fire. This is what ensures
        # "candle blown" or "group photo" are captured even when the engine
        # score would normally fall just below threshold.
        req_boost = 0.0
        if hasattr(session, "required_tracker"):
            req_boost = session.required_tracker.priority_boost(
                moment.detected_moment)
        # The generic "emotional moment" required shot must see actual EMOTION.
        # Without this, ANY two people in frame get boosted over the threshold
        # (a basketball video fired it with zero emotion signal).
        if moment.detected_moment in _GAZE_GENERIC and float(shot.emotion) < 0.5:
            req_boost = 0.0

        # ── Audio cue boost ───────────────────────────────────────────────
        # Sound tells us WHEN a moment lands: a cheer/applause spike marks the
        # instant of the blow/cut/reveal; singing means a ceremony is underway.
        # Add that as a timing boost so the camera fires on the right frames.
        # Empty (live, silent clip, or no model) → 0.0 → unchanged behaviour.
        audio_events = session.current_audio_events(time.time())
        audio_boost  = audio_capture_boost(audio_events)

        decision = decide_capture(
            moment_class      = moment.detected_moment,
            moment_confidence = float(moment.confidence),
            shot_total        = float(shot.total) + req_boost + audio_boost,
            nima_score        = nima_score,
            emotion           = emotion,
            gaze_ratio        = float(gaze.gaze_ratio),
            face_count        = int(face_count),
            # A background-flat frame's CLIP scores are forced-choice noise (a
            # pool table "matching" cake at 0.65) — they must not push the
            # shutter. Real signals (emotion spike, gaze, audio) still can.
            clip_per_prompt   = (None if _bg_flat else clip_per_prompt),
            kept_centroid_sim = kept_sim,
            predictor_result  = pred,
            learner_keep_rate = learner_keep_rate,
            calibrator        = CALIBRATOR,
            active_classes    = list(session.vlm.active_profiles),
            faces_looking     = int(getattr(gaze, "faces_looking", 0) or 0),
        )
        reasons = list(decision.reasons)

        # ── Detail / object shot (no people, but a clear birthday OBJECT) ─────
        # A photographer also grabs establishing shots — the cake, the gift, the
        # decorated table — with nobody in frame. When there are NO faces yet CLIP
        # confidently sees a birthday object, fire a modest (ELEVATED) capture so
        # these aren't missed. Cooldown still applies (not never-miss) and the
        # per-class time-aware dedup keeps it to a few. Object-set scoped, so it
        # never changes people-moment behaviour.
        # Gate faceless object shots against the BACKGROUND contrast prompts: if
        # the scene matches "pool table / empty room / screen" as well as (or
        # better than) the birthday object, it is NOT a cake — don't capture it.
        _pp0 = (clip_scores.per_prompt or {}) if clip_scores and clip_scores.available else {}
        _bg_best = max((v for k, v in _pp0.items() if k.startswith("bg_")), default=0.0)
        if (face_count == 0 and not decision.triggered
                and moment.detected_moment in _OBJECT_SCENES
                and float(moment.confidence) >= _OBJECT_SHOT_CONF
                and float(moment.confidence) >= _bg_best + _OBJECT_BG_MARGIN):
            decision.triggered = True
            decision.priority_tier = TIER_ELEVATED
            # NOTE: no auto-bypass here — an object shot respects the cooldown so
            # a continuously-moving scene (hands opening a gift) is rate-limited.
            # A genuinely NEW object (a different cake) bypasses via the colour
            # new-scene check below, so each distinct one is still captured.
            reasons.insert(0, "object_detail")

        if req_boost > 0:
            reasons.insert(0, "required_shot_boost")
        if audio_boost > 0.03:
            # Record which sound drove the boost (for the review UI / debugging).
            cue = max(audio_events, key=audio_events.get) if audio_events else "audio"
            reasons.insert(0, f"audio:{cue}")

        # ── Capture the important VOICE ───────────────────────────────────
        # When SOUND alone marks an important moment — a cheer/applause peak
        # (the reaction to candles-blown / cake-cut) or the birthday SONG — we
        # escalate so the moment is never missed, even if vision was quiet.
        # CRITICAL/HIGH bypass the cooldown like any other never-miss moment.
        _A_RANK = {"": 0, TIER_NORMAL: 0, TIER_ELEVATED: 1, TIER_HIGH: 2, TIER_CRITICAL: 3}
        a_tier = audio_priority_tier(audio_events)
        if a_tier and _A_RANK.get(a_tier, 0) > _A_RANK.get(decision.priority_tier, 0):
            decision.priority_tier = a_tier
            cue = max(audio_events, key=audio_events.get) if audio_events else "audio"
            reasons.insert(0, f"voice:{cue}")
            if a_tier in (TIER_CRITICAL, TIER_HIGH):
                decision.triggered = True            # never miss the voice moment
                decision.bypass_throttle = True

        # Event awareness — "can the model feel a birthday is happening?" The
        # audio ceremony cue (the song/cheer) is the reliable half; surfaced so
        # the dashboard shows the system sensing the event in real time.
        session.birthday_awareness = round(float(birthday_audio_awareness(audio_events)), 3)
        # The birthday song / a cheer IS event evidence — arm the people-
        # triggers even before anything event-ish is visible on camera.
        # Only a cheer/applause PEAK proves an event by sound. Singing alone
        # (ELEVATED) must not — every music reel has vocals, and it was arming
        # the gate for random romantic/serial clips with a soundtrack.
        if a_tier in (TIER_CRITICAL, TIER_HIGH):
            session._event_evidence = True
        # Also include the heuristic best-shot rule as a reason if the
        # ensemble triggered and the frame is genuinely a peak — keeps
        # backward-compat with existing capture_reason consumers.
        if decision.triggered and session.quality.is_best_shot(shot):
            if "best_shot" not in reasons: reasons.insert(0, "best_shot")

        # Priority-tier gating of the cooldown.
        # Normal/Elevated must still wait for can_capture(); Critical/High
        # bypass it — the system promised the user we'd never miss a
        # cake-cutting or a deliberately posed group, so we honor that
        # even if we just took a photo 200ms ago.
        #
        # IMPORTANT: bypass is NOT unconditional. Even priority captures
        # respect a small "priority floor" interval (default 0.4s) so we
        # don't fire on every single frame of a long cake-cutting
        # sequence. Without this floor a 10-second priority moment would
        # produce ~80 captures, which is terrible UX. Configurable via
        # SNAPPY_PRIORITY_FLOOR_SEC (default 0.4).
        # New-scene bypass — only a real scene CUT (consecutive-frame jump, not
        # "differs from the last capture") counts as new content. Hash-distance
        # from the last capture is trivially large under a handheld camera and
        # was machine-gunning 47 captures in 41s; a cut is content actually
        # changing (measured: cuts 30+, handheld pan 5–20).
        if (decision.triggered and not decision.bypass_throttle
                and (_clock - getattr(session, "_last_cut_ts", -999.0)) <= 1.0):
            decision.bypass_throttle = True
            if "new_scene" not in reasons:
                reasons.insert(0, "new_scene")

        # Use the SAME deterministic clock as everything else (_clock: media
        # time for uploads, a frame-count-based synthetic clock live) — NOT
        # wall-clock. An uploaded video's PROCESSING speed varies run to run
        # (server load, warm vs cold model caches), so time.time() made the
        # cooldown/priority-floor gates non-deterministic: identical video
        # content produced a different capture count each time it was
        # (re-)uploaded, purely from incidental server timing.
        cooldown_ok = session.quality.can_capture(_clock)
        priority_bypass = bool(decision.bypass_throttle)
        if priority_bypass:
            try:
                floor_sec = float(os.environ.get("SNAPPY_PRIORITY_FLOOR_SEC", "0.4"))
            except Exception:
                floor_sec = 0.4
            last_cap = getattr(session, "_last_priority_cap_ts", float("-inf"))
            now_ts   = _clock
            if (now_ts - last_cap) < floor_sec:
                # Within the priority floor — don't bypass even though we
                # are CRITICAL/HIGH. The cooldown is honored. This still
                # gives us 2-3 captures per second on priority moments,
                # which is what photographers do.
                priority_bypass = False
        gated_ok = cooldown_ok or priority_bypass

        if decision.triggered and not gated_ok:
            log.debug(
                f"[{session.sid}] capture suppressed by cooldown "
                f"(tier={decision.priority_tier}, score={decision.final_score:.2f})"
            )

        # Personalization gate — keep the album to the HOST group: suppress a
        # capture that contains ONLY strangers (with a tiny budget for a stranger
        # gazing right at the lens). No-op until host inference is confident, so
        # it never hurts cold-start or single-subject sessions.
        gaze_strong = (int(getattr(gaze, "faces_looking", 0) or 0) >= 1
                       or float(getattr(gaze, "gaze_ratio", 0.0)) >= 0.6)
        # Personalization filters PEOPLE shots (strangers). A faceless detail /
        # object shot (the cake, a gift) has no person to filter, so it always
        # clears this gate — only people frames are subject to host filtering.
        host_ok = (face_count == 0) or session.host_group.should_capture(
            frame_host_score, n_host_faces, gaze_strong)
        if decision.triggered and gated_ok and capture_allowed and not host_ok:
            log.debug(f"[{session.sid}] capture suppressed — only strangers "
                      f"(host={frame_host_score:.2f}, n_host={n_host_faces})")

        # FACELESS object scenes get ONE shot per ~4s unless a scene CUT showed
        # new content. A handheld drift over the same gift box must not machine-
        # gun 17 near-identical captures; a montage's cuts (new cake) unlock
        # immediately, and people moments are never rate-limited here.
        _obj_ok = True
        if face_count == 0 and decision.triggered:
            _since_cap = _clock - getattr(session, "_last_objcap_ts", -999.0)
            _since_cut = _clock - getattr(session, "_last_cut_ts", -999.0)
            _obj_ok = (_since_cap >= 3.5) or (_since_cut <= 1.0)
            if not _obj_ok:
                log.debug(f"[{session.sid}] faceless capture rate-limited "
                          f"(since_cap={_since_cap:.1f}s, no recent cut)")

        # EVENT-EVIDENCE gate (birthday): capture NOTHING until the session has
        # actually seen the event — some class (cake/gifts/candles/group…) held
        # 2 strong frames, or the birthday song/cheer was heard. People-triggers
        # (emotion/gaze/required "emotional moment") otherwise fire on any video
        # of people — a random interview is not a birthday. At a real party the
        # evidence arrives within the first seconds, so nothing real is lost.
        _ctx_ok = True
        if (decision.triggered
                and _resolve_event(getattr(session, "event_type", "")) == "birthday"
                and not getattr(session, "_event_evidence", False)):
            _ctx_ok = False
            log.debug(f"[{session.sid}] capture suppressed — no event evidence "
                      f"yet (moment={moment.detected_moment})")

        if decision.triggered and gated_ok and capture_allowed and host_ok and _obj_ok and _ctx_ok:
            # For a faceless OBJECT/DETAIL shot (a cake, a gift) use the CURRENT
            # frame — the thing on screen NOW — not a preroll pick, which for a
            # fast montage of different cakes would grab a stale earlier cake and
            # collapse distinct ones. For people moments keep the preroll peak
            # (the exact smile/blow).
            if face_count == 0:
                best_frame, best_score = full_frame, float(shot.total)
            else:
                # Photo source = the FULL-RES ring (models ran on a proxy; the
                # photo carries every pixel the camera delivered). Proxy preroll
                # only as a last resort.
                _ring = getattr(session, "_fullres_ring", None)
                if _ring:
                    best_frame, best_score = max(_ring, key=lambda t: t[1])
                else:
                    best_frame, best_score = session.predictor.preroll.best()
                    if best_frame is None:
                        best_frame, best_score = full_frame, float(shot.total)
            session._last_cap_hash = _phash(best_frame)    # for the new-scene bypass
            session._last_cap_color = _csig(best_frame)

            # ── 1. Enhance the wide frame (auto brightness/contrast/sat/sharp)
            wide_enh, wide_report = enhance(best_frame, auto=True)
            wide_h, wide_w = wide_enh.shape[:2]
            face_boxes = [f.box for f in face_frame.faces] if face_frame.available else []
            # Faces were detected on the PROXY — map boxes to the full-res photo.
            _sf = wide_w / max(1.0, float(frame.shape[1]))
            if abs(_sf - 1.0) > 0.01:
                face_boxes = [(int(x * _sf), int(y * _sf), int(w * _sf), int(h * _sf))
                              for (x, y, w, h) in face_boxes]

            # ── 2. Determine which zoom-hints to save.
            # Normal path: one photo with auto zoom (subject-distance logic).
            # Multi-angle path: when a required_shots multi-angle set is active
            # (e.g. cake_cutting + 3+ faces = person cutting + knife + wide),
            # save all three angles in one trigger.  Each angle counts as a
            # separate captured photo.
            zoom_hints_to_save: list[str] = []
            if (hasattr(session, "required_tracker")
                    and session.required_tracker.needs_multi_angle(
                        moment.detected_moment, face_count)):
                zoom_hints_to_save = session.required_tracker.get_multi_angle_zoom_hints(
                    moment.detected_moment)
                log.info(f"[{session.sid}] multi-angle burst "
                         f"({len(zoom_hints_to_save)} shots) for "
                         f"{moment.detected_moment}")
            if not zoom_hints_to_save:
                zoom_hints_to_save = ["auto"]   # single standard photo

            # Required-shot bypass classes for dedup: classes whose required
            # shots are still uncaptured bypass the duplicate check so the
            # very first genuine instance is always captured.
            req_bypass = set()
            if hasattr(session, "required_tracker"):
                req_bypass = session.required_tracker.get_uncaptured_classes()

            if priority_bypass and not cooldown_ok:
                reasons.insert(0, f"bypass_cooldown:{decision.priority_tier}")
            if decision.priority_tier in ("critical", "high"):
                session._last_priority_cap_ts = _clock

            # Dedup "held-moment" refresh clock: MEDIA time for uploads (so a clip
            # deterministically yields spaced shots across its whole duration —
            # box → opening → reveal — no matter how fast it's processed), and
            # wall-clock for live/replay (real time). This is what makes the gift
            # SEQUENCE captured consistently instead of clustering on one phase.
            _cap_ts = (float(getattr(session, "_media_ts", 0.0)) if source == "upload"
                       else session.frame_count / 5.0)   # deterministic ~5fps live clock

            # Smoothed capture TAG — a one-frame label blip must not mis-tag the
            # photo. The locally dominant label (conf-weighted over the ~1.4s
            # history) replaces the current one only when it CLEARLY dominates
            # (1.5×), so genuine new moments aren't lagged. candle_blowing and
            # cake_smashing are exempt: brief, phase/geometry-confirmed moments.
            _tag = str(moment.detected_moment)
            if _tag not in ("candle_blowing", "cake_smashing"):
                _sums: dict = {}
                for _l, _c in getattr(session, "_label_hist", []):
                    _sums[_l] = _sums.get(_l, 0.0) + max(0.05, _c)
                if _sums:
                    _dom = max(_sums, key=_sums.get)
                    if _dom != _tag and _sums[_dom] >= 1.5 * _sums.get(_tag, 0.0):
                        _tag = _dom

            for zoom_hint in zoom_hints_to_save:
                # ── 3. Build the final image for this angle ─────────────────
                roi = compute_zoom_roi(
                    face_boxes   = face_boxes,
                    moment_class = moment.detected_moment,
                    frame_w      = wide_w, frame_h = wide_h,
                    zoom_hint    = zoom_hint,
                )
                zoom_info: dict | None = None
                if roi is not None:
                    crop  = wide_enh[roi.y:roi.y + roi.h, roi.x:roi.x + roi.w]
                    # Cap digital upscale at 2× — beyond that a crop just looks
                    # soft (interpolation can't add detail that isn't there).
                    # On SMALL sources (<720p) don't upscale at all: blowing up
                    # a low-res crop + sharpening it is what makes photos look
                    # "processed"/degraded. A small sharp crop beats a big soft one.
                    scale = max(1.0, min(wide_h / max(1, crop.shape[0]), 2.0))
                    if wide_h < 720 or crop.shape[0] < 300:
                        scale = 1.0
                    if scale > 1.05:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                                          interpolation=cv2.INTER_LANCZOS4)
                    final_img, final_report = enhance(crop, auto=True)
                    if scale > 1.05:
                        # Gentle extra unsharp to counter digital-zoom softness —
                        # kept mild (1.35) so it doesn't halo/amplify JPEG noise.
                        _b = cv2.GaussianBlur(final_img, (0, 0), 1.2)
                        final_img = np.clip(
                            cv2.addWeighted(final_img, 1.35, _b, -0.35, 0), 0, 255
                        ).astype(np.uint8)
                    zoom_info = roi.to_dict()
                else:
                    final_img, final_report = wide_enh, wide_report

                # ── 4. Near-duplicate guard — per-class, with required bypass
                if session.dup_guard.is_duplicate(
                        final_img, _tag, req_bypass, now=_cap_ts,
                        color_img=best_frame):   # judge colour on the RAW frame (pre-enhance)
                    log.info(f"[{session.sid}] skipped near-duplicate capture "
                             f"(moment={_tag}, hint={zoom_hint}, "
                             f"tier={decision.priority_tier})")
                    continue   # try next angle

                # ── 5. Save the image (dir created on first use) ────────────
                session.ensure_dir()
                stamp = int(time.time() * 1000)
                fname = f"snap_{session.capture_count:04d}_{stamp}.jpg"
                fpath = str(session.dir / fname)
                # 95: captures are the product — minimise re-encode generation
                # loss (browser JPEG → enhance → save is already 2 generations).
                cv2.imwrite(fpath, final_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                session.dup_guard.remember(final_img, _tag, now=_cap_ts,
                                           color_img=best_frame)
                if face_count == 0:
                    session._last_objcap_ts = _clock
                if zoom_info is not None:
                    log.info(f"[{session.sid}] zoomed capture {fname} "
                             f"({roi.zoom_factor:.1f}×, hint={zoom_hint})")

                # ── 6. Thumbnail ────────────────────────────────────────────
                small = cv2.resize(final_img, (160, 120))
                _, enc = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 60])
                thumb_b64 = base64.b64encode(enc.tobytes()).decode()

                tags = list(set(reasons + [session.event_type, _tag]))
                photo_url = f"/captures/{session.dir.name}/{fname}"

                pe = PhotoEntry(
                    filepath=fpath, url=photo_url,
                    timestamp=float(time.time()), quality_score=float(best_score),
                    face_count=face_count, emotion_score=float(shot.emotion),
                    gaze_triggered=bool(gaze.is_group_looking),
                    moment_type=str(_tag),
                    moment_conf=float(moment.confidence), tags=tags,
                )
                setattr(pe, "zoom_url", None)
                setattr(pe, "zoom_info", zoom_info)
                setattr(pe, "enhance_report", final_report.to_dict())
                # RAW-frame colour signature (pre-enhance) — album dedups on this
                # so different-coloured cakes stay distinct even after white-balance.
                _cakesig = _csig(best_frame)
                setattr(pe, "color_sig", _cakesig.tolist() if _cakesig is not None else None)
                session.photos.append(pe)
                session.capture_count += 1
                session.quality.record_capture(_clock)
                session.predictor.confirm_capture()
                captured       = True
                capture_reason = reasons[0] if reasons else "engine"
                captured_url   = photo_url
                captured_zoom_url  = None
                captured_zoom_info = zoom_info
                session.last_capture_frame = best_frame.copy()

                # Advance the event-progress cursor (monotonic) so the
                # sequence prior knows how far the event has come.
                if _seq_event and _seq_stage != _UNKNOWN_STAGE \
                        and _seq_stage > session.event_stage_cursor:
                    session.event_stage_cursor = _seq_stage

                # ── 7. Mark required shot as fulfilled ──────────────────────
                if hasattr(session, "required_tracker"):
                    marked = session.required_tracker.on_captured(
                        moment_class   = moment.detected_moment,
                        zoom_hint_used = zoom_hint,
                        capture_url    = photo_url,
                        face_count     = face_count,
                    )
                    if marked:
                        # Tag the photo so album curation can force-include it
                        # and never drop a captured required shot in filtering.
                        for shot_id in marked:
                            tag = f"required:{shot_id}"
                            if tag not in pe.tags:
                                pe.tags.append(tag)
                        log.info(f"[{session.sid}] required shot(s) fulfilled: "
                                 f"{marked} (hint={zoom_hint})")

                if STORE is not None:
                    try:
                        STORE.add_photo(session.sid, {
                            "url": pe.url, "filepath": pe.filepath, "ts": pe.timestamp,
                            "score": pe.quality_score, "faces": pe.face_count,
                            "emotion": pe.emotion_score, "gaze": pe.gaze_triggered,
                            "moment": pe.moment_type, "moment_conf": pe.moment_conf,
                            "tags": pe.tags,
                        })
                    except Exception as ex:
                        log.warning(f"Persist photo failed: {ex}")
                # end for zoom_hint in zoom_hints_to_save

    # ── Live frame gallery ────────────────────────────────────────────────
    small_frame = cv2.resize(frame, (160, 120))
    _, enc = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 45])
    frame_b64_small = base64.b64encode(enc.tobytes()).decode()
    gallery_entry = safe({
        "id":      session.frame_count,
        "b64":     frame_b64_small,
        "score":   float(shot.total),
        "faces":   face_count,
        "moment":  str(moment.detected_moment),
        "mconf":   float(moment.confidence),
        "gaze":    bool(gaze.is_group_looking),
        "captured": bool(captured),
        "ts":      float(time.time()),
    })
    session.frame_gallery.append(gallery_entry)
    if len(session.frame_gallery) > session.MAX_GALLERY:
        session.frame_gallery.pop(0)

    # ── Rescue reservoir (live only) ──────────────────────────────────────
    # A sparse rolling JPEG buffer of the whole session (~1 frame / 2s, thinned
    # by halving when full). If the session ends with ZERO captures, the album
    # can ask the VLM to look at these and rescue moments frame-level vision
    # missed (e.g. a video-call birthday where a screen dominates the frame).
    if source == "live":
        if not hasattr(session, "_reservoir"):
            session._reservoir = []; session._res_last = -999.0; session._res_gap = 1.0
        if _clock - session._res_last >= session._res_gap:
            _im = full_frame
            _rh, _rw = _im.shape[:2]
            if _rw > 960:
                _im = cv2.resize(_im, (960, int(_rh * 960 / _rw)))
            _okE, _buf = cv2.imencode(".jpg", _im, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if _okE:
                session._reservoir.append((float(_clock), _buf.tobytes()))
                session._res_last = _clock
                if len(session._reservoir) > 96:
                    session._reservoir = session._reservoir[::2]
                    session._res_gap *= 2

    # ── Setup coach (live camera only) ────────────────────────────────────
    # The photographer's assistant: light, steadiness, framing, and keeping
    # the cake table in view — computed from signals this frame already has.
    setup_info = None
    if source == "live":
        _pp_s = clip_scores.per_prompt if clip_scores and clip_scores.available else None
        _bg_s = max((v for k, v in (_pp_s or {}).items() if k.startswith("bg_")),
                    default=0.0)
        _boxes_s = [f.box for f in face_frame.faces] if face_frame.available else []
        # THROTTLED cake-locating scan (3 extra CLIP crops) — only every ~4s, and
        # only while the cake isn't already in frame. That's what lets the coach
        # say "the cake is to your LEFT, turn that way" instead of just "not
        # visible". Cost at 5fps: ~3 CLIP calls per 20 frames.
        _side = getattr(session, "_setup_side", None)
        _obj_now = max((( _pp_s or {}).get(k, 0.0) for k in
                        ("cake", "cake_with_candles", "cake_person", "food_table",
                         "birthday_gifting")), default=0.0)
        if (_obj_now - _bg_s) >= 0.05:
            _side = None                       # already framed — nothing to point at
            session._setup_side = None
        elif _clock - float(getattr(session, "_setup_scan_ts", -999.0)) >= 4.0:
            session._setup_scan_ts = _clock
            try:
                _side = _setup_locate(frame, CLIP,
                                      _resolve_event(getattr(session, "event_type", "")))
            except Exception:
                _side = None
            session._setup_side = _side
        setup_info = _setup_advise(frame, _boxes_s, _pp_s, _bg_s,
                                   int(getattr(session, "_last_jitter", 0)),
                                   event_side=_side)

    # ── Response payload ─────────────────────────────────────────────────
    return safe({
        "type":     "frame_result",
        "frame_id": int(session.frame_count),
        "analysis": shot.to_dict(),
        "face_detection": {
            "available": bool(face_frame.available),
            "backend":   str(face_frame.backend),
            "face_count": int(face_count),
            "boxes":     _display_face_boxes(face_frame, frame, full_frame),
            "error":     face_frame.error,
        },
        "gaze":       gaze.to_dict(),
        "moment":     moment.to_dict(),
        "emotion":    emotion.to_dict(),
        "prediction": pred.to_dict(),
        "clip": {
            "available": bool(CLIP.available),
            "backend":   "transformers" if CLIP.available else "heuristic",
            "best":      clip_scores.best if clip_scores and clip_scores.available else None,
            "best_score": (clip_scores.best_score
                           if clip_scores and clip_scores.available else None),
            "infer_ms":  (round(clip_scores.inference_ms, 1)
                          if clip_scores and clip_scores.available else None),
            "kept_centroid_sim": (clip_scores.similarity_to_kept_centroid
                                  if clip_scores else None),
            "init_error": CLIP.init_error,
        },
        "nima": {
            "available": bool(NIMA.available),
            "score":     float(nima_score) if nima_score is not None else None,
        },
        "hsemotion": {"available": bool(HSEMO.available)},
        "engine": decision.to_dict() if decision is not None else None,
        "calibration": ({"p": decision.calibrated_p}
                        if decision is not None and decision.calibrated_p is not None
                        else None),
        "setup":            setup_info,
        "captured":         bool(captured),
        "capture_reason":   capture_reason,
        "captured_url":     captured_url,
        "captured_zoom_url":  captured_zoom_url,
        "captured_zoom_info": captured_zoom_info,
        "captured_thumb":   thumb_b64,
        "total_captures":   int(session.capture_count),
        "score_timeline":   [round(float(s), 3) for s in session.score_timeline[-60:]],
        "gallery_frame":    gallery_entry,
        "active_moments":   [str(m) for m in session.vlm.get_active_moment_names()],
        # Developer-mode: surface the audio + sequence models too, so the dev
        # dashboard can show every model's live activity, not just vision.
        "audio": {
            "events": {k: round(float(v), 3) for k, v in (audio_events or {}).items()},
            "boost":  round(float(audio_boost), 3),
            "voice_tier": audio_priority_tier(audio_events) or "",
            "birthday_awareness": float(getattr(session, "birthday_awareness", 0.0)),
        },
        "sequence": {
            "event": _seq_event or "",
            "stage": (None if _seq_stage >= _UNKNOWN_STAGE else int(_seq_stage)),
            "cursor": int(getattr(session, "event_stage_cursor", -1)),
        },
    })


# ── Album-time speech context ─────────────────────────────────────────────
def session_speech_text(session) -> "str | None":
    """Transcribe what was SAID during the session (album-time, never live).

    Audio source: the uploaded video's track (kept at extraction) or the
    accumulated live/replay mic chunks. 'Happy birthday to you' heard = the
    strongest occasion evidence there is. None when no audio / model off.
    """
    try:
        # Check whether there's any audio BEFORE paying Whisper's model-load
        # cost (multi-second CPU load first call) — a pure photo-import
        # session (Camera/Pro tier batch-import, no live mic/video) has
        # neither, and used to pay that cost for nothing every single time.
        wav = getattr(session, "_speech_wav", None)
        if wav is None:
            chunks = list(getattr(session, "_speech_pcm", []) or [])
            if not chunks:
                return None
            wav = np.concatenate(chunks)

        from models.speech_transcriber import SpeechTranscriber
        st = SpeechTranscriber.get()
        if not st.ok:
            return None
        text = st.transcribe(wav)
        if text:
            log.info(f"[{session.sid}] speech heard: {text[:120]!r}")
        return text
    except Exception as e:
        log.debug(f"speech context skipped: {e}")
        return None


# ── Zero-capture rescue ───────────────────────────────────────────────────
def rescue_zero_capture(session) -> None:
    """When a session ends with NO captures, let the VLM re-examine the video.

    Frame-level vision has a floor: a video-call birthday (real cake, but a
    phone screen dominating the frame) scores like junk to CLIP. The VLM
    understands the scene — so at album time, when there is NOTHING to curate,
    it answers a strict YES/NO ("is this a birthday celebration?") on ~10
    sampled frames and rescues the ones it confirms. Junk stays junk: measured,
    the binary check said NO on every junk fixture. Album-time only.
    """
    try:
        if session.photos:
            return
        from models.vlm_tagger import VLMTagger
        vlm = VLMTagger.get()
        if not vlm.ok:
            log.info(f"[{session.sid}] zero-capture rescue skipped — VLM unavailable")
            return
        # Collect candidate frames: live/replay → reservoir; upload → the file.
        frames = []
        res = list(getattr(session, "_reservoir", []) or [])
        if res:
            idxs = np.linspace(0, len(res) - 1, min(10, len(res))).astype(int)
            for i in idxs:
                arr = np.frombuffer(res[int(i)][1], np.uint8)
                im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if im is not None:
                    frames.append((res[int(i)][0], im))
        elif getattr(session, "_video_path", None):
            cap = cv2.VideoCapture(session._video_path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            for fi in np.linspace(5, max(6, n - 5), 10).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, im = cap.read()
                if ok:
                    frames.append((fi / fps, im))
            cap.release()
        if not frames:
            return
        yes = [(t, im) for (t, im) in frames if vlm.is_birthday(im) is True]
        if len(yes) == 1:
            # Borderline: one confirmed frame could be a fluke. Arbitrate with
            # the SET-level question — the VLM sees all sampled frames together
            # (a dark cinema + a Happy-Birthday montage + a person in tears
            # reads as a birthday surprise as a WHOLE, not frame by frame).
            occ = vlm.occasion([im for (_, im) in frames]) or ""
            if any(k in occ.lower() for k in ("birthday", "party", "celebr",
                                              "anniversar", "wedding", "surprise")):
                log.info(f"[{session.sid}] zero-capture rescue: 1 frame + "
                         f"set-level occasion {occ!r} — confirmed")
                idx = frames.index(yes[0])
                for j in (idx - 1, idx + 1):
                    if 0 <= j < len(frames):
                        yes.append(frames[j])
                yes.sort(key=lambda t: t[0])
        if len(yes) < 2:                     # junk stays out
            log.info(f"[{session.sid}] zero-capture rescue: "
                     f"{len(yes)}/{len(frames)} frames confirmed — not rescuing")
            return
        # Keep up to 4, spread across the video.
        keep = yes if len(yes) <= 4 else [yes[i] for i in
                                          np.linspace(0, len(yes) - 1, 4).astype(int)]
        session.ensure_dir()
        for t, im in keep:
            lbl = vlm.tag(im)
            if not lbl or lbl == "not_birthday":
                lbl = "cake_with_candles"
            enh, rep = enhance(im, auto=True)
            stamp = int(time.time() * 1000)
            fname = f"snap_{session.capture_count:04d}_{stamp}.jpg"
            fpath = str(session.dir / fname)
            cv2.imwrite(fpath, enh, [cv2.IMWRITE_JPEG_QUALITY, 95])
            pe = PhotoEntry(
                filepath=fpath, url=f"/captures/{session.dir.name}/{fname}",
                timestamp=float(time.time()) + t, quality_score=0.55,
                face_count=0, emotion_score=0.0, gaze_triggered=False,
                moment_type=str(lbl), moment_conf=0.75,
                tags=["rescue", session.event_type, str(lbl)],
            )
            setattr(pe, "zoom_url", None); setattr(pe, "zoom_info", None)
            setattr(pe, "enhance_report", rep.to_dict())
            session.photos.append(pe)
            session.capture_count += 1
        log.info(f"[{session.sid}] zero-capture rescue: saved {len(keep)} "
                 f"VLM-confirmed moments")
    except Exception as e:
        log.debug(f"zero-capture rescue skipped: {e}")


# ── Finale (closing shot) ─────────────────────────────────────────────────
def capture_finale(session) -> None:
    """Photographer's CLOSING SHOT, taken when the session ends / album builds.

    The last thing the camera saw is often the payoff — the FINISHED cake after
    a decorating sequence, the full table, the final pose. If the stream ended
    while the faceless rate-limit was still cooling, that ending was never
    captured. The full-res ring still holds the final ~1.2s of frames: when the
    tail is event-relevant and NOT a near-duplicate of an existing photo, save
    the best of it as one final shot.
    """
    try:
        ring = list(getattr(session, "_fullres_ring", []) or [])
        if not ring or not getattr(session, "_event_evidence", False):
            return
        # Majority label over the final frames — must be a real event moment.
        hist = list(getattr(session, "_label_hist", []) or [])[-6:]
        if not hist:
            return
        sums: dict = {}
        for lbl, cf in hist:
            sums[lbl] = sums.get(lbl, 0.0) + max(0.05, float(cf))
        label = max(sums, key=sums.get)
        if label in ("general_peak", "gazing_moments", "watching", ""):
            return
        best_frame, best_score = max(ring, key=lambda t: t[1])
        enh, rep = enhance(best_frame, auto=True)
        # Near-dup of something already captured → the ending is already covered.
        if session.dup_guard.is_duplicate(enh, label, None, color_img=best_frame):
            return
        session.ensure_dir()
        stamp = int(time.time() * 1000)
        fname = f"snap_{session.capture_count:04d}_{stamp}.jpg"
        fpath = str(session.dir / fname)
        cv2.imwrite(fpath, enh, [cv2.IMWRITE_JPEG_QUALITY, 95])
        session.dup_guard.remember(enh, label, color_img=best_frame)
        pe = PhotoEntry(
            filepath=fpath, url=f"/captures/{session.dir.name}/{fname}",
            timestamp=float(time.time()), quality_score=float(max(best_score, 0.5)),
            face_count=0, emotion_score=0.0, gaze_triggered=False,
            moment_type=str(label), moment_conf=0.7,
            tags=["finale", session.event_type, str(label)],
        )
        setattr(pe, "zoom_url", None); setattr(pe, "zoom_info", None)
        setattr(pe, "enhance_report", rep.to_dict())
        session.photos.append(pe)
        session.capture_count += 1
        log.info(f"[{session.sid}] finale shot saved ({label}) → {fname}")
    except Exception as e:
        log.debug(f"finale capture skipped: {e}")


# ── Async dispatch ────────────────────────────────────────────────────────
async def process_frame_async(session, frame_b64: str) -> dict:
    """Run process_frame on the worker pool. Asyncio loop never blocks."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, process_frame, session, frame_b64)


async def run_in_pipeline_executor(fn, *args):
    """Run any blocking call (e.g. Custom Mode's VLM intent-parse) on the SAME
    worker pool frames use, off the asyncio event loop. Public wrapper so
    callers outside this module (api/server.py) don't need to reach into the
    module-private _executor directly."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, fn, *args)


# ── Video upload processing ──────────────────────────────────────────────
async def process_video_async(session, video_path: str, sample_fps: float = 2.0):
    """Extract frames from an uploaded video and run them through the SAME
    per-frame pipeline as the live camera. Runs on a dedicated worker so live
    frames keep moving on the other workers.

    Live vs. recorded — the MODEL is identical, the CADENCE differs
    ──────────────────────────────────────────────────────────────────────
    Every frame — live or recorded — goes through the exact same models
    (faces, gaze, CLIP, NIMA, emotion), the same ensemble `decide_capture`,
    the same priority tiers, the same smart-zoom and the same near-duplicate
    guard. A cake-cutting is judged by the identical logic either way. Only
    HOW frames arrive differs, and that creates three deliberate differences:

      1. Sampling. Live processes whatever the camera streams (~10-15 fps).
         Recorded is sub-sampled to `sample_fps` (default 2 fps) — we don't
         need every frame of a file to find its peaks, and 2 fps keeps a long
         clip fast to analyse.
      2. Thresholds. Recorded clips (phone/browser-recompressed) are usually
         softer than a live sensor feed, so video mode lowers the capture
         interval (0.5 s) and the best-shot threshold (0.42) so we don't miss
         genuine moments to compression artefacts. Restored afterwards.
      3. Timing pressure. Live must decide in real time and leans on the
         pre-roll buffer to recover the best nearby frame. Recorded has the
         whole file, so timing is relaxed.

    WHY they still feel the same: the capture DECISION is threshold-driven and
    identical; only the input rate and two thresholds move. So a moment that
    fires live will fire on the same footage recorded (and vice-versa), just
    sampled coarser. Recorded mode is our test harness for the live product.

    On completion this auto-generates the curated album (named after the
    uploaded video) and ends the session — one recorded video, one album.
    """
    loop = asyncio.get_event_loop()

    def _run():
        log.info(f"Video [{session.sid}]: opening {video_path}")
        session._video_path = video_path   # kept for the zero-capture rescue
        try:
            cap = cv2.VideoCapture(video_path)
        except Exception as e:
            session.video_status = "error"
            session.video_error = f"VideoCapture raised: {e}"
            log.exception(f"Video [{session.sid}] open failed")
            return

        if not cap.isOpened():
            try: size = os.path.getsize(video_path)
            except Exception: size = -1
            session.video_status = "error"
            session.video_error = (
                f"Could not open video ({size} bytes). Likely an unsupported codec. "
                "Try converting with ffmpeg: "
                "ffmpeg -i input.mov -c:v libx264 -preset fast output.mp4"
            )
            log.warning(f"Video [{session.sid}] open failed — file unreadable by OpenCV")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        step         = max(1, round(src_fps / sample_fps))
        sampled_total = max(1, total_frames // step) if total_frames > 0 else 1
        session.video_total_frames     = sampled_total
        session.video_processed_frames = 0
        session.video_status           = "processing"
        log.info(f"Video [{session.sid}]: {total_frames} frames @ {src_fps:.1f} fps, "
                 f"{width}x{height}, sampling every {step} → ~{sampled_total} frames "
                 f"(target {sample_fps} fps)")

        # Each uploaded video is a fresh, self-contained job: re-arm capture
        # (a previous video may have auto-ended this session), start the
        # near-duplicate window clean, and remember where THIS video's photos
        # begin so the album below covers only this video.
        session.active = True
        session.video_album = None
        session.dup_guard.reset()
        if hasattr(session, "required_tracker"):
            session.required_tracker.reset()
        if hasattr(session, "candle_flame_history"):
            session.candle_flame_history.clear()
        if hasattr(session, "candle_face_y_history"):
            session.candle_face_y_history.clear()
        if hasattr(session, "clip_offprompt_history"):
            session.clip_offprompt_history.clear()
        session.event_stage_cursor = -1
        session.audio_timeline = []
        session._media_ts = 0.0
        album_start_idx = len(session.photos)

        # ── Audio cue timeline ────────────────────────────────────────────
        # If the clip carries sound, classify its whole soundtrack ONCE into a
        # time-stamped event timeline (cheering/applause/singing/laughter). The
        # per-frame pipeline then looks these up by timestamp to boost capture
        # timing — a cheer/applause spike marks the instant a moment lands
        # (blow, cut, reveal). No audio / no model → empty timeline → no-op.
        if AUDIO is not None:
            try:
                wav = extract_audio_16k_mono(video_path)
                if wav is not None:
                    session._speech_wav = wav       # kept for album-time speech
                    session.audio_timeline = AUDIO.classify_timeline(wav)
                    n_peaks = sum(1 for w in session.audio_timeline
                                  if celebration_intensity(w["events"]) > 0.4)
                    log.info(f"Video [{session.sid}] audio: "
                             f"{len(session.audio_timeline)} windows, "
                             f"{n_peaks} celebration peaks")
                else:
                    log.info(f"Video [{session.sid}] no audio track — "
                             f"vision-only capture")
            except Exception as ex:
                log.warning(f"Video [{session.sid}] audio analysis skipped: {ex}")

        # Video mode: relax the capture cooldown AND lower the quality
        # best-shot threshold a bit. Live camera footage is usually higher
        # quality than recorded/uploaded video (screen recordings, phone
        # clips downscaled by browsers, etc.), so the bar should be lower
        # in video mode or we'll miss everything.
        original_min_interval = session.quality.MIN_INTERVAL
        original_best_thr     = getattr(session.quality, "BEST_SHOT_THRESHOLD", None)
        session.quality.MIN_INTERVAL = 0.5      # half-second floor for video
        if hasattr(session.quality, "BEST_SHOT_THRESHOLD"):
            session.quality.BEST_SHOT_THRESHOLD = 0.42
        log.info(f"Video [{session.sid}] using capture interval "
                 f"{session.quality.MIN_INTERVAL}s (live default was "
                 f"{original_min_interval}s) + relaxed best-shot threshold")

        idx = 0
        consecutive_errors = 0
        first_error_logged = False
        while True:
            try:
                ret, frame = cap.read()
            except Exception as e:
                log.exception(f"Video [{session.sid}] cap.read() raised at frame {idx}")
                session.video_status = "error"
                session.video_error = f"cv2.read failed at frame {idx}: {e}"
                break
            if not ret:
                break
            if idx % step == 0:
                if frame is None or frame.size == 0:
                    consecutive_errors += 1
                else:
                    try:
                        # Tell process_frame where we are in media time so it can
                        # look up the audio cue for this frame.
                        session._media_ts = idx / float(src_fps)
                        # process_frame skips its own decode when frame is supplied.
                        # source="upload" → takes session ownership; live frames
                        # into this session are refused (no mixing).
                        res = process_frame(session, "", frame=frame, source="upload")
                        # Stream this frame's telemetry to the dev dashboard so
                        # the model-activity bars animate through the uploaded
                        # clip in step with playback (no-op if nobody watching).
                        try:
                            from api.globals import push_to_session
                            res["media_ts"] = round(session._media_ts, 2)
                            res["source"] = "upload"
                            push_to_session(session.sid, res)
                        except Exception:
                            pass
                        consecutive_errors = 0
                    except Exception as e:
                        consecutive_errors += 1
                        if not first_error_logged:
                            log.exception(f"Video [{session.sid}] frame {idx}: {e}")
                            first_error_logged = True
                        elif consecutive_errors == 25:
                            log.warning(f"Video [{session.sid}] {consecutive_errors} "
                                        f"consecutive frame errors — bailing out")
                            session.video_status = "error"
                            session.video_error = f"25+ consecutive frame errors — last: {e}"
                            break
                session.video_processed_frames += 1
                if session.video_total_frames > 0:
                    session.video_progress = min(99, int(
                        100 * session.video_processed_frames / session.video_total_frames))
                # Periodic progress log so a hung video is obvious in the log
                if session.video_processed_frames % 25 == 0:
                    log.info(f"Video [{session.sid}] progress: "
                             f"{session.video_processed_frames}/{session.video_total_frames} "
                             f"= {session.video_progress}%, captures={session.capture_count}")
            idx += 1

        cap.release()
        # Restore the original (live) capture interval + threshold so any
        # later live session on this Session object isn't trigger-happy.
        session.quality.MIN_INTERVAL = original_min_interval
        if original_best_thr is not None and hasattr(session.quality, "BEST_SHOT_THRESHOLD"):
            session.quality.BEST_SHOT_THRESHOLD = original_best_thr
        # NOTE: video_status flips to "done" at the very END of this function
        # (after album generation), not here. This used to be set right after
        # frame scoring finished, which was VISIBLE to a concurrent request
        # the instant it was set — since this whole function runs on a worker
        # thread while album generation (finale shot, VLM rescue, curation)
        # still had real work left to do. A second video uploaded into the
        # SAME session (e.g. "upload another") could start — and its
        # reset_for_new_video() reassigns session.quality/predictor/etc. —
        # WHILE the first video's own finale/rescue/album-gen was still
        # running on those very same objects. Two videos' pipelines briefly
        # overlapped on shared session state: exactly the kind of run-to-run
        # inconsistency ("upload video A, upload video B right after, B's
        # result looks wrong") that shouldn't be possible.
        _frames_ok = (session.video_status != "error")
        if _frames_ok:
            session.video_progress = 100

        # Zero-captures diagnostic — when a video processed cleanly but
        # nothing fired, dump the last frame's ensemble breakdown so the
        # user can tell whether the threshold is wrong or signals are weak.
        if (_frames_ok
                and session.capture_count == 0
                and session.frame_gallery):
            last = session.frame_gallery[-1] if session.frame_gallery else {}
            log.warning(
                f"Video [{session.sid}] processed {session.video_processed_frames} "
                f"frames but captured 0. Last frame summary: "
                f"score={last.get('score', 0):.2f} faces={last.get('faces', 0)} "
                f"moment={last.get('moment', '?')} mconf={last.get('mconf', 0):.2f}. "
                f"Try lowering SNAPPY_CAPTURE_THRESHOLD (default 0.55) or "
                f"SNAPPY_CLIP_CONFIDENCE_FLOOR (default 0.42)."
            )

        log.info(f"Video [{session.sid}] frames {'ok' if _frames_ok else 'error'} — "
                 f"{session.video_processed_frames} frames processed, "
                 f"{session.capture_count} captures")
        # Zero-capture RESCUE must run while the uploaded file still exists —
        # it re-samples frames from it (a video-call birthday scores like junk
        # to frame-level CLIP but the VLM confirms it). Only then clean up.
        if _frames_ok and session.capture_count == 0:
            rescue_zero_capture(session)
        try: os.remove(video_path)
        except Exception: pass

        # Don't leave an empty folder behind when a video captured nothing
        # (the _upload file is gone and no snaps were saved).
        try:
            if session.dir.exists() and not any(session.dir.iterdir()):
                session.dir.rmdir()
                log.info(f"Video [{session.sid}] removed empty captures folder "
                         f"{session.dir.name} (0 captures)")
        except OSError:
            pass

        # ── Recorded-video album + auto-end ─────────────────────────────────
        # A recorded upload is a one-shot job: when it finishes we generate the
        # album automatically (named after the uploaded video) and end the
        # session — the same curation the live "Generate Album" action runs.
        # Only THIS video's captures are included (album_start_idx).
        if _frames_ok:
            capture_finale(session)        # closing shot (finished cake) if uncaptured
            rescue_zero_capture(session)   # VLM second look when NOTHING captured
            video_photos = session.photos[album_start_idx:]
            if video_photos:
                album_name = session.source_name or session.event_name or session.sid
                try:
                    album = album_gen.generate(video_photos, album_name,
                                               session.event_type, session.prompt,
                                               speech=session_speech_text(session))
                    session.video_album = safe({
                        "event_name":     album.event_name,
                        "total_selected": album.total_selected,
                        "total_captured": album.total_captured,
                        "cover":          album.cover_photo or "",
                        "stats":          album_gen.stats(album),
                    })
                    log.info(f"Video [{session.sid}] album '{album_name}': "
                             f"{album.total_selected}/{album.total_captured} "
                             f"photos selected → albums/{album_name}/")
                except Exception as ex:
                    log.warning(f"Video [{session.sid}] album generation failed: {ex}")
            else:
                log.info(f"Video [{session.sid}] no captures — no album generated")
            session.active = False   # one recorded video → one finished session

        # THE completion signal — only now is it safe for a poller (or a new
        # upload into this same session) to proceed. See the long note above
        # `_frames_ok` for why this moved from partway through frame scoring
        # to here, after finale/rescue/album-gen have truly finished.
        session.video_status = "done" if _frames_ok else "error"

    await loop.run_in_executor(_executor, _run)
