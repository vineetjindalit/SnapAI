"""
backend/api/routes.py — pure HTTP handlers as (status, body_dict) functions.

Why pure functions?
  - Trivially unit-testable (no transport mocking).
  - The protocol class (server.py) just dispatches by (method, path) and
    emits the response — no business logic in the network layer.

All handlers take dict-shaped JSON bodies (already parsed where applicable)
and return (status_code, json_safe_dict). HTTP errors are returned as
(4xx/5xx, {"error": "..."}).

Sections (in order):
  Health / device
  Sessions   POST/GET/DELETE
  Photos / gallery / stats
  Album
  Feedback / learning
  Prompt / voice
  Discoveries
  Video upload
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple
import os

import cv2
import numpy as np

from gpu.device                  import device_summary
from models.album_generator      import PhotoEntry
from models.category_requirements import (
    merge_prompt, get_profile, parse_negations,
)
from utils.safe_types            import safe

from api.globals import (
    ALBUMS_DIR, AUTO_CLASS, BILLING, CACHE, CALIBRATOR, CAPTURES_DIR,
    CLIP, FACE_PROVIDER, HSEMO, LEARNER, L2CS, NIMA,
    PROMPT_ROUTER, SESSIONS, STORAGE, STORE, WHISPER, _CONT_LEARNER,
    album_gen,
)
from api.session import Session
from auth.middleware import auth_enabled, require_user, ErrorResult
from auth import routes as auth_routes
from billing.stripe_integration import enforce_session_quota, TIERS

log = logging.getLogger("snappy.routes")

Status = int
Body   = Dict[str, Any]
Result = Tuple[Status, Body]


# ═══════════════════════════════════════════════════════════════════════════
# Owner identity
# ═══════════════════════════════════════════════════════════════════════════
def is_owner(headers: Optional[Dict[str, str]] = None) -> bool:
    """True only for the person running this Mac.

    SNAPPY_DEV_UI=1 alone is NOT enough: `share.sh` sets it on the one backend
    that every friend's tunnel link hits, so gating on it alone showed the
    Developer tab (and its dev-only test tools) to every friend. The owner is
    the account named by SNAPPY_OWNER_EMAIL, defaulting to user id 1 — the
    first account created on this Mac, i.e. yours.
    """
    if os.environ.get("SNAPPY_DEV_UI", "0") != "1":
        return False
    # TEMPORARY, explicit opt-in override — everyone gets Developer/Admin for
    # a while (the user asked to show it to all testers, then turn it back
    # off). Never the default: only set when this env var is present.
    if os.environ.get("SNAPPY_DEV_UI_PUBLIC", "0") == "1":
        return True
    if not auth_enabled():
        return True                      # local dev, no login at all
    from auth.jwt_auth import JWTAuth
    tok = JWTAuth.from_authorization_header((headers or {}).get("authorization", ""))
    ok, payload = (JWTAuth().verify(tok) if tok else (False, {}))
    if not ok:
        return False

    owner_id = 1
    owner_email = os.environ.get("SNAPPY_OWNER_EMAIL", "").strip().lower()
    if owner_email and STORE is not None:
        row = STORE.find_user_by_email(owner_email)
        if row:
            owner_id = int(row["id"])
    return int(payload.get("sub", 0)) == owner_id


# ═══════════════════════════════════════════════════════════════════════════
# Health
# ═══════════════════════════════════════════════════════════════════════════
def health(headers: Optional[Dict[str, str]] = None) -> Result:
    return 200, {
        "status":  "ok",
        "version": "v2.5",
        "sessions": len(SESSIONS),
        # The frontend AuthGate probes this to decide whether to show the
        # Login/Register flow (SNAPPY_AUTH=1 turns the whole auth product on).
        "auth_enabled": auth_enabled(),
        # Developer/Admin are OWNER-ONLY — see is_owner(). Friends hitting the
        # same share.sh backend get False here, so they never see the tabs.
        "dev_ui": is_owner(headers),
        "persistence": STORE is not None,
        "compute":     device_summary(),
        "continuous_learning": _CONT_LEARNER is not None,
        "models": {
            "face":      {"backend": FACE_PROVIDER.backend},
            "gaze":      {"backend": "l2cs" if (L2CS and L2CS.available) else "iris"},
            "clip":      {"available": CLIP.available, "error": CLIP.init_error,
                          "trained_centroids": CLIP.has_trained_centroids,
                          "trained_meta": CLIP.trained_meta},
            "nima":      {"available": NIMA.available, "error": NIMA.init_error},
            "hsemotion": {"available": HSEMO.available, "error": HSEMO.init_error},
            "whisper":   {"available": WHISPER.available, "error": WHISPER.init_error},
            "calibration": {"available": CALIBRATOR is not None},
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# Sessions
# ═══════════════════════════════════════════════════════════════════════════
def create_session(body: bytes, headers: Optional[Dict[str, str]] = None) -> Result:
    try:
        # Auth gate (skipped when SNAPPY_AUTH=0)
        user = require_user(headers or {})
        if isinstance(user, ErrorResult):
            return user.status, user.body

        # Tier quota enforcement
        if STORE is not None and not user.anonymous:
            tier = STORE.get_user_tier(user.user_id) if hasattr(STORE, "get_user_tier") else "free"
            allowed, err = enforce_session_quota(STORE, user.user_id, tier)
            if not allowed:
                return 402, {"error": err, "tier": tier}

        data = json.loads(body or b"{}")
        sid  = str(uuid.uuid4())[:8]
        event_name = str(data.get("event_name", "My Event"))
        event_type = str(data.get("event_type", "general"))
        raw_prompt = str(data.get("prompt", ""))

        # ── Category intelligence ──────────────────────────────────────────
        # Merge category defaults with the user's prompt.  The effective prompt
        # is what VLMDetector + PromptRouter will actually use.  Negations
        # ("no candle") are extracted and stored as excluded_classes so the
        # pipeline can skip those moments cleanly.
        effective_prompt, excluded_classes, required_shots = merge_prompt(
            raw_prompt, event_type
        )
        # Keep raw_prompt on the session for display; effective_prompt drives ML.
        prompt = raw_prompt or effective_prompt   # show user what they typed

        SESSIONS[sid] = Session(
            sid, event_name, event_type, prompt,
            captures_root=CAPTURES_DIR,
            effective_prompt=effective_prompt,
            excluded_classes=excluded_classes,
            required_shots=required_shots,
        )
        PROMPT_ROUTER.init_session(sid, effective_prompt)

        # ── Optional real-camera source (DSLR / RTSP / phone-IP / GoPro / USB) ──
        # Additive: a request with no "camera_source" behaves exactly as before
        # (browser WebSocket frames). When present, frames from the named
        # camera are fed through the SAME process_frame() pipeline a browser
        # session uses — event_engine.py / host_group.py / everything else
        # applies unchanged. See backend/camera/camera_session_bridge.py.
        camera_error = None
        camera_source = data.get("camera_source")
        if camera_source:
            try:
                from camera.camera_session_bridge import start_camera_for_session
                cam_kwargs = {k: v for k, v in dict(camera_source).items() if k != "type"}
                cam_result = start_camera_for_session(
                    sid, SESSIONS[sid],
                    str(camera_source.get("type", "usb")), **cam_kwargs)
                if isinstance(cam_result, str):
                    camera_error = cam_result
                    log.warning(f"[{sid}] camera_source failed: {camera_error}")
                else:
                    log.info(f"[{sid}] camera_source attached: "
                             f"{camera_source.get('type', 'usb')}")
            except Exception as ex:
                camera_error = str(ex)
                log.warning(f"[{sid}] camera_source setup error: {camera_error}")

        # Custom Mode's NLU runs on the VLM (Qwen2.5-VL), which — unlike CLIP
        # — is lazy-loaded (measured: ~24s cold, one-time per server process,
        # by design: album generation is its only OTHER caller and doesn't
        # need it instantly). Without this, a custom session's very FIRST
        # prompt would silently eat that whole 24s with nothing but
        # "Understanding…" on screen. Firing the load here, the moment the
        # session is created, overlaps it with the user getting their camera
        # going / typing their first request instead.
        if event_type in ("custom", "smart_event"):
            from models.vlm_tagger import VLMTagger
            threading.Thread(target=VLMTagger.get, daemon=True).start()

        if STORE is not None:
            try:
                STORE.create_session(sid, event_name, event_type, effective_prompt,
                                     owner_id=user.user_id)
            except Exception as ex: log.warning(f"persist session: {ex}")
        profile = get_profile(event_type)
        log.info(
            f"Session created: {sid} | {event_name} | owner={user.user_id} "
            f"| category={event_type} | required={len(required_shots)} shots "
            f"| excluded={sorted(excluded_classes)} | prompt='{effective_prompt[:80]}'"
        )
        return 201, safe({
            "session_id":       sid,
            "event_name":       event_name,
            "event_type":       event_type,
            "prompt":           prompt,
            "effective_prompt": effective_prompt,
            "excluded_classes": sorted(excluded_classes),
            "required_shots":   len(required_shots),
            "ws_url":           f"/ws/{sid}",
            "owner_id":         user.user_id,
            "camera_error":     camera_error,
        })
    except Exception as e:
        log.exception("create_session failed")
        return 400, {"error": str(e)}


def list_sessions() -> Result:
    return 200, safe({"sessions": [
        {"sid": sid, "event_name": s.event_name, "active": bool(s.active),
         "captures": int(len(s.photos)), "frames": int(s.frame_count)}
        for sid, s in SESSIONS.items()
    ]})


def end_session(sid: str) -> Result:
    if sid in SESSIONS: SESSIONS[sid].active = False
    try:
        from camera.camera_session_bridge import stop_camera_for_session
        stop_camera_for_session(sid)
    except Exception as ex:
        log.warning(f"stop camera for {sid}: {ex}")
    if STORE is not None:
        try: STORE.end_session(sid)
        except Exception as ex: log.warning(f"persist end_session: {ex}")
    return 200, {"status": "ended"}


def ingest_photo_frame(sid: str, frame, tags: list, captured_at: Optional[float] = None,
                        fname_prefix: str = "manual") -> Result:
    """Shared ingestion path for ANY photo that did NOT come through the
    live per-frame pipeline (api/pipeline.py's process_frame): a
    photographer's manual shutter press (manual_photo, tags=["manual"]), or
    a DSLR's full-resolution shutter trigger fired after the AI already
    decided a live-preview frame was capture-worthy
    (camera_session_bridge.py, tags=["dslr_fullres"]).

    Enhances, scores and tags the frame EXACTLY like an auto-capture, then
    drops it into the SAME session.photos list models/album_generator.py
    already dedups/quality-floors at album time — so a manual shot, a DSLR
    full-res shot, and an AI auto-capture of the SAME moment are reconciled
    by the one importance-based best-of-cluster selection the whole system
    already trusts, never a second/duplicated "which one is better" path.

    `frame` is an already-decoded BGR numpy array (both callers have one —
    manual_photo decodes the uploaded JPEG, the camera bridge reads gphoto2's
    saved file with cv2.imread before calling this).
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    if frame is None:
        return 400, {"error": "Could not decode image"}

    from models.enhance import enhance
    enhanced, enh_report = enhance(frame, auto=True)

    face_count = 0
    boxes = None
    try:
        face_frame = FACE_PROVIDER.detect(enhanced)
        if face_frame.available:
            boxes = [f.box for f in face_frame.faces]
            face_count = face_frame.count
    except Exception as ex:
        log.debug(f"[{sid}] ingest_photo_frame face detect skipped: {ex}")

    nima_score = None
    try:
        if NIMA.available:
            nima_score = NIMA.score(enhanced)
    except Exception:
        pass

    try:
        shot = s.quality.analyze(enhanced, face_boxes=boxes, nima_score=nima_score)
        quality_score = float(shot.total)
    except Exception as ex:
        log.warning(f"[{sid}] ingest_photo_frame quality analyze failed: {ex}")
        quality_score = 0.5   # neutral — never silently drop a photo on a scoring bug

    moment_type, moment_conf = tags[0] if tags else "manual", 0.0
    try:
        moment = s.vlm.detect(enhanced)
        moment_type, moment_conf = moment.detected_moment, float(moment.confidence)
    except Exception as ex:
        log.debug(f"[{sid}] ingest_photo_frame moment detect skipped: {ex}")

    ts = float(captured_at or time.time())
    try:
        capdir = s.ensure_dir()
    except Exception as e:
        return 500, {"error": f"Could not prepare captures dir: {e}"}
    fname = f"{fname_prefix}_{int(ts * 1000)}.jpg"
    fpath = capdir / fname
    ok, buf = cv2.imencode(".jpg", enhanced, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return 500, {"error": "JPEG encode failed"}
    try:
        fpath.write_bytes(buf.tobytes())
    except OSError as e:
        return 500, {"error": f"Could not save photo: {e}"}

    pe = PhotoEntry(
        filepath=str(fpath), url=f"/captures/{s.dir.name}/{fname}",
        timestamp=ts, quality_score=quality_score, face_count=int(face_count),
        emotion_score=0.0, gaze_triggered=False,
        moment_type=str(moment_type), moment_conf=moment_conf,
        # tag is how album_generator/the frontend tell this apart from a
        # live-pipeline AI auto-capture — same convention pipeline.py uses
        # for its own mode tags (e.g. "custom", "general").
        tags=list(tags),
    )
    s.photos.append(pe)
    s.capture_count = len(s.photos)

    if STORE is not None:
        try:
            STORE.add_photo(sid, {
                "url": pe.url, "filepath": pe.filepath, "ts": pe.timestamp,
                "score": pe.quality_score, "faces": pe.face_count,
                "emotion": pe.emotion_score, "gaze": pe.gaze_triggered,
                "moment": pe.moment_type, "moment_conf": pe.moment_conf,
                "tags": pe.tags,
            })
        except Exception as ex:
            log.warning(f"[{sid}] persist photo ({tags}) failed: {ex}")

    log.info(f"[{sid}] photo ingested ({tags}): {fname} "
             f"quality={quality_score:.2f} faces={face_count}")
    return 201, safe({
        "ok": True, "url": pe.url, "quality_score": round(quality_score, 3),
        "enhancement": enh_report.to_dict(), "total_photos": len(s.photos),
    })


def manual_photo(sid: str, body: bytes) -> Result:
    """Ingest a photo the PHOTOGRAPHER captured on their own camera/phone —
    not through the live AI pipeline — into this session's photo pool. See
    ingest_photo_frame() for the shared reconciliation logic this delegates
    to (also used by camera_session_bridge.py for DSLR full-res triggers).

    Body (JSON): {"image": "<base64 JPEG>", "captured_at": <unix seconds, optional>}
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    try:
        data = json.loads(body or b"{}")
        img_b64 = str(data.get("image", ""))
        if not img_b64:
            return 400, {"error": "Missing 'image' (base64 JPEG)"}
        raw = base64.b64decode(img_b64)
        arr = np.frombuffer(raw, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return 400, {"error": f"Bad request: {e}"}

    return ingest_photo_frame(sid, frame, tags=["manual"],
                               captured_at=data.get("captured_at"))


def batch_photos(sid: str, body: bytes) -> Result:
    """Bulk-import an already-captured batch (200-600+ photos) in ONE request
    — the real-world shape for Camera/Pro tier (and any Mobile session the
    partner reviews afterward): the partner's phone pulls photos off a DSLR
    via USB-C/SD-reader (OS-level file access, no camera SDK — see
    partner-capture-architecture.md §4a) into its own camera roll / app
    sandbox, then the app POSTs them here as one batch instead of one HTTP
    round-trip per photo (which would be 200-600 requests for a single event).

    Body (JSON): {"images": ["<base64 JPEG>", ...], "captured_at": [<unix
    seconds per image, optional>, ...]}
    Each image runs through the EXACT same ingest_photo_frame() pipeline a
    single manual_photo() call uses — same enhance/score/tag/dedup-ready
    path, just looped. No new reconciliation logic: an imported DSLR batch
    and a live Mobile auto-capture of the same moment still resolve through
    album_generator's one best-of-cluster selection at album time.

    Returns per-image success/failure so a partner's bad file doesn't sink
    the whole batch, plus a running total so the app can show import progress.
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    try:
        data = json.loads(body or b"{}")
        images = data.get("images") or []
        captured_ats = data.get("captured_at") or []
        if not isinstance(images, list) or not images:
            return 400, {"error": "Missing 'images' (list of base64 JPEGs)"}
    except Exception as e:
        return 400, {"error": f"Bad request: {e}"}

    results = []
    ok_count = 0
    for i, img_b64 in enumerate(images):
        try:
            raw = base64.b64decode(str(img_b64))
            arr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            cap_at = captured_ats[i] if i < len(captured_ats) else None
            status, resp = ingest_photo_frame(sid, frame, tags=["manual", "batch_import"],
                                               captured_at=cap_at)
            if status == 201:
                ok_count += 1
                results.append({"index": i, "ok": True, "url": resp.get("url")})
            else:
                results.append({"index": i, "ok": False, "error": resp.get("error", "ingest failed")})
        except Exception as e:
            results.append({"index": i, "ok": False, "error": str(e)})

    s = SESSIONS[sid]
    log.info(f"[{sid}] batch import: {ok_count}/{len(images)} photos ingested")
    return 201, safe({
        "ok": True, "imported": ok_count, "failed": len(images) - ok_count,
        "total_photos": len(s.photos), "results": results,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Admin (owner-only Mac dashboard — every user, every session, any issues)
# ═══════════════════════════════════════════════════════════════════════════
def admin_overview(headers: Optional[Dict[str, str]] = None) -> Result:
    """Everything the Mac owner needs to see who's using SnapAI right now:
    every registered friend, every session they've run, live capture counts,
    and flagged issues (stuck processing, errors, zero-capture sessions).

    Gated exactly like Developer mode (see is_owner): needs SNAPPY_DEV_UI=1
    AND the owner's own login. Friends on the same share.sh backend get a 404
    — this exposes every user's email, so it must never leak.
    """
    if not is_owner(headers):
        return 404, {"error": "Not found"}

    if STORE is None:
        return 200, {"users": [], "sessions": []}

    users = STORE.list_users_admin()
    email_by_id = {u["id"]: u["email"] for u in users}

    sessions = []
    for r in STORE.list_all_sessions_admin():
        sid = r["sid"]
        live = SESSIONS.get(sid)
        issues: list = []

        if live is not None:
            captures     = int(len(live.photos))
            video_status = getattr(live, "video_status", None)
            video_error  = getattr(live, "video_error", None)
            active       = bool(live.active)
        else:
            # Not resident in memory (e.g. server restarted since) — fall
            # back to what SQLite remembers.
            captures     = STORE.photo_count(sid)
            video_status = None
            video_error  = None
            active       = False

        if video_error:
            issues.append(f"video error: {video_error}")
        if video_status == "processing" and (time.time() - float(r["created"])) > 900:
            issues.append("stuck processing >15min")
        if r["ended"] and captures == 0:
            issues.append("ended with zero captures")

        sessions.append({
            "sid": sid,
            "owner_id": r["owner_id"] or 0,
            "owner_email": email_by_id.get(r["owner_id"], None) if r["owner_id"] else None,
            "event_name": r["event_name"], "event_type": r["event_type"],
            "created": float(r["created"]), "ended": r["ended"],
            "active": active, "captures": captures,
            "video_status": video_status, "video_error": video_error,
            "issues": issues,
        })

    return 200, safe({"users": users, "sessions": sessions})


def stats(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    last = s.quality._hist[-1].to_dict() if s.quality._hist else {}
    return 200, safe({
        "sid": sid, "active": bool(s.active),
        "frame_count": int(s.frame_count),
        "capture_count": int(s.capture_count),
        "last_score": last,
        "score_timeline": [round(float(v), 3) for v in s.score_timeline[-60:]],
    })


# ═══════════════════════════════════════════════════════════════════════════
# Photos / gallery
# ═══════════════════════════════════════════════════════════════════════════
def list_photos(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    return 200, safe({"total": int(len(s.photos)), "photos": [
        {"url": p.url, "ts": float(p.timestamp), "score": float(p.quality_score),
         "faces": int(p.face_count), "moment": str(p.moment_type),
         "gaze": bool(p.gaze_triggered), "tags": list(p.tags),
         # Dual-shot extras (wide already lives at "url"; zoom is optional)
         "zoom_url":  getattr(p, "zoom_url",  None),
         "zoom_info": getattr(p, "zoom_info", None),
         "enhance_report": getattr(p, "enhance_report", None)}
        for p in s.photos
    ]})


def gallery(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    return 200, safe({"frames": list(SESSIONS[sid].frame_gallery)})


# ═══════════════════════════════════════════════════════════════════════════
# Album
# ═══════════════════════════════════════════════════════════════════════════
def album(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    s.active = False
    # Closing shot + zero-capture rescue run BEFORE the empty check — they can
    # create the very photos an empty session is missing (the finished cake; a
    # video-call birthday the frame-level models couldn't score).
    from api.pipeline import capture_finale, rescue_zero_capture, session_speech_text
    capture_finale(s)
    rescue_zero_capture(s)
    if not s.photos:
        return 400, {"error": "No photos yet"}
    # Recorded-video sessions name the album after the uploaded video so the
    # manual "Generate Album" action matches the auto-generated one.
    album_name = s.source_name or s.event_name
    # Pass the CLIP engine so the curator can score each capture's RELEVANCE to
    # this event (per-event centroids) and gate out off-event / mislabelled
    # shots, on top of dedup + sequence ordering. Speech = occasion context.
    a = album_gen.generate(s.photos, album_name, s.event_type, s.prompt, clip=CLIP,
                           speech=session_speech_text(s))

    # OPTIONAL photographer-grade pass (VLM) — runs ONLY at album time and ONLY
    # if the user opted in (SNAPPY_VLM_API_KEY set). It re-tags + curates the
    # already-curated shots; otherwise it's a pure no-op and we keep the
    # on-device classifier's tags. Fully isolated — any error degrades silently.
    try:
        from models.vlm_curator import VLMCurator
        vlm = VLMCurator.get()
        if vlm.available:
            refined = vlm.refine_album(
                [{"url": p.url, "filepath": p.filepath, "moment_type": p.moment_type}
                 for p in a.photos])
            if refined:
                kept = []
                for p in a.photos:
                    r = refined.get(p.url)
                    if r is None:
                        kept.append(p); continue
                    if r.get("moment"):  p.moment_type = r["moment"]
                    if r.get("caption"): p.tags = list(p.tags) + [f"caption:{r['caption']}"]
                    if r.get("keep", True): kept.append(p)
                if kept:                       # never let the VLM empty the album
                    a.photos = kept; a.total_selected = len(kept)
                log.info(f"VLM curator refined album → kept {len(a.photos)}")
    except Exception as ex:
        log.warning(f"VLM curator skipped ({ex}) — using on-device tags")

    return 200, safe({
        "event_name": str(a.event_name),
        "event_type": str(a.event_type),
        "occasion":   str(getattr(a, "occasion", "") or ""),
        "total_selected": int(a.total_selected),
        "total_captured": int(a.total_captured),
        "low_confidence": bool(getattr(a, "low_confidence", False)),
        "cover": str(a.cover_photo or ""),
        "photos": [{"url": p.url, "score": float(p.quality_score),
                    "moment": str(p.moment_type), "tags": list(p.tags),
                    "relevance": round(float(p.relevance), 3)} for p in a.photos],
        "stats": album_gen.stats(a),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Feedback / learning
# ═══════════════════════════════════════════════════════════════════════════
def feedback(sid: str, body: bytes) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    try:
        data      = json.loads(body or b"{}")
        photo_url = str(data.get("photo_url", ""))
        kept      = bool(data.get("kept", True))
        moment    = str(data.get("moment", "general_peak"))
    except Exception as e:
        return 400, {"error": f"Bad payload: {e}"}

    photo = next((p for p in s.photos if p.url == photo_url), None)
    if photo is None:
        return 404, {"error": "Photo not found in session"}
    if not moment or moment == "general_peak":
        moment = photo.moment_type or "general_peak"

    if STORE is not None:
        try: STORE.record_feedback(sid, photo_url, moment, kept)
        except Exception as ex: log.warning(f"persist feedback: {ex}")
    prior = LEARNER.record(sid, moment, kept)

    # Calibration update (Platt scaling per class)
    if CALIBRATOR is not None:
        try:
            CALIBRATOR.observe(moment, raw_score=float(photo.moment_conf), kept=kept)
        except Exception as ex:
            log.debug(f"calibration observe: {ex}")

    if kept and CLIP.available:
        try:
            img = cv2.imread(photo.filepath)
            if img is not None:
                CLIP.record_kept_frame(sid, img)
        except Exception as ex:
            log.warning(f"CLIP centroid: {ex}")

    return 200, safe({
        "ok": True, "sid": sid, "moment": moment, "kept": kept,
        "kept_count": prior.kept, "trashed_count": prior.trashed,
        "keep_rate": round(prior.keep_rate, 3),
    })


# ── Album review: per-photo reactions + overall review ─────────────────────────

# How each reaction maps to learning.
#   keep   : True  → positive signal for the moment class (Beta α+1)
#            False → negative signal (Beta β+1)
#            None  → moment is RIGHT, only the execution was off → no class change
#   reinforce_visual : feed the frame into the CLIP kept-centroid (only for
#            genuinely good keepers — never for blurry/bad-framing photos).
_REACTION_SEMANTICS: Dict[str, Dict[str, Any]] = {
    "love":         {"keep": True,  "reinforce_visual": True},
    "up":           {"keep": True,  "reinforce_visual": True},
    "down":         {"keep": False, "reinforce_visual": False},
    "wrong_moment": {"keep": False, "reinforce_visual": False},
    # "right moment, but not captured the way I want" — keep the moment, don't
    # reinforce this exact (blurry/badly-framed) frame as the visual ideal.
    "blurry":       {"keep": True,  "reinforce_visual": False},
    "bad_framing":  {"keep": True,  "reinforce_visual": False},
}


def photo_reaction(sid: str, body: bytes) -> Result:
    """POST /sessions/{sid}/photo_reaction

    One reaction (+ optional note) for one album photo.

    Body:
        photo_url : str   — the photo being reacted to (required)
        reaction  : str   — love|up|down|blurry|wrong_moment|bad_framing
        note      : str   — optional per-photo review (typed or transcribed voice)

    Effect: persists the reaction, then feeds the learner per _REACTION_SEMANTICS
    so the model captures that moment better (or stops over-capturing it) next
    time. "blurry"/"bad_framing" keep the moment wanted but don't reinforce the
    poorly-executed frame as the visual ideal.
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    try:
        data      = json.loads(body or b"{}")
        photo_url = str(data.get("photo_url", "")).strip()
        reaction  = str(data.get("reaction", "")).strip().lower()
        note      = str(data.get("note", "")).strip()
    except Exception as e:
        return 400, {"error": f"Bad payload: {e}"}
    if not photo_url:
        return 400, {"error": "photo_url is required"}
    if reaction and reaction not in _REACTION_SEMANTICS:
        return 400, {"error": f"unknown reaction '{reaction}'"}

    photo  = next((p for p in s.photos if p.url == photo_url), None)
    moment = (photo.moment_type if photo else "general_peak") or "general_peak"

    if STORE is not None:
        try: STORE.add_photo_reaction(sid, photo_url, reaction, note, moment)
        except Exception as ex: log.warning(f"persist photo_reaction: {ex}")

    # Feed the learner
    sem = _REACTION_SEMANTICS.get(reaction, {})
    keep = sem.get("keep", None)
    if keep is not None:
        try:
            LEARNER.record(sid, moment, bool(keep))
            if CALIBRATOR is not None and photo is not None:
                CALIBRATOR.observe(moment, raw_score=float(photo.moment_conf), kept=bool(keep))
            if keep and sem.get("reinforce_visual") and CLIP.available and photo is not None:
                img = cv2.imread(photo.filepath)
                if img is not None:
                    CLIP.record_kept_frame(sid, img)
        except Exception as ex:
            log.warning(f"photo_reaction learn failed: {ex}")

    log.info(f"photo_reaction [{sid}] {reaction or '∅'} on {photo_url.split('/')[-1]} "
             f"(moment={moment}, note={'yes' if note else 'no'})")
    return 200, safe({"ok": True, "reaction": reaction, "moment": moment})


def album_review(sid: str, body: bytes) -> Result:
    """POST /sessions/{sid}/album_review

    Overall review of the whole album (text and/or 1-5 rating). The text may be
    typed or transcribed from voice on the client. Any "you missed X" phrasing
    in the text is also parsed into a missed-moment learning signal so a spoken
    overall review still teaches the model.

    Body: { text: str, rating: int(0-5) }
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    try:
        data   = json.loads(body or b"{}")
        text   = str(data.get("text", "")).strip()
        rating = int(data.get("rating", 0) or 0)
    except Exception as e:
        return 400, {"error": f"Bad payload: {e}"}
    if not text and not rating:
        return 400, {"error": "Provide text and/or a rating"}
    rating = max(0, min(5, rating))

    if STORE is not None:
        try: STORE.add_album_review(sid, text, rating)
        except Exception as ex: log.warning(f"persist album_review: {ex}")

    # Mine the free text for "missed X" mentions → missed-moment learning.
    # Only scan the text AFTER the first miss-trigger word so praise before it
    # ("great candle shots, but you missed the hug") isn't mistaken for a miss.
    learned_hints: list = []
    if text:
        from models.category_requirements import _NEGATION_CLASS_MAP
        low = text.lower()
        triggers = ("missed", "forgot", "didn't capture", "didnt capture", "missing")
        first = min((low.find(t) for t in triggers if t in low), default=-1)
        if first >= 0:
            tail = low[first:]
            seen: set = set()
            for key, classes in _NEGATION_CLASS_MAP.items():
                if key in tail:
                    cls = next(iter(classes), "general_peak")
                    if cls in seen:
                        continue
                    seen.add(cls)
                    try:
                        LEARNER.record(sid, cls, True)
                        if STORE is not None:
                            STORE.add_missed_moment(sid, text, "important", cls)
                        learned_hints.append(cls)
                    except Exception as ex:
                        log.warning(f"album_review learn failed: {ex}")
    log.info(f"album_review [{sid}] rating={rating} chars={len(text)} "
             f"learned={learned_hints}")
    return 200, safe({"ok": True, "rating": rating, "learned": learned_hints})


def transcribe(sid: str, body: bytes) -> Result:
    """POST /sessions/{sid}/transcribe

    Pure speech-to-text for review dictation. Unlike /voice this does NOT touch
    the prompt — it just returns the transcript so the client can drop it into a
    review box (per-photo or overall).
    """
    if not WHISPER.available:
        return 503, {"error": f"Server STT unavailable: {WHISPER.init_error}"}
    text = WHISPER.transcribe(body or b"")
    if not text:
        return 422, {"error": "Could not transcribe audio"}
    return 200, safe({"ok": True, "transcript": text})


def learning(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    payload = {
        "sid": sid,
        "priors": LEARNER.snapshot(sid).get(sid, {}),
        "clip_available": bool(CLIP.available),
        "clip_init_error": CLIP.init_error,
        "face_backend": FACE_PROVIDER.backend,
    }
    if CALIBRATOR is not None:
        payload["calibration"] = CALIBRATOR.snapshot()
    return 200, safe(payload)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt / voice
# ═══════════════════════════════════════════════════════════════════════════
def update_prompt(sid: str, body: bytes) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    try:
        data   = json.loads(body or b"{}")
        text   = str(data.get("text", "")).strip()
        source = str(data.get("source", "text"))
    except Exception as e:
        return 400, {"error": f"Bad payload: {e}"}
    if not text:
        return 400, {"error": "Empty prompt text"}

    transition = PROMPT_ROUTER.update(sid, text)
    if transition is None:
        return 500, {"error": "PromptRouter rejected update"}

    s = SESSIONS[sid]
    s.vlm.update_prompt(text)
    s.prompt = text

    if STORE is not None:
        try: STORE.log_prompt(sid, transition.intent, text,
                              transition.matched, transition.active, source)
        except Exception as ex: log.warning(f"prompt log: {ex}")

    return 200, safe({
        "ok": True, "transition": transition.to_dict(),
        "active_classes": transition.active,
    })


def voice(sid: str, body: bytes) -> Result:
    if not WHISPER.available:
        return 503, {"error": f"Server STT unavailable: {WHISPER.init_error}"}
    text = WHISPER.transcribe(body or b"")
    if not text:
        return 422, {"error": "Could not transcribe audio"}
    transition = PROMPT_ROUTER.update(sid, text)
    if transition and sid in SESSIONS:
        SESSIONS[sid].vlm.update_prompt(text)
        SESSIONS[sid].prompt = text
        if STORE is not None:
            try: STORE.log_prompt(sid, transition.intent, text,
                                  transition.matched, transition.active, "voice")
            except Exception: pass
    return 200, safe({
        "ok": True, "transcript": text,
        "transition": transition.to_dict() if transition else None,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Discoveries (auto-classifier)
# ═══════════════════════════════════════════════════════════════════════════
def list_discoveries(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    return 200, safe({"discoveries": [d.to_dict() for d in AUTO_CLASS.list_discoveries(sid)]})


def name_discovery(sid: str, cid: str, body: bytes) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    try:
        data = json.loads(body or b"{}")
        name = str(data.get("name", "")).strip()
    except Exception as e:
        return 400, {"error": f"Bad payload: {e}"}
    if not name:
        return 400, {"error": "Empty name"}
    d = AUTO_CLASS.name_discovery(cid, name)
    if d is None:
        return 404, {"error": "Discovery not found"}
    return 200, safe({"ok": True, "discovery": d.to_dict()})


# ═══════════════════════════════════════════════════════════════════════════
# Video upload progress
# ═══════════════════════════════════════════════════════════════════════════
def video_progress(sid: str) -> Result:
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    return 200, safe({
        "status":           str(s.video_status),
        "progress":         int(s.video_progress),
        "total_frames":     int(s.video_total_frames),
        "processed_frames": int(s.video_processed_frames),
        "captures":         int(s.capture_count),
        "error":            s.video_error,
        # When status == "done" the album is auto-generated and the session
        # auto-ended; `album` carries the curated summary (None until then).
        "album":            s.video_album,
        "active":           bool(s.active),
    })


# ═══════════════════════════════════════════════════════════════════════════
# Auth (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════
def auth_register(body: bytes, headers: Optional[Dict[str, str]] = None) -> Result:
    return auth_routes.register(STORE, body, headers)


def auth_login(body: bytes, headers: Optional[Dict[str, str]] = None) -> Result:
    return auth_routes.login(STORE, body, headers)


def auth_logout(headers: Dict[str, str]) -> Result:
    return auth_routes.logout(headers)


def auth_refresh(headers: Dict[str, str]) -> Result:
    return auth_routes.refresh(headers)


def auth_me(headers: Dict[str, str]) -> Result:
    return auth_routes.me(headers, STORE)


def auth_verify(body: bytes) -> Result:
    return auth_routes.verify_email(STORE, body)


def auth_forgot(body: bytes, headers: Optional[Dict[str, str]] = None) -> Result:
    return auth_routes.forgot_password(STORE, body, headers)


def auth_reset(body: bytes) -> Result:
    return auth_routes.reset_password(STORE, body)


# ═══════════════════════════════════════════════════════════════════════════
# Billing (Phase 4)
# ═══════════════════════════════════════════════════════════════════════════
def billing_tiers() -> Result:
    return 200, {"tiers": [
        {"name": t.name, "price_usd": t.price_usd,
         "events_per_month": t.events_per_month,
         "captures_per_event": t.captures_per_event,
         "simultaneous_cameras": t.simultaneous_cameras,
         "api_access": t.api_access, "custom_branding": t.custom_branding}
        for t in TIERS.values()
    ]}


def billing_checkout(headers: Dict[str, str], body: bytes) -> Result:
    if BILLING is None:
        return 503, {"error": "Billing module not loaded"}
    user = require_user(headers)
    if isinstance(user, ErrorResult):
        return user.status, user.body
    if user.anonymous:
        return 401, {"error": "Login required to upgrade"}
    try:
        data = json.loads(body or b"{}")
    except Exception as e:
        return 400, {"error": f"Bad JSON: {e}"}
    tier = str(data.get("tier", "")).lower()
    success_url = str(data.get("success_url", "https://your-snappy-host/billing?status=ok"))
    cancel_url  = str(data.get("cancel_url",  "https://your-snappy-host/billing?status=cancel"))
    return BILLING.create_checkout(
        user_id=user.user_id, email=user.email, tier=tier,
        success_url=success_url, cancel_url=cancel_url,
    )


def billing_webhook(headers: Dict[str, str], body: bytes) -> Result:
    """Stripe sends event payloads here. We verify the signature and
    update the user's tier in the DB."""
    if BILLING is None or not BILLING.available:
        return 503, {"error": "Billing not configured"}
    sig = headers.get("stripe-signature", "")
    ok, event = BILLING.parse_webhook(body, sig)
    if not ok:
        return 400, event

    typ = event.get("type", "")
    obj = event.get("data", {}).get("object", {}) or {}
    log.info(f"stripe webhook: {typ}")

    if typ in ("checkout.session.completed",
               "customer.subscription.created",
               "customer.subscription.updated"):
        user_id = None
        try:
            # Stripe puts the user id in metadata on subscription objects;
            # checkout.session has client_reference_id.
            user_id = (int(obj.get("client_reference_id", 0))
                       or int((obj.get("metadata") or {}).get("user_id", 0)))
        except Exception:
            pass
        sub_id  = (obj.get("subscription") if "subscription" in obj
                   else obj.get("id"))
        cust_id = obj.get("customer")
        tier    = (obj.get("metadata") or {}).get("tier", "pro")
        status  = obj.get("status", "active")
        renews  = obj.get("current_period_end")

        if user_id and STORE is not None:
            try:
                STORE.update_user_subscription(
                    user_id=user_id, tier=tier,
                    stripe_customer_id=cust_id,
                    stripe_subscription_id=sub_id,
                    status=status, renews=float(renews) if renews else None,
                )
            except Exception as ex:
                log.warning(f"persist subscription: {ex}")

    elif typ == "customer.subscription.deleted":
        sub_id = obj.get("id")
        if STORE is not None and sub_id:
            user = STORE.find_user_by_stripe_subscription(sub_id)
            if user:
                STORE.update_user_subscription(
                    user_id=user["id"], tier="free",
                    status="canceled", renews=None,
                )

    return 200, {"received": True}


# ── Category / review endpoints ───────────────────────────────────────────────

def required_shots(sid: str) -> Result:
    """GET /sessions/{sid}/required_shots

    Returns the live required-shots checklist for the session: which shots
    were captured and which are still missing.  Used by the review panel.
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s = SESSIONS[sid]
    return 200, safe(s.required_tracker.to_summary())


def report_missed_moment(sid: str, headers: Dict[str, str],
                         body: bytes) -> Result:
    """POST /sessions/{sid}/missed_moment

    User reports a moment they noticed was not captured.

    Body:
        description  : str  — free text ("missed candle blown", "you forgot the hug")
        severity     : str  — "critical" | "important" | "casual"  (default "important")
        moment_hint  : str  — optional; caller can specify the class name directly

    The server derives moment_hint from the description if not supplied, then:
      • Persists the report in SQLite.
      • Instructs the OnlineLearner to trust this moment class more next time.
      • Returns the row_id so the UI can reference it.
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    try:
        data        = json.loads(body or b"{}")
        description = str(data.get("description", "")).strip()
        severity    = str(data.get("severity", "important")).lower()
        if severity not in ("critical", "important", "casual"):
            severity = "important"
        if not description:
            return 400, {"error": "description is required"}

        # Derive moment_hint from description if not supplied
        moment_hint = str(data.get("moment_hint", "")).strip()
        if not moment_hint:
            from models.category_requirements import _NEGATION_CLASS_MAP
            desc_low = description.lower()
            for key, classes in _NEGATION_CLASS_MAP.items():
                if key in desc_low:
                    moment_hint = next(iter(classes), "general_peak")
                    break
            if not moment_hint:
                moment_hint = "general_peak"

        row_id = 0
        if STORE is not None:
            row_id = STORE.add_missed_moment(sid, description, severity, moment_hint)

        # Feed into OnlineLearner: a missed moment is a positive signal that we
        # SHOULD have captured this class — record it as kept so the class's
        # trust factor rises. Severity controls how many positive observations
        # we add (critical counts more), since the learner is count-based.
        reps = {"critical": 3, "important": 2, "casual": 1}.get(severity, 1)
        try:
            for _ in range(reps):
                LEARNER.record(sid, moment_hint, True)
            if STORE is not None and row_id:
                STORE.mark_missed_learned(row_id)
            log.info(
                f"Missed moment [{sid}] severity={severity} "
                f"class={moment_hint} (+{reps} keep): {description!r}"
            )
        except Exception as ex:
            log.warning(f"missed_moment learner update failed: {ex}")

        return 201, safe({
            "row_id":      row_id,
            "moment_hint": moment_hint,
            "severity":    severity,
            "learned":     True,
            "message":     (
                f"Got it — I'll capture {moment_hint.replace('_', ' ')} "
                "more aggressively next time."
            ),
        })
    except Exception as e:
        log.exception("report_missed_moment failed")
        return 500, {"error": str(e)}


def review_summary(sid: str) -> Result:
    """GET /sessions/{sid}/review_summary

    Combined post-album review data:
      • required_shots checklist (captured / missed)
      • missed_moment reports filed for this session
      • album stats (if album generated)
    """
    if sid not in SESSIONS:
        return 404, {"error": "Session not found"}
    s   = SESSIONS[sid]
    rqs = s.required_tracker.to_summary()
    missed: list = []; reactions: list = []; reviews: list = []
    if STORE is not None:
        try: missed    = STORE.get_missed_moments(sid)
        except Exception: pass
        try: reactions = STORE.get_photo_reactions(sid)
        except Exception: pass
        try: reviews   = STORE.get_album_reviews(sid)
        except Exception: pass
    return 200, safe({
        "session_id":    sid,
        "event_name":    s.event_name,
        "event_type":    s.event_type,
        "required_shots": rqs,
        "missed_moments": missed,
        "photo_reactions": reactions,
        "album_reviews":   reviews,
        "album":         s.video_album,
        "total_captures": s.capture_count,
    })
