"""
models/setup_advisor.py — the photographer's ASSISTANT: guide the user to place
the phone so the event is captured perfectly.

A real photographer's first job is POSITION. This advisor watches the live frame
and gives DIRECTIONAL instructions ("pan left", "step back", "raise it to chest
height") rather than vague complaints — plus a written setup checklist and a
placement brief the user can read or have spoken.

Signals used (all already computed by the pipeline — no extra real-time cost):
faces, CLIP per-prompt scores, brightness, shake. The one extra thing is an
optional THROTTLED 3-crop CLIP scan (~once every few seconds) that locates the
cake table left/centre/right so the coach can say WHICH WAY to turn.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Static placement brief — the "where to put the phone" guidelines, shown as
# written steps in the UI (and read aloud in speak mode).
BIRTHDAY_PLACEMENT_GUIDE: List[str] = [
    "Place the phone OPPOSITE the cake, so faces turn toward the camera when everyone sings.",
    "Stand it at chest height, tilted slightly down over the cake.",
    "Leave 2–3 steps of distance so the cake AND the people around it both fit.",
    "Prop it against something solid (a bottle, a stack of books) or use a stand — no hand-holding.",
    "Keep the main light in FRONT of people's faces; avoid a bright window or lamp behind them.",
    "Turn the phone sideways (landscape) for a group; upright is fine for one person.",
    "Pick a spot where nobody will walk between the camera and the cake.",
    "Once it's placed, leave it — SnapAI watches and captures on its own.",
]

_EVENT_KEYS = ("cake", "cake_with_candles", "cake_person", "food_table",
               "birthday_gifting")


def locate_event(frame, clip_engine, event: str = "birthday") -> Optional[str]:
    """Which THIRD of the frame holds the cake/table: 'left' | 'centre' | 'right'
    | None. Runs CLIP on three crops, so callers MUST throttle it (a few seconds
    apart). Returns None when nothing event-like is found anywhere.
    """
    if clip_engine is None or frame is None:
        return None
    try:
        H, W = frame.shape[:2]
        thirds = {
            "left":   frame[:, : int(W * 0.42)],
            "centre": frame[:, int(W * 0.29): int(W * 0.71)],
            "right":  frame[:, int(W * 0.58):],
        }
        best_where, best_margin = None, 0.0
        for where, crop in thirds.items():
            if crop.size == 0:
                continue
            pp = clip_engine.score_frame(crop, event=event).per_prompt or {}
            bg = max((v for k, v in pp.items() if k.startswith("bg_")), default=0.0)
            margin = max((pp.get(k, 0.0) for k in _EVENT_KEYS), default=0.0) - bg
            if margin > best_margin:
                best_where, best_margin = where, margin
        # Only trust a clear signal — otherwise say "not found".
        return best_where if best_margin >= 0.045 else None
    except Exception:
        return None


def advise(frame,
           face_boxes: list,          # [(x, y, w, h)] in PROXY pixels
           per_prompt: Optional[dict],
           bg_best: float,
           jitter: int,               # consecutive-frame hash distance (0-64)
           event_side: Optional[str] = None,   # from locate_event(), if scanned
           ) -> Dict[str, Any]:
    """Return {tips, score, checklist, guide}.

    tips      — prioritized, directional, at most 2 (the live coach line)
    score     — readiness 0-100
    checklist — [{label, ok, detail}] for the written setup guide panel
    guide     — static placement steps
    """
    tips: List[str] = []
    checks: List[Dict[str, Any]] = []
    score = 100
    H, W = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    def check(label: str, ok: bool, detail: str):
        checks.append({"label": label, "ok": bool(ok), "detail": detail})

    # ── 1. Light ───────────────────────────────────────────────────────────
    mean_b = float(gray.mean())
    backlit = False
    if face_boxes and mean_b > 110:
        fvals = []
        for (x, y, w, h) in face_boxes:
            roi = gray[max(0, int(y)): int(y) + int(h), max(0, int(x)): int(x) + int(w)]
            if roi.size:
                fvals.append(float(roi.mean()))
        backlit = bool(fvals and min(fvals) < mean_b - 55)

    if mean_b < 60:
        tips.append("Too dark — turn on the room lights or move the phone near a lamp")
        score -= 35
        check("Lighting", False, "Too dark for clean photos")
    elif mean_b < 85:
        tips.append("A little dim — add light and the photos get much better")
        score -= 15
        check("Lighting", False, "Dim — more light recommended")
    elif backlit:
        tips.append("Light is BEHIND the people — move the phone to the other side so faces are lit")
        score -= 25
        check("Lighting", False, "Backlit — faces are in shadow")
    else:
        check("Lighting", True, "Good light on faces")

    # ── 2. Steadiness ──────────────────────────────────────────────────────
    if jitter >= 14:
        tips.append("Camera is shaky — prop the phone against something solid or use a stand")
        score -= 20
        check("Steady", False, "Hand-held / moving")
    else:
        check("Steady", True, "Held steady")

    # ── 3. Framing the people — DIRECTIONAL ────────────────────────────────
    if face_boxes:
        boxes = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in face_boxes]
        biggest = max(h for (_, _, _, h) in boxes) / float(H)
        cxs = [(x + w / 2) / float(W) for (x, _, w, _) in boxes]
        cys = [(y + h / 2) / float(H) for (_, y, _, h) in boxes]
        avg_x, avg_y = sum(cxs) / len(cxs), sum(cys) / len(cys)
        cut = any(x <= 2 or y <= 2 or (x + w) >= W - 2 for (x, y, w, h) in boxes)

        # Distance
        if biggest < 0.07:
            tips.append("People are far away — move the phone a few steps closer")
            score -= 20
            check("Distance", False, "Too far — subjects look small")
        elif biggest > 0.45:
            tips.append("Too close — step back so the cake and everyone fit in frame")
            score -= 15
            check("Distance", False, "Too close — scene is cropped")
        else:
            check("Distance", True, "Good distance")

        # Centring — say WHICH WAY to turn.
        if cut:
            tips.append("Someone is cut off at the edge — step back or re-centre the phone")
            score -= 15
            check("Framing", False, "A person is cut off at the frame edge")
        elif avg_x < 0.35:
            tips.append("Turn the phone slightly LEFT — the group is off to that side")
            score -= 10
            check("Framing", False, "Group sits left of centre")
        elif avg_x > 0.65:
            tips.append("Turn the phone slightly RIGHT — the group is off to that side")
            score -= 10
            check("Framing", False, "Group sits right of centre")
        else:
            check("Framing", True, "Group is centred")

        # Height / tilt
        if avg_y < 0.30:
            tips.append("Tilt the phone UP a little — faces are near the top edge")
            score -= 10
            check("Height", False, "Aimed too low / faces near top")
        elif avg_y > 0.75:
            tips.append("Raise the phone to chest height and tilt slightly DOWN")
            score -= 10
            check("Height", False, "Aimed too high / faces near bottom")
        else:
            check("Height", True, "Good camera height")

        # Orientation — a group fits far better in landscape.
        if len(boxes) >= 3 and H > W:
            tips.append("Turn the phone SIDEWAYS (landscape) — it fits the whole group")
            score -= 10
            check("Orientation", False, "Portrait with a group — landscape fits more")
        else:
            check("Orientation", True, "Orientation suits the scene")
    else:
        check("People in frame", False, "No one in view yet")

    # ── 4. The cake table — where the moments actually happen ──────────────
    pp = per_prompt or {}
    obj_best = max((pp.get(k, 0.0) for k in _EVENT_KEYS), default=0.0)
    sees_event = (obj_best - bg_best) >= 0.05
    if sees_event:
        check("Cake in view", True, "Cake / table is in frame")
        if face_boxes:
            check("Cake + guests together", True, "Both in the same shot")
    else:
        check("Cake in view", False, "Cake table not detected in frame")
        # If a throttled scan found it off to one side, say which way to turn.
        if event_side in ("left", "right"):
            tips.insert(0, f"The cake is toward the {event_side.upper()} — "
                           f"turn the phone {event_side} to bring it into frame")
            score -= 15
        elif not face_boxes:
            tips.append("Aim at where the celebration will happen — the cake table is the perfect spot")
            score -= 25
        else:
            tips.append("Keep the cake table in view — that's where the key moments happen")
            score -= 10

    if not tips:
        tips.append("Great setup — leave the phone here, SnapAI will do the rest ✓")

    return {
        "tips": tips[:2],
        "score": int(max(0, min(100, score))),
        "checklist": checks,
        "guide": BIRTHDAY_PLACEMENT_GUIDE,
    }
