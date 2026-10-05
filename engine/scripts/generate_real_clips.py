#!/usr/bin/env python3
"""
generate_real_clips.py — Realistic synthetic clips for manual SnapAI frontend testing.

These clips simulate real Indian celebration and candid moments at 1280×720 30fps.
They are placed in tests/datasets/clips/real/ so they can be directly uploaded to
the SnapAI frontend for manual testing.

Scenarios (15 clips):
  R01  Cake cutting          — birthday/anniversary, candles, knife, group lean-in
  R02  Ring ceremony (Sagai) — couple facing each other, ring exchange motion
  R03  Varmala / Jaimala     — garland exchange, couple + crowd
  R04  Sangeet dance         — 4-6 faces, energetic movement, coloured lights
  R05  First dance (couple)  — slow sway, close faces, dim warm light
  R06  Emotional parent hug  — parent + child embrace at ceremony
  R07  Group photo moment    — 8+ faces all facing camera, countdown moment
  R08  Diwali diya lighting  — hands near flame, warm glow, multiple faces
  R09  Baby milestone        — small figure, adult faces watching with joy
  R10  Champagne / toast     — glasses clinking, upward motion, faces smiling
  R11  Bride walking in      — motion, crowd turning, focus shift
  R12  Mehendi close-up      — hands detail, faces admiring, low motion
  R13  Kids candid playing   — fast random motion, occasional laugh face
  R14  Outdoor fireworks     — bright light burst + silhouetted faces looking up
  R15  Award / felicitation  — handshake, stage lighting, applause motion

Run:
    cd snappy_final
    python scripts/generate_real_clips.py
"""
from __future__ import annotations
import json, math, os, random
from pathlib import Path

import cv2
import numpy as np

random.seed(7)
np.random.seed(7)

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "datasets" / "clips" / "real"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H, FPS = 1280, 720, 30

# ── Palette (BGR) ─────────────────────────────────────────────────────────────
CREAM       = (230, 240, 250)
WARM_BG     = (170, 190, 215)
GOLD        = ( 30, 165, 215)
MARIGOLD    = ( 20, 130, 255)
DEEP_RED    = ( 35,  30, 170)
NAVY        = (100,  50,  20)
PINK_LIGHT  = (200, 185, 255)
IVORY       = (240, 245, 255)
CHAMPAGNE   = (190, 215, 240)
TEAL        = (160, 160,  40)
FOREST      = ( 40,  90,  40)

# ── Skin tones (BGR) ──────────────────────────────────────────────────────────
SKIN_FAIR   = (155, 185, 215)
SKIN_MED    = (110, 150, 195)
SKIN_WARM   = ( 90, 130, 185)
SKIN_DARK   = ( 65,  95, 140)
SKINS       = [SKIN_FAIR, SKIN_MED, SKIN_WARM, SKIN_DARK]


# ── Helpers ───────────────────────────────────────────────────────────────────

def writer(name: str) -> tuple:
    path   = OUT_DIR / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw     = cv2.VideoWriter(str(path), fourcc, FPS, (W, H))
    if not vw.isOpened():
        raise RuntimeError(f"Cannot open writer: {path}")
    return vw, path


def bg(color, noise_amp: int = 6) -> np.ndarray:
    """Create a background frame with subtle noise."""
    img = np.full((H, W, 3), color, dtype=np.uint8)
    if noise_amp:
        noise = np.random.randint(-noise_amp, noise_amp, (H, W, 3), dtype=np.int16)
        img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def draw_face(img, cx, cy, r, skin=SKIN_MED, gaze=True,
              expr="smile", alpha=1.0, tilt=0):
    """Draw a realistic-ish stylised face.

    Parameters
    ----------
    cx, cy  : centre of face
    r       : radius (head half-width)
    skin    : BGR skin tone
    gaze    : True = looking at camera (wide iris, centred pupils)
    expr    : "neutral" | "smile" | "laugh" | "surprised"
    alpha   : opacity (0..1) for blending into the image
    tilt    : horizontal tilt offset for pupils (simulates looking sideways)
    """
    overlay = img.copy()

    # ── Head ──────────────────────────────────────────────────────────────────
    cv2.ellipse(overlay, (cx, cy), (r, int(r * 1.25)), 0, 0, 360, skin, -1)

    # ── Hair ──────────────────────────────────────────────────────────────────
    hair = (max(0, skin[0]-80), max(0, skin[1]-110), max(0, skin[2]-120))
    hair = tuple(max(0, min(255, v)) for v in (35, 28, 22))
    cv2.ellipse(overlay, (cx, cy - r//2), (r, int(r * 0.65)), 0, 180, 360, hair, -1)
    cv2.ellipse(overlay, (cx, cy - r),    (r, r//3),           0, 180, 360, hair, -1)

    # ── Eyebrows ──────────────────────────────────────────────────────────────
    brow_c = (max(0, hair[0]-10), max(0, hair[1]-10), max(0, hair[2]-10))
    for ex in [cx - r//3, cx + r//3]:
        cv2.ellipse(overlay, (ex, cy - r//2 - 4),
                    (r//4, r//12), 0, 0, 180, brow_c, 2)

    # ── Eyes ──────────────────────────────────────────────────────────────────
    eye_rx = int(r * 0.20) if gaze else int(r * 0.13)
    eye_ry = int(r * 0.13) if gaze else int(r * 0.06)
    for ex in [cx - r//3, cx + r//3]:
        ey = cy - r//4
        cv2.ellipse(overlay, (ex, ey), (eye_rx, eye_ry), 0, 0, 360, (255,255,255), -1)
        if gaze:
            px = ex + tilt
            cv2.circle(overlay, (px, ey), max(2, eye_rx//2), (40, 35, 30), -1)
            cv2.circle(overlay, (px - 2, ey - 2), max(1, eye_rx//6), (230,230,230), -1)

    # ── Nose ──────────────────────────────────────────────────────────────────
    nose_c = tuple(max(0, int(v * 0.78)) for v in skin)
    cv2.circle(overlay, (cx, cy + r//8), r//8, nose_c, -1)

    # ── Mouth ──────────────────────────────────────────────────────────────────
    my = cy + r//3
    if expr == "smile":
        cv2.ellipse(overlay, (cx, my), (r//3, r//7), 0, 0, 180, (50,40,40), 2)
        cv2.ellipse(overlay, (cx, my), (r//3 - 4, r//8 - 2), 0, 0, 180, (140,100,100), 1)
    elif expr == "laugh":
        cv2.ellipse(overlay, (cx, my), (int(r*0.38), int(r*0.22)), 0, 0, 180, (40,35,35), -1)
        cv2.ellipse(overlay, (cx, my), (int(r*0.32), int(r*0.16)), 0, 0, 180, (200,170,170), -1)
        # Teeth
        cv2.ellipse(overlay, (cx, my), (int(r*0.30), int(r*0.12)), 0, 0, 180, (240,240,245), -1)
    elif expr == "surprised":
        cv2.ellipse(overlay, (cx, my), (r//5, r//4), 0, 0, 360, (40,35,35), -1)
        cv2.ellipse(overlay, (cx, my), (r//6, r//5), 0, 0, 360, (200,180,180), -1)
    else:  # neutral
        cv2.line(overlay, (cx - r//4, my), (cx + r//4, my), (50,40,40), 2)

    # ── Neck + ears ──────────────────────────────────────────────────────────
    ny = cy + int(r * 1.25)
    cv2.rectangle(overlay, (cx - r//4, ny), (cx + r//4, ny + r//2), skin, -1)
    for ex, ey in [(cx - r, cy), (cx + r, cy)]:
        cv2.ellipse(overlay, (ex, ey), (r//6, r//4), 0, 0, 360, skin, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def draw_gradient_bg(img, top_color, bot_color):
    """Vertical gradient background."""
    for y in range(H):
        t = y / H
        row = tuple(int(top_color[i] * (1-t) + bot_color[i] * t) for i in range(3))
        img[y, :] = row


def add_bokeh(img, n=18, color=(220, 215, 200), radius_range=(8, 28)):
    """Scatter soft bokeh circles (out-of-focus lights)."""
    overlay = img.copy()
    for _ in range(n):
        cx = random.randint(0, W)
        cy = random.randint(0, H // 3)
        r  = random.randint(*radius_range)
        cv2.circle(overlay, (cx, cy), r, color, -1)
    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    img[:] = cv2.GaussianBlur(img, (15, 15), 0)


def add_string_lights(img, y_base, n_bulbs=18, flicker=0.0):
    """Draw a row of warm fairy lights."""
    spacing = W // n_bulbs
    for i in range(n_bulbs):
        cx = i * spacing + spacing // 2
        sag = int(30 * math.sin(i * math.pi / n_bulbs))
        cy  = y_base + sag
        on  = random.random() > flicker
        c   = (50, 200, 255) if on else (20, 80, 110)
        cv2.circle(img, (cx, cy), 6, c, -1)
        # Glow halo
        cv2.circle(img, (cx, cy), 12, (int(c[0]*0.3), int(c[1]*0.3), int(c[2]*0.3)), -1)
    # Wire
    pts = [(i * spacing + spacing//2, y_base + int(30*math.sin(i*math.pi/n_bulbs)))
           for i in range(n_bulbs)]
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, (40, 40, 40), 1)


def flame(img, cx, cy, h=32, flicker=2):
    """Draw an animated candle flame."""
    dx = random.randint(-flicker, flicker)
    # Outer flame (orange)
    cv2.ellipse(img, (cx+dx, cy), (9, h), 0, 0, 360, (30, 130, 255), -1)
    # Inner flame (yellow)
    cv2.ellipse(img, (cx+dx, cy + h//4), (5, int(h*0.65)), 0, 0, 360, (30, 220, 255), -1)
    # Core (white)
    cv2.ellipse(img, (cx+dx, cy + h//3), (3, int(h*0.40)), 0, 0, 360, (240, 245, 255), -1)
    # Glow
    glow = np.zeros_like(img)
    cv2.circle(glow, (cx, cy + h//2), h+10, (10, 60, 100), -1)
    cv2.addWeighted(glow, 0.25, img, 1.0, 0, img)


def add_crowd_silhouette(img, y_start, n=12, color=(40, 35, 30)):
    """Draw a row of silhouetted heads at bottom to simulate audience."""
    spacing = W // n
    for i in range(n):
        cx = i * spacing + spacing // 2 + random.randint(-20, 20)
        cy = y_start + random.randint(-15, 15)
        r  = random.randint(25, 40)
        cv2.ellipse(img, (cx, cy), (r, int(r*1.3)), 0, 0, 360, color, -1)
        cv2.rectangle(img, (cx - r//2, cy + int(r*1.3)), (cx + r//2, cy + int(r*2.5)), color, -1)


def draw_garland(img, x1, y1, x2, y2, t, color=MARIGOLD):
    """Draw a hanging garland between two hands."""
    pts = []
    for i in range(30):
        s  = i / 29
        gx = int(x1 + (x2 - x1) * s)
        gy = int(y1 + (y2 - y1) * s + 50 * math.sin(s * math.pi) + 5 * math.sin(t * 0.2))
        pts.append((gx, gy))
    for a, b in zip(pts, pts[1:]):
        cv2.line(img, a, b, color, 4)
    for pt in pts[::3]:
        cv2.circle(img, pt, 6, color, -1)
        cv2.circle(img, pt, 3, (255,255,200), -1)


def draw_ring(img, cx, cy, r=18, gold=(30, 165, 215)):
    """Draw a gold ring."""
    cv2.circle(img, (cx, cy), r,     gold,  5)
    cv2.circle(img, (cx, cy), r - 3, (200, 235, 250), 2)
    # Diamond
    dia_pts = np.array([(cx, cy-r-12), (cx+8, cy-r-5),
                         (cx, cy-r+2), (cx-8, cy-r-5)], np.int32)
    cv2.fillPoly(img, [dia_pts], (240, 248, 255))
    cv2.polylines(img, [dia_pts], True, (150, 200, 230), 1)


def draw_hand(img, cx, cy, pointing_up=True, skin=SKIN_MED):
    """Simple hand/finger proxy."""
    direction = -1 if pointing_up else 1
    palm_y    = cy + direction * 10
    cv2.ellipse(img, (cx, palm_y), (22, 18), 0, 0, 360, skin, -1)
    for i, (fx, fy_off) in enumerate([(-18, -40), (-7, -48), (5, -46), (16, -40), (26, -28)]):
        fy = palm_y + direction * (-fy_off)
        cv2.ellipse(img, (cx + fx, fy), (7, 20), 0, 0, 360, skin, -1)


def vignette(img, strength=0.45):
    """Apply a circular vignette to simulate camera lens."""
    mask = np.zeros((H, W), dtype=np.float32)
    cv2.ellipse(mask, (W//2, H//2), (W//2, H//2), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (W//4*2+1, H//4*2+1), W//4)
    mask = 1.0 - (1.0 - mask) * strength
    img[:] = (img * mask[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)


def warm_tint(img, strength=0.15):
    """Add a warm (golden-hour) colour tint."""
    warm = np.zeros_like(img, dtype=np.float32)
    warm[:, :, 0] = -10   # reduce blue
    warm[:, :, 2] = +20   # boost red
    img[:] = np.clip(img.astype(np.float32) + warm * strength * (1/0.15), 0, 255).astype(np.uint8)


SPECS: list[dict] = []


# ═══════════════════════════════════════════════════════════════════════════════
# R01 — CAKE CUTTING
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r01_cake_cutting():
    vw, path = writer("R01_cake_cutting.mp4")
    for t in range(FPS * 10):  # 10 sec
        img = bg(WARM_BG)
        draw_gradient_bg(img, (195, 215, 235), (160, 185, 215))
        add_string_lights(img, 80, n_bulbs=22, flicker=0.05)

        # Cake body (3-tier)
        for tier, (tw, th, ty) in enumerate([(240, 60, H-180), (180, 55, H-235), (130, 50, H-280)]):
            fill = (235, 242, 250) if tier % 2 == 0 else (220, 230, 248)
            cv2.ellipse(img, (W//2, ty + th//2), (tw, th//2), 0, 0, 360, fill, -1)
            cv2.ellipse(img, (W//2, ty - th//2), (tw, th//2), 0, 0, 360, fill, -1)
            cv2.rectangle(img, (W//2-tw, ty-th//2), (W//2+tw, ty+th//2), fill, -1)
            # Frosting border
            cv2.ellipse(img, (W//2, ty - th//2), (tw, th//2), 0, 0, 360, (200,170,200), 3)
            # Decoration dots
            for dx in range(-tw+20, tw-10, 30):
                cv2.circle(img, (W//2+dx, ty), 5, MARIGOLD, -1)

        # Candles (5)
        candle_xs = [W//2 - 60, W//2 - 30, W//2, W//2 + 30, W//2 + 60]
        candle_y  = H - 305
        for cx in candle_xs:
            cv2.rectangle(img, (cx-4, candle_y), (cx+4, candle_y+30), (240,220,200), -1)
            flame(img, cx, candle_y - 20, h=28)

        # Faces: 3 people leaning in — centre birthday person, two sides
        lean = int(6 * math.sin(t / 12))
        draw_face(img, W//2 - 280, H//2 - 30 + lean, 68, SKIN_MED,  gaze=True,  expr="laugh")
        draw_face(img, W//2,       H//2 - 80,         80, SKIN_FAIR, gaze=True,  expr="smile" if t < FPS*4 else "laugh")
        draw_face(img, W//2 + 280, H//2 - 30 - lean, 68, SKIN_WARM, gaze=True,  expr="smile")

        # Knife sweeping down after 4 sec
        if t > FPS * 3:
            progress  = min(1.0, (t - FPS*3) / (FPS * 3))
            knife_y   = int((H - 330) + progress * 80)
            knife_tip = knife_y + 80
            cv2.rectangle(img, (W//2 - 5, knife_y), (W//2 + 5, knife_tip), (200,210,220), -1)
            cv2.line(img, (W//2, knife_tip), (W//2, knife_tip + 20), (180,190,210), 2)
            # Handle
            cv2.rectangle(img, (W//2 - 12, knife_y - 35), (W//2 + 12, knife_y), (80,60,50), -1)

        # At cut moment (t ~ FPS*5), flash + confetti
        if FPS*5 <= t <= FPS*5 + 8:
            flash = np.full((H, W, 3), 255, dtype=np.uint8)
            alpha = 1.0 - (t - FPS*5) / 8
            cv2.addWeighted(flash, alpha * 0.3, img, 1 - alpha * 0.3, 0, img)

        vignette(img)
        warm_tint(img)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R01_cake_cutting.mp4",
    "scenario": "Birthday/anniversary cake cutting — 3-tier cake, 5 candles, 3 faces leaning in, knife sweep at sec 4",
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 3,
    "expected_captures_max": 40,
    "notes": "Run with prompt='cake cutting'. All 3 faces look at camera = HIGH. Knife motion at sec 4 = CRITICAL.",
    "prompt_hint": "cake cutting",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R02 — RING CEREMONY (SAGAI)
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r02_ring_ceremony():
    vw, path = writer("R02_ring_ceremony.mp4")
    for t in range(FPS * 12):
        img = bg((185, 200, 225))
        draw_gradient_bg(img, (205, 215, 235), (160, 178, 210))
        add_bokeh(img, n=22, color=(200, 210, 230), radius_range=(10, 35))
        add_string_lights(img, 70, n_bulbs=20, flicker=0.08)

        # Couple facing each other
        progress = min(1.0, t / (FPS * 6))
        lx = int(W//2 - 240 + 60 * progress)  # groom moves slightly right
        rx = int(W//2 + 240 - 60 * progress)  # bride moves slightly left

        # Groom (left, looking right toward bride = tilt right)
        draw_face(img, lx, H//2 - 20, 78, SKIN_MED,  gaze=True, expr="smile", tilt=8)
        # Bride (right, looking left toward groom = tilt left)
        draw_face(img, rx, H//2 - 20, 75, SKIN_FAIR, gaze=True, expr="smile", tilt=-8)

        # Hands meeting in centre
        hand_cx = W//2
        hand_cy = H//2 + 120
        draw_hand(img, lx + 80, hand_cy, pointing_up=True, skin=SKIN_MED)
        draw_hand(img, rx - 80, hand_cy, pointing_up=True, skin=SKIN_FAIR)

        # Ring exchange — ring slides from groom's hand to bride's finger after sec 5
        ring_progress = min(1.0, max(0.0, (t - FPS*5) / (FPS * 3)))
        ring_x = int((lx + 80) + (rx - 80 - (lx + 80)) * ring_progress)
        ring_y = int(hand_cy - 30 + 10 * ring_progress)
        draw_ring(img, ring_x, ring_y)

        # Reaction moment: at sec 8 both look at camera
        if t > FPS * 8:
            draw_face(img, lx, H//2 - 20, 78, SKIN_MED,  gaze=True, expr="laugh", tilt=0)
            draw_face(img, rx, H//2 - 20, 75, SKIN_FAIR, gaze=True, expr="laugh", tilt=0)

        # Family faces in background (smaller, blurred)
        for i, (fx, expr) in enumerate([
            (W//2 - 490, "smile"), (W//2 - 390, "smile"), (W//2 + 390, "smile"), (W//2 + 490, "laugh")
        ]):
            draw_face(img, fx, H//2 + 30, 45, SKINS[i % 4], gaze=False, expr=expr, alpha=0.75)

        vignette(img, strength=0.5)
        warm_tint(img, strength=0.2)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R02_ring_ceremony.mp4",
    "scenario": "Sagai / ring ceremony — couple facing, ring slides at sec 5, both look at camera at sec 8",
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 2,
    "expected_captures_max": 35,
    "notes": "Run with prompt='ring ceremony'. Camera-gaze at sec 8 = HIGH/CRITICAL. Ring transfer = peak signal.",
    "prompt_hint": "ring ceremony",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R03 — VARMALA / JAIMALA (Garland Exchange)
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r03_varmala():
    vw, path = writer("R03_varmala.mp4")
    for t in range(FPS * 12):
        img = bg((170, 185, 215))
        draw_gradient_bg(img, (185, 195, 225), (145, 165, 205))
        add_bokeh(img, n=30, color=GOLD, radius_range=(8, 22))

        # Stage floor
        cv2.rectangle(img, (0, H - 120), (W, H), (140, 155, 180), -1)
        cv2.rectangle(img, (0, H - 122), (W, H - 118), (160, 175, 200), 2)

        # Groom (left) and Bride (right)
        groom_x = int(W//2 - 200 + 20 * math.sin(t / 15))
        bride_x = int(W//2 + 200 - 20 * math.sin(t / 15))
        draw_face(img, groom_x, H//2 - 30, 82, SKIN_WARM, gaze=True, expr="smile")
        draw_face(img, bride_x, H//2 - 30, 78, SKIN_FAIR, gaze=True, expr="smile")

        # Garland in each person's hands, moving toward other person
        phase = min(1.0, t / (FPS * 5))
        # Groom's garland (left → right)
        g1x = int(groom_x + 120 * phase)
        g1y = int(H//2 + 80 + 40 * phase)
        draw_garland(img, groom_x + 40, H//2 + 60, g1x, g1y, t, color=MARIGOLD)
        # Bride's garland (right → left)
        g2x = int(bride_x - 120 * phase)
        g2y = int(H//2 + 80 + 40 * phase)
        draw_garland(img, bride_x - 40, H//2 + 60, g2x, g2y, t, color=DEEP_RED)

        # Crowd silhouette at back
        add_crowd_silhouette(img, H - 200, n=16)

        # Confetti burst at garland exchange moment (sec 5-6)
        if FPS*5 <= t <= FPS*6:
            for _ in range(30):
                cx_ = random.randint(W//2 - 200, W//2 + 200)
                cy_ = random.randint(H//4, 3*H//4)
                c_  = random.choice([MARIGOLD, GOLD, DEEP_RED, (100, 200, 100)])
                cv2.circle(img, (cx_, cy_), random.randint(4, 9), c_, -1)

        vignette(img)
        warm_tint(img, strength=0.25)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R03_varmala.mp4",
    "scenario": "Varmala / Jaimala — couple facing, garlands exchanged at sec 5, confetti burst",
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 3,
    "expected_captures_max": 40,
    "notes": "Run with prompt='varmala'. Garland exchange at sec 5 is the peak moment.",
    "prompt_hint": "varmala",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R04 — SANGEET DANCE
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r04_sangeet():
    vw, path = writer("R04_sangeet_dance.mp4")
    stage_lights = [(W//6, 0), (W//3, 0), (W//2, 0), (2*W//3, 0), (5*W//6, 0)]
    light_colors = [(40, 60, 200), (30, 200, 80), (200, 60, 60), (180, 60, 180), (40, 180, 220)]

    for t in range(FPS * 12):
        img = bg((30, 25, 20))  # dark stage

        # Stage lights (sweeping)
        for i, ((lx, ly), lc) in enumerate(zip(stage_lights, light_colors)):
            sweep  = int(60 * math.sin(t / 18 + i * 1.2))
            end_x  = lx + sweep
            end_y  = H - 100
            pts    = np.array([(lx - 30, ly), (lx + 30, ly), (end_x + 80, end_y), (end_x - 80, end_y)])
            overlay= img.copy()
            cv2.fillPoly(overlay, [pts], lc)
            cv2.addWeighted(overlay, 0.18, img, 0.82, 0, img)

        # 5 dancers with energetic movement
        dancer_positions = [
            (W//2 - 360, H//2),
            (W//2 - 180, H//2 - 20),
            (W//2,       H//2 + 10),
            (W//2 + 180, H//2 - 20),
            (W//2 + 360, H//2),
        ]
        exprs  = ["laugh", "smile", "laugh", "smile", "laugh"]
        skin_l = [SKIN_FAIR, SKIN_MED, SKIN_WARM, SKIN_FAIR, SKIN_DARK]
        for i, (dx, dy) in enumerate(dancer_positions):
            bob   = int(25 * math.sin(t / 6 + i * 0.8))
            sway  = int(15 * math.sin(t / 9 + i * 1.1))
            gaze  = (abs(t // FPS - 3) < 2) or (abs(t // FPS - 8) < 2)
            draw_face(img, dx + sway, dy + bob, 62, skin_l[i],
                      gaze=gaze, expr=exprs[i])

        # Crowd silhouette at front/bottom
        add_crowd_silhouette(img, H - 150, n=14, color=(20, 18, 15))

        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R04_sangeet_dance.mp4",
    "scenario": "Sangeet performance — dark stage, 5 coloured spotlights, 5 dancers, group gaze at sec 3 and 8",
    "expected_priority": ["high", "elevated", "critical"],
    "expected_captures_min": 2,
    "expected_captures_max": 40,
    "notes": "Run with prompt='sangeet'. Stage lights + group gaze moments = HIGH/CRITICAL.",
    "prompt_hint": "sangeet",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R05 — FIRST DANCE (COUPLE)
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r05_first_dance():
    vw, path = writer("R05_first_dance.mp4")
    for t in range(FPS * 12):
        img = bg((120, 110, 95))
        draw_gradient_bg(img, (135, 120, 105), (90, 80, 70))
        add_bokeh(img, n=25, color=(180, 195, 210), radius_range=(12, 40))

        # Slow circular sway
        angle  = t / (FPS * 4) * 2 * math.pi * 0.5
        cx     = int(W//2 + 30 * math.sin(angle))
        cy     = int(H//2 + 10 * math.cos(angle))
        sway_r = int(30 * math.sin(t / 20))

        draw_face(img, cx - 65 + sway_r, cy - 15, 78, SKIN_MED,  gaze=False, expr="smile", tilt=6)
        draw_face(img, cx + 65 + sway_r, cy - 15, 72, SKIN_FAIR, gaze=False, expr="smile", tilt=-6)

        # Look at camera moment — sec 7-9
        if FPS*7 <= t <= FPS*9:
            draw_face(img, cx - 65 + sway_r, cy - 15, 78, SKIN_MED,  gaze=True, expr="smile")
            draw_face(img, cx + 65 + sway_r, cy - 15, 72, SKIN_FAIR, gaze=True, expr="laugh")

        add_string_lights(img, H - 90, n_bulbs=28, flicker=0.03)
        vignette(img, strength=0.65)
        warm_tint(img, strength=0.3)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R05_first_dance.mp4",
    "scenario": "First dance — couple swaying together, dim warm light, both look at camera at sec 7",
    "expected_priority": ["high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Camera gaze at sec 7-9 is the key moment. Heavy vignette simulates real low-light dance floor.",
    "prompt_hint": "first dance",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R06 — EMOTIONAL PARENT HUG
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r06_parent_hug():
    vw, path = writer("R06_parent_hug.mp4")
    for t in range(FPS * 10):
        img = bg(WARM_BG)
        draw_gradient_bg(img, (210, 220, 235), (175, 190, 215))
        add_string_lights(img, 100, n_bulbs=18, flicker=0.04)

        progress = min(1.0, t / (FPS * 4))

        # Parent (left, taller/older skin tone)
        px = int(W//2 - 160 + 100 * progress)
        py = H//2 - 10
        draw_face(img, px, py, 82, SKIN_DARK, gaze=False, expr="laugh")

        # Child (right, smaller, moving toward parent)
        cx_ = int(W//2 + 200 - 140 * progress)
        cy_ = H//2 + 30
        draw_face(img, cx_, cy_, 60, SKIN_MED,  gaze=False, expr="laugh")

        # After hug (sec 4+): both look at camera
        if t > FPS * 4:
            draw_face(img, px,  py,  82, SKIN_DARK, gaze=True, expr="laugh")
            draw_face(img, cx_, cy_, 60, SKIN_MED,  gaze=True, expr="smile")

        vignette(img, strength=0.55)
        warm_tint(img, strength=0.2)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R06_parent_hug.mp4",
    "scenario": "Emotional parent-child hug — converging then both look at camera at sec 4",
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Post-hug camera gaze = HIGH. Convergence motion = ELEVATED pre-hug.",
    "prompt_hint": "emotional moment",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R07 — GROUP PHOTO MOMENT (8 FACES)
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r07_group_photo():
    vw, path = writer("R07_group_photo.mp4")

    # Fixed face positions for a group of 8
    positions = [
        (W//2 - 490, H//2 + 20, 52, SKIN_FAIR),
        (W//2 - 350, H//2 - 20, 60, SKIN_MED),
        (W//2 - 200, H//2 + 10, 65, SKIN_WARM),
        (W//2 -  60, H//2 - 15, 68, SKIN_DARK),
        (W//2 +  60, H//2 - 15, 68, SKIN_MED),
        (W//2 + 200, H//2 + 10, 65, SKIN_FAIR),
        (W//2 + 350, H//2 - 20, 60, SKIN_WARM),
        (W//2 + 490, H//2 + 20, 52, SKIN_DARK),
    ]
    exprs = ["smile", "laugh", "smile", "laugh", "smile", "laugh", "smile", "smile"]

    for t in range(FPS * 10):
        img = bg((200, 210, 225))
        draw_gradient_bg(img, (210, 218, 232), (180, 195, 218))
        add_bokeh(img, n=15, color=(220, 225, 235), radius_range=(8, 20))

        # Countdown text overlay (sec 0-3 people settling; sec 3+ all gaze)
        phase = "settling" if t < FPS * 3 else "gaze"

        for (fx, fy, fr, fskin), expr in zip(positions, exprs):
            bob  = int(4 * math.sin(t / 8 + fx * 0.005))
            gaze = (phase == "gaze")
            draw_face(img, fx, fy + bob, fr, fskin, gaze=gaze, expr=expr)

        # "Say cheese!" text after sec 3
        if phase == "gaze":
            cv2.putText(img, "Say Cheese!", (W//2 - 140, H - 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (60, 80, 200), 3, cv2.LINE_AA)

        vignette(img, strength=0.35)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R07_group_photo.mp4",
    "scenario": "Group photo — 8 faces, first 3 sec settling/random gaze, then all look at camera with 'Say Cheese!'",
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 3,
    "expected_captures_max": 50,
    "notes": "All 8 faces gazing after sec 3 = CRITICAL. This tests the group-gaze saturation behavior.",
    "prompt_hint": "group photo",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R08 — DIWALI DIYA LIGHTING
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r08_diwali():
    vw, path = writer("R08_diwali_diya.mp4")
    for t in range(FPS * 10):
        img = bg((35, 30, 25))  # dark indoor

        # Multiple diyas in a row
        diya_positions = [W//2 - 200, W//2 - 100, W//2, W//2 + 100, W//2 + 200]
        diya_y = H - 200

        for dx in diya_positions:
            # Diya plate (clay)
            cv2.ellipse(img, (dx, diya_y), (30, 12), 0, 0, 360, (60, 100, 160), -1)
            cv2.ellipse(img, (dx, diya_y), (30, 12), 0, 0, 360, (40, 70, 120), 2)
            # Oil pool
            cv2.ellipse(img, (dx, diya_y - 3), (15, 5), 0, 0, 360, (20, 50, 100), -1)
            # Flame (lit progressively)
            lit_at = int(dx / 100) + 2
            if t > FPS * lit_at:
                flame(img, dx, diya_y - 15, h=24)

        # Rangoli pattern on floor
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            ex  = int(W//2 + 80 * math.cos(rad))
            ey  = int(diya_y + 60 + 20 * math.sin(rad))
            c   = [MARIGOLD, DEEP_RED, GOLD, (60, 200, 60)][(angle // 30) % 4]
            cv2.circle(img, (ex, ey), 6, c, -1)
        cv2.circle(img, (W//2, diya_y + 60), 8, (30, 160, 255), -1)

        # 2 people's faces lit by diya glow from below
        glow_strength = min(1.0, max(0.0, (t - FPS*2) / (FPS * 3)))
        face_skin_l   = tuple(min(255, int(SKIN_MED[i] + 30 * glow_strength)) for i in range(3))
        face_skin_r   = tuple(min(255, int(SKIN_FAIR[i] + 30 * glow_strength)) for i in range(3))
        draw_face(img, W//2 - 200, H//2 - 60, 72, face_skin_l, gaze=False, expr="smile")
        draw_face(img, W//2 + 200, H//2 - 60, 68, face_skin_r, gaze=True,  expr="smile")

        # Both look at camera after sec 7
        if t > FPS * 7:
            draw_face(img, W//2 - 200, H//2 - 60, 72, face_skin_l, gaze=True, expr="smile")
            draw_face(img, W//2 + 200, H//2 - 60, 68, face_skin_r, gaze=True, expr="laugh")

        vignette(img, strength=0.7)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R08_diwali_diya.mp4",
    "scenario": "Diwali diya lighting — 5 diyas, rangoli, 2 faces lit from below, both look at camera at sec 7",
    "expected_priority": ["high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Low ambient light tests the quality floor. Diya glow = warm tint. Both gaze at sec 7 = HIGH.",
    "prompt_hint": "diwali",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R09 — BABY FIRST STEPS
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r09_baby_steps():
    vw, path = writer("R09_baby_steps.mp4")
    for t in range(FPS * 10):
        img = bg(CREAM)
        draw_gradient_bg(img, (240, 245, 255), (215, 225, 240))

        # Baby (small, waddling across frame)
        bx    = int(W//4 + (t / (FPS * 9)) * W//2)
        by    = H - 200
        waddle= int(12 * math.sin(t / 4))
        draw_face(img, bx, by + waddle, 42, SKIN_FAIR, gaze=False, expr="smile")
        # Baby body
        cv2.rectangle(img, (bx-22, by+50), (bx+22, by+110), (180,200,230), -1)
        # Legs waddling
        leg = int(12 * math.sin(t / 4))
        cv2.line(img, (bx - 10, by+110), (bx - 10 + leg, by+155), SKIN_FAIR, 8)
        cv2.line(img, (bx + 10, by+110), (bx + 10 - leg, by+155), SKIN_FAIR, 8)

        # Two watching adults (excited faces)
        # Left parent — comes into frame from left
        parent_l_x = max(60, W//4 - 200 + int((W//4 - 120) * min(1.0, t/(FPS*2))))
        draw_face(img, parent_l_x, H//2 - 30, 72, SKIN_DARK, gaze=False, expr="laugh" if t>FPS*3 else "smile")
        # Right parent — already in frame, kneeling right side
        draw_face(img, 3*W//4 + 80, H//2 + 20, 68, SKIN_MED, gaze=False,
                  expr="surprised" if FPS*4 <= t <= FPS*6 else "laugh")

        # Both look at camera at sec 8 (proud moment)
        if t > FPS * 8:
            draw_face(img, parent_l_x, H//2 - 30, 72, SKIN_DARK, gaze=True, expr="laugh")
            draw_face(img, 3*W//4 + 80, H//2 + 20, 68, SKIN_MED, gaze=True, expr="laugh")

        vignette(img, strength=0.4)
        warm_tint(img, strength=0.1)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R09_baby_steps.mp4",
    "scenario": "Baby's first steps — small waddling figure, 2 excited parents watching, both look at camera at sec 8",
    "expected_priority": ["high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Joyful candid. Baby motion + parent expressions = ELEVATED. Camera gaze at sec 8 = HIGH.",
    "prompt_hint": "baby steps",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R10 — CHAMPAGNE / TOAST
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r10_toast():
    vw, path = writer("R10_champagne_toast.mp4")
    for t in range(FPS * 10):
        img = bg(WARM_BG)
        draw_gradient_bg(img, (200, 215, 230), (160, 180, 210))
        add_bokeh(img, n=20, color=CHAMPAGNE, radius_range=(8, 28))

        # 4 faces in a semi-circle
        face_data = [
            (W//2 - 360, H//2 - 10, 62, SKIN_FAIR, "smile"),
            (W//2 - 150, H//2 - 20, 68, SKIN_MED,  "laugh"),
            (W//2 + 150, H//2 - 20, 68, SKIN_WARM, "smile"),
            (W//2 + 360, H//2 - 10, 62, SKIN_DARK, "laugh"),
        ]
        for fx, fy, fr, fskin, fexpr in face_data:
            draw_face(img, fx, fy, fr, fskin, gaze=(t > FPS*5), expr=fexpr)

        # Champagne glasses converging to centre
        progress = min(1.0, (t - FPS) / (FPS * 3)) if t > FPS else 0
        glass_ys = [H//2 + 130, H//2 + 130, H//2 + 130, H//2 + 130]
        glass_xs_start = [W//2 - 380, W//2 - 140, W//2 + 140, W//2 + 380]
        glass_xs_end   = [W//2 - 80,  W//2 - 30,  W//2 + 30,  W//2 + 80]

        for i, (sx, ex, gy) in enumerate(zip(glass_xs_start, glass_xs_end, glass_ys)):
            gx = int(sx + (ex - sx) * progress)
            # Stem
            cv2.line(img, (gx, gy), (gx, gy - 80), (200, 215, 230), 3)
            # Base
            cv2.ellipse(img, (gx, gy), (25, 8), 0, 0, 360, (200, 215, 230), 2)
            # Bowl
            bowl_pts = np.array([(gx-20, gy-80), (gx+20, gy-80),
                                  (gx+28, gy-130), (gx-28, gy-130)])
            cv2.fillPoly(img, [bowl_pts], (185, 205, 225))
            cv2.polylines(img, [bowl_pts], True, (210, 225, 240), 2)
            # Bubbles
            if t > FPS * 2:
                for b in range(4):
                    bby = gy - 90 - int((t % FPS) * 1.5) - b * 12
                    if gy - 130 < bby < gy - 80:
                        cv2.circle(img, (gx + random.randint(-8, 8), bby), 2, (220, 230, 240), -1)

        # Clink flash at glass-meeting moment
        if abs(progress - 1.0) < 0.07 and t > FPS:
            flash = np.full_like(img, 255)
            cv2.addWeighted(flash, 0.35, img, 0.65, 0, img)
            # Sparkle lines
            cx_, cy_ = W//2, H//2 + 50
            for ang in range(0, 360, 30):
                rad = math.radians(ang)
                cv2.line(img, (cx_, cy_),
                         (int(cx_ + 40*math.cos(rad)), int(cy_ + 40*math.sin(rad))),
                         (240, 240, 200), 2)

        vignette(img)
        warm_tint(img, strength=0.2)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R10_champagne_toast.mp4",
    "scenario": "Champagne toast — 4 faces, glasses converge at sec 4, all look at camera at sec 5, clink flash",
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 2,
    "expected_captures_max": 40,
    "notes": "Run with prompt='toast'. Glass clink = peak signal. Group gaze = HIGH/CRITICAL.",
    "prompt_hint": "toast",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R11 — BRIDE WALKING IN
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r11_bride_entry():
    vw, path = writer("R11_bride_entry.mp4")
    for t in range(FPS * 12):
        img = bg((160, 175, 205))
        draw_gradient_bg(img, (175, 188, 215), (135, 155, 195))
        add_string_lights(img, 60, n_bulbs=30, flicker=0.04)
        add_bokeh(img, n=18, color=(215, 220, 235), radius_range=(10, 30))

        # Crowd turning to look at entrance (left side)
        crowd_positions = [
            (W//2 - 100, H//2 + 40, 58, SKIN_MED),
            (W//2 + 80,  H//2 + 20, 62, SKIN_WARM),
            (W//2 + 250, H//2 + 10, 65, SKIN_DARK),
            (W//2 + 400, H//2 + 30, 55, SKIN_FAIR),
        ]
        for fx, fy, fr, fskin in crowd_positions:
            # Crowd gradually turns left (tilt increases negatively)
            tilt_amount = int(-12 * min(1.0, t / (FPS * 3)))
            draw_face(img, fx, fy, fr, fskin, gaze=False, expr="surprised", tilt=tilt_amount)

        # Bride entering from left, walking right
        bride_x = int(-80 + (t / (FPS * 10)) * (W//2 + 80))
        bride_x = min(W//2 + 50, bride_x)
        draw_face(img, bride_x, H//2 - 20, 82, SKIN_FAIR, gaze=False, expr="smile")

        # Aisle flower petals falling
        for _ in range(5):
            px_ = random.randint(0, bride_x + 80)
            py_ = random.randint(0, H)
            cv2.circle(img, (px_, py_), random.randint(3, 7), MARIGOLD, -1)

        # Bride looks at camera at sec 9
        if t > FPS * 9:
            draw_face(img, bride_x, H//2 - 20, 82, SKIN_FAIR, gaze=True, expr="smile")
            # Crowd also looks forward
            for fx, fy, fr, fskin in crowd_positions:
                draw_face(img, fx, fy, fr, fskin, gaze=True, expr="smile")

        add_crowd_silhouette(img, H - 160, n=14)
        vignette(img, strength=0.5)
        warm_tint(img, strength=0.2)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R11_bride_entry.mp4",
    "scenario": "Bride walking in — bride enters from left, crowd turns to watch, all look at camera at sec 9",
    "expected_priority": ["critical", "high", "elevated"],
    "expected_captures_min": 2,
    "expected_captures_max": 40,
    "notes": "Run with prompt='bride entry'. Lateral motion + crowd gaze shift = ELEVATED. Full gaze at sec 9 = HIGH/CRITICAL.",
    "prompt_hint": "bride entry",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R12 — MEHENDI CLOSE-UP
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r12_mehendi():
    vw, path = writer("R12_mehendi_closeup.mp4")
    for t in range(FPS * 10):
        img = bg(CREAM)
        draw_gradient_bg(img, (245, 248, 255), (225, 232, 248))

        # Hands resting (centre)
        hand_y = H - 250
        # Decorated hand (bride)
        cv2.ellipse(img, (W//2, hand_y), (140, 55), 0, 0, 360, SKIN_FAIR, -1)
        # Mehendi design (dark brown henna patterns)
        henna = (30, 40, 80)
        cv2.ellipse(img, (W//2, hand_y), (50, 18), 0, 0, 360, henna, 2)
        cv2.circle(img, (W//2, hand_y), 10, henna, -1)
        for ang in range(0, 360, 45):
            rad = math.radians(ang)
            cv2.line(img, (W//2, hand_y),
                     (int(W//2 + 45*math.cos(rad)), int(hand_y + 18*math.sin(rad))),
                     henna, 2)
        for i, finger_x in enumerate(range(W//2 - 80, W//2 + 90, 40)):
            cv2.rectangle(img, (finger_x - 12, hand_y - 55 - i*5),
                          (finger_x + 12, hand_y - 15), SKIN_FAIR, -1)
            cv2.circle(img, (finger_x, hand_y - 55 - i*5), 5, henna, 1)

        # Mehendi artist's hand in corner
        draw_hand(img, W//2 + 160, hand_y + 10, pointing_up=False, skin=SKIN_MED)

        # 2 admiring faces looking at hands (looking down = gaze off)
        bob = int(4 * math.sin(t / 10))
        draw_face(img, W//2 - 300, H//2 - 100 + bob, 68, SKIN_MED,  gaze=False, expr="smile")
        draw_face(img, W//2 + 320, H//2 - 100 - bob, 65, SKIN_DARK, gaze=False, expr="smile")

        # Both look at camera at sec 6 (spontaneous moment)
        if t > FPS * 6:
            draw_face(img, W//2 - 300, H//2 - 100, 68, SKIN_MED,  gaze=True, expr="laugh")
            draw_face(img, W//2 + 320, H//2 - 100, 65, SKIN_DARK, gaze=True, expr="smile")

        warm_tint(img, strength=0.15)
        vignette(img, strength=0.4)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R12_mehendi_closeup.mp4",
    "scenario": "Mehendi / henna — decorated hands in focus, 2 admiring faces, both look at camera at sec 6",
    "expected_priority": ["high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Low motion. Camera gaze at sec 6 = HIGH. Tests SnapAI candid-vs-posed distinction.",
    "prompt_hint": "mehendi",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R13 — KIDS CANDID PLAYING
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r13_kids_playing():
    vw, path = writer("R13_kids_candid.mp4")
    kid_state = [(random.randint(100, W-100), random.randint(H//3, H-150),
                   random.uniform(-6, 6), random.uniform(-4, 4),
                   random.choice(SKINS)) for _ in range(4)]

    for t in range(FPS * 10):
        img = bg((195, 215, 200))
        draw_gradient_bg(img, (210, 225, 210), (175, 200, 180))

        # Grass stripe at bottom
        cv2.rectangle(img, (0, H-120), (W, H), FOREST, -1)
        for gx in range(0, W, 20):
            cv2.line(img, (gx, H-120), (gx + 5, H-155), (30, 110, 40), 2)

        new_states = []
        for (kx, ky, vx, vy, kskin) in kid_state:
            # Update position
            kx = int(kx + vx)
            ky = int(ky + vy)
            # Bounce off walls
            if kx < 60 or kx > W - 60:
                vx = -vx
            if ky < H//4 or ky > H - 160:
                vy = -vy
            kx = max(60, min(W-60, kx))
            ky = max(H//4, min(H-160, ky))
            new_states.append((kx, ky, vx, vy, kskin))

            # Random gaze (candid — rarely look at camera)
            gaze = random.random() < 0.12
            expr = random.choice(["laugh", "laugh", "smile", "neutral"])
            draw_face(img, kx, ky, 48, kskin, gaze=gaze, expr=expr)

        kid_state = new_states

        # Occasional all-gaze moment at sec 7 (they spot the camera)
        if FPS*7 <= t <= FPS*7 + FPS:
            for (kx, ky, _, _, kskin) in kid_state:
                draw_face(img, kx, ky, 48, kskin, gaze=True, expr="laugh")

        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R13_kids_candid.mp4",
    "scenario": "Kids playing candid — 4 children with random fast motion, random gaze, all notice camera at sec 7",
    "expected_priority": ["elevated", "high", "normal"],
    "expected_captures_min": 1,
    "expected_captures_max": 35,
    "notes": "Tests candid non-posed behavior. Random gaze = ELEVATED. Group gaze at sec 7 = HIGH.",
    "prompt_hint": "kids playing",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R14 — OUTDOOR FIREWORKS
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r14_fireworks():
    vw, path = writer("R14_outdoor_fireworks.mp4")

    bursts: list[dict] = []

    for t in range(FPS * 12):
        img = bg((15, 12, 10))

        # Spawn new burst every ~1 sec
        if t % FPS == 0 and t > 0:
            bursts.append({
                "cx": random.randint(W//4, 3*W//4),
                "cy": random.randint(H//6, H//2),
                "birth": t,
                "color": random.choice([DEEP_RED, MARIGOLD, TEAL,
                                        (200, 80, 200), (80, 200, 200), (200, 200, 80)])
            })

        # Draw bursts
        for b in bursts:
            age   = t - b["birth"]
            r_max = 100
            r     = int(r_max * age / (FPS * 1.2))
            alpha = max(0.0, 1.0 - age / (FPS * 1.5))
            if r > 0 and alpha > 0:
                overlay = img.copy()
                for ang in range(0, 360, 12):
                    rad = math.radians(ang)
                    ex  = int(b["cx"] + r * math.cos(rad))
                    ey  = int(b["cy"] + r * math.sin(rad))
                    cv2.circle(overlay, (ex, ey), max(2, int(6 * alpha)), b["color"], -1)
                # Centre glow
                cv2.circle(overlay, (b["cx"], b["cy"]), max(1, r//3), (240,240,255), -1)
                cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)
                # Screen ambient from burst
                ambient = np.zeros_like(img)
                cv2.circle(ambient, (b["cx"], b["cy"]), 200,
                           tuple(int(v*0.2) for v in b["color"]), -1)
                img = cv2.addWeighted(ambient, 0.3, img, 0.7, 0)

        # Silhouetted crowd watching from bottom
        add_crowd_silhouette(img, H - 160, n=14, color=(8, 7, 5))

        # 3 faces lit by fireworks glow (looking up — no camera gaze)
        glow = min(1.0, max(0.0, (t - FPS) / (FPS * 3))) * 0.5
        gs   = tuple(min(255, int(SKIN_MED[i] + 40 * glow)) for i in range(3))
        draw_face(img, W//2 - 250, H - 260, 58, gs,       gaze=False, expr="surprised")
        draw_face(img, W//2,       H - 250, 65, SKIN_DARK, gaze=False, expr="smile")
        draw_face(img, W//2 + 250, H - 260, 55, SKIN_FAIR, gaze=False, expr="surprised")

        # Brief camera look at sec 10
        if FPS*10 <= t <= FPS*10 + FPS:
            draw_face(img, W//2 - 250, H-260, 58, gs,        gaze=True, expr="smile")
            draw_face(img, W//2,       H-250, 65, SKIN_DARK,  gaze=True, expr="laugh")
            draw_face(img, W//2 + 250, H-260, 55, SKIN_FAIR,  gaze=True, expr="surprised")

        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R14_outdoor_fireworks.mp4",
    "scenario": "Outdoor fireworks — dark sky, burst animations, silhouetted crowd, 3 faces looking up, camera gaze at sec 10",
    "expected_priority": ["elevated", "high", "normal"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Tests low-light with bright transient sources. Faces look up (no gaze) = ELEVATED. Camera gaze at sec 10 = HIGH.",
    "prompt_hint": "fireworks",
})


# ═══════════════════════════════════════════════════════════════════════════════
# R15 — AWARD / FELICITATION
# ═══════════════════════════════════════════════════════════════════════════════
def clip_r15_award():
    vw, path = writer("R15_award_ceremony.mp4")
    for t in range(FPS * 10):
        img = bg((30, 25, 20))  # dark stage
        draw_gradient_bg(img, (40, 35, 28), (20, 17, 14))

        # Stage spotlights
        for sx, sc in [(W//3, (150, 150, 200)), (2*W//3, (150, 200, 150))]:
            cone = img.copy()
            pts  = np.array([(sx - 15, 0), (sx + 15, 0),
                             (sx + 140, H - 100), (sx - 140, H - 100)])
            cv2.fillPoly(cone, [pts], sc)
            cv2.addWeighted(cone, 0.2, img, 0.8, 0, img)

        # Stage floor
        cv2.rectangle(img, (0, H - 90), (W, H), (50, 45, 40), -1)
        cv2.line(img, (0, H-90), (W, H-90), (80, 75, 65), 2)

        # Presenter (left) and recipient (right)
        progress = min(1.0, (t - FPS) / (FPS * 3)) if t > FPS else 0
        # Handshake: hands move toward each other
        hand_x_l = int(W//2 - 180 + 80 * progress)
        hand_x_r = int(W//2 + 180 - 80 * progress)

        draw_face(img, W//2 - 250, H//2 - 20, 72, SKIN_MED,  gaze=False, expr="smile", tilt=6)
        draw_face(img, W//2 + 250, H//2 - 20, 68, SKIN_WARM, gaze=False, expr="smile", tilt=-6)

        # Both look at camera at sec 6 (pose for photo)
        if t > FPS * 6:
            draw_face(img, W//2 - 250, H//2 - 20, 72, SKIN_MED,  gaze=True, expr="smile")
            draw_face(img, W//2 + 250, H//2 - 20, 68, SKIN_WARM, gaze=True, expr="laugh")

        # Trophy / plaque
        if t > FPS:
            ty_ = H//2 + 40
            cv2.rectangle(img, (W//2 - 18, ty_ - 60), (W//2 + 18, ty_), GOLD, -1)
            cv2.ellipse(img, (W//2, ty_ - 60), (25, 18), 0, 0, 360, GOLD, -1)
            cv2.ellipse(img, (W//2, ty_ - 60), (15, 10), 0, 0, 360, (200, 230, 240), -1)

        # Audience applause motion (silhouettes raising hands)
        add_crowd_silhouette(img, H - 160, n=14, color=(20, 18, 14))
        if t > FPS * 5:  # applause — arms up
            for i in range(14):
                ax = i * (W // 14) + W//28
                cv2.line(img, (ax, H - 160), (ax - 20, H - 200), (20, 18, 14), 8)
                cv2.line(img, (ax, H - 160), (ax + 20, H - 200), (20, 18, 14), 8)

        vignette(img, strength=0.6)
        vw.write(img)
    vw.release()
    return path

SPECS.append({
    "name": "R15_award_ceremony.mp4",
    "scenario": "Award/felicitation on stage — spotlight, trophy handoff, both faces look at camera at sec 6, audience applause",
    "expected_priority": ["high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "Run with prompt='award'. Camera gaze at sec 6 = HIGH. Applause motion = ELEVATED prior.",
    "prompt_hint": "award",
})


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
CLIP_FUNCS = [
    clip_r01_cake_cutting,
    clip_r02_ring_ceremony,
    clip_r03_varmala,
    clip_r04_sangeet,
    clip_r05_first_dance,
    clip_r06_parent_hug,
    clip_r07_group_photo,
    clip_r08_diwali,
    clip_r09_baby_steps,
    clip_r10_toast,
    clip_r11_bride_entry,
    clip_r12_mehendi,
    clip_r13_kids_playing,
    clip_r14_fireworks,
    clip_r15_award,
]


def main():
    print(f"\nSnapAI — Real Scenario Clip Generator")
    print(f"Output: {OUT_DIR}\n")
    print(f"{'Clip':<35} {'Size':>8}  Scenario")
    print("-" * 90)
    for fn in CLIP_FUNCS:
        p    = fn()
        size = p.stat().st_size / 1024
        spec = next(s for s in SPECS if s["name"] == p.name)
        print(f"  ✓ {p.name:<33} {size:6.0f} KB  {spec['scenario'][:52]}")

    spec_path = OUT_DIR / "_real_specs.json"
    spec_path.write_text(json.dumps(SPECS, indent=2))
    print(f"\n  ✓ {'_real_specs.json':<33} (spec sidecar)")
    print(f"\n{'─'*90}")
    print(f"Done. {len(CLIP_FUNCS)} real-scenario clips generated.")
    print(f"\nUpload these to the SnapAI frontend for manual testing.")
    print(f"Each clip has a 'prompt_hint' in the spec — set that as the active prompt")
    print(f"in SnapAI before uploading for best CRITICAL-tier triggering.\n")


if __name__ == "__main__":
    main()
