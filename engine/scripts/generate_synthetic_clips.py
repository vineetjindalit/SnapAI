#!/usr/bin/env python3
"""
generate_synthetic_clips.py — Build a deterministic test corpus.

Why this exists
---------------
We need a small, fully-reproducible set of test clips that exercise
specific cases of the SnapAI pipeline: face presence/absence, motion,
lighting, group gaze, peak signals, etc. Real-world clips are great
for full validation (see download_test_clips.py) but they're not
reproducible across machines and they carry licensing concerns.

These synthetic clips are produced from pure OpenCV draw calls — no
external assets, no network, no copyright issues. Each clip is 5–8
seconds long, 640x480, 24 fps, mp4v-encoded.

Output
------
tests/datasets/clips/synthetic/
    01_static_face.mp4           — one face, low motion, neutral light
    02_group_gaze.mp4            — 3 faces all "looking at camera"
    03_low_light_motion.mp4      — dim scene + waving motion
    04_cake_cutting_proxy.mp4    — bright candle-like blobs + motion
    05_high_motion_no_face.mp4   — scrolling pattern, no faces
    06_dark_scene.mp4            — very dark, intermittent flicker
    07_color_burst.mp4           — confetti-like color variance
    08_smile_to_laugh.mp4        — single face rising "happiness"
    09_two_people_hug.mp4        — two faces converging then close
    10_blank_scene.mp4           — empty, no signals (negative control)

Each clip's expected behavior is documented in CLIP_SPECS below and
becomes part of the benchmark assertions.
"""
from __future__ import annotations
import os, math, random
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "tests" / "datasets" / "clips" / "synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT, FPS = 640, 480, 24

# Each entry is the spec the benchmark harness will check against.
CLIP_SPECS = []


def writer(name: str):
    """Open an mp4 VideoWriter. Fourcc 'mp4v' is the most portable inside OpenCV."""
    path = OUT_DIR / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, FPS, (WIDTH, HEIGHT))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open writer for {path}")
    return vw, path


def draw_face(img, cx, cy, size, looking_at_cam=True, expression="neutral",
              skin=(125, 165, 200)):     # BGR! warm peach skin tone
    """Draw a stylised face proxy.

    The pipeline's heuristic face detectors (Haar / MediaPipe) won't
    necessarily fire on these, but YOLOv8-face will pick up well-formed
    ellipsoid skin-tones if it's installed. The intent is to make
    clips that exercise our motion + color + composition signals,
    not to fool a face detector. For real face benchmarks use the
    Pexels downloader.
    """
    # Head
    cv2.ellipse(img, (cx, cy), (size, int(size * 1.25)), 0, 0, 360, skin, -1)
    # Hair
    cv2.ellipse(img, (cx, cy - size//2), (size, size//2), 0, 180, 360, (40, 30, 25), -1)
    # Eyes — bigger if looking at camera, narrower otherwise
    eye_w = int(size * 0.18) if looking_at_cam else int(size * 0.10)
    eye_h = int(size * 0.10) if looking_at_cam else int(size * 0.04)
    cv2.ellipse(img, (cx - size//3, cy - size//4), (eye_w, eye_h), 0, 0, 360, (255,255,255), -1)
    cv2.ellipse(img, (cx + size//3, cy - size//4), (eye_w, eye_h), 0, 0, 360, (255,255,255), -1)
    if looking_at_cam:
        cv2.circle(img, (cx - size//3, cy - size//4), max(2, eye_w//3), (30, 30, 30), -1)
        cv2.circle(img, (cx + size//3, cy - size//4), max(2, eye_w//3), (30, 30, 30), -1)
    # Mouth
    if expression == "smile":
        cv2.ellipse(img, (cx, cy + size//3), (size//3, size//6), 0, 0, 180, (50, 50, 50), 2)
    elif expression == "laugh":
        cv2.ellipse(img, (cx, cy + size//3), (size//3, size//4), 0, 0, 180, (30, 30, 30), -1)
        cv2.ellipse(img, (cx, cy + size//3), (int(size//3 * 0.8), int(size//4 * 0.6)), 0, 0, 180, (200, 80, 80), -1)
    else:
        cv2.line(img, (cx - size//4, cy + size//3), (cx + size//4, cy + size//3), (50,50,50), 2)


# ── Clip 01 — static face, neutral lighting ───────────────────────────────
def clip_01_static_face():
    vw, path = writer("01_static_face.mp4")
    for t in range(FPS * 6):  # 6 sec
        img = np.full((HEIGHT, WIDTH, 3), 210, dtype=np.uint8)  # warm grey bg
        # Very small bob to simulate real human stillness — never zero motion
        bob = int(3 * math.sin(t / 6))
        draw_face(img, WIDTH//2, HEIGHT//2 + bob, 80, looking_at_cam=True, expression="smile")
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "01_static_face.mp4",
    "scenario": "Single static face, gentle motion, good lighting",
    "expected_priority": ["high", "elevated", "normal"],
    "expected_captures_min": 1,
    "expected_captures_max": 20,
    "notes": "Should capture some frames via gaze + face signals.",
})


# ── Clip 02 — group gaze (3 faces, all looking at camera) ─────────────────
def clip_02_group_gaze():
    vw, path = writer("02_group_gaze.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
        # Gentle pulse — simulates breathing in a group photo
        pulse = int(2 * math.sin(t / 8))
        draw_face(img, WIDTH//2 - 160, HEIGHT//2 + pulse, 70, True, "smile")
        draw_face(img, WIDTH//2,        HEIGHT//2 - pulse, 80, True, "smile")
        draw_face(img, WIDTH//2 + 160, HEIGHT//2 + pulse, 70, True, "laugh")
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "02_group_gaze.mp4",
    "scenario": "Three faces all looking at camera — should fire HIGH priority "
                "(or CRITICAL when run with a matching prompt)",
    # critical is permitted because we benchmark this with prompt='group photo',
    # which makes the prompt-matched class fire critical instead of high.
    "expected_priority": ["critical", "high"],
    "expected_captures_min": 1,
    "expected_captures_max": 30,
    "notes": "HIGH priority requires gaze_ratio >= 0.65 and face_count >= 1. "
             "Priority floor (~0.4s) keeps capture count reasonable.",
})


# ── Clip 03 — low light + waving motion ───────────────────────────────────
def clip_03_low_light_motion():
    vw, path = writer("03_low_light_motion.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 40, dtype=np.uint8)  # very dim
        # Hand-waving motion: a bright streak moving left-to-right
        cx = int(60 + (t * 8) % (WIDTH - 120))
        cv2.rectangle(img, (cx, 220), (cx + 60, 280), (90, 90, 110), -1)
        # Faint face in low light (BGR-correct dim skin)
        draw_face(img, WIDTH//2, HEIGHT//2 + 60, 70, looking_at_cam=False, expression="neutral",
                  skin=(50, 60, 80))
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "03_low_light_motion.mp4",
    "scenario": "Dim scene with horizontal motion (someone waving in low light)",
    # 'high' is permitted because if the face proxy briefly catches the dim
    # face, gaze-proxy registers and bumps tier. The 0.4s priority floor
    # keeps total captures in a reasonable range.
    "expected_priority": ["normal", "elevated", "high"],
    "expected_captures_min": 0,
    "expected_captures_max": 25,
    "notes": "Quality scores will be low. Capture count depends heavily on "
             "whether the dim face is detected. In production the NIMA + "
             "blur thresholds would tighten this further; this benchmark "
             "uses a lightweight proxy so the count runs a bit higher.",
})


# ── Clip 04 — cake-cutting proxy (bright blobs + brown/white + motion) ────
def clip_04_cake_cutting():
    vw, path = writer("04_cake_cutting_proxy.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 180, dtype=np.uint8)
        # Cake = brown/white blob in the center
        cv2.ellipse(img, (WIDTH//2, HEIGHT//2 + 80), (120, 60), 0, 0, 360, (240, 235, 220), -1)
        cv2.ellipse(img, (WIDTH//2, HEIGHT//2 + 80), (120, 60), 0, 0, 360, (130, 90, 60), 4)
        # Candles = bright blobs on top, flickering
        for cx in [WIDTH//2 - 60, WIDTH//2 - 20, WIDTH//2 + 20, WIDTH//2 + 60]:
            cy = HEIGHT//2 + 30
            flicker = random.randint(-3, 3)
            cv2.circle(img, (cx, cy), 4, (200, 200, 240), -1)               # wick
            cv2.ellipse(img, (cx, cy - 12 + flicker), (8, 16), 0, 0, 360,    # flame
                        (60, 180, 255), -1)
            cv2.ellipse(img, (cx, cy - 12 + flicker), (4, 10), 0, 0, 360,
                        (200, 240, 255), -1)
        # Two faces leaning in
        lean = int(5 * math.sin(t / 12))
        draw_face(img, WIDTH//2 - 140, HEIGHT//2 - 40 + lean, 60, True, "smile")
        draw_face(img, WIDTH//2 + 140, HEIGHT//2 - 40 - lean, 60, True, "smile")
        # Hand with a knife sweeping down at second 3
        if t > FPS * 3:
            knife_y = int(HEIGHT//2 + 30 + (t - FPS*3) * 4)
            cv2.rectangle(img, (WIDTH//2 - 5, knife_y - 40), (WIDTH//2 + 5, knife_y), (220, 220, 230), -1)
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "04_cake_cutting_proxy.mp4",
    "scenario": "Cake-cutting analogue: brown/white blob + bright candle "
                "blobs + faces leaning in + downward knife motion. Tests "
                "the visual heuristic + CRITICAL priority when prompt is "
                "'cake cutting'.",
    "expected_priority": ["critical", "high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 25,
    "notes": "Benchmark should be run with prompt='cake cutting' to test "
             "CRITICAL tier triggering.",
})


# ── Clip 05 — high motion, no faces (sports proxy) ────────────────────────
def clip_05_high_motion_no_face():
    vw, path = writer("05_high_motion_no_face.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
        # Multiple moving rectangles
        for i in range(5):
            x = int((t * 12 + i * 130) % (WIDTH + 80) - 40)
            y = 80 + i * 60
            color = (50 + i*40, 100 + i*20, 200 - i*30)
            cv2.rectangle(img, (x, y), (x + 60, y + 30), color, -1)
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "05_high_motion_no_face.mp4",
    "scenario": "High motion, no faces (sports-action proxy)",
    "expected_priority": ["normal", "elevated"],
    "expected_captures_min": 0,
    "expected_captures_max": 15,
    "notes": "Predictor 'peak' may trigger ELEVATED. No HIGH (no faces).",
})


# ── Clip 06 — dark scene with intermittent flicker ────────────────────────
def clip_06_dark_scene():
    vw, path = writer("06_dark_scene.mp4")
    for t in range(FPS * 6):
        # Most frames very dark
        base = 12
        # Flicker every ~30 frames (1.25 sec)
        if t % 30 == 0:
            base = 180
        img = np.full((HEIGHT, WIDTH, 3), base, dtype=np.uint8)
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "06_dark_scene.mp4",
    "scenario": "Very dark, occasional flicker — tests low-light reject pipeline",
    "expected_priority": ["normal"],
    "expected_captures_min": 0,
    "expected_captures_max": 5,
    "notes": "Should produce few or zero captures. Validates quality discard.",
})


# ── Clip 07 — confetti-like color burst ───────────────────────────────────
def clip_07_color_burst():
    vw, path = writer("07_color_burst.mp4")
    palette = [
        (60, 80, 230), (90, 200, 230), (60, 230, 100),
        (230, 200, 60), (230, 100, 200), (200, 60, 230),
    ]
    particles = [(random.randint(0, WIDTH), random.randint(0, HEIGHT//2),
                  random.choice(palette), random.uniform(2, 6)) for _ in range(60)]
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 240, dtype=np.uint8)
        for i, (x, y, c, v) in enumerate(particles):
            ny = (y + int(v * t)) % HEIGHT
            cv2.circle(img, (x, ny), 4, c, -1)
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "07_color_burst.mp4",
    "scenario": "Confetti-like falling color particles",
    "expected_priority": ["normal", "elevated"],
    "expected_captures_min": 0,
    "expected_captures_max": 15,
    "notes": "High color variance — confetti_burst heuristic may match.",
})


# ── Clip 08 — single face, smile → laugh (rising emotion) ─────────────────
def clip_08_smile_to_laugh():
    vw, path = writer("08_smile_to_laugh.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 215, dtype=np.uint8)
        expr = "smile" if t < FPS * 3 else "laugh"
        draw_face(img, WIDTH//2, HEIGHT//2, 90, looking_at_cam=True, expression=expr)
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "08_smile_to_laugh.mp4",
    "scenario": "Single face transitioning from smile to open laugh",
    "expected_priority": ["high", "elevated"],
    "expected_captures_min": 1,
    "expected_captures_max": 25,
    "notes": "Looking at camera = HIGH priority; emotion rises mid-clip.",
})


# ── Clip 09 — two people converging into hug ──────────────────────────────
def clip_09_two_people_hug():
    vw, path = writer("09_two_people_hug.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 220, dtype=np.uint8)
        progress = min(1.0, t / (FPS * 4))
        left_x  = int(WIDTH//2 - 200 + 120 * progress)
        right_x = int(WIDTH//2 + 200 - 120 * progress)
        draw_face(img, left_x,  HEIGHT//2, 75, looking_at_cam=False, expression="smile")
        draw_face(img, right_x, HEIGHT//2, 75, looking_at_cam=False, expression="smile")
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "09_two_people_hug.mp4",
    "scenario": "Two faces converging into a hug — emotional moment",
    # critical permitted when run with prompt='hug moment' (default benchmark
    # prompt mapping) since prompt match + face presence triggers CRITICAL.
    "expected_priority": ["critical", "normal", "elevated"],
    "expected_captures_min": 0,
    "expected_captures_max": 30,
    "notes": "With a 'hug moment' prompt this fires CRITICAL throttled to "
             "the 0.4s priority floor. Without a prompt it stays NORMAL.",
})


# ── Clip 10 — blank negative control ──────────────────────────────────────
def clip_10_blank_scene():
    vw, path = writer("10_blank_scene.mp4")
    for t in range(FPS * 6):
        img = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        vw.write(img)
    vw.release()
    return path

CLIP_SPECS.append({
    "name": "10_blank_scene.mp4",
    "scenario": "Empty grey frame, no signals (negative control)",
    "expected_priority": ["normal"],
    "expected_captures_min": 0,
    "expected_captures_max": 0,
    "notes": "MUST produce zero captures. Anything else is a precision bug.",
})


CLIP_FUNCS = [
    clip_01_static_face, clip_02_group_gaze, clip_03_low_light_motion,
    clip_04_cake_cutting, clip_05_high_motion_no_face, clip_06_dark_scene,
    clip_07_color_burst, clip_08_smile_to_laugh, clip_09_two_people_hug,
    clip_10_blank_scene,
]


def main():
    print(f"Writing synthetic clips to: {OUT_DIR}")
    for fn in CLIP_FUNCS:
        p = fn()
        size_kb = p.stat().st_size / 1024
        print(f"  ✓ {p.name:35s} ({size_kb:6.0f} KB)")

    # Write spec sidecar so the benchmark harness has authoritative
    # expected-behavior rules.
    import json
    spec_path = OUT_DIR / "_specs.json"
    spec_path.write_text(json.dumps(CLIP_SPECS, indent=2))
    print(f"  ✓ {spec_path.name} (spec sidecar)")
    print(f"\nDone. {len(CLIP_FUNCS)} clips + 1 spec file created.")


if __name__ == "__main__":
    main()
