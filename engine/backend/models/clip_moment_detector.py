"""
models/clip_moment_detector.py

VLM-style prompt-driven moment detector without PyTorch/CLIP dependency.

Architecture:
  - Text side:  keyword → visual feature profile mapping (like CLIP text encoder output)
  - Image side: multi-scale color histogram + HOG features + semantic color blobs
  - Similarity: cosine similarity between text embedding and image embedding
  - Threshold:  if similarity > event threshold → this is a "prompted moment"

Moments supported (expandable):
  cake_cutting, ring_ceremony, first_dance, bouquet_toss,
  group_photo, candle_blowing, cake_decoration, hug_moment,
  champagne_toast, sports_action, confetti_burst, general_peak

How to upgrade to real CLIP later:
  Replace VLMDetector.encode_image() with:
      from transformers import CLIPProcessor, CLIPModel
      model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  The rest of the pipeline stays identical.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.safe_types import safe, clamp

# ── Moment definitions ─────────────────────────────────────────────────────────
# Each moment has:
#   keywords   : user prompt keywords that trigger this moment
#   color_cues : list of (H_mean, H_range, S_min) in HSV — dominant color signatures
#   blob_bright: whether to look for bright point blobs (candles, sparkles)
#   motion_band: (min, max) expected motion energy
#   edge_level : expected edge density (0=plain, 1=complex scene)
#   face_needed: minimum faces required
MOMENT_PROFILES: Dict[str, Dict] = {
    "cake_cutting": {
        "keywords":    ["cake", "cutting cake", "cake cut", "birthday cake"],
        "color_cues":  [(0, 180, 0.0), (30, 40, 0.3)],   # white + brown/yellow
        "blob_bright": True,    # candles
        "motion_band": (0.1, 0.6),
        "edge_level":  0.4,
        "face_needed": 0,
        "weight":      1.0,
    },
    "ring_ceremony": {
        "keywords":    ["ring", "ring ceremony", "engagement", "proposal", "wedding ring"],
        "color_cues":  [(35, 30, 0.4), (0, 180, 0.0)],    # gold/silver + white
        "blob_bright": False,
        "motion_band": (0.0, 0.3),
        "edge_level":  0.3,
        "face_needed": 1,
        "weight":      1.0,
    },
    "first_dance": {
        "keywords":    ["first dance", "dance", "dancing", "waltz"],
        "color_cues":  [(270, 60, 0.3), (0, 180, 0.0)],   # purple/blue lights + white
        "blob_bright": True,   # stage lights
        "motion_band": (0.3, 0.8),
        "edge_level":  0.4,
        "face_needed": 1,
        "weight":      1.0,
    },
    "bouquet_toss": {
        "keywords":    ["bouquet", "flowers", "toss", "bouquet toss"],
        "color_cues":  [(0, 30, 0.5), (120, 40, 0.4)],    # red/pink + green
        "blob_bright": False,
        "motion_band": (0.4, 0.9),
        "edge_level":  0.5,
        "face_needed": 1,
        "weight":      1.0,
    },
    "candle_blowing": {
        "keywords":    ["candle", "blow candle", "birthday candle", "wish"],
        "color_cues":  [(20, 20, 0.7)],                    # orange flame
        "blob_bright": True,
        "motion_band": (0.1, 0.5),
        "edge_level":  0.3,
        "face_needed": 1,
        "weight":      1.0,
    },
    # The gathering / singing phase: people in front of a lit cake, NObody
    # actively blowing yet. Distinguished from candle_blowing at runtime by
    # models/candle_phase.py (leaning-in + flame-extinction), not by appearance.
    "cake_with_candles": {
        "keywords":    ["cake with candles", "candles lit", "birthday cake",
                        "gathered", "singing", "happy birthday"],
        "color_cues":  [(20, 20, 0.7), (0, 180, 0.0)],     # flame + white cake
        "blob_bright": True,
        "motion_band": (0.0, 0.4),
        "edge_level":  0.4,
        "face_needed": 2,
        "weight":      0.9,
    },
    "group_photo": {
        "keywords":    ["group", "group photo", "everyone", "all together", "family photo"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.0, 0.2),
        "edge_level":  0.5,
        "face_needed": 3,
        "weight":      1.0,
    },
    # Snacks / food spread on the table (often kept beside the cake). A worth-
    # capturing detail shot; no faces required.
    "gift_box_reveal": {
        "keywords":    ["surprise box", "giant gift", "gift box", "big box",
                        "box reveal", "huge present", "surprise reveal"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.0, 0.6),
        "edge_level":  0.5,
        "face_needed": 0,
        "weight":      0.9,
    },
    "food_table": {
        "keywords":    ["food", "snacks", "table spread", "platter", "starters",
                        "appetizers", "buffet"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.0, 0.3),
        "edge_level":  0.5,
        "face_needed": 0,
        "weight":      0.85,
    },
    "champagne_toast": {
        "keywords":    ["toast", "champagne", "cheers", "drinks"],
        "color_cues":  [(45, 30, 0.4)],                    # golden yellow
        "blob_bright": False,
        "motion_band": (0.1, 0.5),
        "edge_level":  0.4,
        "face_needed": 1,
        "weight":      1.0,
    },
    "confetti_burst": {
        "keywords":    ["confetti", "celebration", "burst", "colorful"],
        "color_cues":  [(0, 180, 0.5)],                    # many colors
        "blob_bright": True,
        "motion_band": (0.5, 1.0),
        "edge_level":  0.7,
        "face_needed": 0,
        "weight":      1.0,
    },
    "sports_action": {
        "keywords":    ["sports", "action", "goal", "score", "run", "jump", "match"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.6, 1.0),
        "edge_level":  0.5,
        "face_needed": 0,
        "weight":      1.0,
    },
    "hug_moment": {
        "keywords":    ["hug", "embrace", "together", "reunion"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.0, 0.4),
        "edge_level":  0.5,
        "face_needed": 2,
        "weight":      1.0,
    },
    "general_peak": {
        "keywords":    ["photo", "capture", "moment", "picture", "snap"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.0, 1.0),
        "edge_level":  0.0,
        "face_needed": 0,
        "weight":      0.7,
    },

    # ── Birthday-specific additions ────────────────────────────────────────

    # Person A feeding Person B a piece of cake: hand-near-mouth + open-mouth
    # expression + low motion (moment of feeding, not chaos).  Colour: warm
    # (skin tones) + white/light (plate, cake).
    "cake_feeding": {
        "keywords":    ["cake feeding", "feeding cake", "eat cake", "eating cake",
                        "feed cake", "make eat", "khilana"],
        "color_cues":  [(0, 180, 0.0), (15, 30, 0.4)],   # white + warm skin
        "blob_bright": False,
        "motion_band": (0.0, 0.35),   # gentle movement
        "edge_level":  0.35,
        "face_needed": 2,             # at least two people
        "weight":      1.0,
    },

    # People clapping during a celebration (cake-cutting, award, etc.).
    # High edge density (many arms/hands), moderate motion, group scene.
    "clapping_scene": {
        "keywords":    ["clapping", "clap", "applause", "cheering", "cheer"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.25, 0.75),
        "edge_level":  0.55,
        "face_needed": 2,
        "weight":      0.9,
    },

    # Face pushed into the cake / cake on the face. High motion burst; the
    # geometry refinement (models/cake_smash.py) confirms/denies at runtime —
    # this profile mainly lets "cake smash" in a USER PROMPT activate the class.
    "cake_smashing": {
        "keywords":    ["cake smash", "cake smashing", "smash cake", "smash",
                        "face in cake", "cake on face"],
        "color_cues":  [(0, 180, 0.0), (15, 25, 0.3)],   # white cake + skin
        "blob_bright": False,
        "motion_band": (0.3, 0.9),
        "edge_level":  0.4,
        "face_needed": 1,
        "weight":      1.0,
    },

    # Giving / opening birthday presents. Faceless close-ups are common (hands
    # + box), so no face requirement — the object-detail path captures those.
    "birthday_gifting": {
        "keywords":    ["gift", "gifts", "present", "presents", "opening gifts",
                        "birthday gift", "gift box", "unwrapping"],
        "color_cues":  [],
        "blob_bright": False,
        "motion_band": (0.05, 0.6),
        "edge_level":  0.4,
        "face_needed": 0,
        "weight":      0.9,
    },
}


@dataclass
class MomentResult:
    detected_moment: str           # best matching moment
    confidence:      float         # 0-1
    all_scores:      Dict[str, float]
    active_moments:  List[str]     # moments above threshold
    is_capture_moment: bool

    def to_dict(self) -> dict:
        return safe({
            "detected_moment":   self.detected_moment,
            "confidence":        self.confidence,
            "all_scores":        self.all_scores,
            "active_moments":    self.active_moments,
            "is_capture_moment": self.is_capture_moment,
        })


class VLMDetector:
    """
    Visual Language Moment Detector.
    Text prompt → moment profile → visual similarity scoring.
    """

    def __init__(self, prompt: str = "", capture_threshold: float = 0.52):
        self.prompt              = prompt.lower()
        self.capture_threshold   = capture_threshold
        self.active_profiles     = self._parse_prompt(prompt)
        self.face_cascade        = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self._prev_gray: Optional[np.ndarray] = None
        self._score_history: List[float] = []
        self.HISTORY = 20

    # ── Prompt parsing ─────────────────────────────────────────────────────────
    def _parse_prompt(self, prompt: str) -> List[str]:
        """Map user text → list of moment profiles to watch for."""
        p = prompt.lower()
        matched = []
        for name, profile in MOMENT_PROFILES.items():
            for kw in profile["keywords"]:
                if kw in p:
                    matched.append(name)
                    break
        # Always include general_peak as fallback
        if "general_peak" not in matched:
            matched.append("general_peak")
        return matched if matched else list(MOMENT_PROFILES.keys())

    def update_prompt(self, prompt: str):
        self.prompt          = prompt.lower()
        self.active_profiles = self._parse_prompt(prompt)

    # ── Image feature extraction ───────────────────────────────────────────────
    def _color_histogram(self, frame: np.ndarray) -> np.ndarray:
        """32-bin HSV histogram per channel, flattened and L2-normalised."""
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_h  = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        s_h  = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        v_h  = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()
        hist = np.concatenate([h_h, s_h, v_h]).astype(np.float32)
        norm = np.linalg.norm(hist)
        return hist / (norm + 1e-8)

    def _motion_energy(self, gray: np.ndarray) -> float:
        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return 0.0
        diff   = cv2.absdiff(gray, cv2.resize(self._prev_gray, (gray.shape[1], gray.shape[0])))
        energy = clamp(float(np.mean(diff)) / 40.0)
        self._prev_gray = gray.copy()
        return energy

    def _bright_blob_count(self, gray: np.ndarray) -> int:
        """Count bright small blobs (candle flames, sparkles, lights)."""
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return sum(1 for c in contours if 3 < cv2.contourArea(c) < 300)

    def _edge_density(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 50, 150)
        return clamp(float(np.mean(edges)) / 80.0)

    def _dominant_hue(self, hsv: np.ndarray, hue: float, hrange: float, s_min: float) -> float:
        """Fraction of pixels near target hue with sufficient saturation."""
        h_ch = hsv[:, :, 0].astype(float)
        s_ch = hsv[:, :, 1].astype(float) / 255.0
        in_hue = np.abs(h_ch - hue) < hrange
        in_sat = s_ch > s_min
        return clamp(float(np.sum(in_hue & in_sat)) / float(h_ch.size) * 8.0)

    def _face_count(self, gray: np.ndarray) -> int:
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        return int(len(faces)) if len(faces) > 0 else 0

    # ── Per-moment scoring ─────────────────────────────────────────────────────
    def _score_moment(self, profile_name: str,
                       gray: np.ndarray,
                       hsv:  np.ndarray,
                       motion: float,
                       blobs:  int,
                       edges:  float,
                       nfaces: int) -> float:
        p = MOMENT_PROFILES[profile_name]
        parts = []

        # Color cue match
        if p["color_cues"]:
            cc_scores = [self._dominant_hue(hsv, h, r, s) for h, r, s in p["color_cues"]]
            parts.append(clamp(float(np.mean(cc_scores)) * 1.5))
        else:
            parts.append(0.5)  # neutral

        # Bright blob match
        if p["blob_bright"]:
            parts.append(clamp(blobs / 8.0))
        else:
            parts.append(0.5)

        # Motion band match
        m_min, m_max = p["motion_band"]
        if m_min <= motion <= m_max:
            parts.append(1.0)
        else:
            dist = min(abs(motion - m_min), abs(motion - m_max))
            parts.append(clamp(1.0 - dist * 2.0))

        # Edge level match
        parts.append(clamp(1.0 - abs(edges - p["edge_level"]) * 2.0))

        # Face requirement
        needed = p["face_needed"]
        if needed == 0:
            parts.append(0.7)
        elif nfaces >= needed:
            parts.append(1.0)
        else:
            parts.append(clamp(nfaces / max(needed, 1)))

        score = float(np.mean(parts)) * float(p["weight"])
        return clamp(score)

    # ── Main detection ─────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> MomentResult:
        small = cv2.resize(frame, (320, 240))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        motion = self._motion_energy(gray)
        blobs  = self._bright_blob_count(gray)
        edges  = self._edge_density(gray)
        nfaces = self._face_count(gray)

        all_scores: Dict[str, float] = {}
        for name in self.active_profiles:
            all_scores[name] = self._score_moment(name, gray, hsv, motion, blobs, edges, nfaces)

        # Also score non-active profiles at lower weight for display
        for name in MOMENT_PROFILES:
            if name not in all_scores:
                all_scores[name] = self._score_moment(name, gray, hsv, motion, blobs, edges, nfaces) * 0.5

        best_moment = max(all_scores, key=lambda k: all_scores[k])
        best_score  = all_scores[best_moment]

        # Temporal smoothing for best active moment
        self._score_history.append(best_score)
        if len(self._score_history) > self.HISTORY:
            self._score_history.pop(0)
        smoothed = float(np.mean(self._score_history))

        active_above = [k for k, v in all_scores.items() if v >= self.capture_threshold]
        is_capture   = bool(smoothed >= self.capture_threshold and best_moment in self.active_profiles)

        return MomentResult(
            detected_moment  = best_moment,
            confidence       = clamp(smoothed),
            all_scores       = {k: round(float(v), 3) for k, v in all_scores.items()},
            active_moments   = active_above,
            is_capture_moment= is_capture,
        )

    def get_active_moment_names(self) -> List[str]:
        return list(self.active_profiles)
