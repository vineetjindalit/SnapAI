"""
models/clip_engine.py — REAL OpenAI CLIP via HuggingFace transformers.

If `torch` and `transformers` are installed, we lazy-load CLIP ViT-B/32 once
on first use, embed the moment-prompt set, and cache those text embeddings.
Each frame is encoded and we return cosine sim to every prompt.

If those libs are missing, `CLIPEngine.available` is False and the heuristic
VLMDetector still runs — the system never crashes from a missing dep.

Few-shot personalisation:
  * For each session, we maintain a running mean embedding of frames the
    user marked as "kept" (👍). The next frame's score gets a boost
    proportional to its cosine similarity to that "kept centroid".
  * That gives genuine on-device learning per event, with no training loop.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import os as _os
import numpy as np

log = logging.getLogger("snappy.clip")

_CENTROIDS_PATH = Path(__file__).with_name("clip_class_centroids.npz")
# Per-event centroid files live here: event_centroids/<event>.npz. Training one
# event writes ONLY its own file, so tuning birthday can never shift wedding's
# accuracy. The global clip_class_centroids.npz above is the shared "general"
# fallback used for unknown events / free-form prompts.
_EVENT_CENTROIDS_DIR = Path(__file__).with_name("event_centroids")
# Blend weight for trained centroids when scoring frames.
# Lowered from 0.5 → 0.25 so the text prompt stays the primary signal and
# the centroid only nudges it. At 0.5 the new cake_with_candles centroid
# (built from many cake-with-lit-candle domain images) was dominating all
# cake scenes, including genuine cake_cutting clips.
_CENTROID_WEIGHT = float(_os.environ.get("SNAPPY_CENTROID_WEIGHT", "0.25"))

# Classes that are INTERNAL — used during scoring (so the model can say
# "this looks like a non-event") but never displayed to the user as a
# detected moment. If `negative` wins, the system reports `watching`.
HIDDEN_CLASSES = {"negative"}

# Minimum cosine-sim (mapped to [0,1]) to consider a class a real match.
# Below this floor, the CLASS LABEL collapses to "watching" — but unlike
# the previous behavior, we DO NOT zero out the confidence. Other signals
# (quality, NIMA, emotion, face count) can still trigger a capture via
# `general_peak`. The floor only suppresses the *label*, not the *decision*.
#
# 0.42 is empirically a sane default for real-world (not curated) videos.
# Screen recordings and amateur phone footage tend to score in the 0.40-0.55
# range. Setting this too high (0.55+) starves the model on genuine events.
CONFIDENCE_FLOOR = float(_os.environ.get("SNAPPY_CLIP_CONFIDENCE_FLOOR", "0.40"))
# How much the best class must beat the second-best by to count as decisive.
# 0.01 is tight on purpose — sibling classes (cake_cutting vs candle_blowing)
# typically differ by 0.02-0.05 and we don't want every cake scene to flip
# to "watching" because two related classes scored close to each other.
DECISIVENESS_MARGIN = float(_os.environ.get("SNAPPY_CLIP_MARGIN", "0.01"))

# Default prompt list — keep aligned with MOMENT_PROFILES keys in
# clip_moment_detector.py so heuristic + CLIP scores can blend cleanly.
DEFAULT_MOMENT_PROMPTS: Dict[str, str] = {
    # Sibling cake/candle prompts are written to be maximally discriminative:
    # cake_cutting emphasises the KNIFE/slicing, candle_blowing emphasises the
    # LIT flames + blowing — so CLIP separates the two phases of a birthday.
    "cake_cutting":    "a person slicing a cake with a knife at a celebration",
    "candle_blowing":  "a person leaning in and blowing out the candles on a cake",
    "cake_with_candles": "people gathered around a birthday cake with lit candles",
    "cake_feeding":    "a person feeding a bite of cake to another person",
    "clapping_scene":  "a group of people clapping and applauding together",
    # Extended birthday taxonomy — text baseline so trained centroids blend
    # symmetrically (no class is scored by centroid alone). Global vocabulary;
    # only active for an event whose class set includes them.
    "cake":               "a birthday cake by itself on a table",
    "food_table":         "a table spread with party snacks and food platters",
    "gift_box_reveal":     "people opening a huge decorated surprise box at a celebration",
    "cake_person":        "a person posing next to a birthday cake",
    "cake_smashing":      "someone smashing their face into a birthday cake",
    "birthday_gifting":   "a person giving or opening birthday gifts",
    "surprise_celebration": "people surprising someone with raised hands and shock",
    "pre_preparation":    "decorating and setting up for a birthday party",
    "person_arrival":     "a person arriving and being greeted at a party",
    "individual_people":  "a portrait of a single person",
    "smiling_moments":    "a person smiling happily",
    "laughing_moments":   "people laughing out loud",
    "crying_moments":     "a person crying with tears of emotion",
    "hugging_moments":    "two people hugging warmly",
    "dancing_moments":    "people dancing at a party",
    "gazing_moments":     "a person looking directly into the camera",
    "ring_ceremony":   "a wedding ring exchange or engagement proposal",
    "first_dance":     "a couple dancing on a stage with lights",
    "bouquet_toss":    "a bride throwing a flower bouquet into a crowd",
    "group_photo":     "a group of people posing together for a photo",
    "champagne_toast": "people raising glasses for a toast at a celebration",
    "confetti_burst":  "colorful confetti exploding over a crowd",
    "sports_action":   "a player making a dramatic action move at a sporting event",
    "hug_moment":      "two people hugging warmly",
    "general_peak":    "a beautiful candid photograph of an important moment",
    # BACKGROUND CONTRAST — never active, never captured. Zero-shot CLIP is a
    # forced choice: without a "none of the above", a pool table or an empty
    # hallway WINS some birthday class (all prompts flat ~0.65) and gets
    # captured as 'cake'. These give junk scenes something honest to win.
    "bg_room":    "an empty room with plain furniture, nobody celebrating",
    "bg_game":    "a games table or sports equipment, like billiards or a pool table",
    "bg_screen":  "a television, computer or phone screen",
    "bg_street":  "an ordinary street or outdoor scene with no celebration",
    # COOKING/FOOD contrast. Measured 2026-08-07: a cooking clip (wok, chopped
    # vegetables) beat the four prompts above by +0.050..0.079 and was captured
    # as `cake_cutting` on 8/10 frames — a knife-and-food scene genuinely
    # resembles "slicing a cake" more than it resembles a pool table or an
    # empty room. Without a kitchen prompt to win, the forced choice had to
    # pick a birthday class. These give food/prep scenes an honest home.
    "bg_cooking":  "cooking food in a frying pan or wok on a stove in a kitchen",
    "bg_chopping": "chopping raw vegetables on a cutting board, food preparation",
    "bg_meal":     "a plate of ordinary savoury food, a normal meal, not a dessert",
    # Other common off-domain footage a phone camera wanders into.
    "bg_nature":   "landscape, plants or animals with nobody celebrating",
    "bg_vehicle":  "a car, bus or vehicle on a road",
    "bg_document": "a document, book, whiteboard or page of text",
    "bg_shop":     "a shop, office or workplace interior during ordinary business",
}


@dataclass
class CLIPScores:
    available:   bool
    backend:     str                              # "transformers" | "none"
    per_prompt:  Dict[str, float] = field(default_factory=dict)
    best:        Optional[str]   = None
    best_score:  float           = 0.0
    inference_ms: float          = 0.0
    similarity_to_kept_centroid: Optional[float] = None


class CLIPEngine:
    """Singleton lazy-loaded CLIP wrapper."""

    _instance: Optional["CLIPEngine"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._init_error: Optional[str] = None
        self._text_emb: Optional[np.ndarray] = None    # (N, D)
        self._prompts: List[str] = list(DEFAULT_MOMENT_PROMPTS.keys())
        self._prompt_text: List[str] = list(DEFAULT_MOMENT_PROMPTS.values())

        # Per-session "kept frame" centroids — keyed by session id.
        self._kept_centroids: Dict[str, np.ndarray] = {}
        self._kept_counts: Dict[str, int] = {}

        # Trained class centroids loaded from clip_class_centroids.npz
        # (produced by backend.training.finetune_clip).
        self._trained_classes: List[str] = []
        self._trained_centroids: Optional[np.ndarray] = None
        self._trained_meta: Dict = {}
        self._try_load_trained_centroids()
        # Lazy per-event centroid cache: event_key → (classes, centroids|None).
        # None marks "checked, no file" so we don't re-stat every frame.
        self._event_centroids: Dict[str, tuple] = {}

    def _try_load_trained_centroids(self) -> None:
        if not _CENTROIDS_PATH.exists():
            log.info("No trained class centroids found "
                     f"(expected at {_CENTROIDS_PATH}). "
                     "Run: python -m backend.training.finetune_clip")
            return
        try:
            data = np.load(_CENTROIDS_PATH, allow_pickle=True)
            self._trained_classes = list(data["classes"].tolist())
            self._trained_centroids = data["centroids"].astype(np.float32)
            try:
                import json
                self._trained_meta = json.loads(str(data["metrics"]))
            except Exception:
                self._trained_meta = {}
            acc = self._trained_meta.get("val_acc")
            log.info(f"Loaded trained centroids: {len(self._trained_classes)} classes"
                     f"{f' (val_acc={acc:.2f})' if acc else ''}")
        except Exception as e:
            log.warning(f"Failed to load trained centroids: {e}")
            self._trained_centroids = None

    def _centroids_for(self, event: str):
        """Return (classes, centroids) for an event — its OWN trained set if one
        exists, else the shared 'general' default. This is the per-event
        isolation: birthday's file can't influence wedding and vice-versa.
        """
        try:
            from models.event_sequence import resolve_event
            key = resolve_event(event) or (event or "").strip().lower()
        except Exception:
            key = (event or "").strip().lower()
        if not key:
            return self._trained_classes, self._trained_centroids
        if key not in self._event_centroids:
            path = _EVENT_CENTROIDS_DIR / f"{key}.npz"
            loaded = None
            if path.exists():
                try:
                    data = np.load(path, allow_pickle=True)
                    loaded = (list(data["classes"].tolist()),
                              data["centroids"].astype(np.float32))
                    log.info(f"Loaded per-event centroids for '{key}': "
                             f"{len(loaded[0])} classes")
                except Exception as e:
                    log.warning(f"Failed to load event centroids '{key}': {e}")
            self._event_centroids[key] = loaded
        ev = self._event_centroids[key]
        return ev if ev is not None else (self._trained_classes, self._trained_centroids)

    def event_classes(self, event: str) -> List[str]:
        """The classes an event was TRAINED on (empty if it has no own file).

        The pipeline treats these as the event's in-prompt vocabulary, so a
        birthday session knows cake_person / birthday_gifting / etc. are valid
        moments. Returns [] for events with no per-event centroids, so events
        that haven't been trained keep their existing behaviour — strict
        isolation, no cross-event effect.
        """
        try:
            from models.event_sequence import resolve_event
            key = resolve_event(event) or (event or "").strip().lower()
        except Exception:
            key = (event or "").strip().lower()
        if not key:
            return []
        self._centroids_for(key)                     # populate cache
        ev = self._event_centroids.get(key)
        return list(ev[0]) if ev else []

    @classmethod
    def get(cls) -> "CLIPEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = CLIPEngine()
            return cls._instance

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    @property
    def has_trained_centroids(self) -> bool:
        return self._trained_centroids is not None

    @property
    def trained_meta(self) -> Dict:
        return {
            "classes":  self._trained_classes,
            "val_acc":  self._trained_meta.get("val_acc"),
        } if self.has_trained_centroids else {}

    # ── Lazy load on first use ────────────────────────────────────────────────
    def warmup(self) -> bool:
        if self._model is not None:
            return True
        if self._init_error is not None:
            return False
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except Exception as e:
            self._init_error = (
                f"CLIP unavailable ({e}). Install with: "
                "pip install torch transformers pillow"
            )
            log.warning(self._init_error)
            return False

        try:
            from gpu.device import get_device
            self._device = get_device()
            log.info(f"Loading CLIP ViT-B/32 on {self._device} "
                     "(first call may download ~600MB)...")
            t0 = time.time()
            self._model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            if self._device != "cpu":
                try:
                    self._model = self._model.to(self._device)
                except Exception as ex:
                    log.warning(f"Failed to move CLIP to {self._device}, falling back to CPU: {ex}")
                    self._device = "cpu"
            self._model.eval()

            # Pre-compute text embeddings
            with torch.no_grad():
                inputs = self._processor(text=self._prompt_text, return_tensors="pt",
                                         padding=True).to(self._device)
                emb = self._model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                self._text_emb = emb.cpu().numpy().astype(np.float32)
            log.info(f"CLIP ready on {self._device} in {time.time()-t0:.1f}s")
            return True
        except Exception as e:
            self._init_error = f"CLIP load failed: {e}"
            log.exception(self._init_error)
            self._model = None
            return False

    # ── Per-frame inference ──────────────────────────────────────────────────
    def score_frame(self, bgr_frame: np.ndarray, sid: Optional[str] = None,
                    event: str = "") -> CLIPScores:
        if not self.warmup():
            return CLIPScores(available=False, backend="none")

        try:
            import torch
            from PIL import Image
        except Exception as e:
            return CLIPScores(available=False, backend="none")

        try:
            t0 = time.time()
            rgb = bgr_frame[:, :, ::-1]
            pil = Image.fromarray(rgb)
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                img = self._model.get_image_features(**inputs)
                img = img / img.norm(dim=-1, keepdim=True)
                img_np = img.cpu().numpy().astype(np.float32)[0]   # (D,)

            sims = (self._text_emb @ img_np)                       # cosine since both unit
            sims_dict: Dict[str, float] = {
                k: float((s + 1.0) / 2.0)                          # map [-1,1] → [0,1]
                for k, s in zip(self._prompts, sims.tolist())
            }

            # Blend trained class centroids if present — this is what makes
            # CLIP "fine-tuned" on your event style. Up-weights classes
            # whose centroid your bootstrapped photos cluster near.
            # Uses the ACTIVE EVENT's own centroids when available (per-event
            # isolation), else the shared general set.
            ev_classes, ev_centroids = self._centroids_for(event)
            if ev_centroids is not None:
                tsims = (ev_centroids @ img_np)                    # (T,)
                tsims = (tsims + 1.0) / 2.0
                for cls, ts in zip(ev_classes, tsims.tolist()):
                    if cls in sims_dict:
                        sims_dict[cls] = float(
                            (1 - _CENTROID_WEIGHT) * sims_dict[cls]
                            + _CENTROID_WEIGHT * float(ts)
                        )
                    else:
                        sims_dict[cls] = float(ts)

            # Exclude internal classes (e.g. `negative`) from the user-visible
            # winner. They still participate in scoring — if `negative` is
            # the true winner, the visible result becomes `watching`.
            visible = {k: v for k, v in sims_dict.items() if k not in HIDDEN_CLASSES}
            sorted_classes = sorted(visible.items(), key=lambda kv: -kv[1])
            top_class, top_score = sorted_classes[0]
            second_score = sorted_classes[1][1] if len(sorted_classes) > 1 else 0.0

            # Did `negative` actually win? If so, the user-visible label
            # should reflect that the frame is uninteresting.
            negative_score = sims_dict.get("negative", 0.0)
            negative_won = negative_score > top_score

            # Apply confidence floor + decisiveness margin
            decisive = (top_score >= CONFIDENCE_FLOOR
                        and (top_score - second_score) >= DECISIVENESS_MARGIN)

            if negative_won or not decisive:
                best = "watching"           # neutral state — no event found
            else:
                best = top_class

            kept_sim: Optional[float] = None
            if sid and sid in self._kept_centroids:
                c = self._kept_centroids[sid]
                kept_sim = float(((img_np @ c) + 1.0) / 2.0)

            # best_score reflects the *visible* winner — if best is
            # "watching", report the highest legitimate score so the UI can
            # show "watching (top match: cake_cutting 48%)" if it wants.
            best_score_val = sims_dict.get(best, top_score)
            scores = CLIPScores(
                available=True, backend="transformers",
                per_prompt=visible, best=best, best_score=best_score_val,
                inference_ms=(time.time() - t0) * 1000.0,
                similarity_to_kept_centroid=kept_sim,
            )
            # Stash raw embedding for downstream consumers (auto-classifier etc.)
            scores._image_emb = img_np.tolist()  # type: ignore[attr-defined]
            return scores
        except Exception as e:
            log.exception("CLIP inference failed")
            return CLIPScores(available=False, backend="none")

    # ── Online learning hooks ─────────────────────────────────────────────────
    def record_kept_frame(self, sid: str, bgr_frame: np.ndarray) -> None:
        """User signalled this capture was good. Add to per-session centroid."""
        if not self.warmup():
            return
        try:
            import torch
            from PIL import Image
            rgb = bgr_frame[:, :, ::-1]
            pil = Image.fromarray(rgb)
            inputs = self._processor(images=pil, return_tensors="pt").to(self._device)
            with torch.no_grad():
                img = self._model.get_image_features(**inputs)
                img = img / img.norm(dim=-1, keepdim=True)
                v = img.cpu().numpy().astype(np.float32)[0]
            n = self._kept_counts.get(sid, 0)
            if sid in self._kept_centroids:
                c = self._kept_centroids[sid]
                c = (c * n + v) / (n + 1)
                c = c / (np.linalg.norm(c) + 1e-8)
            else:
                c = v
            self._kept_centroids[sid] = c
            self._kept_counts[sid] = n + 1
            log.info(f"CLIP kept-centroid updated for {sid} (n={n+1})")
        except Exception as e:
            log.warning(f"record_kept_frame failed: {e}")

    def reset_session(self, sid: str) -> None:
        self._kept_centroids.pop(sid, None)
        self._kept_counts.pop(sid, None)

    # ── Dynamic prompt vocabulary (mid-event prompt change) ────────────
    def add_prompt(self, class_name: str, prompt_text: str) -> bool:
        """Add a brand-new moment class at runtime. Re-embeds the prompt
        and appends to the active prompt set. Returns True on success.

        Example: user says "also catch saree pulling" → backend extracts
        a class id + prompt and calls this. CLIP can now score frames
        against the new class on the very next frame.
        """
        if not self.warmup():
            return False
        if class_name in self._prompts:
            return True   # already known
        try:
            import torch
            inputs = self._processor(text=[prompt_text], return_tensors="pt",
                                     padding=True).to(self._device)
            with torch.no_grad():
                emb = self._model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            new_row = emb.cpu().numpy().astype(np.float32)
            if self._text_emb is None:
                self._text_emb = new_row
            else:
                self._text_emb = np.vstack([self._text_emb, new_row])
            self._prompts.append(class_name)
            self._prompt_text.append(prompt_text)
            log.info(f"CLIP vocabulary +1: '{class_name}' → '{prompt_text}'")
            return True
        except Exception as e:
            log.warning(f"add_prompt failed: {e}")
            return False

    def update_prompt(self, class_name: str, prompt_text: str) -> bool:
        """Replace an existing class's text prompt (re-embed it)."""
        if not self.warmup():
            return False
        if class_name not in self._prompts:
            return self.add_prompt(class_name, prompt_text)
        try:
            import torch
            inputs = self._processor(text=[prompt_text], return_tensors="pt",
                                     padding=True).to(self._device)
            with torch.no_grad():
                emb = self._model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
            idx = self._prompts.index(class_name)
            self._text_emb[idx] = emb.cpu().numpy().astype(np.float32)[0]
            self._prompt_text[idx] = prompt_text
            log.info(f"CLIP prompt updated: '{class_name}' → '{prompt_text}'")
            return True
        except Exception as e:
            log.warning(f"update_prompt failed: {e}")
            return False

    def vocabulary(self) -> List[str]:
        """Currently-known moment class names."""
        return list(self._prompts)
