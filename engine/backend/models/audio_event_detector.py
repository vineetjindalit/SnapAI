"""
backend/models/audio_event_detector.py — what the EARS hear, for capture timing.

A photographer hears a moment before they see it: the "Happy Birthday" song
ends, everyone cheers, glasses clink. Audio is the strongest *timing* signal we
have — omnidirectional (it catches off-camera events) and predictive (the song
precedes the blow). This module turns a mic / video-audio stream into a small
set of event probabilities the capture engine can act on.

Model: AST (Audio Spectrogram Transformer), fine-tuned on AudioSet (527 sound
classes), via HuggingFace transformers — the SAME stack CLIP already uses, so
no new heavy framework. 16 kHz mono in, per-class sigmoid probabilities out.

We collapse the 527 AudioSet labels into a compact event vocabulary that maps
to photographic moments:

    applause / cheering  → a peak JUST happened (blow, cut, reveal, speech end)
    singing              → a ceremony is in progress (pre-arm the camera)
    laughter             → an emotional/candid moment
    music / speech       → ambient context

Design mirrors CLIPEngine: a lazy singleton with `.available` + `.warmup()`.
If torchaudio/transformers are missing or the model fails to load, `available`
is False and `classify()` returns {} — the pipeline then behaves exactly as it
does today (audio is purely additive, never required).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, List, Optional

import numpy as np

log = logging.getLogger("snappy.audio")

# Default model — AudioSet-trained AST. ~350MB, cached after first load.
_AST_MODEL_ID = os.environ.get(
    "SNAPPY_AUDIO_MODEL", "MIT/ast-finetuned-audioset-10-10-0.4593")
AUDIO_SR = 16000   # AST expects 16 kHz mono

# Compact event vocabulary → AudioSet label substrings that map into it.
# Matching is case-insensitive substring against model.config.id2label.
_EVENT_KEYWORDS: Dict[str, List[str]] = {
    "applause":  ["applause", "clapping"],
    "cheering":  ["cheering", "crowd", "chatter, crowd", "shout", "yell", "children shouting"],
    "singing":   ["singing", "choir", "chant", "child singing", "humming"],
    "laughter":  ["laughter", "giggle", "chuckle", "snicker", "baby laughter"],
    "music":     ["music"],
    "speech":    ["speech", "conversation", "narration", "babbling"],
}


class AudioEventDetector:
    """Lazy AST-based audio tagger. Singleton via .get()."""

    _instance: Optional["AudioEventDetector"] = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._extractor = None
        self._device = "cpu"
        self._lock = threading.Lock()
        self._warmed = False
        self.available = False
        self.init_error: Optional[str] = None
        # bucket → list[int] of AudioSet class indices (built at warmup)
        self._bucket_idx: Dict[str, List[int]] = {}

    @classmethod
    def get(cls) -> "AudioEventDetector":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Model load ─────────────────────────────────────────────────────────
    def warmup(self) -> bool:
        """Load AST once. Returns True if the model is ready, False otherwise.

        Never raises — a failure just leaves `available` False so the rest of
        the system runs audio-free.
        """
        if self._warmed:
            return self.available
        with self._lock:
            if self._warmed:
                return self.available
            self._warmed = True
            if os.environ.get("SNAPPY_AUDIO", "1") == "0":
                self.init_error = "disabled via SNAPPY_AUDIO=0"
                log.info("Audio detector disabled (SNAPPY_AUDIO=0)")
                return False
            try:
                import torch
                from transformers import (ASTForAudioClassification,
                                          ASTFeatureExtractor)
                t0 = time.time()
                self._device = ("mps" if torch.backends.mps.is_available()
                                else "cuda" if torch.cuda.is_available() else "cpu")
                self._extractor = ASTFeatureExtractor.from_pretrained(_AST_MODEL_ID)
                self._model = (ASTForAudioClassification
                               .from_pretrained(_AST_MODEL_ID).eval().to(self._device))
                self._build_bucket_index()
                self.available = True
                log.info(f"Audio (AST) ready on {self._device} in "
                         f"{time.time()-t0:.1f}s — {self._model.config.num_labels} "
                         f"AudioSet classes")
            except Exception as ex:
                self.init_error = str(ex)
                self.available = False
                log.warning(f"Audio detector unavailable: {ex}")
            return self.available

    def _build_bucket_index(self) -> None:
        """Map each compact event bucket → the AudioSet class indices in it."""
        id2label = self._model.config.id2label
        self._bucket_idx = {b: [] for b in _EVENT_KEYWORDS}
        for idx, label in id2label.items():
            low = str(label).lower()
            for bucket, kws in _EVENT_KEYWORDS.items():
                if any(kw in low for kw in kws):
                    self._bucket_idx[bucket].append(int(idx))

    # ── Inference ────────────────────────────────────────────────────────────
    def classify(self, wav_16k: np.ndarray) -> Dict[str, float]:
        """Classify a mono 16 kHz waveform → {event_bucket: probability in 0-1}.

        Returns {} when the model isn't available so callers can no-op cleanly.
        A bucket's probability is the MAX over its member AudioSet classes.
        """
        if not self.warmup() or wav_16k is None or len(wav_16k) < AUDIO_SR // 2:
            return {}
        try:
            import torch
            wav = np.asarray(wav_16k, dtype=np.float32)
            inp = self._extractor(wav, sampling_rate=AUDIO_SR, return_tensors="pt")
            inp = {k: v.to(self._device) for k, v in inp.items()}
            with torch.no_grad():
                logits = self._model(**inp).logits
            probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
            out: Dict[str, float] = {}
            for bucket, idxs in self._bucket_idx.items():
                out[bucket] = float(max((probs[i] for i in idxs), default=0.0))
            return out
        except Exception as ex:
            log.debug(f"audio classify failed: {ex}")
            return {}

    def classify_timeline(self, wav_16k: np.ndarray,
                          window_s: float = 1.0,
                          hop_s: float = 0.5) -> List[Dict]:
        """Slide a window over a full waveform → a time-stamped event timeline.

        Returns a list of {"t": centre_seconds, "events": {bucket: prob}}.
        Used by the recorded-video path to pre-compute audio cues for the whole
        clip, which the per-frame pipeline then looks up by timestamp.
        """
        if not self.warmup() or wav_16k is None or len(wav_16k) < AUDIO_SR // 2:
            return []
        win = int(window_s * AUDIO_SR)
        hop = max(1, int(hop_s * AUDIO_SR))
        timeline: List[Dict] = []
        for start in range(0, max(1, len(wav_16k) - win + 1), hop):
            chunk = wav_16k[start:start + win]
            ev = self.classify(chunk)
            if ev:
                timeline.append({"t": (start + win / 2) / AUDIO_SR, "events": ev})
        return timeline


# ── Capture-cue mapping ────────────────────────────────────────────────────
# Translate raw event probabilities into the signals the capture engine wants:
# a priority/timing BOOST and an optional moment hint.

def celebration_intensity(events: Dict[str, float]) -> float:
    """0-1 'something is peaking' score — applause/cheering spike when a moment
    lands (the blow, the cut, the reveal). This is the core timing signal."""
    if not events:
        return 0.0
    return float(max(events.get("applause", 0.0), events.get("cheering", 0.0)))


def audio_capture_boost(events: Dict[str, float]) -> float:
    """Additive boost to the capture score from audio (0 when no audio).

    Tuned to be a NUDGE, not an override: a clear applause/cheer adds ~0.12,
    singing (ceremony in progress) ~0.05, laughter ~0.06. Capped so audio can
    push a borderline frame over the line but never force a bad capture.
    """
    if not events:
        return 0.0
    boost = 0.0
    boost += 0.12 * celebration_intensity(events)
    boost += 0.10 * float(events.get("singing", 0.0))    # the song is a strong cue
    boost += 0.06 * float(events.get("laughter", 0.0))
    return float(min(0.22, boost))


# ── "Capture the important voice" + "feel the birthday" ────────────────────────
_TIER_CRITICAL, _TIER_HIGH, _TIER_ELEVATED = "critical", "high", "elevated"


def audio_priority_tier(events: Dict[str, float]) -> str:
    """Priority tier implied by SOUND ALONE (""=nothing special). Lets the
    capture engine NEVER MISS a moment that the voice/crowd marks:

      • sharp applause/cheer = the REACTION to a peak just landing (candles
        blown, cake cut, gift revealed) → CRITICAL.
      • sustained singing = the birthday ceremony itself (the 'Happy Birthday'
        song) → HIGH: capture the people singing around the cake.
      • a softer cheer/sing/laugh → ELEVATED nudge.
    """
    if not events:
        return ""
    peak = max(float(events.get("applause", 0.0)), float(events.get("cheering", 0.0)))
    sing = float(events.get("singing", 0.0))
    laugh = float(events.get("laughter", 0.0))
    # A cheer/applause PEAK is a brief transient (the reaction) → bypass-cooldown
    # tiers are safe and "never miss" matters most.
    if peak >= 0.55:
        return _TIER_CRITICAL
    if peak >= 0.35:
        return _TIER_HIGH
    # Singing is SUSTAINED (the whole song); ELEVATED respects the cooldown so we
    # get a few good shots of people singing around the cake, not a 75-shot flood.
    if sing >= 0.45 or laugh >= 0.45:
        return _TIER_ELEVATED
    return ""


def birthday_audio_awareness(events: Dict[str, float]) -> float:
    """0-1: how strongly the SOUND says a celebration is underway — the audio
    half of "the model can feel a birthday is happening". The song dominates;
    cheering/applause/laughter add."""
    if not events:
        return 0.0
    sing = float(events.get("singing", 0.0))
    peak = max(float(events.get("cheering", 0.0)), float(events.get("applause", 0.0)))
    laugh = float(events.get("laughter", 0.0))
    return float(min(1.0, 0.80 * sing + 0.45 * peak + 0.25 * laugh))
