"""
models/speech_transcriber.py — LISTEN to what people SAY (album-time only).

The audio event model (AST) hears SOUNDS — applause, cheering, singing — but
words carry the context: "happy birthday to you" proves a birthday; "happy
anniversary" proves the cake is for something else; names get heard. Real-time
transcription would cost the live loop, so this runs ONCE, at album build.

Contract (same shape as the other optional models):
  • ON by default; disable with SNAPPY_SPEECH=0.
    Model: SNAPPY_SPEECH_MODEL (default openai/whisper-base, ~290 MB,
    downloaded once on first use; multilingual — handles Hindi + English).
  • Lazy singleton; degrades to .ok=False and returns None — the album simply
    builds without speech context. Never raises into the pipeline.
"""
from __future__ import annotations
import os
import logging

log = logging.getLogger("snappy.speech")


class SpeechTranscriber:
    _inst = None

    def __init__(self):
        self.ok = False
        self._pipe = None
        if os.environ.get("SNAPPY_SPEECH", "1") != "1":
            log.info("speech transcription OFF (SNAPPY_SPEECH=0)")
            return
        try:
            import torch
            from transformers import pipeline
            name = os.environ.get("SNAPPY_SPEECH_MODEL", "openai/whisper-base")
            log.info(f"speech transcriber: loading {name} (one-time download on first run)…")
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self._pipe = pipeline("automatic-speech-recognition", model=name,
                                  device=dev, chunk_length_s=30)
            self.ok = True
            log.info(f"speech transcriber ready ({name} on {dev})")
        except Exception as e:
            self.ok = False
            log.info(f"speech transcriber unavailable ({e}) — album builds without speech")

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def transcribe(self, wav_16k) -> "str | None":
        """16 kHz mono float32 → text (or None). Capped at ~5 min for speed."""
        if not self.ok or wav_16k is None:
            return None
        try:
            import numpy as np
            wav = np.asarray(wav_16k, dtype=np.float32).reshape(-1)
            if wav.size < 16000:                       # <1s — nothing to hear
                return None
            wav = wav[: 16000 * 300]
            out = self._pipe({"array": wav, "sampling_rate": 16000})
            text = (out.get("text") or "").strip()
            return text or None
        except Exception as e:
            log.debug(f"transcription failed: {e}")
            return None
