"""
backend/voice/transcription.py — server-side STT fallback (Whisper).

Primary path is browser-side: Web Speech API runs locally and sends text
transcripts via WebSocket. That's free, low-latency, and works in
Chrome/Edge/Safari. This server-side fallback is for clients that can't
do browser STT (e.g. headless capture, Firefox without dictation).

Uses faster-whisper (CPU-friendly C++ port of Whisper):
    pip install faster-whisper

If not installed, the WhisperTranscriber.available is False and the
voice WS endpoint just rejects audio uploads with a clear error.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from typing import Optional

log = logging.getLogger("snappy.voice")


class WhisperTranscriber:
    _instance: Optional["WhisperTranscriber"] = None
    _lock = threading.Lock()

    def __init__(self, model_size: str = "base"):
        self._model = None
        self._init_error: Optional[str] = None
        self._model_size = model_size
        self._try_load()

    @classmethod
    def get(cls) -> "WhisperTranscriber":
        with cls._lock:
            if cls._instance is None:
                cls._instance = WhisperTranscriber()
            return cls._instance

    def _try_load(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as e:
            self._init_error = (
                f"faster-whisper not installed ({e}). "
                "Install with: pip install faster-whisper"
            )
            log.info("Whisper STT optional and not loaded — browser handles voice")
            return
        try:
            from gpu.device import get_device
            dev = get_device()
            compute = "int8" if dev == "cpu" else "float16"
            log.info(f"Loading faster-whisper {self._model_size} on {dev}/{compute}…")
            t0 = time.time()
            self._model = WhisperModel(self._model_size, device=dev if dev in ("cpu","cuda") else "cpu",
                                       compute_type=compute)
            log.info(f"Whisper ready in {time.time()-t0:.1f}s")
        except Exception as e:
            self._init_error = f"Whisper load failed: {e}"
            log.warning(self._init_error)
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def init_error(self) -> Optional[str]:
        return self._init_error

    def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Accepts WebM/Opus/WAV bytes. Returns transcript or None."""
        if self._model is None:
            return None
        try:
            segments, _info = self._model.transcribe(io.BytesIO(audio_bytes),
                                                     beam_size=1)
            return " ".join(s.text for s in segments).strip()
        except Exception as e:
            log.warning(f"Whisper transcribe failed: {e}")
            return None
