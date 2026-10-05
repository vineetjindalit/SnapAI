"""
backend/models/audio_extract.py — pull a 16 kHz mono waveform out of a video.

Used by the recorded-video path so the audio detector can analyse an uploaded
clip's soundtrack. Uses the ffmpeg binary bundled with imageio-ffmpeg (no
system install needed). Returns None when the clip has no audio stream or
anything goes wrong — the caller then simply runs audio-free.

Live capture doesn't use this: there the mic delivers PCM directly (browser /
mobile), which goes straight into AudioEventDetector.classify().
"""
from __future__ import annotations

import logging
import subprocess
from typing import Optional

import numpy as np

log = logging.getLogger("snappy.audio")

SR = 16000  # target sample rate (AST expects 16 kHz)


def _ffmpeg_exe() -> Optional[str]:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as ex:
        log.debug(f"imageio-ffmpeg unavailable: {ex}")
        return None


def has_audio_stream(video_path: str) -> bool:
    """True if the container has at least one audio stream (cheap probe)."""
    exe = _ffmpeg_exe()
    if not exe:
        return False
    try:
        # ffmpeg prints stream info to stderr; look for "Audio:".
        proc = subprocess.run([exe, "-i", video_path], capture_output=True,
                              timeout=20)
        return b"Audio:" in proc.stderr
    except Exception:
        return False


def extract_audio_16k_mono(video_path: str,
                           max_seconds: float = 0.0) -> Optional[np.ndarray]:
    """Decode the video's audio to a mono float32 array at 16 kHz in [-1, 1].

    Returns None if there is no audio stream or extraction fails. `max_seconds`
    > 0 caps the duration read (0 = whole clip).
    """
    exe = _ffmpeg_exe()
    if not exe:
        return None
    cmd = [exe, "-v", "error", "-i", video_path,
           "-vn",                  # no video
           "-ac", "1",             # mono
           "-ar", str(SR),         # 16 kHz
           "-f", "f32le"]          # raw 32-bit float PCM
    if max_seconds and max_seconds > 0:
        cmd += ["-t", str(max_seconds)]
    cmd += ["pipe:1"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
        if proc.returncode != 0 or not proc.stdout:
            return None
        wav = np.frombuffer(proc.stdout, dtype=np.float32).copy()
        if wav.size < SR // 2:      # < 0.5s of audio → treat as none
            return None
        # Guard against NaNs/clipping noise from odd codecs.
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
        return wav
    except Exception as ex:
        log.debug(f"audio extract failed for {video_path}: {ex}")
        return None
