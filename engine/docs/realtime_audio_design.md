# Real-time audio for capture timing — design

Audio is SnapAI's strongest *timing* signal: a cheer/applause spike marks the
instant a moment lands (the blow, the cut, the reveal), and the "Happy Birthday"
song *predicts* the blow seconds early. This doc defines the one **contract**
both the backend (now) and the mobile app (later) implement, so the audio brain
can move on-device without changing anything downstream.

---

## The contract (shared by every implementation)

Any audio source — ffmpeg track, browser mic, or on-device mic — must produce:

1. **Event probabilities** — a dict over a fixed vocabulary, each in `[0,1]`:

   ```
   {applause, cheering, singing, laughter, music, speech}
   ```

   (Defined once in `backend/models/audio_event_detector.py :: _EVENT_KEYWORDS`,
   collapsed from AudioSet's 521 classes.)

2. **A capture boost** — fed additively into the capture decision, identical
   everywhere:

   ```python
   celebration_intensity(events) = max(applause, cheering)          # the "peak"
   audio_capture_boost(events)   = 0.12*celebration_intensity
                                 + 0.05*singing + 0.06*laughter      # capped 0.20
   ```

As long as a source emits the same `{bucket: prob}` dict, the rest of the
pipeline (priority tiers, album curation) is unchanged. Backend ↔ mobile is a
drop-in swap.

---

## Implemented now — browser mic → backend (testable today)

```
browser mic ─(WebAudio, 16kHz mono, ~1s)→ WS {"type":"audio","pcm":b64}
   → server._handle_ws_audio → AudioEventDetector.classify (AST/AudioSet)
   → session.live_audio_events  → process_frame audio_capture_boost
```

- Frontend: `hooks/useMicAudio.ts` captures + downsamples + base64-streams 1s
  float32 chunks; `useLiveStream` sends them via `SnappyWs.sendAudio`.
- Backend: `AudioEventDetector` (HuggingFace AST, reuses the existing torch
  stack); live events expire after 1.5s (`Session.current_audio_events`).
- Same boost as the recorded-video path → live == upload behaviour.
- Verified: real WS round-trip (audio chunk + frame) processes cleanly.

**Why backend-first:** zero new mobile deps, works in the current browser test
rig, and lets us validate the *signal's value* on real events before committing
the phone work.

---

## Mobile-native — on-device (production target)

```
phone mic ─→ on-device VAD ─→ TFLite YAMNet (3.7 MB, ~10ms/inf)
   → 521 AudioSet scores → SAME 6 buckets → SAME audio_capture_boost
   → on-device capture engine (no audio ever leaves the device)
```

| Concern | Plan |
|---------|------|
| **Model** | YAMNet TFLite (or MobileNet-audio). AudioSet-trained → identical label space to the backend AST, so the bucket map is reused verbatim. |
| **Latency** | ~10 ms/inference on a modern phone; run every ~0.5 s on a 1 s window. |
| **Battery** | Mic + a 3.7 MB model is cheap vs the camera/vision stack already running. |
| **Privacy** | Audio is classified on-device and **discarded** — only the 6 probabilities (never raw audio) touch the capture logic. Add a clear mic toggle + indicator. |
| **Parity** | Port `_EVENT_KEYWORDS` + `audio_capture_boost` to the mobile codebase (≈30 lines). A shared JSON of the bucket→keyword map keeps them in lock-step. |

### What the mobile app must build
1. Mic capture + 16 kHz mono framing (platform AudioRecord / AVAudioEngine).
2. YAMNet TFLite inference on a sliding 1 s window @ ~2 Hz.
3. AudioSet-score → 6-bucket reducer (port of `_EVENT_KEYWORDS`).
4. `audio_capture_boost` (port) → feed the on-device capture decision exactly
   where `process_frame` adds `audio_boost` today.

### Migration
The backend path stays as the **fallback / web** implementation. When the
mobile model ships, the phone simply stops sending `{"type":"audio"}` chunks and
applies the boost locally — no backend or pipeline change required, because both
honour the same contract above.

---

## Files
- `backend/models/audio_event_detector.py` — AST classifier + bucket map + boost (the contract).
- `backend/models/audio_extract.py` — ffmpeg audio for uploaded video.
- `backend/api/server.py :: _handle_ws_audio` — live mic ingress.
- `backend/api/session.py :: current_audio_events` — live-vs-recorded selection + TTL.
- `frontend-react/src/hooks/useMicAudio.ts` — browser mic capture/stream.
