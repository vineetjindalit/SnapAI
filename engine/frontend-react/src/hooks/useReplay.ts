// frontend-react/src/hooks/useReplay.ts
// DEV TEST TOOL — replay a recorded clip AS the live camera.
//
// It decodes a chosen video file and pushes its frames through the live
// WebSocket (ws.sendFrame) at real-time fps, so the whole LIVE path runs —
// capture reflex, priority tiers, personalization, dedup, the live dashboard —
// fed by a reproducible clip instead of a camera.
//
// ISOLATION: the caller (DevMode) starts a replay in a FRESH session with the
// camera unmounted, so a clip's captures and the real camera can never share a
// session. We read the socket from a ref so frames always go to the CURRENT
// session's ws (a fresh session reconnects the socket).

import { useCallback, useRef, useState } from "react";
import { useSession } from "@/store/session";
import type { SnappyWs } from "@/api/ws";

// BUG FIXED: the backend processes frames on a worker POOL (4 workers), so the
// last handful of frames sent are still being scored when the sender loop
// finishes. The old code called stop() — flipping the UI to "REPLAY FINISHED"
// — the instant the last frame was SENT, not when its result came back. On a
// slower phone (more queued frames at the end) this gap was big enough that
// captures kept appearing (or "capturing…" kept showing) after "FINISHED" had
// already printed. Fix: track frames sent-but-not-yet-resulted and only
// finish once that drains to zero (or a safety timeout, so a dropped message
// can never hang the UI forever).
const DRAIN_TIMEOUT_MS = 6000;
const DRAIN_POLL_MS = 150;
// Phones (esp. iOS Safari) can silently drop a video seek — under memory
// pressure, thermal throttling, or backgrounding — and `onseeked` then never
// fires again. Before this watchdog existed, that meant the send loop just
// stopped forever with the UI stuck on "▶ REPLAYING CLIP AS LIVE" and no
// error, no timeout, nothing: the #1 reported cause of "gets stuck, nothing
// happens" on mobile. One retry, then a visible abort.
const SEEK_STALL_MS = 4000;
// If the clip never loads at all (bad codec, huge 4K file on a phone GPU,
// corrupt upload) `onloadeddata` never fires and the loop never starts.
const LOAD_TIMEOUT_MS = 12000;

export function useReplay(ws: SnappyWs | null, fps = 5) {
  const setReplaying = useSession((s) => s.setReplaying);
  const wsRef = useRef<SnappyWs | null>(ws);
  wsRef.current = ws;                                   // always the latest socket
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const timerRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const urlRef = useRef<string | null>(null);

  // In-flight accounting for the drain (see note above).
  const pendingRef = useRef(0);
  const unsubRef = useRef<(() => void) | null>(null);
  const drainTimerRef = useRef<number | null>(null);
  const [draining, setDraining] = useState(false);
  const [pendingCount, setPendingCount] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Seek-stall + load watchdogs (see constants above).
  const stallTimerRef   = useRef<number | null>(null);
  const stallRetriedRef = useRef(false);
  const loadTimerRef    = useRef<number | null>(null);

  const clearDrainTimer = () => {
    if (drainTimerRef.current != null) { clearInterval(drainTimerRef.current); drainTimerRef.current = null; }
  };
  const clearStallTimer = () => {
    if (stallTimerRef.current != null) { clearTimeout(stallTimerRef.current); stallTimerRef.current = null; }
  };
  const clearLoadTimer = () => {
    if (loadTimerRef.current != null) { clearTimeout(loadTimerRef.current); loadTimerRef.current = null; }
  };

  const stop = useCallback(() => {
    if (timerRef.current != null) { clearInterval(timerRef.current); timerRef.current = null; }
    if (videoRef.current) { videoRef.current.pause(); videoRef.current.remove(); videoRef.current = null; }
    if (urlRef.current) { URL.revokeObjectURL(urlRef.current); urlRef.current = null; }
    unsubRef.current?.(); unsubRef.current = null;
    clearDrainTimer();
    clearStallTimer();
    clearLoadTimer();
    stallRetriedRef.current = false;
    pendingRef.current = 0;
    setPendingCount(0);
    setDraining(false);
    setReplaying(false);
  }, [setReplaying]);

  // Stop AND surface a reason — every failure path below goes through this
  // instead of hanging silently, so the UI can show an error + a way out
  // rather than an indefinite "▶ REPLAYING CLIP AS LIVE".
  const abort = useCallback((message: string) => {
    stop();
    setError(message);
  }, [stop]);

  // Called once all frames have been SENT. Waits for the backend to actually
  // finish scoring them before flipping the UI to "finished".
  const finishReplay = useCallback(() => {
    if (pendingRef.current <= 0) { stop(); return; }
    setDraining(true);
    const deadline = Date.now() + DRAIN_TIMEOUT_MS;
    clearDrainTimer();
    drainTimerRef.current = window.setInterval(() => {
      if (pendingRef.current <= 0 || Date.now() >= deadline) stop();
    }, DRAIN_POLL_MS);
  }, [stop]);

  // The clip's AUDIO track, decoded to 16 kHz mono float32. Sent alongside the
  // frames so the backend HEARS the replay too (song/cheers → capture timing +
  // event evidence; speech → album-time occasion context). Seeking a <video>
  // plays no sound, so we decode the file once and stream slices ourselves.
  const pcmRef = useRef<Float32Array | null>(null);
  const pcmPosRef = useRef(0);

  const f32ToB64 = (f: Float32Array): string => {
    const u8 = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
    let bin = "";
    for (let i = 0; i < u8.length; i += 0x8000)
      bin += String.fromCharCode(...u8.subarray(i, i + 0x8000));
    return btoa(bin);
  };

  const decodeAudio16k = async (file: File): Promise<Float32Array | null> => {
    try {
      const buf = await file.arrayBuffer();
      const probe = new (window.AudioContext || (window as any).webkitAudioContext)();
      const decoded = await probe.decodeAudioData(buf.slice(0));
      probe.close();
      const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * 16000), 16000);
      const src = off.createBufferSource();
      src.buffer = decoded; src.connect(off.destination); src.start();
      const out = await off.startRendering();
      return out.getChannelData(0);
    } catch { return null; }              // silent clip / unsupported codec
  };

  const replay = useCallback((file: File) => {
    stop();
    setError(null);
    pcmRef.current = null; pcmPosRef.current = 0;
    decodeAudio16k(file).then((pcm) => { pcmRef.current = pcm; });
    const url = URL.createObjectURL(file);
    urlRef.current = url;
    const v = document.createElement("video");
    v.preload = "auto";
    v.src = url; v.muted = true; (v as any).playsInline = true;
    // A DETACHED <video> often refuses to play() (stays paused), so the frame
    // loop would send nothing. Attach it hidden to the DOM so muted autoplay
    // actually starts and frames flow. Removed again in stop().
    v.style.cssText = "position:fixed;left:-9999px;top:0;width:2px;height:2px;opacity:0;pointer-events:none;";
    document.body.appendChild(v);
    videoRef.current = v;
    const c = canvasRef.current ?? document.createElement("canvas");
    canvasRef.current = c;

    // If the clip never becomes decodable at all (bad codec, huge 4K file on
    // a phone GPU, a corrupt pick from the share sheet) `onloadeddata` never
    // fires and the whole feature used to just sit there forever with no
    // feedback. Bound it.
    clearLoadTimer();
    loadTimerRef.current = window.setTimeout(() => {
      abort("Couldn't load this clip on this device (unsupported format, or too large to decode). Try “Upload video” instead — it doesn't need real-time playback.");
    }, LOAD_TIMEOUT_MS);

    // Any decode failure the browser can name explicitly.
    v.onerror = () => {
      abort("This clip failed to play on this device (unsupported format/codec). Try “Upload video” instead.");
    };

    v.onloadeddata = () => {
      clearLoadTimer();
      // iOS Safari in particular can silently no-op currentTime seeks on a
      // <video> that has never actually played — seeking alone sometimes
      // isn't enough to prime the decoder. A muted play()+immediate pause()
      // "warms up" playback so the seek loop below reliably produces frames
      // instead of the video just sitting on its first frame forever.
      v.play().then(() => v.pause()).catch(() => {});
      setReplaying(true);
      // Subscribe HERE (not earlier) — by now the fresh session's socket is
      // definitely the one in wsRef.current, so we count results for THIS
      // replay only, never a stale/previous session's.
      pendingRef.current = 0; setPendingCount(0); setDraining(false);
      unsubRef.current?.();
      unsubRef.current = wsRef.current?.on((msg: any) => {
        if (msg?.type === "frame_result") {
          pendingRef.current = Math.max(0, pendingRef.current - 1);
          setPendingCount(pendingRef.current);
        }
      }) ?? null;
      // Cap width so each JPEG stays small — never upscale the source. (Large
      // frames were the failure: they arrive split across TCP reads.)
      // Full HD ingest (matches the live camera): 1080p holds 5 fps real-time;
      // 4K measured at ~1 fps. Never upscale — small sources stay native.
      const w = Math.min(v.videoWidth || 1920, 1920);
      c.width = w; c.height = Math.round(w * ((v.videoHeight / v.videoWidth) || 1.0));
      const ctx = c.getContext("2d");
      // Drive by SEEKING (not play()) so it never depends on autoplay/visibility
      // policy — step through the clip at `fps`, draw+send each frame. Paced with
      // setTimeout so it streams at real time, mirroring a live camera.
      const dur = (isFinite(v.duration) && v.duration > 0) ? v.duration : 20;
      const stepT = 1 / fps;
      let t = 0;
      const advance = () => {
        if (!videoRef.current) return;
        v.currentTime = Math.min(t, dur);
        // Arm the stall watchdog for THIS seek. Phones can drop a seek
        // request under memory/thermal pressure or while backgrounded —
        // `onseeked` then never fires and the loop used to just hang. One
        // retry, then a visible, recoverable error instead of an infinite
        // "REPLAYING CLIP AS LIVE".
        clearStallTimer();
        stallTimerRef.current = window.setTimeout(() => {
          if (!videoRef.current) return;
          if (!stallRetriedRef.current) {
            stallRetriedRef.current = true;
            advance();
          } else {
            abort("Playback stalled partway through the clip — this device couldn't keep up. Try “Upload video” instead (it doesn't need real-time playback).");
          }
        }, SEEK_STALL_MS);
      };
      v.onseeked = () => {
        clearStallTimer();
        stallRetriedRef.current = false;
        if (!videoRef.current || !ctx) return;
        if (v.videoWidth) {
          ctx.drawImage(v, 0, 0, c.width, c.height);
          const b64 = c.toDataURL("image/jpeg", 0.95).split(",")[1];
          const sock = wsRef.current;
          if (b64 && sock) {
            sock.sendFrame(b64);
            pendingRef.current += 1;
            setPendingCount(pendingRef.current);
          }
          // Ship the audio elapsed since the last send (~1s batches — the
          // sound model wants ≥0.5s windows; speech accumulates album-side).
          const pcm = pcmRef.current;
          if (pcm && sock) {
            const end = Math.min(Math.floor(t * 16000), pcm.length);
            if (end - pcmPosRef.current >= 16000) {
              sock.sendAudio(f32ToB64(pcm.slice(pcmPosRef.current, end)));
              pcmPosRef.current = end;
            }
          }
        }
        t += stepT;
        if (t <= dur) timerRef.current = window.setTimeout(advance, Math.max(50, Math.round(1000 / fps)));
        else finishReplay();   // all frames SENT — wait for them to be SCORED before "finished"
      };
      advance();   // kick off the first seek
    };
  }, [fps, stop, abort, finishReplay, setReplaying]);

  return { replay, stop, draining, pendingCount, error };
}
