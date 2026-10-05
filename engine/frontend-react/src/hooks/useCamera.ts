// frontend-react/src/hooks/useCamera.ts
// Camera capture + JPEG-frame stream → callback. ~5 fps default.
//
// Robust against:
//   - React 18 StrictMode (mount → unmount → mount triggers play() AbortError)
//   - Mobile: front/back switch, tolerant constraints, and `ready` set the
//     MOMENT the stream is live (not after play()) so granting permission
//     always clears the "waiting…" spinner
//   - getUserMedia rejection (no permission) — exposed as `error` state
//   - Stale callbacks (onFrame ref-stored, not in deps array)

import { useCallback, useEffect, useRef, useState } from "react";

export type Facing = "environment" | "user";  // back / front

export interface UseCameraOptions {
  fps?: number;
  width?: number;
  height?: number;
  /** Initial camera. Defaults to the BACK camera — right for filming an event. */
  preferBackCamera?: boolean;
  onFrame?: (b64: string) => void;
  onError?: (err: Error) => void;
}

export interface UseCameraResult {
  videoRef:  React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  ready: boolean;
  error: string | null;
  facing: Facing;
  hasMultipleCameras: boolean;
  start: () => Promise<void>;
  stop:  () => void;
  switchCamera: () => Promise<void>;
}

export function useCamera({
  fps = 5, width = 1920, height = 1080,
  preferBackCamera = true,   // events → back camera by default
  onFrame, onError,
}: UseCameraOptions = {}): UseCameraResult {
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<number | null>(null);
  const mountedRef = useRef<boolean>(true);
  const startingRef = useRef<boolean>(false);
  const facingRef = useRef<Facing>(preferBackCamera ? "environment" : "user");

  const onFrameRef = useRef(onFrame);
  const onErrorRef = useRef(onError);
  useEffect(() => { onFrameRef.current = onFrame; }, [onFrame]);
  useEffect(() => { onErrorRef.current = onError; }, [onError]);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [facing, setFacing] = useState<Facing>(facingRef.current);
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);

  // Probe for a second camera so the flip button only shows when it helps.
  useEffect(() => {
    navigator.mediaDevices?.enumerateDevices?.()
      .then((ds) => setHasMultipleCameras(
        ds.filter((d) => d.kind === "videoinput").length > 1))
      .catch(() => { /* labels hidden until permission — leave flip available */ });
  }, [ready]);

  const stopStream = useCallback(() => {
    if (intervalRef.current != null) { clearInterval(intervalRef.current); intervalRef.current = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const stop = useCallback(() => { stopStream(); setReady(false); }, [stopStream]);

  // Try the requested facing, then progressively looser constraints so a phone
  // that dislikes an exact mode/resolution still opens SOME camera.
  const getStream = useCallback(async (facingMode: Facing): Promise<MediaStream> => {
    const attempts: MediaStreamConstraints[] = [
      { audio: false, video: { facingMode: { ideal: facingMode }, width: { ideal: width }, height: { ideal: height } } },
      { audio: false, video: { facingMode: { ideal: facingMode } } },
      { audio: false, video: true },
    ];
    let lastErr: any;
    for (const c of attempts) {
      try { return await navigator.mediaDevices.getUserMedia(c); }
      catch (e) { lastErr = e; }
    }
    throw lastErr;
  }, [width, height]);

  const startLoop = useCallback(() => {
    const v = videoRef.current;
    const c = canvasRef.current ?? document.createElement("canvas");
    const ctx = c.getContext("2d");
    const tickMs = Math.max(50, Math.round(1000 / fps));
    if (intervalRef.current != null) clearInterval(intervalRef.current);
    intervalRef.current = window.setInterval(() => {
      if (!ctx || !v || !v.videoWidth) return;
      try {
        if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
          c.width = v.videoWidth; c.height = v.videoHeight;
        }
        ctx.drawImage(v, 0, 0, c.width, c.height);
        const dataUrl = c.toDataURL("image/jpeg", 0.95);
        onFrameRef.current?.(dataUrl.slice(dataUrl.indexOf(",") + 1));
      } catch (e) { console.warn("frame capture failed", e); }
    }, tickMs);
  }, [fps]);

  const start = useCallback(async () => {
    if (startingRef.current) return;
    startingRef.current = true;
    setError(null);

    let stream: MediaStream;
    try {
      stream = await getStream(facingRef.current);
    } catch (e: any) {
      setError(e?.message ?? String(e));
      startingRef.current = false;
      onErrorRef.current?.(e);
      return;
    }
    if (!mountedRef.current) { stream.getTracks().forEach((t) => t.stop()); startingRef.current = false; return; }

    streamRef.current = stream;
    const v = videoRef.current;
    if (v) {
      v.srcObject = stream;
      v.muted = true;         // autoplay policy
      v.playsInline = true;   // iOS/Safari
      v.play().catch(() => { /* AbortError/NotAllowed are fine; frames still flow */ });
    }
    // Permission is granted and the stream is LIVE → clear the spinner NOW,
    // regardless of play()'s outcome. (The old code set this only at the very
    // end, past mobile early-returns, so it could spin forever after granting.)
    setReady(true);
    startingRef.current = false;
    startLoop();
  }, [getStream, startLoop]);

  const switchCamera = useCallback(async () => {
    facingRef.current = facingRef.current === "environment" ? "user" : "environment";
    setFacing(facingRef.current);
    stopStream();
    startingRef.current = false;
    await new Promise((r) => setTimeout(r, 150));  // let old tracks release
    await start();
  }, [stopStream, start]);

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; stop(); };
  }, [stop]);

  return { videoRef, canvasRef, ready, error, facing, hasMultipleCameras, start, stop, switchCamera };
}
