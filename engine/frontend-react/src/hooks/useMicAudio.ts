// frontend-react/src/hooks/useMicAudio.ts
// Capture mic audio → mono 16 kHz float32 → ~1s base64 chunks via onChunk.
// Used by the live stream to feed audio event cues (cheering/applause/singing)
// to the backend in real time. Mirrors what ffmpeg extracts for uploaded video.

import { useCallback, useEffect, useRef, useState } from "react";

const TARGET_SR = 16000;   // AST expects 16 kHz
const CHUNK_SEC = 1.0;     // one classification window per second

function downsample(buf: Float32Array, inSR: number, outSR: number): Float32Array {
  if (outSR >= inSR) return buf;
  const ratio = inSR / outSR;
  const out = new Float32Array(Math.floor(buf.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = buf[Math.floor(i * ratio)] || 0;
  return out;
}

function f32ToB64(f: Float32Array): string {
  const bytes = new Uint8Array(f.buffer, f.byteOffset, f.byteLength);
  let bin = "";
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) {
    bin += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CH)) as any);
  }
  return btoa(bin);
}

export function useMicAudio(
  onChunk: (b64: string) => void,
  opts: { enabled?: boolean } = {},
) {
  const { enabled = true } = opts;
  const onChunkRef = useRef(onChunk);
  useEffect(() => { onChunkRef.current = onChunk; }, [onChunk]);

  const ctxRef    = useRef<AudioContext | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const accRef    = useRef<number[]>([]);
  const [active, setActive] = useState(false);
  const [error,  setError]  = useState<string | null>(null);

  const start = useCallback(async () => {
    if (ctxRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const Ctx = window.AudioContext || (window as any).webkitAudioContext;
      const ctx = new Ctx();
      ctxRef.current = ctx;
      const src  = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      const inSR = ctx.sampleRate;
      const need = Math.floor(TARGET_SR * CHUNK_SEC);
      proc.onaudioprocess = (e) => {
        const ds = downsample(e.inputBuffer.getChannelData(0), inSR, TARGET_SR);
        const acc = accRef.current;
        for (let i = 0; i < ds.length; i++) acc.push(ds[i]);
        if (acc.length >= need) {
          const chunk = Float32Array.from(acc.slice(0, need));
          accRef.current = acc.slice(need);
          try { onChunkRef.current(f32ToB64(chunk)); } catch { /* noop */ }
        }
      };
      // Route through a muted gain node so onaudioprocess fires without
      // echoing the mic back to the speakers.
      const mute = ctx.createGain(); mute.gain.value = 0;
      src.connect(proc); proc.connect(mute); mute.connect(ctx.destination);
      setActive(true); setError(null);
    } catch (e: any) {
      setError(e?.message ?? String(e));   // e.g. mic permission denied
    }
  }, []);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    ctxRef.current?.close().catch(() => {});
    ctxRef.current = null; streamRef.current = null; accRef.current = [];
    setActive(false);
  }, []);

  useEffect(() => {
    if (enabled) start();
    return () => stop();
  }, [enabled, start, stop]);

  return { active, error, start, stop };
}
