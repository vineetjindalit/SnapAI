import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { CameraOff, Loader2, SwitchCamera } from "lucide-react";
import type { useCamera } from "@/hooks/useCamera";
import { useReducedMotion } from "@/lib/motionUtils";

interface Props {
  camera: ReturnType<typeof useCamera>;
  faceBoxes?: Array<{ x: number; y: number; w: number; h: number }>;
  flash?: boolean;
  className?: string;
  /**
   * Viewfinder overlay (corner brackets + sweep). OFF by default so DevMode,
   * which shares this component, renders exactly as it did before.
   */
  viewfinder?: boolean;
  /**
   * How close the pipeline REALLY is to firing a capture, 0..1. Supplied by
   * the caller from actual frame_result values (engine.final_score /
   * engine.threshold where the ensemble runs, moment.confidence otherwise).
   * `undefined` means "no meaningful signal for this event type" — the
   * overlay then sits at its resting state instead of inventing a ramp.
   */
  confidence?: number;
  /**
   * Changes exactly once per REAL capture (pass the captured frame's id).
   * The shutter flash is keyed off this value changing — never off a timer —
   * so the flash cannot fire without the backend having actually captured,
   * and fires within one animation frame of the frame_result arriving.
   */
  captureKey?: number | string | null;
}

export function CameraView({ camera, faceBoxes = [], flash = false, className = "",
                             viewfinder = false, confidence, captureKey = null }: Props) {
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const [autoStarted, setAutoStarted] = useState(false);
  const reduced = useReducedMotion();

  // ── Shutter flash, driven by a real capture ────────────────────────────
  // Keyed on captureKey (the captured frame's id): a NEW id means the backend
  // genuinely captured, so this fires once per capture and self-clears. The
  // previous approach in UserMode set state during render and could re-fire
  // in a loop while the same captured frame remained the latest one.
  const [shutter, setShutter] = useState(false);
  const lastCaptureKey = useRef(captureKey);
  // useLayoutEffect, not useEffect: this must be committed in the SAME paint
  // as the frame_result that reported the capture. useEffect runs after
  // paint, which would put the flash one frame behind the event it is
  // supposed to be synced to.
  useLayoutEffect(() => {
    if (captureKey === null || captureKey === undefined) return;
    if (captureKey === lastCaptureKey.current) return;
    lastCaptureKey.current = captureKey;
    setShutter(true);
    const t = setTimeout(() => setShutter(false), reduced ? 90 : 420);
    return () => clearTimeout(t);
  }, [captureKey, reduced]);

  // Resting → armed. Clamped, and only ever as high as the real signal says.
  const conf = typeof confidence === "number" && Number.isFinite(confidence)
    ? Math.max(0, Math.min(1, confidence))
    : 0;
  const armed = conf >= 0.75;   // visibly "about to shoot"

  useEffect(() => {
    if (!autoStarted) {
      camera.start();
      setAutoStarted(true);
    }
  }, [autoStarted, camera]);

  // Draw face boxes
  useEffect(() => {
    const v = camera.videoRef.current;
    const c = overlayRef.current;
    if (!v || !c) return;
    c.width  = v.clientWidth  * window.devicePixelRatio;
    c.height = v.clientHeight * window.devicePixelRatio;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, c.width, c.height);
    if (!faceBoxes.length || v.videoWidth === 0) return;
    // The video renders with object-fit:cover, which scales the feed
    // UNIFORMLY (one scale factor, not independent x/y stretch) to fill the
    // box and crops whichever axis overflows, centered. Treating it as a
    // simple per-axis stretch (old: sx = canvasWidth/videoWidth, sy =
    // canvasHeight/videoHeight, drawn from (0,0)) ignores that crop offset
    // entirely — whenever the camera's aspect ratio doesn't exactly match
    // the container's, every box lands off by the cropped amount (measured:
    // a face box rendered well above the actual face). Uniform scale + a
    // centered offset matches what's actually on screen.
    const scale = Math.max(c.width / v.videoWidth, c.height / v.videoHeight);
    const offsetX = (c.width  - v.videoWidth  * scale) / 2;
    const offsetY = (c.height - v.videoHeight * scale) / 2;
    ctx.strokeStyle = "rgba(99,102,241,0.85)";
    ctx.lineWidth = 2 * window.devicePixelRatio;
    for (const b of faceBoxes) {
      ctx.strokeRect(offsetX + b.x * scale, offsetY + b.y * scale,
                      b.w * scale, b.h * scale);
    }
  }, [faceBoxes, camera.videoRef]);

  return (
    <div className={"relative w-full overflow-hidden " + className}>
      <video ref={camera.videoRef}
             autoPlay playsInline muted
             className="w-full h-full object-cover bg-black" />
      <canvas ref={overlayRef}
              className="absolute inset-0 w-full h-full pointer-events-none" />

      {/* ── Viewfinder ──────────────────────────────────────────────────
          Corner brackets that tighten and brighten as the pipeline actually
          approaches its capture threshold. Every number below is a function
          of `conf`, which the caller derives from real frame_result values —
          with no signal (conf = 0) this sits at a dim resting state and does
          not move, so it can never suggest activity that isn't happening.
          Raw scores stay in DevMode; this is ambient, unitless feedback. */}
      {viewfinder && camera.ready && (
        <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
          {([
            ["top-0 left-0",     "border-t-2 border-l-2 rounded-tl-xl"],
            ["top-0 right-0",    "border-t-2 border-r-2 rounded-tr-xl"],
            ["bottom-0 left-0",  "border-b-2 border-l-2 rounded-bl-xl"],
            ["bottom-0 right-0", "border-b-2 border-r-2 rounded-br-xl"],
          ] as const).map(([pos, edges]) => (
            <motion.span
              key={pos}
              className={`absolute ${pos} ${edges}`}
              animate={{
                // Brackets close in toward the frame as confidence rises.
                width:   28 + 26 * (1 - conf),
                height:  28 + 26 * (1 - conf),
                margin:  10 + 10 * (1 - conf),
                opacity: 0.3 + 0.7 * conf,
                borderColor: armed ? "rgb(52,211,153)" : "rgba(255,255,255,0.9)",
              }}
              transition={reduced
                ? { duration: 0 }
                : { type: "spring", stiffness: 200, damping: 26 }}
              style={
                // The breathing pulse only runs once genuinely near threshold.
                armed && !reduced
                  ? { animation: "viewfinder-breathe 1.1s ease-in-out infinite" }
                  : undefined
              }
            />
          ))}

          {/* A single sweep line, shown only when actually close to firing. */}
          <AnimatePresence>
            {armed && !reduced && (
              <motion.span
                className="absolute inset-x-6 h-px bg-gradient-to-r from-transparent
                           via-emerald-300/80 to-transparent"
                initial={{ top: "12%", opacity: 0 }}
                animate={{ top: ["12%", "88%"], opacity: [0, 1, 0] }}
                exit={{ opacity: 0 }}
                transition={{ duration: 1.4, repeat: Infinity, ease: "easeInOut" }}
              />
            )}
          </AnimatePresence>
        </div>
      )}

      {/* Legacy prop-driven flash (DevMode still passes `flash`). */}
      {flash && (
        <div className="absolute inset-0 bg-white pointer-events-none
                        animate-capture-flash" />
      )}

      {/* Shutter flash — fires from a real capture event (captureKey), with a
          quick aperture-style contraction. Reduced motion gets a plain, brief
          opacity blink instead of the scale move. */}
      <AnimatePresence>
        {shutter && (
          <motion.div
            key="shutter"
            className="absolute inset-0 bg-white pointer-events-none"
            initial={{ opacity: reduced ? 0.5 : 0.85 }}
            animate={{ opacity: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduced ? 0.09 : 0.4, ease: "easeOut" }}
          />
        )}
      </AnimatePresence>
      <AnimatePresence>
        {shutter && !reduced && (
          <motion.div
            key="shutter-ring"
            className="absolute inset-0 pointer-events-none border-[3px]
                       border-white/80 rounded-3xl"
            initial={{ opacity: 0.9, scale: 1 }}
            animate={{ opacity: 0, scale: 0.94 }}
            transition={{ duration: 0.4, ease: "easeOut" }}
          />
        )}
      </AnimatePresence>

      {/* Front / back camera switch — shown once the camera is live and the
          device actually has more than one camera. */}
      {camera.ready && camera.hasMultipleCameras && (
        <button
          onClick={() => camera.switchCamera()}
          title="Switch camera"
          className="absolute top-3 right-3 z-10 flex items-center gap-2
                     px-4 py-2.5 rounded-full bg-black/55 text-white text-sm
                     backdrop-blur active:scale-95 transition min-h-[44px]">
          <SwitchCamera className="w-5 h-5" />
          {camera.facing === "environment" ? "Back" : "Front"}
        </button>
      )}

      {!camera.ready && !camera.error && (
        <div className="absolute inset-0 grid place-items-center
                        bg-black/50 text-white text-sm">
          <div className="flex flex-col items-center gap-3">
            <div className="flex items-center gap-2">
              <Loader2 className="w-5 h-5 animate-spin" />
              Starting camera…
            </div>
            {/* Safety net: a real tap re-triggers getUserMedia, which some
                mobile browsers require after the permission prompt. */}
            <button onClick={camera.start}
                    className="px-5 py-2.5 rounded-full bg-white/15 text-sm
                               backdrop-blur active:scale-95 min-h-[44px]">
              Tap to retry
            </button>
          </div>
        </div>
      )}
      {camera.error && (
        <div className="absolute inset-0 grid place-items-center bg-black/70 px-6">
          <div className="text-center text-white max-w-md">
            <CameraOff className="w-10 h-10 mx-auto mb-3 text-rose-400" />
            <div className="text-sm font-medium">Camera unavailable</div>
            <p className="text-xs text-slate-300 mt-1">{camera.error}</p>
            <button onClick={camera.start} className="btn-ghost mt-3 text-slate-900">
              Try again
            </button>
          </div>
        </div>
      )}

      {/* Hidden canvas for frame extraction */}
      <canvas ref={camera.canvasRef} className="hidden" />
    </div>
  );
}
