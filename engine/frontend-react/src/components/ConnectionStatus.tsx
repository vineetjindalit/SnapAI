// frontend-react/src/components/ConnectionStatus.tsx
//
// The user-facing connection indicator. Deliberately NOT LiveStatus.tsx —
// that one is DevMode's telemetry banner (moment / tier / score) and stays a
// debug view.
//
// TRUTHFULNESS CONTRACT — the whole point of this component:
//   • The dot's colour and pulse rate are a pure function of the REAL socket
//     phase from api/ws.ts (set in onopen/onclose), never a decorative loop.
//   • "Receiving" is derived from the actual arrival time of the last WS
//     message, so an open-but-idle socket does NOT claim to be streaming.
//   • The reconnect countdown counts down to the retry the socket genuinely
//     scheduled (status.nextRetryAt), not an invented interval.
//   • There is no state in which this renders "connected" while the socket is
//     down: every phase maps to exactly one visual, and the fallback is the
//     disconnected visual, not the connected one.

import { useEffect, useState } from "react";
import { motion } from "motion/react";
import type { WsStatus } from "@/api/ws";
import { useReducedMotion } from "@/lib/motionUtils";

interface Props {
  status: WsStatus;
  /** True only while WS messages are actually arriving (useReceivingFrames). */
  receiving: boolean;
  className?: string;
}

/** Visual + copy for each REAL socket phase. `pulseSec` is the breathing
 *  period: slower when idle, faster when frames are genuinely flowing. */
function present(status: WsStatus, receiving: boolean) {
  switch (status.phase) {
    case "open":
      return receiving
        ? { label: "Live",        dot: "bg-emerald-400", ring: "bg-emerald-400/40",
            text: "text-emerald-700", pulseSec: 0.9 }
        : { label: "Connected",   dot: "bg-emerald-400", ring: "bg-emerald-400/30",
            text: "text-emerald-700", pulseSec: 2.6 };
    case "connecting":
      return { label: "Connecting…", dot: "bg-sky-400", ring: "bg-sky-400/30",
               text: "text-sky-700", pulseSec: 1.2 };
    case "reconnecting":
      return { label: "Reconnecting", dot: "bg-amber-400", ring: "bg-amber-400/40",
               text: "text-amber-700", pulseSec: 0.7 };
    case "fatal":
      return { label: "Session ended", dot: "bg-rose-500", ring: "bg-rose-500/30",
               text: "text-rose-700", pulseSec: 0 };
    // "closed" and any unforeseen value both fall through to the OFFLINE
    // visual on purpose — an unknown state must never look connected.
    default:
      return { label: "Offline", dot: "bg-slate-400", ring: "bg-slate-400/20",
               text: "text-slate-600", pulseSec: 0 };
  }
}

/** Seconds until the socket's genuinely-scheduled retry. Ticks once a second
 *  purely to re-render the number — it does not drive the reconnect itself
 *  (api/ws.ts owns that), so it can never desync the actual retry. */
function useRetryCountdown(nextRetryAt: number): number | null {
  const [secs, setSecs] = useState<number | null>(null);

  useEffect(() => {
    if (!nextRetryAt) { setSecs(null); return; }
    const tick = () =>
      setSecs(Math.max(0, Math.ceil((nextRetryAt - Date.now()) / 1000)));
    tick();
    const id = setInterval(tick, 1000);
    return () => clearInterval(id);
  }, [nextRetryAt]);

  return secs;
}

export function ConnectionStatus({ status, receiving, className = "" }: Props) {
  const reduced = useReducedMotion();
  const p = present(status, receiving);
  const countdown = useRetryCountdown(
    status.phase === "reconnecting" ? status.nextRetryAt : 0);
  const animated = p.pulseSec > 0 && !reduced;

  return (
    <div className={"inline-flex items-center gap-2 " + className}>
      <span className="relative grid place-items-center w-2.5 h-2.5 shrink-0">
        {/* Expanding halo — only while genuinely connected or retrying. */}
        {animated && (
          <motion.span
            className={"absolute inset-0 rounded-full " + p.ring}
            animate={{ scale: [1, 2.1, 1], opacity: [0.7, 0, 0.7] }}
            transition={{ duration: p.pulseSec, repeat: Infinity, ease: "easeOut" }}
          />
        )}
        {/* Core dot. Brightness breathes at the same real-state-derived rate. */}
        <motion.span
          className={"relative w-2.5 h-2.5 rounded-full " + p.dot}
          animate={animated ? { opacity: [0.65, 1, 0.65] } : { opacity: 1 }}
          transition={animated
            ? { duration: p.pulseSec, repeat: Infinity, ease: "easeInOut" }
            : { duration: 0 }}
        />
      </span>

      <span className={"text-xs font-medium " + p.text}>
        {p.label}
        {status.phase === "reconnecting" && countdown !== null && (
          <span className="ml-1 tabular-nums text-amber-600/80">
            · retrying in {countdown}s
          </span>
        )}
      </span>
    </div>
  );
}
