// frontend-react/src/hooks/useWsStatus.ts
// Subscribes to the REAL socket lifecycle (api/ws.ts's onStatus) so the UI can
// render connection state truthfully. There is no polling and no synthetic
// timer here: every field originates in an actual onopen/onclose/onmessage
// event, or in the retry delay the socket genuinely scheduled.

import { useEffect, useState } from "react";
import type { SnappyWs, WsStatus } from "@/api/ws";

const IDLE: WsStatus = { phase: "closed", lastMessageAt: 0, nextRetryAt: 0, attempt: 0 };

/** Live connection status for a socket (null-safe: returns a closed status
 *  when there's no socket yet, e.g. before a session exists). */
export function useWsStatus(ws: SnappyWs | null): WsStatus {
  const [status, setStatus] = useState<WsStatus>(() => ws?.getStatus() ?? IDLE);

  useEffect(() => {
    if (!ws) { setStatus(IDLE); return; }
    // onStatus fires immediately with the current value, so there's no gap
    // between subscribing and having accurate state.
    return ws.onStatus(setStatus);
  }, [ws]);

  return status;
}

/**
 * True when frames are genuinely flowing right now — derived from the real
 * arrival time of the last WS message, not from an assumption that "open
 * means receiving". A socket can be open and completely idle (the 25s
 * heartbeat is the only traffic), and the UI must be able to tell those apart.
 *
 * `windowMs` should be comfortably longer than the frame interval (5fps =
 * 200ms) but far shorter than the heartbeat, so heartbeat pongs alone never
 * masquerade as an active stream.
 */
export function useReceivingFrames(status: WsStatus, windowMs = 1200): boolean {
  const [receiving, setReceiving] = useState(false);

  useEffect(() => {
    if (status.phase !== "open" || !status.lastMessageAt) {
      setReceiving(false);
      return;
    }
    const age = Date.now() - status.lastMessageAt;
    if (age > windowMs) { setReceiving(false); return; }
    setReceiving(true);
    // Fall back to "not receiving" if nothing else arrives within the window.
    // Re-armed on every message because lastMessageAt changes each time.
    const t = setTimeout(() => setReceiving(false), windowMs - age);
    return () => clearTimeout(t);
  }, [status.phase, status.lastMessageAt, windowMs]);

  return receiving;
}
