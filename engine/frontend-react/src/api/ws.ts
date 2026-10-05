// frontend-react/src/api/ws.ts
// WebSocket client for the SnapAI live-frame stream.
// Self-managing reconnect, typed messages, per-instance reference count.

import type { FrameResult, PromptTransition, AssistantReply } from "./types";

export type SnappyWsMessage =
  | FrameResult
  | { type: "pong" }
  | { type: "error"; message: string }
  | { type: "prompt_changed"; transition: PromptTransition }
  | { type: "assistant_reply"; data: AssistantReply };

type Handler = (msg: SnappyWsMessage) => void;

/** Real socket lifecycle, surfaced so the UI can show connection state
 *  TRUTHFULLY. Every value here is set from an actual WebSocket event
 *  (onopen/onclose/onmessage) or from the retry delay this class actually
 *  scheduled — nothing is inferred, polled, or invented. */
export type WsPhase = "connecting" | "open" | "reconnecting" | "fatal" | "closed";

export interface WsStatus {
  phase: WsPhase;
  /** epoch ms of the last message ACTUALLY received (any type), or 0. Lets the
   *  UI distinguish "connected and receiving frames" from "connected but idle"
   *  without inventing a timer. */
  lastMessageAt: number;
  /** epoch ms when the next reconnect attempt is genuinely scheduled to fire
   *  (only meaningful while phase === "reconnecting"), or 0. */
  nextRetryAt: number;
  /** How many consecutive reconnect attempts have been made. */
  attempt: number;
}

type StatusHandler = (s: WsStatus) => void;

/** The heartbeat period the socket really uses (see startHeartbeat). Exported
 *  so the UI can surface the true cadence rather than hard-coding a guess. */
export const WS_HEARTBEAT_MS = 25_000;

// The backend rejects a dead/unknown session at the raw HTTP-upgrade level —
// a plain 404 with the TCP connection closed immediately, before any WS frame
// is ever exchanged (see api/server.py _upgrade_ws: "sid not in SESSIONS").
// The browser's WebSocket API can't tell JS WHY a handshake failed — onclose
// fires with no reason either way — so a session that's gone for good (ended,
// or the backend restarted) used to retry FOREVER with no error and no way
// out. Distinguish it from a real network outage by TIMING: an active
// server-side reject completes near-instantly (local/tunnel round trip);
// a genuine "no network" failure takes much longer to time out. A couple of
// fast, back-to-back failed handshakes is a reliable signal the session
// itself is gone, not that connectivity is flaky.
const FATAL_REJECT_MS = 800;
const FATAL_REJECT_STREAK = 2;

export class SnappyWs {
  private url:    string;
  private ws:     WebSocket | null = null;
  private handlers = new Set<Handler>();
  private reconnectDelayMs = 1000;
  private closed = false;
  private heartbeat?: ReturnType<typeof setInterval>;
  private connectStartedAt = 0;
  private openedThisAttempt = false;
  private fastRejectStreak = 0;
  private fatalHandler?: () => void;
  private statusHandlers = new Set<StatusHandler>();
  private status: WsStatus = {
    phase: "closed", lastMessageAt: 0, nextRetryAt: 0, attempt: 0,
  };

  /** Called when the session looks permanently gone (not a network blip) —
   *  the caller should drop it and return to Setup for a fresh one. */
  onFatal(cb: () => void) { this.fatalHandler = cb; }

  /** Subscribe to real connection-state changes. Fires immediately with the
   *  current status so a late subscriber isn't blank until the next event.
   *  Returns an unsubscribe fn. */
  onStatus(cb: StatusHandler): () => void {
    this.statusHandlers.add(cb);
    cb(this.status);
    return () => { this.statusHandlers.delete(cb); };
  }

  getStatus(): WsStatus { return this.status; }

  private setStatus(patch: Partial<WsStatus>) {
    this.status = { ...this.status, ...patch };
    this.statusHandlers.forEach((h) => h(this.status));
  }

  constructor(sid: string) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    // In dev (vite :5173) vite proxies /ws → :8765. In prod same-origin.
    // Pass the bearer token via query (browsers can't set Authorization on
    // new WebSocket()). Backend reads it in _upgrade_ws.
    const token = localStorage.getItem("snappy.auth_token");
    const q = token ? `?token=${encodeURIComponent(token)}` : "";
    this.url = `${proto}://${location.host}/ws/${sid}${q}`;
  }

  connect() {
    // Re-arm on every connect() so React 18 StrictMode's setup→cleanup→setup
    // (which calls close() in between) doesn't leave the socket permanently
    // closed. Without this, the second setup hit `if (this.closed) return` and
    // NO frames ever flowed — the camera ran but nothing was detected.
    this.closed = false;
    // Don't open a duplicate socket if one is already connecting/open.
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;
    this.connectStartedAt = Date.now();
    this.openedThisAttempt = false;
    // Only "connecting" on a FIRST attempt; a retry stays "reconnecting" so
    // the UI doesn't flip to a hopeful state on every failed retry loop.
    if (this.status.phase !== "reconnecting") {
      this.setStatus({ phase: "connecting", nextRetryAt: 0 });
    }
    this.ws = new WebSocket(this.url);
    this.ws.onopen    = () => {
      this.openedThisAttempt = true;
      this.fastRejectStreak = 0;
      this.reconnectDelayMs = 1000;
      this.startHeartbeat();
      this.setStatus({ phase: "open", nextRetryAt: 0, attempt: 0 });
    };
    this.ws.onclose   = () => {
      this.stopHeartbeat();
      if (this.closed) { this.setStatus({ phase: "closed", nextRetryAt: 0 }); return; }
      if (!this.openedThisAttempt && Date.now() - this.connectStartedAt < FATAL_REJECT_MS) {
        this.fastRejectStreak += 1;
        if (this.fastRejectStreak >= FATAL_REJECT_STREAK) {
          this.closed = true;   // stop retrying — this session is gone, not the network
          this.setStatus({ phase: "fatal", nextRetryAt: 0 });
          this.fatalHandler?.();
          return;
        }
      } else {
        this.fastRejectStreak = 0;   // a slow failure or a drop of a live connection — just a blip
      }
      // Publish the ACTUAL delay being scheduled below, so a countdown in the
      // UI matches when the retry really fires.
      this.setStatus({
        phase: "reconnecting",
        nextRetryAt: Date.now() + this.reconnectDelayMs,
        attempt: this.status.attempt + 1,
      });
      setTimeout(() => this.connect(), this.reconnectDelayMs);
      this.reconnectDelayMs = Math.min(this.reconnectDelayMs * 2, 15_000);
    };
    this.ws.onerror   = () => { /* handled via onclose retry */ };
    this.ws.onmessage = (ev) => {
      // Stamped from the real message arrival, before parsing, so "receiving"
      // reflects actual traffic (frames AND pong keepalives).
      this.setStatus({ lastMessageAt: Date.now() });
      try {
        const msg = JSON.parse(ev.data) as SnappyWsMessage;
        this.handlers.forEach((h) => h(msg));
      } catch { /* ignore non-JSON */ }
    };
  }

  on(h: Handler): () => void {
    this.handlers.add(h);
    return () => this.handlers.delete(h);
  }

  // Keepalive: ping every 25 s so the connection never idles long enough for a
  // proxy/tunnel to drop it (Cloudflare quick-tunnels cut idle WS after ~100 s).
  // The backend replies {type:"pong"} (server.py), keeping traffic flowing both
  // ways — so an open-but-idle tab stops churning through reconnects.
  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeat = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: "ping" }));
      }
    }, WS_HEARTBEAT_MS);
  }

  private stopHeartbeat() {
    if (this.heartbeat) { clearInterval(this.heartbeat); this.heartbeat = undefined; }
  }

  sendFrame(frameB64: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ frame: frameB64 }));
    }
  }

  /** Returns whether the message actually went out. The socket silently
   *  drops sends while reconnecting (tunnel blip, brief network hiccup) —
   *  callers MUST check this and give visible feedback rather than assuming
   *  success, or a prompt can vanish with no sign anything went wrong
   *  (measured: this is what "I typed something and nothing happened"
   *  looks like from the user's side). */
  sendPrompt(text: string, source: "text" | "voice" = "text"): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "prompt_update", text, source }));
      return true;
    }
    return false;
  }

  /** Custom Mode's "Stop watching" button — instant, bypasses the VLM
   *  intent-parsing that a typed "stop" would otherwise wait on. */
  sendStopWatching() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "stop_watching" }));
    }
  }

  /** Send a mono 16 kHz float32 mic chunk (base64) for live audio cues. */
  sendAudio(pcmB64: string) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: "audio", pcm: pcmB64 }));
    }
  }

  close() {
    this.closed = true;
    this.stopHeartbeat();
    this.ws?.close();
    this.ws = null;
    this.handlers.clear();
    this.setStatus({ phase: "closed", nextRetryAt: 0 });
    this.statusHandlers.clear();
  }
}
