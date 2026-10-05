// frontend-react/src/hooks/useLiveStream.ts
// Wires camera frames → WebSocket → store updates.
// Single hook the live pages use.

import { useEffect, useMemo, useRef } from "react";
import { useCamera } from "./useCamera";
import { useMicAudio } from "./useMicAudio";
import { SnappyWs, type SnappyWsMessage } from "@/api/ws";
import { useSession } from "@/store/session";
import type { FrameResult } from "@/api/types";

export function useLiveStream(opts: { fps?: number; audio?: boolean } = {}) {
  const { fps = 5, audio = true } = opts;
  const sid = useSession((s) => s.sid);
  const setLatest    = useSession((s) => s.setLatestFrame);
  const pushCapture  = useSession((s) => s.pushCapture);
  const pushTimeline = useSession((s) => s.pushTimeline);
  const pushPrompt   = useSession((s) => s.pushPrompt);
  const bumpFrames   = useSession((s) => s.bumpFrames);
  const setTotalCaps = useSession((s) => s.setTotalCaptures);
  const clearSession = useSession((s) => s.clearSession);
  const setAssistantReply = useSession((s) => s.setAssistantReply);
  const setCaptureGuide   = useSession((s) => s.setCaptureGuide);
  const setWatchingFor    = useSession((s) => s.setWatchingFor);
  const setMomentsChecklist = useSession((s) => s.setMomentsChecklist);
  const setWatchStatus    = useSession((s) => s.setWatchStatus);

  // Stable WS instance per session id
  const ws = useMemo(() => sid ? new SnappyWs(sid) : null, [sid]);

  // Camera with WS-bound frame callback
  const lastSentRef = useRef<number>(0);
  const camera = useCamera({
    fps,
    onFrame: (b64) => {
      // Pause the live camera while a recorded video is being analysed, OR while
      // a clip OWNS the session as a replay — so the real camera never mixes a
      // single frame into an upload's / replay's session.
      const _st = useSession.getState();
      if (_st.uploading || _st.replayActive) return;
      // Throttle defensively in case the FPS interval is misbehaving.
      const now = performance.now();
      if (now - lastSentRef.current < 1000 / fps - 5) return;
      lastSentRef.current = now;
      ws?.sendFrame(b64);
    },
  });

  // Mic audio → WS (cheering/applause/singing cues for live capture timing).
  // Gated exactly like the camera: live mic must never bleed into an upload's
  // or a replay's session (that was a source of "mixing").
  const mic = useMicAudio((b64) => {
    const _st = useSession.getState();
    if (_st.uploading || _st.replayActive) return;
    ws?.sendAudio(b64);
  }, { enabled: audio && !!ws });

  // Plumb WS messages → store
  useEffect(() => {
    if (!ws) return;
    ws.connect();
    // Belt-and-suspenders with the {type:"error"} handler below: THIS is the
    // path that actually fires for a dead session (see ws.ts) — the backend
    // rejects it before any WS message can ever be sent, so the app used to
    // just sit there retrying forever with the camera running and nothing
    // happening.
    ws.onFatal(() => {
      console.warn("SnapAI: session appears to be gone (server keeps rejecting the connection) — clearing");
      clearSession();
    });
    const off = ws.on((msg: SnappyWsMessage) => {
      if (msg.type === "error") {
        // Stale session id (e.g. left over from before a server restart) →
        // drop it so the app returns to Setup for a fresh session instead of
        // silently sending frames to a dead session.
        if (/not found/i.test((msg as any).message || "")) {
          console.warn("SnapAI: session not found — clearing stale session");
          clearSession();
        }
        return;
      }
      if (msg.type === "prompt_changed") {
        pushPrompt(msg.transition);
        return;
      }
      if (msg.type === "assistant_reply") {
        setAssistantReply(msg.data.reply);
        setCaptureGuide(msg.data.guide);
        setWatchingFor(msg.data.watching_for);
        setMomentsChecklist(msg.data.moments ?? []);
        setWatchStatus(msg.data.status);
        return;
      }
      if (msg.type !== "frame_result") return;
      const f = msg as FrameResult;
      // setLatest/frames/timeline fire for ALL frames so the developer
      // dashboard's live meters animate while a video upload is processing.
      setLatest(f);
      bumpFrames();
      pushTimeline(f.analysis?.total ?? 0);
      // …but the LIVE capture store (the "Recent moments" grid + "Captured"
      // count) must hold LIVE captures only. An upload streams its own
      // captured frames over the same socket; routing those into the live
      // grid is exactly the "live mixed with upload" bug. Skip upload frames —
      // the upload's shots are shown by its own curated album instead.
      if (f.source === "upload") return;
      setTotalCaps(f.total_captures);
      if (f.captured && f.captured_thumb) {
        pushCapture({
          id: f.frame_id,
          thumbB64: f.captured_thumb,
          url:      f.captured_url,
          zoomUrl:  f.captured_zoom_url,
          zoomFactor: f.captured_zoom_info?.zoom_factor,
          moment:   f.moment.detected_moment,
          score:    f.analysis?.total ?? 0,
          tier:     f.engine?.priority_tier,
          reasons:  (f.engine?.reasons && f.engine.reasons.length)
                      ? f.engine.reasons
                      : (f.capture_reason ? [f.capture_reason] : []),
          ts:       Date.now(),
          feedback: null,
        });
      }
    });
    return () => { off(); ws.close(); };
  }, [ws, setLatest, bumpFrames, pushTimeline, setTotalCaps,
      pushCapture, pushPrompt, clearSession,
      setAssistantReply, setCaptureGuide, setWatchingFor, setMomentsChecklist, setWatchStatus]);

  return { camera, mic, ws };
}
