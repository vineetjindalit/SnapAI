// frontend-react/src/store/session.ts
// Global state — session info + the latest live-frame result + capture history.

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { FrameResult, HealthResponse, PromptTransition } from "@/api/types";

export type AppMode = "user" | "developer" | "admin";

export interface CaptureEntry {
  id: number;
  thumbB64: string;
  url:      string | null;            // wide shot
  zoomUrl:  string | null;            // zoomed/cropped shot (may be null)
  zoomFactor?: number;
  moment: string;
  score: number;
  tier?: string;                      // priority tier that let it through
  reasons?: string[];                 // WHY it fired (engine + audio/voice/required/best_shot)
  ts: number;
  feedback: "kept" | "trashed" | null;
}

interface PromptHistoryEntry {
  ts: number;
  text: string;
  intent: string;
  matched: string[];
  active: string[];
}

interface SessionState {
  // Mode (persisted)
  mode: AppMode;
  setMode: (m: AppMode) => void;

  // Active session
  sid: string | null;
  eventName: string;
  eventType: string;
  prompt: string;
  setSession: (sid: string, eventName: string, eventType: string, prompt: string) => void;
  clearSession: () => void;

  // Latest frame result (the entire WS payload)
  latestFrame: FrameResult | null;
  setLatestFrame: (f: FrameResult) => void;

  // Health (model availability)
  health: HealthResponse | null;
  setHealth: (h: HealthResponse) => void;

  // Captures
  captures: CaptureEntry[];
  pushCapture: (c: CaptureEntry) => void;
  markFeedback: (id: number, kept: boolean) => void;

  // Score timeline (for charts)
  timeline: number[];
  pushTimeline: (s: number) => void;

  // Prompt history (developer mode)
  promptHistory: PromptHistoryEntry[];
  pushPrompt: (t: PromptTransition) => void;

  // Stats summary (running averages for the user-mode HUD)
  totalFrames: number;
  totalCaptures: number;
  bumpFrames: () => void;
  setTotalCaptures: (n: number) => void;

  // True while a recorded video is uploading/processing — the live camera
  // pauses sending frames so an upload and the live feed never mix captures.
  uploading: boolean;
  setUploading: (b: boolean) => void;

  // replayActive: a replay OWNS the live session for its whole duration (from
  // start until the user exits to live/upload) — the camera AND mic are gated
  // off the entire time, so nothing live can mix into the clip's session.
  // replaying: the clip is actively streaming right now (UI indicator only).
  replayActive: boolean;
  setReplayActive: (b: boolean) => void;
  replaying: boolean;
  setReplaying: (b: boolean) => void;

  // Custom Mode's conversational state (see api/types.ts's AssistantReply) —
  // SnapAI's reply, the live "watching for" guide, and status indicator.
  assistantReply: string;
  setAssistantReply: (assistantReply: string) => void;
  captureGuide: string;
  setCaptureGuide: (captureGuide: string) => void;
  watchingFor: string;
  setWatchingFor: (watchingFor: string) => void;
  // General Mode's live checklist — mirrors AssistantReply.moments. Empty
  // for Custom Mode (single objective, no list to check off).
  momentsChecklist: Array<{ label: string; captured: boolean }>;
  setMomentsChecklist: (moments: Array<{ label: string; captured: boolean }>) => void;
  watchStatus: "idle" | "understanding" | "watching" | "captured";
  setWatchStatus: (watchStatus: SessionState["watchStatus"]) => void;
}

const SCORE_HISTORY_MAX = 120;
const CAPTURES_MAX      = 60;
const PROMPTS_MAX       = 40;

export const useSession = create<SessionState>()(
  persist(
    (set) => ({
      mode: "user",
      setMode: (mode) => set({ mode }),

      sid: null, eventName: "", eventType: "", prompt: "",
      setSession: (sid, eventName, eventType, prompt) =>
        set({ sid, eventName, eventType, prompt, captures: [], timeline: [],
              promptHistory: [], totalFrames: 0, totalCaptures: 0,
              assistantReply: "", captureGuide: "", watchingFor: "",
              momentsChecklist: [], watchStatus: "idle" }),
      clearSession: () =>
        set({ sid: null, eventName: "", eventType: "", prompt: "",
              latestFrame: null, captures: [], timeline: [],
              promptHistory: [], totalFrames: 0, totalCaptures: 0,
              assistantReply: "", captureGuide: "", watchingFor: "",
              momentsChecklist: [], watchStatus: "idle" }),

      latestFrame: null,
      setLatestFrame: (latestFrame) => set({ latestFrame }),

      health: null,
      setHealth: (health) => set({ health }),

      captures: [],
      pushCapture: (c) => set((s) => ({
        captures: [c, ...s.captures].slice(0, CAPTURES_MAX),
      })),
      markFeedback: (id, kept) => set((s) => ({
        captures: s.captures.map((c) =>
          c.id === id ? { ...c, feedback: kept ? "kept" : "trashed" } : c),
      })),

      timeline: [],
      pushTimeline: (s2) => set((s) => ({
        timeline: [...s.timeline, s2].slice(-SCORE_HISTORY_MAX),
      })),

      promptHistory: [],
      pushPrompt: (t) => set((s) => ({
        promptHistory: [
          { ts: t.ts, text: t.text, intent: t.intent,
            matched: t.matched, active: t.active },
          ...s.promptHistory,
        ].slice(0, PROMPTS_MAX),
      })),

      totalFrames: 0, totalCaptures: 0,
      bumpFrames: () => set((s) => ({ totalFrames: s.totalFrames + 1 })),
      setTotalCaptures: (n) => set({ totalCaptures: n }),

      uploading: false,
      setUploading: (uploading) => set({ uploading }),

      replayActive: false,
      setReplayActive: (replayActive) => set({ replayActive }),
      replaying: false,
      setReplaying: (replaying) => set({ replaying }),

      assistantReply: "",
      setAssistantReply: (assistantReply) => set({ assistantReply }),
      captureGuide: "",
      setCaptureGuide: (captureGuide) => set({ captureGuide }),
      watchingFor: "",
      setWatchingFor: (watchingFor) => set({ watchingFor }),
      momentsChecklist: [],
      setMomentsChecklist: (momentsChecklist) => set({ momentsChecklist }),
      watchStatus: "idle",
      setWatchStatus: (watchStatus) => set({ watchStatus }),
    }),
    {
      name: "snappy-session",
      partialize: (s) => ({ mode: s.mode }), // only persist the mode
    }
  )
);
