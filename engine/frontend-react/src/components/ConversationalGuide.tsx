// ConversationalGuide — the live assistant overlay for both Custom Mode
// (open-ended, per-moment capture requests) and General Mode (name just the
// occasion, an LLM derives the whole moment list) — shown over the camera
// instead of SetupCoach (which stays birthday-only). Mirrors SetupCoach's
// visual language (same position, same black/70 backdrop-blur pill styling)
// but is driven by the conversational WS flow instead of the per-frame
// birthday setup advisor — see hooks/useLiveStream.ts's "assistant_reply"
// handling and backend api/server.py's _dispatch_custom_prompt /
// _dispatch_general_event for where this state comes from.
import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";
import { Eye, Sparkles, Camera as CameraIcon, MessageCircle, Square,
         CheckCircle2, Circle } from "lucide-react";
import { useSession } from "@/store/session";
import { useReducedMotion } from "@/lib/motionUtils";
import type { SnappyWs } from "@/api/ws";

const STATUS_META: Record<string, Record<string, { icon: typeof Eye; label: string }>> = {
  custom: {
    idle:          { icon: MessageCircle, label: "Tell me what to capture" },
    understanding: { icon: Sparkles,      label: "Understanding…" },
    watching:      { icon: Eye,           label: "Watching" },
    captured:      { icon: CameraIcon,    label: "Captured!" },
  },
  smart_event: {
    idle:          { icon: MessageCircle, label: "Tell me the event" },
    understanding: { icon: Sparkles,      label: "Working out what to capture…" },
    watching:      { icon: Eye,           label: "Watching for key moments" },
    captured:      { icon: CameraIcon,    label: "Captured!" },
  },
};

// Full sentences (not bare keywords) so a tap gives the VLM the same quality
// of input as someone actually typing a real request.
const CUSTOM_SUGGESTIONS = [
  "capture me when I pose",
  "capture when I smile",
  "capture when the lights change",
  "capture when I wave",
];

// Occasion NAMES, not capture requests — General Mode derives the moment
// list itself from just the name.
const EVENT_SUGGESTIONS = ["Diwali", "Rakhi", "Janmashtami", "a wedding"];

interface Props { ws: SnappyWs | null }

export function ConversationalGuide({ ws }: Props) {
  const eventType       = useSession((s) => s.eventType);
  const eventName       = useSession((s) => s.eventName);
  const watchStatus     = useSession((s) => s.watchStatus);
  const assistantReply  = useSession((s) => s.assistantReply);
  const captureGuide    = useSession((s) => s.captureGuide);
  // Lighting/steadiness coaching — Custom and General used to render nothing
  // here at all (the backend sent setup: null), so a room too dark to
  // photograph gave the user no explanation for why the picture looked black
  // and nothing was being captured.
  const setupTips       = useSession((s) => s.latestFrame?.setup?.tips ?? []);
  const momentsChecklist = useSession((s) => s.momentsChecklist);
  const reduced = useReducedMotion();
  const isGeneral = eventType === "smart_event";
  const SUGGESTIONS = isGeneral ? EVENT_SUGGESTIONS : CUSTOM_SUGGESTIONS;

  const meta = (STATUS_META[isGeneral ? "smart_event" : "custom"][watchStatus])
               ?? STATUS_META.custom.idle;
  const Icon = meta.icon;
  const understanding = watchStatus === "understanding";
  const watching      = watchStatus === "watching";
  const showSuggestions = watchStatus === "idle" || watchStatus === "captured";

  // General Mode: the user already named the occasion back on the Setup
  // screen ("Event name" field) — don't make them type it again. Auto-fire
  // it as the very first prompt the moment this mounts idle, exactly once
  // per session (the ref guards against StrictMode's double-mount and
  // against re-firing after a later "Stop" returns to idle).
  const autoSent = useRef(false);
  useEffect(() => {
    if (isGeneral && !autoSent.current && watchStatus === "idle" && eventName?.trim() && ws) {
      autoSent.current = true;
      ws.sendPrompt(eventName.trim(), "text");
    }
  }, [isGeneral, watchStatus, eventName, ws]);

  // Re-trigger a slide-in every time the reply text actually changes, so each
  // new turn visibly ARRIVES instead of just silently swapping in place —
  // the difference between "a UI updated" and "it just said something".
  const [replyKey, setReplyKey] = useState(0);
  const lastReply = useRef(assistantReply);
  useEffect(() => {
    if (assistantReply !== lastReply.current) {
      lastReply.current = assistantReply;
      setReplyKey((k) => k + 1);
    }
  }, [assistantReply]);

  return (
    // Renders BELOW the camera box (UserMode.tsx), not over it — as an
    // overlay this panel sat right across the middle of the frame, covering
    // whoever was being photographed. A normal in-flow card here instead of
    // a dark translucent overlay, since it's no longer floating on video.
    <div className="mt-3">
      <div className="rounded-2xl bg-slate-900 text-white
                      px-4 py-3 text-sm shadow-md">
        <div className="flex items-center gap-2">
          {/* A breathing ring while actively watching — the one moment this
              overlay most needs to feel ALIVE, since nothing else on screen
              signals "the camera is being analyzed right now". */}
          <span className={"shrink-0 relative grid place-items-center w-5 h-5 " +
                           (understanding ? "animate-pulse" : "")}>
            {watching && (
              <span className="absolute inline-flex h-full w-full rounded-full
                               bg-emerald-400/40 animate-ping" />
            )}
            <Icon className={"w-4 h-4 relative " +
                             (watching ? "text-emerald-400" : "")} />
          </span>
          <span className="font-medium leading-snug">{meta.label}</span>

          {watching && (
            <button onClick={() => ws?.sendStopWatching()}
              className="ml-auto shrink-0 flex items-center gap-1 px-2.5 py-1
                         rounded-full bg-white/15 hover:bg-white/25 active:bg-white/30
                         text-xs font-medium transition">
              <Square className="w-3 h-3" fill="currentColor" />
              Stop
            </button>
          )}
        </div>

        {assistantReply && (
          <p key={replyKey} className="mt-1.5 leading-snug text-white/95 animate-slide-in-up">
            {assistantReply}
          </p>
        )}

        {/* General Mode: a real, filling-in-live checklist instead of flat
            "watching for: a, b, c" text — this is what turns "the app is
            doing something invisible" into visible progress. Each chip's
            checked state comes straight from the backend's per-moment
            last_capture_ts (server.py's _general_moments_payload), so it
            can never show "captured" before the backend actually agrees. */}
        {watching && isGeneral && momentsChecklist.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {momentsChecklist.map((m) => (
              <motion.span
                key={m.label}
                layout={!reduced}
                animate={m.captured && !reduced ? { scale: [1, 1.12, 1] } : undefined}
                transition={{ duration: 0.35, ease: "easeOut" }}
                className={"inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs " +
                           (m.captured
                             ? "bg-emerald-400/20 text-emerald-200"
                             : "bg-white/10 text-white/60")}>
                {m.captured
                  ? <CheckCircle2 className="w-3 h-3 shrink-0" />
                  : <Circle className="w-3 h-3 shrink-0" />}
                {m.label}
              </motion.span>
            ))}
          </div>
        )}

        {watching && !isGeneral && captureGuide && (
          <p className="mt-1.5 text-white/70 text-xs leading-snug">
            {captureGuide}
          </p>
        )}

        {setupTips.length > 0 && (
          <div className="mt-2 rounded-lg bg-amber-400/15 border border-amber-300/30
                          px-2.5 py-1.5">
            {setupTips.map((tip) => (
              <p key={tip} className="text-amber-200 text-xs leading-snug">⚠ {tip}</p>
            ))}
          </div>
        )}

        {watchStatus === "idle" && !assistantReply && (
          <p className="mt-1.5 text-white/70 text-xs leading-snug">
            {isGeneral
              ? 'Type or speak the occasion — e.g. "Diwali", "Rakhi", "a retirement party".'
              : 'Type or speak below — e.g. "capture me when I pose", "capture when the lights turn on".'}
          </p>
        )}

        {/* One-tap re-engagement — matches "What would you like me to
            capture next?": don't make someone re-type a full sentence when
            they just want to try another example. */}
        {showSuggestions && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => ws?.sendPrompt(s, "text")}
                className="px-2.5 py-1 rounded-full bg-white/10 hover:bg-white/20
                           active:bg-white/25 text-xs text-white/90 transition">
                {s.replace(/^capture /, "")}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
