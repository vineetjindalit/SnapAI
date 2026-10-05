import { useEffect, useRef, useState } from "react";
import { Loader2, Mic, MicOff, Send } from "lucide-react";
import type { SnappyWs } from "@/api/ws";

interface Props {
  ws: SnappyWs | null;
  variant?: "user" | "developer";
  onTranscript?: (text: string, source: "voice" | "text") => void;
  placeholder?: string;
  /**
   * True while a previous prompt for this session is still being understood
   * (Custom/General's VLM step). Disables sending a new one entirely rather
   * than letting an impatient re-tap queue a second VLM call behind the
   * first — measured live: 8 duplicate submissions in 13s dragged what
   * should've been a ~30s wait out to 191s, because each one contended with
   * every other one for the same GPU. The backend now also refuses a
   * duplicate on its own (defense in depth), but stopping it here means the
   * user sees WHY nothing happens instead of silently losing taps.
   */
  busy?: boolean;
}

// Browser Web Speech API types — vendor-prefixed in Chrome
declare global {
  interface Window { webkitSpeechRecognition?: any; SpeechRecognition?: any; }
}

export function PromptControl({ ws, variant = "user", onTranscript, placeholder, busy = false }: Props) {
  const [text, setText]   = useState("");
  const [voiceOn, setVoiceOn] = useState(false);
  const [heard,   setHeard]   = useState("");
  const [sendFailed, setSendFailed] = useState(false);
  const recRef = useRef<any>(null);

  const send = (final: string, source: "text" | "voice") => {
    if (busy) return;   // already understanding a previous prompt — see Props.busy
    const t = final.trim(); if (!t) return;
    // sendPrompt silently drops the message while the socket is
    // reconnecting — clearing the box regardless used to make that look
    // exactly like a successful send with nothing happening after.
    // Keep the text and show an error instead so a retry is one tap away.
    const ok = ws?.sendPrompt(t, source) ?? false;
    if (!ok) { setSendFailed(true); return; }
    setSendFailed(false);
    onTranscript?.(t, source);
    if (source === "text") setText("");
  };

  // ── Voice via Web Speech API ────────────────────────────────────────
  const toggleVoice = () => {
    const SR = window.SpeechRecognition ?? window.webkitSpeechRecognition;
    if (!SR) {
      alert("Voice not supported in this browser. Try Chrome / Edge / Safari.");
      return;
    }
    if (voiceOn) {
      try { recRef.current?.stop(); } catch {}
      setVoiceOn(false);
      setHeard("");
      return;
    }
    const rec = new SR();
    rec.lang = "en-US";
    rec.continuous = true;
    rec.interimResults = true;
    let lastFinal = "";
    rec.onresult = (ev: any) => {
      let interim = "", final = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        if (ev.results[i].isFinal) final += ev.results[i][0].transcript;
        else interim += ev.results[i][0].transcript;
      }
      setHeard(interim || final);
      if (final && final !== lastFinal) {
        lastFinal = final;
        send(final, "voice");
      }
    };
    rec.onend = () => { if (voiceOn) try { rec.start(); } catch {} };
    rec.onerror = (e: any) => console.warn("voice error", e);
    try { rec.start(); recRef.current = rec; setVoiceOn(true); }
    catch (e) { console.error(e); }
  };

  useEffect(() => () => { try { recRef.current?.stop(); } catch {} }, []);

  // ── User-mode minimal UI ────────────────────────────────────────────
  if (variant === "user") {
    return (
      <div className="flex items-center gap-2">
        <button onClick={toggleVoice}
                className={"shrink-0 w-12 h-12 rounded-full grid place-items-center " +
                           "shadow-lg transition " +
                           (voiceOn ? "bg-rose-500 hover:bg-rose-600 text-white animate-pulse-slow"
                                    : "bg-white hover:bg-slate-50 border border-slate-200 text-brand-600")}>
          {voiceOn ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
        </button>
        <input
          type="text"
          placeholder={busy ? "Working on it…" : (placeholder ?? "Tell SnapAI what to capture…")}
          value={text}
          disabled={busy}
          onChange={(e) => { setText(e.target.value); setSendFailed(false); }}
          onKeyDown={(e) => e.key === "Enter" && send(text, "text")}
          className={"flex-1 px-5 py-3 rounded-full bg-white border outline-none text-sm shadow-sm " +
                     "disabled:bg-slate-50 disabled:text-slate-400 transition-colors " +
                     (sendFailed
                       ? "border-rose-400 focus:border-rose-500 focus:ring-2 focus:ring-rose-500/20"
                       : "border-slate-200 focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20")} />
        <button onClick={() => send(text, "text")} disabled={busy || !text.trim()}
                title={busy ? "Still understanding your last request…" : undefined}
                className="shrink-0 w-12 h-12 rounded-full bg-brand-600 hover:bg-brand-700
                           disabled:opacity-40 grid place-items-center text-white shadow-lg
                           transition">
          {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
        </button>
        {heard && voiceOn && (
          <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2
                          px-3 py-1 bg-black/80 text-white text-xs rounded-full
                          backdrop-blur whitespace-nowrap">
            heard: "{heard}"
          </div>
        )}
        {sendFailed && (
          <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2
                          px-3 py-1.5 bg-rose-600 text-white text-xs rounded-full
                          backdrop-blur whitespace-nowrap shadow-lg">
            Not connected — tap send to retry
          </div>
        )}
      </div>
    );
  }

  // ── Developer-mode wider variant ────────────────────────────────────
  return (
    <div className="space-y-2">
      <div className="flex gap-2">
        <input
          type="text" placeholder="prompt_update text…"
          value={text} onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(text, "text")}
          className="flex-1 bg-ink-900 border border-ink-700 rounded px-3 py-2
                     font-mono text-sm text-slate-200" />
        <button onClick={() => send(text, "text")}
                className="px-3 py-2 bg-brand-600 hover:bg-brand-700 text-white
                           font-mono rounded text-sm">↵</button>
        <button onClick={toggleVoice}
                className={"px-3 py-2 rounded text-sm " +
                           (voiceOn ? "bg-rose-600 text-white"
                                    : "bg-ink-700 text-slate-200 hover:bg-ink-700/70")}>
          {voiceOn ? "🛑 stop" : "🎙 voice"}
        </button>
      </div>
      {heard && voiceOn && (
        <div className="text-xs font-mono text-slate-400">heard: {heard}</div>
      )}
    </div>
  );
}
