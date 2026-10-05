import { useState } from "react";
import { motion } from "motion/react";
import { Sessions } from "@/api/client";
import { useSession } from "@/store/session";
import { Camera, Sparkles, Lock } from "lucide-react";
import { useReducedMotion, withViewTransition, SPRING } from "@/lib/motionUtils";

// `defaults` are the moments auto-loaded (and used) for the event — shown as
// read-only chips so the user knows what's covered without editing them. The
// backend always applies the full category defaults; the prompt box is only
// for EXTRA moments the user wants on top.
const EVENT_TYPES = [
  { id: "wedding",   label: "Wedding",   emoji: "💍",
    defaults: ["ring ceremony", "first dance", "cake cutting", "group photo", "hugs"] },
  { id: "birthday",  label: "Birthday",  emoji: "🎂",
    defaults: ["pre-preparation", "person arrival", "surprise celebration",
               "cake", "cake person", "cake with candles", "candle blowing",
               "cake cutting", "cake smashing", "cake feeding", "birthday gifting",
               "group photo", "individual people", "smiling moments",
               "laughing moments", "crying moments", "hugging moments",
               "dancing moments", "gazing moments", "food table"] },
  { id: "party",     label: "Party",     emoji: "🎉",
    defaults: ["champagne toast", "group photo", "hug moment", "dancing"] },
  { id: "corporate", label: "Corporate", emoji: "🏢",
    defaults: ["champagne toast", "group photo", "applause"] },
  { id: "sports",    label: "Sports",    emoji: "🏆",
    defaults: ["sports action", "group photo", "celebration"] },
  // No fixed moments at all — the user defines the objective live, by voice
  // or text, once the session starts (see components/ConversationalGuide.tsx
  // + hooks/useLiveStream.ts's assistant_reply handling). Lets people test
  // SnapAI any day with ordinary actions instead of only at a real event.
  { id: "custom",    label: "Custom",    emoji: "✨",
    defaults: [] },
  // "General" in the UI, but its event_type is "smart_event" — NOT the
  // pre-existing "general" backend category (that one's the plain group-
  // photo/highlight fallback, still used silently for any unrecognized event
  // type). Here the user just names the occasion ("Diwali", "Rakhi") and an
  // LLM derives what's worth capturing at request time — no defaults list to
  // show because there's nothing pre-defined until the event name is given.
  { id: "smart_event", label: "General", emoji: "🎇",
    defaults: [] },
];

export function Setup() {
  const setSession = useSession((s) => s.setSession);
  const mode = useSession((s) => s.mode);

  const [eventName, setEventName] = useState("");
  const [eventType, setEventType] = useState("birthday");
  const [prompt,    setPrompt]    = useState("");   // EXTRA moments only
  const [busy, setBusy] = useState(false);
  const [err,  setErr]  = useState<string | null>(null);
  const [lockedShake, setLockedShake] = useState<string | null>(null);
  const reduced = useReducedMotion();

  const start = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await Sessions.create(
        eventName.trim() || `${eventType} session`,
        eventType, prompt.trim() || eventType,
      );
      // Shared-element handoff into the live session: the selected card
      // carries view-transition-name "event-hero", and UserMode's header
      // badge claims the same name, so the browser morphs one into the
      // other. withViewTransition falls back to an instant, identical state
      // update where the API is missing or reduced-motion is on — the
      // session is set either way, never gated on the animation.
      withViewTransition(() =>
        setSession(r.session_id, r.event_name, r.event_type, r.prompt));
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally { setBusy(false); }
  };

  /** A locked card was tapped — shake it briefly so it reads as deliberately
   *  locked rather than broken/unresponsive. */
  const nudgeLocked = (id: string) => {
    setLockedShake(id);
    setTimeout(() => setLockedShake((cur) => (cur === id ? null : cur)), 500);
  };

  if (mode === "developer") return <SetupDev {...{
    eventName, setEventName, eventType, setEventType,
    prompt, setPrompt, busy, err, start }} />;

  // ── User mode setup ─────────────────────────────────────────────────
  return (
    <div className="max-w-3xl mx-auto px-5 py-8 sm:py-14">
      <div className="text-center mb-10">
        <div className="inline-flex items-center gap-2 mb-4 px-4 py-1.5 rounded-full
                        bg-white shadow-sm border border-slate-200 text-xs uppercase
                        tracking-widest text-slate-500">
          <Sparkles className="w-3 h-3 text-brand-500" />
          AI photographer · always watching
        </div>
        <h1 className="text-4xl sm:text-5xl font-bold tracking-tight">
          Capture the moments<br />
          <span className="bg-gradient-to-r from-brand-500 to-rose-500
                           text-transparent bg-clip-text">that matter</span>.
        </h1>
        <p className="mt-4 text-slate-600 max-w-xl mx-auto">
          Tell SnapAI what kind of event this is — it'll watch your camera
          and quietly capture the right moments. You enjoy the day.
        </p>
      </div>

      <div className="card-soft space-y-7">
        <div>
          <label className="text-sm font-medium text-slate-700">
            Event name <span className="text-slate-400">(optional)</span>
          </label>
          <input
            type="text" placeholder="Tara & Sam's wedding"
            value={eventName} onChange={(e) => setEventName(e.target.value)}
            className="mt-2 w-full px-4 py-3 rounded-xl border border-slate-200
                       focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20
                       outline-none transition" />
        </div>

        <div>
          <label className="text-sm font-medium text-slate-700">
            What kind of event is this?
          </label>
          <div className="mt-3 grid grid-cols-2 sm:grid-cols-3 gap-2.5">
            {EVENT_TYPES.map((t, i) => {
              // MVP: Birthday is the tuned, launched event type; Custom is
              // the open-ended live one. The rest are visible as roadmap but
              // not selectable yet.
              const live = t.id === "birthday" || t.id === "custom" || t.id === "smart_event";
              const selected = eventType === t.id;
              const shaking = lockedShake === t.id;
              return (
                <motion.button key={t.id}
                  // Staggered entrance — a short cascade on mount only.
                  initial={reduced ? false : { opacity: 0, y: 10 }}
                  animate={{
                    opacity: 1, y: 0,
                    // Locked cards shake in place when tapped, so the tap is
                    // acknowledged rather than silently ignored.
                    x: shaking && !reduced ? [0, -6, 6, -4, 4, 0] : 0,
                  }}
                  transition={{
                    opacity: reduced ? { duration: 0 } : { delay: i * 0.045, duration: 0.28 },
                    y:       reduced ? { duration: 0 } : { delay: i * 0.045, ...SPRING.ui },
                    x:       { duration: shaking ? 0.42 : 0 },
                  }}
                  whileHover={live && !reduced ? { y: -2 } : undefined}
                  whileTap={live && !reduced ? { scale: 0.97 } : undefined}
                  // Not `disabled`: a disabled button fires no events at all,
                  // so a tap on a locked card would give zero feedback. It
                  // stays focusable/clickable and reports its locked state.
                  aria-disabled={!live}
                  title={live ? undefined : `${t.label} isn't available yet`}
                  onClick={() => {
                    if (live) { setEventType(t.id); setPrompt(""); }
                    else nudgeLocked(t.id);
                  }}
                  style={selected ? { viewTransitionName: "event-hero" } : undefined}
                  className={"relative flex items-center gap-2 px-4 py-3 rounded-xl border-2 " +
                             "transition-colors text-left " +
                             (selected
                               ? "border-brand-500 bg-brand-500/10"
                               : live
                                 ? "border-slate-200 hover:border-slate-300 bg-white"
                                 : "border-slate-100 bg-slate-50 cursor-not-allowed " +
                                   "saturate-[.35] opacity-60")}>
                  <span className={"text-2xl " + (live ? "" : "grayscale")}>{t.emoji}</span>
                  <span className={"font-medium " + (live ? "" : "text-slate-500")}>
                    {t.label}
                  </span>
                  {!live && (
                    <span className="absolute top-1 right-2 flex items-center gap-1
                                     text-[10px] font-semibold uppercase tracking-wide
                                     text-slate-400">
                      <Lock className="w-2.5 h-2.5" />
                      soon
                    </span>
                  )}
                </motion.button>
              );
            })}
          </div>
        </div>

        {eventType === "custom" ? (
          <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-4 py-3">
            <div className="text-sm font-medium text-slate-700">✨ Nothing to set up here</div>
            <p className="mt-1 text-xs text-slate-500">
              Custom mode starts with no fixed moments at all — once you're in, just tell
              SnapAI what to capture by typing or speaking (e.g. "capture me when I pose",
              "capture when the lights turn on"). It'll confirm, guide you, watch, and
              capture the instant it happens.
            </p>
          </div>
        ) : eventType === "smart_event" ? (
          <div className="rounded-xl border border-brand-500/20 bg-brand-500/5 px-4 py-3">
            <div className="text-sm font-medium text-slate-700">🎇 Just name the occasion</div>
            <p className="mt-1 text-xs text-slate-500">
              No fixed moment list — put the occasion in "Event name" above (e.g. "Diwali",
              "Rakhi", "Janmashtami", "a retirement party") and SnapAI will work out what's
              worth capturing on its own, right when the event starts. No pre-training needed
              per event.
            </p>
          </div>
        ) : (
        <div>
          <label className="text-sm font-medium text-slate-700">
            Moments to watch for
          </label>
          <p className="mt-1 text-xs text-slate-500">
            These are pre-loaded for a {EVENT_TYPES.find((t) => t.id === eventType)?.label ?? "event"} and
            captured automatically. Add anything extra below — your additions
            are watched <em>on top of</em> these (or change mid-event by voice).
          </p>
          {/* Read-only default moments — used but not editable, so they can't
              be accidentally deleted. */}
          <div className="mt-2 flex flex-wrap gap-1.5">
            {(EVENT_TYPES.find((t) => t.id === eventType)?.defaults ?? []).map((m) => (
              <span key={m} className="px-2.5 py-1 rounded-full bg-brand-500/10
                                       border border-brand-500/20 text-brand-700
                                       text-xs">
                {m}
              </span>
            ))}
          </div>
          <textarea
            value={prompt} onChange={(e) => setPrompt(e.target.value)}
            rows={2}
            placeholder="Add extra moments (optional) — e.g. catch grandma laughing, the photobooth"
            className="mt-2 w-full px-4 py-3 rounded-xl border border-slate-200
                       focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20
                       outline-none transition resize-none placeholder:text-slate-400" />
        </div>
        )}

        {err && (
          <div className="p-3 rounded-lg bg-rose-50 border border-rose-200
                          text-rose-700 text-sm">
            {err}
          </div>
        )}

        <button onClick={start} disabled={busy}
                className="btn-primary w-full justify-center text-base py-4">
          <Camera className="w-5 h-5" />
          {busy ? "Starting…" : "Start capturing"}
        </button>

        <p className="text-center text-xs text-slate-500">
          We use your browser's camera — nothing is uploaded outside SnapAI.
        </p>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────────────────
// Developer-mode setup — denser, more options visible
// ─────────────────────────────────────────────────────────────────────────
function SetupDev(p: any) {
  return (
    <div className="max-w-2xl mx-auto p-5">
      <div className="card-dev space-y-4">
        <div>
          <h2 className="text-lg font-mono mb-1">/sessions POST</h2>
          <p className="text-xs text-slate-400">
            Create a session via REST. WS connects on /ws/&lt;sid&gt;.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3">
          <input
            type="text" placeholder="event_name"
            value={p.eventName} onChange={(e: any) => p.setEventName(e.target.value)}
            className="bg-ink-900 border border-ink-700 rounded px-3 py-2
                       font-mono text-sm text-slate-200 col-span-2" />
          <select
            value={p.eventType} onChange={(e: any) => p.setEventType(e.target.value)}
            className="bg-ink-900 border border-ink-700 rounded px-3 py-2
                       font-mono text-sm text-slate-200 col-span-2">
            {EVENT_TYPES.map((t) => (
              <option key={t.id} value={t.id}>{t.id}</option>
            ))}
          </select>
          <textarea
            placeholder="prompt"
            value={p.prompt} onChange={(e: any) => p.setPrompt(e.target.value)}
            className="bg-ink-900 border border-ink-700 rounded px-3 py-2
                       font-mono text-sm text-slate-200 col-span-2 h-24" />
        </div>

        {p.err && (
          <div className="p-2 rounded border border-rose-700 bg-rose-900/40
                          text-rose-300 text-xs font-mono">{p.err}</div>
        )}

        <button onClick={p.start} disabled={p.busy}
          className="w-full bg-brand-600 hover:bg-brand-700 text-white
                     font-mono py-2 rounded transition disabled:opacity-50">
          {p.busy ? "POST /sessions ..." : "POST /sessions"}
        </button>
      </div>
    </div>
  );
}
