// SetupCoach — the photographer's assistant, shown over the live camera.
//
// Live coaching line (directional: "turn LEFT", "step back", "tilt up") plus a
// bottom "Setup guide" button that opens the full written guide: a live
// checklist of what's right/wrong plus the placement brief for where to put the
// phone. A Write/Speak toggle chooses whether guidance is READ on screen or
// SPOKEN aloud with the browser's built-in voice (on-device, no network).
import { useEffect, useRef, useState } from "react";
import { BookOpen, ChevronDown, PencilLine, Volume2 } from "lucide-react";

export interface SetupCheck { label: string; ok: boolean; detail: string }
export interface SetupInfo {
  tips: string[];
  score: number;
  checklist?: SetupCheck[];
  guide?: string[];
}

interface Props { setup: SetupInfo | null | undefined }

type Mode = "write" | "speak";

export function SetupCoach({ setup }: Props) {
  const [mode, setMode] = useState<Mode>("write");
  const [open, setOpen] = useState(false);
  const lastSpokenRef = useRef<string>("");
  const lastSpokeAtRef = useRef<number>(0);
  const [shown, setShown] = useState<{ tip: string; score: number } | null>(null);
  const holdRef = useRef<number>(0);

  // Smooth the headline tip so it doesn't flicker frame to frame.
  useEffect(() => {
    if (!setup || !setup.tips?.length) return;
    const now = Date.now();
    const tip = setup.tips[0];
    if (shown?.tip !== tip && now - holdRef.current > 2500) {
      setShown({ tip, score: setup.score });
      holdRef.current = now;
    } else if (shown && shown.tip === tip && shown.score !== setup.score) {
      setShown({ tip, score: setup.score });
    }
  }, [setup, shown]);

  // Speak mode only: read each new tip aloud (throttled), stay quiet once good.
  useEffect(() => {
    if (!shown || mode !== "speak") return;
    const now = Date.now();
    if (shown.score >= 80 || shown.tip === lastSpokenRef.current) return;
    if (now - lastSpokeAtRef.current < 6000) return;
    try {
      window.speechSynthesis?.cancel();
      const u = new SpeechSynthesisUtterance(shown.tip);
      u.rate = 1.02;
      window.speechSynthesis?.speak(u);
      lastSpokenRef.current = shown.tip;
      lastSpokeAtRef.current = now;
    } catch { /* speech unsupported — the on-screen text still shows */ }
  }, [shown, mode]);

  const speakGuide = () => {
    if (!setup?.guide?.length) return;
    try {
      window.speechSynthesis?.cancel();
      const u = new SpeechSynthesisUtterance(
        "How to place your phone. " + setup.guide.join(" "));
      u.rate = 1.0;
      window.speechSynthesis?.speak(u);
    } catch { /* no speech support */ }
  };

  if (!shown) return null;
  const good = shown.score >= 80;
  const checks = setup?.checklist ?? [];

  return (
    // Sits ABOVE the prompt control (which owns the very bottom of the frame).
    <div className="absolute inset-x-0 bottom-16 sm:bottom-24 z-20 pointer-events-none">
      {/* Expanded written guide */}
      {open && (
        <div className="pointer-events-auto mx-2 mb-2 max-h-[52vh] overflow-y-auto
                        rounded-2xl bg-black/85 backdrop-blur text-white
                        p-4 text-sm shadow-2xl">
          <div className="flex items-center justify-between mb-3">
            <span className="font-semibold">Setup guide</span>
            <button onClick={() => setOpen(false)}
                    className="p-1 rounded-full hover:bg-white/10">
              <ChevronDown className="w-5 h-5" />
            </button>
          </div>

          {/* Live checklist — what's right and wrong right now */}
          {checks.length > 0 && (
            <div className="mb-4">
              <div className="text-[11px] uppercase tracking-wide text-white/50 mb-2">
                Right now
              </div>
              <ul className="space-y-1.5">
                {checks.map((c) => (
                  <li key={c.label} className="flex gap-2 items-start">
                    <span className={c.ok ? "text-emerald-400" : "text-amber-400"}>
                      {c.ok ? "✓" : "•"}
                    </span>
                    <span className="flex-1">
                      <span className="font-medium">{c.label}</span>
                      <span className="text-white/60"> — {c.detail}</span>
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Where to place the phone */}
          {!!setup?.guide?.length && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] uppercase tracking-wide text-white/50">
                  Where to place the phone
                </span>
                {mode === "speak" && (
                  <button onClick={speakGuide}
                          className="text-[11px] px-2 py-1 rounded-full bg-white/15">
                    Read aloud
                  </button>
                )}
              </div>
              <ol className="space-y-1.5 list-decimal list-inside text-white/90">
                {setup.guide.map((g, i) => <li key={i}>{g}</li>)}
              </ol>
            </div>
          )}
        </div>
      )}

      {/* Live coaching line */}
      <div className="mx-2 mb-2 flex items-center gap-2 pointer-events-none">
        <div className={"flex-1 flex items-center gap-3 rounded-xl px-3 py-2 text-sm " +
                        (good ? "bg-emerald-600/85 text-white"
                              : "bg-black/70 text-white backdrop-blur")}>
          <span className="shrink-0">{good ? "✓" : "📐"}</span>
          <span className="leading-snug">{shown.tip}</span>
          <span className="ml-auto shrink-0 text-xs opacity-80">{shown.score}</span>
        </div>
      </div>

      {/* Bottom bar: Setup guide button + Write/Speak toggle */}
      <div className="pointer-events-auto mx-2 mb-3 flex items-center gap-2">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-2 px-3.5 py-2.5 rounded-full
                     bg-white/95 text-slate-900 text-sm font-medium
                     shadow-lg active:scale-95 transition">
          <BookOpen className="w-4 h-4" />
          Setup guide
        </button>

        <div className="ml-auto inline-flex p-1 rounded-full bg-black/60 backdrop-blur">
          <button
            onClick={() => { setMode("write"); window.speechSynthesis?.cancel(); }}
            title="Show guidance as text"
            className={"flex items-center gap-1.5 px-3.5 py-2 rounded-full text-sm transition min-h-[40px] " +
                       (mode === "write" ? "bg-white text-slate-900" : "text-white/70")}>
            <PencilLine className="w-4 h-4" /> Write
          </button>
          <button
            onClick={() => setMode("speak")}
            title="Speak guidance aloud"
            className={"flex items-center gap-1.5 px-3.5 py-2 rounded-full text-sm transition min-h-[40px] " +
                       (mode === "speak" ? "bg-white text-slate-900" : "text-white/70")}>
            <Volume2 className="w-4 h-4" /> Speak
          </button>
        </div>
      </div>
    </div>
  );
}
