import type { EnsembleDecision } from "@/api/types";

const COLORS: Record<string, string> = {
  clip:      "bg-violet-500",
  quality:   "bg-blue-500",
  aesthetic: "bg-emerald-500",
  emotion:   "bg-orange-500",
  gaze:      "bg-rose-500",
  kept:      "bg-cyan-500",
  face:      "bg-slate-400",
  predictor: "bg-amber-500",
  learner:   "bg-fuchsia-500",
};
const ORDER = ["clip","quality","aesthetic","emotion","gaze","kept","face","predictor","learner"];

export function EnsembleBars({ engine }: { engine: EnsembleDecision | null }) {
  if (!engine) return (
    <div className="text-xs text-slate-500 italic">
      Ensemble offline (waiting for first frame)
    </div>
  );
  return (
    <div className="space-y-1.5">
      {ORDER.map((k) => {
        const v = engine.contributions[k] ?? 0;
        const w = engine.weights[k] ?? 0;
        return (
          <div key={k} className="flex items-center gap-2 text-xs font-mono">
            <span className="w-20 text-slate-400">{k}</span>
            <div className="flex-1 h-1.5 rounded-full bg-ink-700/50 overflow-hidden">
              <div className={"h-full transition-all duration-200 " + (COLORS[k] ?? "bg-slate-500")}
                   style={{ width: `${Math.round(v * 100)}%` }} />
            </div>
            <span className="w-10 text-right tabular-nums text-slate-200">
              {(v * 100).toFixed(0)}
            </span>
            <span className="w-12 text-right text-slate-500 text-[10px]">
              ×{w.toFixed(2)}
            </span>
          </div>
        );
      })}
      <div className="mt-2 pt-2 border-t border-ink-700 flex justify-between
                      text-xs font-mono">
        <span>final {(engine.final_score * 100).toFixed(0)} / thr {(engine.threshold * 100).toFixed(0)}</span>
        <span className={engine.triggered ? "text-emerald-400" : "text-slate-500"}>
          {engine.triggered ? "▶ TRIGGERED" : "— wait"}
        </span>
      </div>
      {engine.reasons.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {engine.reasons.map((r) => (
            <span key={r} className="px-1.5 py-0.5 bg-ink-700 rounded
                                     text-[10px] font-mono">{r}</span>
          ))}
        </div>
      )}
    </div>
  );
}
