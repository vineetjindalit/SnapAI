// frontend-react/src/components/LiveStatus.tsx
// The "what's happening right now" banner for Developer mode. Shows the feed
// source (LIVE camera vs ▶ analyzing an uploaded clip + its timestamp), the
// current detected moment, the priority tier the models clubbed into, whether
// the frame is being CAPTURED, and the running score vs threshold.

import type { FrameResult } from "@/api/types";

const TIERS: Record<string, { label: string; cls: string }> = {
  critical: { label: "CRITICAL", cls: "bg-rose-500/20 text-rose-300 ring-rose-500/40" },
  high:     { label: "HIGH",     cls: "bg-orange-500/20 text-orange-300 ring-orange-500/40" },
  elevated: { label: "ELEVATED", cls: "bg-amber-500/20 text-amber-300 ring-amber-500/40" },
  normal:   { label: "NORMAL",   cls: "bg-slate-500/20 text-slate-300 ring-slate-500/30" },
};

export function LiveStatus({ frame, captures }:
                           { frame: FrameResult | null; captures: number }) {
  const f = frame;
  const upload = f?.source === "upload";
  const eng = f?.engine ?? null;
  const tier = TIERS[(eng?.priority_tier || "normal").toLowerCase()] ?? TIERS.normal;
  return (
    // Visual cleanup only — this stays a DevMode telemetry banner, with the
    // same props, same fields and same logic. The user-facing connection
    // indicator is components/ConnectionStatus.tsx.
    <div className="flex items-center gap-3 flex-wrap rounded-xl border border-ink-700
                    bg-ink-900/60 px-4 py-3 ring-1 ring-white/5 shadow-sm">
      <span className={"inline-flex items-center gap-1.5 px-3 py-1 rounded-full " +
                       "text-xs font-mono " +
                       (upload ? "bg-amber-500/15 text-amber-300"
                               : "bg-emerald-500/15 text-emerald-300")}>
        <span className={"w-2 h-2 rounded-full " +
                         (upload ? "bg-amber-400" : "bg-emerald-400 animate-pulse")} />
        {upload ? `▶ ANALYZING UPLOAD · t=${(f?.media_ts ?? 0).toFixed(1)}s`
                : (f ? "● LIVE camera" : "idle — start camera or upload")}
      </span>

      <div className="flex-1 min-w-[160px]">
        <div className="text-[10px] uppercase tracking-widest text-slate-500">current moment</div>
        <div className="text-xl font-semibold text-white capitalize leading-tight">
          {(f?.moment?.detected_moment ?? "—").replace(/_/g, " ")}
          <span className="ml-2 text-sm text-slate-400 font-mono">
            {f ? Math.round((f.moment?.confidence ?? 0) * 100) + "%" : ""}
          </span>
        </div>
      </div>

      <span className={"px-3 py-1 rounded-lg text-xs font-bold ring-1 " + tier.cls}>
        {tier.label}
      </span>
      <span className={"px-3 py-1 rounded-lg text-xs font-mono " +
                       (eng?.triggered ? "bg-emerald-500/20 text-emerald-300"
                                       : "bg-ink-700 text-slate-400")}>
        {eng?.triggered ? "▶ CAPTURE" : "— watching"}
      </span>

      <div className="text-right font-mono text-xs text-slate-400 leading-tight">
        <div>captures <span className="text-white">{captures}</span></div>
        <div>score {eng ? Math.round(eng.final_score * 100) : 0}
             /{eng ? Math.round(eng.threshold * 100) : 0}</div>
      </div>
    </div>
  );
}
