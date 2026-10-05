// frontend-react/src/components/ModelActivity.tsx
// Live "how hard is each model working" meter for Developer mode.
// Every model is ALWAYS listed (bars sit at 0 until frames flow), so the panel
// is never blank. Each row: icon, model name, what it detected, a thick
// gradient bar that animates every frame, and a big % — plus a glow when the
// model is firing hard. When a face appears you SEE face/gaze/emotion spike;
// when CLIP locks a moment its bar jumps; a cheer lights the audio bar.

import type { FrameResult } from "@/api/types";

const clamp = (x: number) => Math.max(0, Math.min(1, x || 0));
const pct   = (x: number) => Math.round(clamp(x) * 100);
const trendVal = (t?: string) =>
  t === "peak" ? 1 : t === "rising" ? 0.7 : t === "falling" ? 0.2 : 0.4;

interface Row { k: string; icon: string; grad: string;
                v: number; info: string; group: string; }

function rowsFor(f: FrameResult | null): Row[] {
  const audioEv = f?.audio?.events ?? {};
  const loud = Object.entries(audioEv).sort((a, b) => b[1] - a[1])[0];
  const stage = f?.sequence?.stage;
  return [
    { group: "Perception", k: "Face",      icon: "👤", grad: "from-slate-400 to-slate-300",
      v: clamp((f?.face_detection?.face_count ?? 0) / 5),
      info: f ? `${f.face_detection?.face_count ?? 0} face · ${f.face_detection?.backend ?? "?"}` : "idle" },
    { group: "Perception", k: "Gaze",      icon: "👁", grad: "from-rose-500 to-rose-400",
      v: clamp(f?.gaze?.gaze_ratio ?? 0),
      info: f ? `${f.gaze?.faces_looking ?? 0}/${f.gaze?.total_faces ?? 0} at camera` : "idle" },
    { group: "Perception", k: "Emotion",   icon: "😊", grad: "from-orange-500 to-amber-400",
      v: clamp(f?.emotion?.score ?? 0),
      info: f ? `${f.emotion?.dominant ?? "—"} ${pct(f.emotion?.score ?? 0)}%` : "idle" },
    { group: "Perception", k: "Quality",   icon: "🎯", grad: "from-blue-500 to-sky-400",
      v: clamp(f?.analysis?.total ?? 0),
      info: f ? `shot ${pct(f.analysis?.total ?? 0)}%` : "idle" },
    { group: "Perception", k: "Aesthetic", icon: "🖼", grad: "from-emerald-500 to-green-400",
      v: clamp(f?.nima?.score ?? 0),
      info: f ? `NIMA ${(f.nima?.score ?? 0).toFixed(2)}` : "idle" },
    { group: "Understanding", k: "CLIP",   icon: "🧠", grad: "from-violet-500 to-purple-400",
      v: clamp(f?.clip?.best_score ?? 0),
      info: f ? `${f.clip?.best ?? "—"} ${pct(f.clip?.best_score ?? 0)}% · ${f.clip?.infer_ms ?? "?"}ms` : "idle" },
    { group: "Understanding", k: "Audio",  icon: "🔊", grad: "from-cyan-500 to-teal-400",
      v: clamp((f?.audio?.boost ?? 0) / 0.2),
      info: f ? (loud && loud[1] > 0.05 ? `${loud[0]} ${pct(loud[1])}%` : "quiet / no mic") : "idle" },
    { group: "Understanding", k: "Sequence", icon: "🎬", grad: "from-amber-500 to-yellow-400",
      v: stage != null ? clamp(stage / 20) : 0,
      info: f ? (stage != null ? `stage ${stage} · ${f.sequence?.event || "—"}` : "off-prompt") : "idle" },
    { group: "Timing", k: "Predictor",     icon: "📈", grad: "from-fuchsia-500 to-pink-400",
      v: f ? trendVal(f.prediction?.trend) : 0,
      info: f ? (f.prediction?.trend ?? "?") : "idle" },
  ];
}

export function ModelActivity({ frame }: { frame: FrameResult | null }) {
  const rows = rowsFor(frame);
  const groups = ["Perception", "Understanding", "Timing"];
  return (
    <div className="space-y-3">
      {!frame && (
        <div className="text-[11px] text-slate-500 italic">
          All models listed below · bars move once the camera/feed is live.
        </div>
      )}
      {groups.map((g) => (
        <div key={g}>
          <div className="text-[10px] uppercase tracking-widest text-slate-500 mb-1.5">{g}</div>
          <div className="space-y-1.5">
            {rows.filter((r) => r.group === g).map((r) => {
              const hot = r.v >= 0.6;
              return (
                <div key={r.k}
                     className={"flex items-center gap-2.5 rounded-lg px-2 py-1.5 transition " +
                                (hot ? "bg-white/5 ring-1 ring-white/10" : "")}>
                  <span className="text-base w-6 text-center">{r.icon}</span>
                  <div className="w-20">
                    <div className="text-xs font-semibold text-slate-200 leading-tight">{r.k}</div>
                    <div className="text-[10px] text-slate-500 truncate leading-tight">{r.info}</div>
                  </div>
                  <div className="flex-1 h-2.5 rounded-full bg-ink-700/60 overflow-hidden">
                    <div className={"h-full rounded-full bg-gradient-to-r transition-all duration-150 " + r.grad}
                         style={{ width: `${pct(r.v)}%` }} />
                  </div>
                  <span className={"w-9 text-right text-sm font-mono tabular-nums " +
                                   (hot ? "text-white" : "text-slate-400")}>
                    {pct(r.v)}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
