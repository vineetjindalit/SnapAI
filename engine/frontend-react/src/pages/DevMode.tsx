import { useEffect, useRef, useState } from "react";
import { useSession } from "@/store/session";
import { useLiveStream } from "@/hooks/useLiveStream";
import { useReplay }     from "@/hooks/useReplay";
import { CameraView }    from "@/components/CameraView";
import { PromptControl } from "@/components/PromptControl";
import { EnsembleBars }  from "@/components/EnsembleBars";
import { ModelActivity } from "@/components/ModelActivity";
import { LiveStatus }    from "@/components/LiveStatus";
import { ScoreTimeline } from "@/components/ScoreTimeline";
import { VideoUpload }   from "@/components/VideoUpload";
import { Sessions } from "@/api/client";

export function DevMode() {
  const session  = useSession((s) => s);
  const latest   = session.latestFrame;
  const health   = session.health;
  const captures = session.captures;
  const timeline = session.timeline;
  const promptHistory = session.promptHistory;
  const { camera, ws } = useLiveStream({ fps: 5 });
  const { replay, stop: stopReplay, draining, pendingCount, error: replayError } = useReplay(ws, 5);
  const replaying = useSession((s) => s.replaying);

  // Live ⇄ Upload — mutually exclusive, same as User mode. Switching ENDS the
  // current session and starts a fresh, empty one, so live and upload captures
  // are never in the same session (the backend also hard-locks the source).
  const [feedMode, setFeedMode] = useState<"live" | "upload">("live");
  const [switching, setSwitching] = useState(false);
  // Replay-as-live runs in its OWN fresh session with the camera UNMOUNTED and
  // the camera+mic GATED OFF the whole time (replayActive, in the store), so a
  // replayed clip and the real camera/mic can never share a session / mix
  // captures. `replaying` (also store) = the clip is actively streaming now.
  const replayActive    = useSession((s) => s.replayActive);
  const setReplayActive = useSession((s) => s.setReplayActive);
  useEffect(() => { if (feedMode === "upload") camera.stop(); }, [feedMode, camera.stop]);

  const freshSession = async () => {
    const prev = useSession.getState().sid;
    if (prev) Sessions.end(prev).catch(() => {});
    const r = await Sessions.create(session.eventName, session.eventType, session.prompt);
    useSession.getState().setSession(r.session_id, r.event_name, r.event_type, r.prompt);
  };

  const switchMode = async (target: "live" | "upload") => {
    if (target === feedMode && !replayActive) return;   // re-clicking LIVE exits a replay
    setSwitching(true);
    stopReplay(); setReplayActive(false);
    if (target !== "live") camera.stop();
    try {
      await freshSession();                             // each switch = a clean, empty session
      setFeedMode(target);
    } catch { /* keep current mode if create fails */ }
    finally { setSwitching(false); }
  };

  // Start a replay: stop+unmount the camera, open a FRESH empty session, then
  // stream the clip into it. Nothing the camera saw can leak into the clip's album.
  const startReplay = async (file: File) => {
    setSwitching(true);
    setReplayActive(true);           // FIRST: gate camera+mic immediately, before anything streams
    camera.stop();
    try {
      await freshSession();
      setFeedMode("live");           // context is "live", but camera stays unmounted while replayActive
      replay(file);
    } catch { /* ignore */ }
    finally { setSwitching(false); }
  };

  // Live event log — capped to 200 entries
  const [log, setLog] = useState<{ t: number; lvl: string; msg: string }[]>([]);
  useEffect(() => {
    if (!latest) return;
    const now = Date.now();
    setLog((prev) => {
      const lines: typeof prev = [];
      if (latest.captured) {
        lines.push({ t: now, lvl: "ok",
          msg: `CAPTURE ${latest.moment.detected_moment} score=${latest.analysis?.total?.toFixed(2) ?? "?"} reason=${latest.capture_reason}` });
      }
      const eng = latest.engine;
      if (eng && eng.triggered) {
        lines.push({ t: now, lvl: "info",
          msg: `engine ${eng.moment_class} final=${(eng.final_score*100).toFixed(0)} thr=${(eng.threshold*100).toFixed(0)} reasons=[${eng.reasons.join(",")}]` });
      }
      if (latest.new_discoveries?.length) {
        lines.push({ t: now, lvl: "warn",
          msg: `discovered cluster — ${latest.new_discoveries[0]?.cid}` });
      }
      return [...lines.reverse(), ...prev].slice(0, 200);
    });
  }, [latest]);

  const endSession = async () => {
    if (!session.sid) return;
    await Sessions.end(session.sid).catch(() => {});
    session.clearSession();
  };

  // Generate + SAVE the curated album for the current session. The backend
  // curates (relevancy + dedup + sequence + VLM/classifier tags) and writes it
  // to  snappy_final/albums/<event_name>/  on disk. Works for a live OR a
  // replayed session — whichever owns the current sid.
  const [albumBusy, setAlbumBusy] = useState(false);
  const [albumMsg, setAlbumMsg] = useState<string | null>(null);
  const genAlbum = async () => {
    const sid = useSession.getState().sid;
    if (!sid || albumBusy) return;
    // Stop the camera FIRST — see UserMode.tsx's generateAlbum for why: the
    // album request can run 30-90s+ (VLM model load + per-photo scoring) and
    // used to run entirely with the live camera still streaming, fighting
    // curation for the same GPU and slowing both down.
    if (feedMode === "live") camera.stop();
    setAlbumBusy(true); setAlbumMsg("curating album… (VLM tagging can take a few sec/photo)");
    try {
      const a: any = await Sessions.album(sid);
      setAlbumMsg(`✓ ${a.total_selected}/${a.total_captured} kept → albums/${a.event_name}/`);
    } catch (e: any) {
      setAlbumMsg(`album failed: ${e?.message ?? "is there any capture yet?"}`);
    } finally { setAlbumBusy(false); }
  };

  return (
    <div className="grid grid-cols-12 gap-3 p-3 text-slate-200">

      {/* ── TOP: source toggle + live status banner (full width) ──────── */}
      <div className="col-span-12 flex items-center gap-3">
        <div className="flex rounded-lg bg-ink-800 border border-ink-700 p-0.5
                        text-xs font-mono shrink-0">
          <button onClick={() => switchMode("live")} disabled={switching}
            className={"px-3 py-1 rounded transition disabled:opacity-50 " +
              (feedMode === "live" ? "bg-ink-700 text-emerald-400"
                                   : "text-slate-400 hover:text-slate-200")}>
            ● LIVE
          </button>
          <button onClick={() => switchMode("upload")} disabled={switching}
            className={"px-3 py-1 rounded transition disabled:opacity-50 " +
              (feedMode === "upload" ? "bg-ink-700 text-violet-400"
                                     : "text-slate-400 hover:text-slate-200")}>
            ▶ UPLOAD
          </button>
        </div>
        <div className="flex-1 min-w-0">
          <LiveStatus frame={latest ?? null} captures={session.totalCaptures} />
        </div>
      </div>

      {/* ── LEFT: Camera + ensemble ────────────────────────────────── */}
      <div className="col-span-12 lg:col-span-7 space-y-3">
        <div className="rounded-lg overflow-hidden border border-ink-700 bg-black">
          <div className="relative">
            {replayActive ? (
              // REPLAY owns the view — camera is unmounted, captures land in this
              // clip's OWN fresh session, so nothing merges with the live camera.
              <div className="aspect-video grid place-items-center text-center
                              text-slate-300 text-xs font-mono px-6">
                <div>
                  <div className={replayError ? "text-rose-400 mb-1" : "text-violet-300 mb-1"}>
                    {replayError ? "⚠ REPLAY FAILED"
                      : replaying ? "▶ REPLAYING CLIP AS LIVE"
                      : draining ? `⏳ FINISHING — scoring last ${pendingCount || ""} frame(s)…`
                      : "■ REPLAY FINISHED"}
                  </div>
                  <div className="text-slate-400 mb-3 max-w-md">
                    {replayError
                      ? replayError
                      : replaying
                      ? "Streaming the clip through the live pipeline — camera is OFF. Captures land in this clip's own session."
                      : draining
                      ? "All frames sent — waiting for the backend to finish scoring the last few before the album is final."
                      : `${session.totalCaptures} capture(s) from this clip — its own session, not mixed with the camera.`}
                  </div>
                  <div className="flex gap-2 justify-center">
                    {replaying ? (
                      <button onClick={stopReplay}
                        className="px-2 py-1 rounded bg-rose-600 text-white hover:bg-rose-500">
                        ■ stop replay
                      </button>
                    ) : (
                      <>
                        <label className="px-2 py-1 rounded bg-violet-600/90 text-white
                                          cursor-pointer hover:bg-violet-500">
                          ▶ replay another
                          <input type="file" accept="video/*" className="hidden"
                            onChange={(e) => { const f = e.target.files?.[0];
                              if (f) startReplay(f); e.currentTarget.value = ""; }} />
                        </label>
                        <button onClick={() => switchMode("live")}
                          className="px-2 py-1 rounded bg-ink-700 text-emerald-400 hover:text-emerald-300">
                          ● go live
                        </button>
                      </>
                    )}
                  </div>
                </div>
              </div>
            ) : feedMode === "live" ? (
              <>
                <CameraView camera={camera}
                            flash={!!latest?.captured}
                            faceBoxes={latest?.face_detection.boxes ?? []} />
                <div className="absolute top-2 left-2 flex gap-1.5 text-[10px] font-mono">
                  <Pill val={`fps ~${latest?.frame_id ? Math.min(30, Math.round(session.totalFrames / Math.max(1, (Date.now() - (latest?.analysis?.timestamp ?? Date.now()/1000)*1000)/1000))) : 0}`} />
                  <Pill val={`face ${latest?.face_detection.backend ?? "?"}`} />
                  <Pill val={`device ${health?.compute.device ?? "?"}`} />
                </div>
                {/* Start a replay: opens a fresh isolated session, camera unmounts. */}
                <label className="absolute top-2 right-2 text-[10px] font-mono px-2 py-1
                                  rounded bg-violet-600/90 text-white cursor-pointer hover:bg-violet-500">
                  ▶ replay clip as live
                  <input type="file" accept="video/*" className="hidden"
                    onChange={(e) => { const f = e.target.files?.[0];
                      if (f) startReplay(f); e.currentTarget.value = ""; }} />
                </label>
              </>
            ) : (
              <div className="aspect-video grid place-items-center text-center
                              text-slate-400 text-xs font-mono px-6">
                <div>
                  <div className="text-violet-400 mb-1">▶ ANALYZING UPLOAD</div>
                  Live camera is stopped — the panels below show this video's
                  per-frame analysis.
                </div>
              </div>
            )}
          </div>
          <div className="p-3 border-t border-ink-700">
            <PromptControl ws={ws} variant="developer" />
          </div>
        </div>

        <div className="card-dev">
          <Section title="Models · live activity (each model's signal this frame)" />
          <ModelActivity frame={latest ?? null} />
        </div>

        <div className="card-dev">
          <Section title="Ensemble · how the models club into one decision" />
          <EnsembleBars engine={latest?.engine ?? null} />
        </div>

        <div className="card-dev">
          <Section title="Quality timeline (last 120 frames)" />
          <ScoreTimeline values={timeline} />
          <div className="grid grid-cols-3 gap-2 mt-2 text-xs font-mono">
            <KV k="frame_id"   v={latest?.frame_id ?? 0} />
            <KV k="captures"   v={session.totalCaptures} />
            <KV k="avg_smile"  v={(latest?.emotion.avg_smile ?? 0).toFixed(2)} />
          </div>
        </div>

        <div className="card-dev">
          <Section title="Event log" />
          <div className="max-h-56 overflow-y-auto space-y-0.5">
            {log.map((l, i) => (
              <div key={i} className={"text-xs font-mono py-0.5 " +
                  (l.lvl === "ok"   ? "text-emerald-400" :
                   l.lvl === "warn" ? "text-amber-400"   :
                                       "text-slate-300")}>
                <span className="text-slate-500">
                  {new Date(l.t).toLocaleTimeString()}
                </span>{" "}
                <span className="text-slate-500">[{l.lvl}]</span> {l.msg}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── RIGHT: Models + per-frame JSON-ish breakdown ───────────── */}
      <div className="col-span-12 lg:col-span-5 space-y-3">

        <div className="card-dev">
          <Section title="/health · model zoo" right={
            <span className="text-[10px] font-mono text-slate-500">
              {health?.version ?? "—"}
            </span>
          } />
          <div className="space-y-1 text-xs font-mono">
            <ModelLine name="face"     ok={true}                    detail={health?.models.face.backend} />
            <ModelLine name="gaze"     ok={true}                    detail={health?.models.gaze?.backend ?? "iris"} />
            <ModelLine name="clip"     ok={!!health?.models.clip.available}      detail={health?.models.clip.trained_centroids ? "trained" : "zero-shot"} />
            <ModelLine name="nima"     ok={!!health?.models.nima.available}      detail={health?.models.nima.error ?? "ready"} />
            <ModelLine name="hsemotion" ok={!!health?.models.hsemotion.available} detail={health?.models.hsemotion.error ?? "ready"} />
            <ModelLine name="whisper"  ok={!!health?.models.whisper?.available}  detail={health?.models.whisper?.available ? "ready" : "browser-only"} />
          </div>
        </div>

        <div className="card-dev">
          <Section title="Latest frame · raw signals" />
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
            <KV k="moment"     v={latest?.moment.detected_moment ?? "—"} />
            <KV k="conf"       v={(latest?.moment.confidence ?? 0).toFixed(2)} />
            <KV k="quality"    v={(latest?.analysis?.total ?? 0).toFixed(2)} />
            <KV k="nima"       v={(latest?.nima?.score ?? -1).toFixed(2)} />
            <KV k="emotion"    v={`${latest?.emotion.dominant ?? "?"} ${(latest?.emotion.score ?? 0).toFixed(2)}`} />
            <KV k="gaze"       v={`${latest?.gaze.faces_looking ?? 0}/${latest?.gaze.total_faces ?? 0}`} />
            <KV k="trend"      v={latest?.prediction.trend ?? "?"} />
            <KV k="clip_best"  v={latest?.clip.best ?? "—"} />
            <KV k="clip_score" v={(latest?.clip.best_score ?? 0).toFixed(2)} />
            <KV k="clip_ms"    v={`${latest?.clip.infer_ms ?? "—"}`} />
          </div>
        </div>

        <div className="card-dev">
          <Section title="CLIP per-prompt similarities" />
          <div className="space-y-1">
            {latest && Object.entries(latest.moment.all_scores ?? {})
              .sort((a, b) => b[1] - a[1])
              .slice(0, 8)
              .map(([k, v]) => (
                <div key={k} className="flex items-center gap-2 text-xs font-mono">
                  <span className="w-32 text-slate-400 truncate">{k.replace(/_/g," ")}</span>
                  <div className="flex-1 h-1.5 bg-ink-700/50 rounded">
                    <div className="h-full bg-violet-500 rounded"
                         style={{ width: `${Math.round(v * 100)}%` }} />
                  </div>
                  <span className="w-10 text-right text-slate-200">{(v*100).toFixed(0)}</span>
                </div>
              ))}
          </div>
        </div>

        <div className="card-dev">
          <Section title="Active moment classes" />
          <div className="flex flex-wrap gap-1">
            {(latest?.active_moments ?? []).map((m) => (
              <span key={m} className="px-2 py-0.5 bg-ink-700 rounded
                                       text-[11px] font-mono capitalize">
                {m.replace(/_/g," ")}
              </span>
            ))}
          </div>
        </div>

        <div className="card-dev">
          <Section title="Prompt history" />
          {promptHistory.length === 0 ? (
            <div className="text-xs text-slate-500 italic">No prompt updates yet.</div>
          ) : (
            <div className="space-y-1 max-h-40 overflow-y-auto">
              {promptHistory.map((p, i) => (
                <div key={i} className="text-xs font-mono">
                  <span className="text-slate-500">
                    {new Date(p.ts * 1000).toLocaleTimeString()}
                  </span>{" "}
                  <span className={
                    p.intent === "add"     ? "text-emerald-400" :
                    p.intent === "remove"  ? "text-rose-400"   :
                    p.intent === "replace" ? "text-amber-400"  :
                                              "text-slate-500"
                  }>{p.intent}</span>{" "}
                  <span className="text-slate-200">"{p.text}"</span>{" "}
                  <span className="text-slate-500">
                    → {p.matched.length ? p.matched.join(",") : "—"}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Video upload — only in UPLOAD mode (mutually exclusive with live) */}
        {feedMode === "upload" && <VideoUpload dark />}

        <div className="card-dev">
          <Section title="Recent captures · why each fired" right={
            <span className="text-[10px] font-mono text-slate-500">
              {captures.length}
            </span>
          } />
          <div className="space-y-1 max-h-80 overflow-auto">
            {captures.length === 0 && (
              <div className="text-[11px] font-mono text-slate-500 py-2">
                No captures yet — replay a clip or start the camera.
              </div>
            )}
            {captures.slice(0, 30).map((c) => (
              <div key={c.id} className="flex items-center gap-2 bg-ink-800/60 rounded p-1">
                <img src={`data:image/jpeg;base64,${c.thumbB64}`} alt=""
                     className="w-12 h-12 object-cover rounded shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-1.5">
                    <span className="text-[11px] font-mono text-violet-300 truncate">{c.moment}</span>
                    <TierBadge tier={c.tier} />
                    <span className="text-[9px] font-mono text-slate-500 ml-auto">{Math.round(c.score * 100)}</span>
                  </div>
                  <div className="text-[9px] font-mono text-slate-400 truncate" title={(c.reasons ?? []).join(" · ")}>
                    {(c.reasons && c.reasons.length) ? c.reasons.join(" · ") : "—"}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card-dev">
          <Section title="Session" />
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs font-mono">
            <KV k="sid"        v={session.sid ?? "—"} />
            <KV k="event_type" v={session.eventType} />
            <KV k="prompt"     v={session.prompt} />
            <KV k="frames"     v={session.totalFrames} />
          </div>
          <button onClick={genAlbum} disabled={albumBusy || !session.sid}
            className="mt-3 w-full bg-violet-600 hover:bg-violet-500 disabled:opacity-50
                       text-white text-sm font-mono py-1.5 rounded transition">
            {albumBusy ? "curating…" : "🖼  Generate album → albums/ folder"}
          </button>
          {albumMsg && (
            <div className="mt-1.5 text-[11px] font-mono text-slate-300 break-all">{albumMsg}</div>
          )}
          <button onClick={endSession}
            className="mt-2 w-full bg-rose-700 hover:bg-rose-600 text-white
                       text-sm font-mono py-1.5 rounded transition">
            DELETE /sessions/{session.sid?.slice(0, 6)}…
          </button>
        </div>
      </div>
    </div>
  );
}


// ── tiny helpers ────────────────────────────────────────────────────────
function Section({ title, right }: { title: string; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-2">
      <div className="text-[10px] uppercase tracking-widest text-slate-400">
        {title}
      </div>
      {right}
    </div>
  );
}
function KV({ k, v }: { k: string; v: any }) {
  return (
    <div className="flex justify-between gap-2 truncate">
      <span className="text-slate-500">{k}</span>
      <span className="text-slate-200 truncate text-right">{String(v)}</span>
    </div>
  );
}
function Pill({ val }: { val: string }) {
  return (
    <span className="px-2 py-0.5 bg-black/70 text-white rounded backdrop-blur">
      {val}
    </span>
  );
}
function TierBadge({ tier }: { tier?: string }) {
  if (!tier) return null;
  const t = tier.toLowerCase();
  const cls = t.includes("critical") ? "bg-rose-600"
            : t.includes("high")     ? "bg-orange-600"
            : t.includes("elevated") ? "bg-amber-600"
            :                          "bg-slate-600";
  return <span className={`px-1 rounded text-[8px] font-mono text-white uppercase ${cls}`}>{t}</span>;
}
function ModelLine({ name, ok, detail }:
                   { name: string; ok: boolean; detail?: string | null }) {
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2">
        <span className={"w-1.5 h-1.5 rounded-full " +
                         (ok ? "bg-emerald-400" : "bg-slate-500")} />
        <span className="text-slate-300">{name}</span>
      </div>
      <span className="text-slate-500 truncate">{detail ?? ""}</span>
    </div>
  );
}
