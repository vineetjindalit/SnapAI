import { useEffect, useState } from "react";
import { AnimatePresence, LayoutGroup, motion } from "motion/react";
import { Heart, X, Camera, CheckCircle2, Images, Loader2,
         Radio, FileVideo, Download } from "lucide-react";
import { useSession } from "@/store/session";
import { useLiveStream } from "@/hooks/useLiveStream";
import { useWsStatus, useReceivingFrames } from "@/hooks/useWsStatus";
import { ConnectionStatus } from "@/components/ConnectionStatus";
import { PhotoLightbox, type LightboxPhoto } from "@/components/PhotoLightbox";
import { useReducedMotion, withViewTransition, SPRING } from "@/lib/motionUtils";
import { CameraView } from "@/components/CameraView";
import { SetupCoach } from "@/components/SetupCoach";
import { ConversationalGuide } from "@/components/ConversationalGuide";
import { PromptControl } from "@/components/PromptControl";
import { VideoUpload } from "@/components/VideoUpload";
import { AlbumReview } from "@/components/AlbumReview";
import { Sessions } from "@/api/client";
import { downloadAlbumZip } from "@/utils/zip";
import { savePhotos, isMobile } from "@/lib/savePhotos";

export function UserMode() {
  const session  = useSession((s) => s);
  const captures = session.captures;
  const total    = session.totalCaptures;
  const latest   = session.latestFrame;
  const { camera, ws } = useLiveStream({ fps: 5 });
  const wsStatus  = useWsStatus(ws);
  const receiving = useReceivingFrames(wsStatus);
  const reduced   = useReducedMotion();
  const [lightbox, setLightbox] = useState<LightboxPhoto | null>(null);
  const [albumPhotos, setAlbumPhotos] = useState<any[] | null>(null);
  const [albumBusy, setAlbumBusy] = useState(false);
  const [albumLowConf, setAlbumLowConf] = useState(false);
  const [saveBusy, setSaveBusy]   = useState(false);
  const [phoneBusy, setPhoneBusy] = useState(false);
  const [phoneMsg, setPhoneMsg]   = useState("");
  const [ended, setEnded]         = useState(false);
  const [switching, setSwitching] = useState(false);
  const [modeErr, setModeErr]     = useState<string | null>(null);

  // Live and Upload are MUTUALLY EXCLUSIVE — and they must NEVER share a
  // session, or their captures merge into one album (the backend album is
  // built from a single per-session photo list, s.photos). So switching modes
  // ENDS the current session and starts a fresh, empty one: the other mode's
  // captures are discarded, exactly as intended.
  const [feedMode, setFeedMode] = useState<"live" | "upload">("live");
  useEffect(() => {
    if (feedMode === "upload") camera.stop();
    // Leaving upload mode re-mounts CameraView, which auto-starts the camera.
  }, [feedMode, camera.stop]);

  const switchMode = async (target: "live" | "upload") => {
    if (target === feedMode || switching || ended) return;
    setSwitching(true);
    if (target === "upload") camera.stop();   // kill live frames immediately
    try {
      setAlbumPhotos(null);
      const prev = useSession.getState().sid;
      if (prev) Sessions.end(prev).catch(() => {});   // abandon the current run
      const r = await Sessions.create(
        session.eventName, session.eventType, session.prompt);
      // Fresh, empty session — no captures can carry over between modes.
      useSession.getState().setSession(
        r.session_id, r.event_name, r.event_type, r.prompt);
      setFeedMode(target);
      setModeErr(null);
    } catch (e: any) {
      // Never fail silently: a quota/auth error here used to leave the user on a
      // stale session, so their uploads went nowhere and looked like "0 captures".
      setModeErr(e?.message ?? "Couldn't start a new session — please try again.");
    }
    finally { setSwitching(false); }
  };

  // LIVE "Generate album": curate the best shots, END the event, and show the
  // album. The deep relevancy/dedup/sequence curation runs on the backend.
  const generateAlbum = async () => {
    if (!session.sid || albumBusy) return;
    // Stop the camera FIRST, before the (slow — can be a minute+: the VLM
    // album tagger loads a 3B-param model on first use, then scores every
    // photo) album request even starts. It used to stop only after the
    // request resolved, so the live camera kept streaming the whole time —
    // fighting the album curation for the same GPU and making both slow.
    camera.stop();
    setAlbumBusy(true);
    const sid = session.sid;
    try {
      const a = await Sessions.album(sid);
      setAlbumPhotos(a.photos ?? []);
      setAlbumLowConf(!!a.low_confidence);
      setEnded(true);                      // run complete — end the present event
      Sessions.end(sid).catch(() => {});
    } catch { setAlbumPhotos([]); setEnded(true); }
    finally { setAlbumBusy(false); }
  };

  // Save the album as a single .zip — native "save as" file explorer in
  // Chrome/Edge, a normal download everywhere else (works in every browser).
  const saveAlbum = async () => {
    if (!albumPhotos?.length) return;
    setSaveBusy(true);
    try { await downloadAlbumZip(albumPhotos, session.eventName || "album"); }
    catch { /* ignore */ }
    finally { setSaveBusy(false); }
  };

  // Save to the USER'S device. On a phone this opens the native share sheet →
  // "Save to Photos" puts them in the camera roll (a .zip never gets there).
  // Never claim success silently: some mobile browsers can't do this
  // automatically at all, and the user needs to be told the real fallback
  // (long-press a photo) instead of tapping the button and seeing nothing happen.
  const savePhone = async () => {
    if (!albumPhotos?.length) return;
    setPhoneBusy(true); setPhoneMsg("");
    try {
      const r = await savePhotos(albumPhotos.map((p: any) => p.url));
      setPhoneMsg(
        r === "shared"     ? "Opened share sheet — choose “Save to Photos”."
      : r === "downloaded" ? "Downloading your album…"
      : r === "manual"     ? "Can't save automatically here — press and hold any photo below, then choose “Save Image”."
      :                      "Couldn't load the photos — check your connection and try again.");
    } catch {
      setPhoneMsg("Couldn't save the album.");
    } finally {
      setPhoneBusy(false);
      setTimeout(() => setPhoneMsg(""), 8000);
    }
  };

  // ── Real signals driving the viewfinder + shutter ─────────────────────
  // captureKey changes exactly once per genuine backend capture (the id of
  // the frame that fired). CameraView flashes off this changing — replacing
  // an older setState-during-render flash that also re-fired in a loop for
  // as long as the same captured frame stayed `latest`.
  const captureKey = latest?.captured ? latest.frame_id : null;

  // How close the pipeline REALLY is to capturing, 0..1:
  //   • Birthday runs the ensemble → final_score vs its own threshold is a
  //     genuine "approaching the trigger" ratio.
  //   • Custom/General have no ensemble (engine is null); moment.confidence
  //     is the honest stand-in there.
  //   • Neither available → undefined, and the viewfinder simply rests
  //     rather than animating a number the backend never produced.
  const eng = latest?.engine;
  const captureCloseness =
    eng && eng.threshold > 0 ? Math.min(1, eng.final_score / eng.threshold)
    : typeof latest?.moment?.confidence === "number" ? latest.moment.confidence
    : undefined;

  const endEvent = async () => {
    if (!session.sid) return;
    await Sessions.end(session.sid).catch(() => {});
    session.clearSession();
  };

  const startNewEvent = () => {
    setEnded(false); setAlbumPhotos(null); setFeedMode("live");
    session.clearSession();   // back to Setup to pick the next event
  };

  const giveFeedback = async (id: number, kept: boolean) => {
    const cap = captures.find((c) => c.id === id);
    if (!cap || !cap.url || !session.sid) return;
    session.markFeedback(id, kept);
    Sessions.feedback(session.sid, cap.url, kept, cap.moment).catch(() => {});
  };

  // Friendly status pulled from latest frame. Custom Mode has no fixed
  // moment vocabulary, so the birthday heuristics below (group gaze,
  // rising-score trend) are meaningless there and previously showed random,
  // unrelated text ("Watching a sweet group moment…") no matter what the
  // user actually asked SnapAI to capture — say what's ACTUALLY being
  // watched for instead, from the conversational state ConversationalGuide
  // already drives.
  const statusLine = (session.eventType === "custom" || session.eventType === "smart_event")
    ? (session.watchStatus === "captured" ? "Got it! ✨"
      : session.watchStatus === "understanding"
        ? (session.eventType === "smart_event" ? "Working out what to capture…" : "Understanding your request…")
      : session.watchStatus === "watching" && session.watchingFor
        ? `Watching for: ${session.watchingFor}`
        : (session.eventType === "smart_event" ? "Tell me the event ↓" : "Tell me what to capture ↓"))
    : latest?.captured ? "Got a beautiful one ✨"
    : latest?.gaze.is_group_looking ? "Watching a sweet group moment…"
    : latest?.prediction.trend === "rising" ? "Something good is coming…"
    : "SnapAI is watching.";

  // ── Event-ended view ──────────────────────────────────────────────────
  // After "Generate album" the session is over: show the curated album, a
  // Save-to-folder action, and a way to start a new event.
  if (ended) {
    return (
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <div>
            <div className="text-xs uppercase tracking-widest text-slate-500">
              {session.eventType} · event ended
            </div>
            <div className="font-bold text-2xl">
              {session.eventName || "Your album"}
            </div>
          </div>
          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Phone-first: the share sheet lands photos in the camera roll.
                (A .zip download is useless on a phone gallery.) */}
            <button onClick={savePhone} disabled={phoneBusy || !albumPhotos?.length}
              className="px-5 py-3 sm:px-4 sm:py-2 rounded-xl bg-emerald-600 text-white
                         active:bg-emerald-700 text-base sm:text-sm font-medium transition
                         inline-flex items-center gap-1.5 disabled:opacity-40 min-h-[44px]">
              {phoneBusy ? <Loader2 className="w-4 h-4 animate-spin" />
                         : <Download className="w-4 h-4" />}
              Save to my phone
            </button>
            <button onClick={saveAlbum} disabled={saveBusy || !albumPhotos?.length}
              className="px-5 py-3 sm:px-4 sm:py-2 rounded-xl bg-brand-500 text-white
                         active:bg-brand-600 text-base sm:text-sm font-medium transition
                         inline-flex items-center gap-1.5 disabled:opacity-40 min-h-[44px]">
              {saveBusy ? <Loader2 className="w-4 h-4 animate-spin" />
                        : <Download className="w-4 h-4" />}
              Download album (.zip)
            </button>
            <button onClick={startNewEvent}
              className="px-5 py-3 sm:px-4 sm:py-2 rounded-xl bg-white border border-slate-200
                         text-slate-700 active:bg-slate-50 text-base sm:text-sm transition min-h-[44px]">
              Start new event
            </button>
          </div>
        </div>

        {phoneMsg && (
          <div className="mb-5 rounded-xl border border-emerald-200 bg-emerald-50
                          px-4 py-2.5 text-sm text-emerald-900">
            {phoneMsg}
          </div>
        )}

        {/* GUARANTEED path — always visible on mobile, never hidden behind a
            toast that only appears after the one-tap save fails. */}
        {isMobile() && !phoneMsg && (
          <div className="mb-5 rounded-xl border border-slate-200 bg-slate-50
                          px-4 py-2.5 text-sm text-slate-700">
            📱 “Save to my phone” opens your share sheet — pick “Save to Photos”.
            If nothing opens, <b>press and hold any photo below</b> and choose
            “Save Image” — that always works, on every phone.
          </div>
        )}

        <div className="rounded-2xl bg-emerald-50 border border-emerald-100
                        px-4 py-3 mb-5 text-sm text-emerald-800
                        flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4" />
          Event ended · your album is ready
          {albumPhotos?.length ? ` · ${albumPhotos.length} best moments` : ""}.
        </div>

        {albumLowConf && albumPhotos && albumPhotos.length > 0 && (
          <div className="rounded-2xl bg-amber-50 border border-amber-200
                          px-4 py-3 mb-5 text-sm text-amber-800">
            ⚠ SnapAI couldn't confidently match these to specific {session.eventType}
            {" "}moments — this is a best-effort selection. Adding a little training
            footage of this event type will sharpen the moment recognition.
          </div>
        )}

        {albumPhotos && albumPhotos.length > 0 ? (
          <AlbumReview sessionId={session.sid!} photos={albumPhotos} />
        ) : (
          <div className="rounded-2xl border-2 border-dashed border-slate-200 p-8
                          text-center text-slate-500">
            No moments were captured in this session.
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto px-4 py-6">

      {/* Top stat band */}
      <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
        <div>
          {/* Receiving half of the Setup → session shared-element morph: the
              selected event card in Setup.tsx claims the same
              view-transition-name, so the browser grows that card into this
              header instead of hard-cutting between screens. */}
          <div className="text-xs uppercase tracking-widest text-slate-500"
               style={{ viewTransitionName: "event-hero" }}>
            {session.eventType}
          </div>
          <div className="font-bold text-2xl">{session.eventName}</div>
        </div>
        <div className="flex flex-wrap items-center gap-2.5 sm:gap-3">
          {/* Live ⇄ Upload — mutually exclusive; switching starts a fresh,
              empty session so captures from the other mode are discarded. */}
          <div className="flex rounded-xl bg-slate-100 p-1 text-base sm:text-sm">
            <button onClick={() => switchMode("live")} disabled={switching}
              className={"px-4 py-2.5 sm:px-3 sm:py-1.5 rounded-lg font-medium transition " +
                "inline-flex items-center gap-1.5 disabled:opacity-50 min-h-[44px] sm:min-h-0 " +
                (feedMode === "live"
                  ? "bg-white shadow text-rose-600"
                  : "text-slate-500 active:text-slate-700")}>
              {switching && feedMode === "upload"
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <Radio className="w-4 h-4" />} Live
            </button>
            <button onClick={() => switchMode("upload")} disabled={switching}
              className={"px-4 py-2.5 sm:px-3 sm:py-1.5 rounded-lg font-medium transition " +
                "inline-flex items-center gap-1.5 disabled:opacity-50 min-h-[44px] sm:min-h-0 " +
                (feedMode === "upload"
                  ? "bg-white shadow text-brand-600"
                  : "text-slate-500 active:text-slate-700")}>
              {switching && feedMode === "live"
                ? <Loader2 className="w-4 h-4 animate-spin" />
                : <FileVideo className="w-4 h-4" />} Upload video
            </button>
          </div>

          {feedMode === "live" && (
            <>
              <Stat n={total} label="Captured" accent />
              {/* Frames/Faces are secondary diagnostic counts — hidden on
                  phones so the header isn't fighting for space with the
                  buttons people actually need to tap. */}
              <div className="hidden sm:flex items-center gap-3">
                <Stat n={session.totalFrames}    label="Frames" />
                <Stat n={latest?.face_detection.face_count ?? 0} label="Faces" />
              </div>
              <button onClick={generateAlbum} disabled={albumBusy || total === 0}
                      className="sm:ml-2 px-5 py-3 sm:px-4 sm:py-2 rounded-xl bg-brand-500 text-white
                                 active:bg-brand-600 text-base sm:text-sm font-medium transition
                                 inline-flex items-center gap-1.5 disabled:opacity-40 min-h-[44px]">
                {albumBusy ? <Loader2 className="w-4 h-4 animate-spin" />
                           : <Images className="w-4 h-4" />}
                Generate album
              </button>
            </>
          )}

          <button onClick={endEvent}
                  className="px-5 py-3 sm:px-4 sm:py-2 rounded-xl bg-white border
                             border-slate-200 text-slate-700 active:bg-slate-50
                             text-base sm:text-sm transition min-h-[44px]">
            End event
          </button>
        </div>
      </div>

      {modeErr && (
        <div className="mb-4 rounded-xl border border-rose-200 bg-rose-50
                        px-4 py-3 text-sm text-rose-800">
          {modeErr}
        </div>
      )}

      {feedMode === "live" ? (
       <>
      {/* Camera + status — tall portrait box on phones (fills the screen, no
          16:9 letterbox that crops the view), widescreen on larger screens. */}
      <div className="relative rounded-3xl overflow-hidden shadow-xl bg-black
                      aspect-[3/4] sm:aspect-video">
        <CameraView camera={camera}
                    faceBoxes={latest?.face_detection.boxes ?? []}
                    viewfinder
                    confidence={captureCloseness}
                    captureKey={captureKey}
                    className="h-full" />

        {/* Birthday's setup coach still overlays the camera — its guidance
            ("move left", "cake is off-frame") is about FRAMING, so it has to
            sit on the picture it's talking about. Custom/General's
            ConversationalGuide does NOT: it's a conversation, and it used to
            cover the middle of the frame — squarely on top of whoever was
            being photographed. It now renders BELOW the camera box instead
            (see after this div), leaving the live view unobstructed. */}
        {!(session.eventType === "custom" || session.eventType === "smart_event")
          && <SetupCoach setup={latest?.setup} />}

        {/* Soft top gradient + status — LIVE and status grouped on the LEFT so
            the top-right corner stays clear for the camera-flip button. */}
        <div className="absolute top-0 inset-x-0 h-24 bg-gradient-to-b
                        from-black/55 via-black/20 to-transparent
                        flex items-start p-5 pr-20 pointer-events-none">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-white">
            {/* Connection state — bound to the real socket phase, so this
                never shows "Live" while the socket is actually down. The old
                hard-coded red dot pulsed identically whether connected or
                not, which is exactly the misleading case to avoid. */}
            <ConnectionStatus status={wsStatus} receiving={receiving}
                              className="[&_span]:!text-white/90 drop-shadow" />
            <span className="text-white/90 text-sm font-medium drop-shadow">
              {statusLine}
            </span>
          </div>
        </div>

        {/* Bottom prompt control */}
        <div className="absolute bottom-0 inset-x-0 p-4 sm:p-6
                        bg-gradient-to-t from-black/55 to-transparent
                        pointer-events-none">
          <div className="max-w-2xl mx-auto pointer-events-auto relative">
            <PromptControl ws={ws} variant="user"
              busy={session.watchStatus === "understanding"}
              placeholder={session.eventType === "smart_event"
                ? "Name the occasion (e.g. Diwali)…" : undefined} />
          </div>
        </div>

        {/* Capture toast */}
        {latest?.captured && (
          <div className="absolute top-20 right-4 sm:right-6 animate-slide-in-up">
            <div className="flex items-center gap-2 px-4 py-2.5 rounded-2xl
                            bg-white/95 backdrop-blur shadow-lg">
              <CheckCircle2 className="w-5 h-5 text-emerald-500" />
              <div>
                <div className="text-xs text-slate-500">Captured</div>
                <div className="text-sm font-semibold capitalize">
                  {latest.moment.detected_moment.replace(/_/g, " ")}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Custom / General conversational panel — deliberately OUTSIDE the
          camera box so it can never cover the person being photographed. */}
      {(session.eventType === "custom" || session.eventType === "smart_event") && (
        <ConversationalGuide ws={ws} />
      )}

      {/* Recent captures */}
      <section className="mt-8">
        <h2 className="text-lg font-semibold mb-3">Recent moments</h2>
        {captures.length === 0 ? (
          <div className="rounded-2xl border-2 border-dashed border-slate-200 p-8
                          text-center text-slate-500">
            <Camera className="w-8 h-8 mx-auto mb-2 text-slate-400" />
            SnapAI is warming up. Captures will appear here.
          </div>
        ) : (
          // LayoutGroup + per-card `layout` gives FLIP reflow: when a new
          // capture lands at the front, the existing cards slide to their new
          // slots instead of snapping.
          <LayoutGroup>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4
                            lg:grid-cols-5 gap-3">
              <AnimatePresence initial={false}>
                {captures.map((c) => (
                  <CaptureCard key={c.id} c={c} reduced={reduced}
                               onOpen={() => setLightbox({
                                 url: c.url ?? `data:image/jpeg;base64,${c.thumbB64}`,
                                 moment: c.moment,
                               })}
                               onKeep={() => giveFeedback(c.id, true)}
                               onTrash={() => giveFeedback(c.id, false)} />
                ))}
              </AnimatePresence>
            </div>
          </LayoutGroup>
        )}
      </section>

      <PhotoLightbox photo={lightbox} onClose={() => withViewTransition(
        () => setLightbox(null))} />

      <p className="mt-10 text-center text-sm text-slate-400">
        Tap 💚 on captures you love or ✕ to skip — SnapAI learns your taste.
      </p>
       </>
      ) : (
        <section className="mt-2">
          <div className="mb-3 flex items-center gap-2 text-sm text-slate-500">
            <Radio className="w-4 h-4 text-slate-400" />
            Live capture is paused — the camera is off while you analyze a recorded video.
          </div>
          <VideoUpload />
        </section>
      )}
    </div>
  );
}


function Stat({ n, label, accent = false }:
              { n: number; label: string; accent?: boolean }) {
  return (
    <div className="text-right leading-tight">
      <div className={"font-bold text-2xl tabular-nums " +
                      (accent ? "text-rose-500" : "text-slate-700")}>{n}</div>
      <div className="text-xs uppercase tracking-widest text-slate-500">{label}</div>
    </div>
  );
}


function CaptureCard({ c, onKeep, onTrash, onOpen, reduced }:
                     { c: any; onKeep: () => void; onTrash: () => void;
                       onOpen: () => void; reduced: boolean }) {
  const kept    = c.feedback === "kept";
  const trashed = c.feedback === "trashed";
  const hasZoom = !!c.zoomUrl;
  const [view,  setView]  = useState<"wide" | "zoom">("wide");

  // Use the wide URL by default; switch to the zoom URL when the user
  // toggles. Falls back to the embedded thumbnail if neither URL works.
  const currentUrl =
    view === "zoom" && c.zoomUrl ? c.zoomUrl :
    c.url            ? c.url     :
    `data:image/jpeg;base64,${c.thumbB64}`;

  // "Developing" pass — a just-arrived capture starts soft and desaturated
  // and resolves over ~600ms. Pure CSS filter animation on a single element
  // (GPU-composited, no WebGL, no per-frame JS), so it stays cheap even when
  // captures arrive back-to-back on the hot path. Runs once, on mount, which
  // for this grid means "when the capture actually arrived over the WS".
  const developStyle = reduced ? undefined
    : { animation: "photo-develop 600ms ease-out both" };

  return (
    <motion.div
      layout={!reduced}
      initial={reduced ? false : { opacity: 0, scale: 0.94, y: 8 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={reduced ? { opacity: 0 } : { opacity: 0, scale: 0.94 }}
      transition={reduced ? { duration: 0 } : SPRING.photo}
      className={"group relative rounded-xl overflow-hidden bg-white " +
                    "shadow-sm border border-slate-100 " +
                    (kept ? "ring-2 ring-emerald-400 " : "") +
                    (trashed ? "opacity-50 grayscale " : "")}>
      <button type="button" onClick={onOpen}
              aria-label={`Open ${String(c.moment).replace(/_/g, " ")} full size`}
              className="block w-full">
        <img src={currentUrl} alt={c.moment}
             style={developStyle}
             className="w-full aspect-square object-cover" />
      </button>

      {/* Moment label */}
      <div className="absolute top-2 left-2">
        <span className="px-2.5 py-1 bg-white/90 backdrop-blur rounded-full
                         text-xs font-medium capitalize">
          {c.moment.replace(/_/g, " ")}
        </span>
      </div>

      {/* Wide/Zoom toggle — only when a zoom shot exists */}
      {hasZoom && (
        <div className="absolute top-2 right-2 flex bg-white/85 backdrop-blur
                        rounded-full p-0.5 text-xs font-medium">
          <button onClick={(e) => { e.stopPropagation(); setView("wide"); }}
            className={"px-2.5 py-1 rounded-full transition min-h-[28px] " +
              (view === "wide" ? "bg-ink-900 text-white" : "text-slate-700")}>
            wide
          </button>
          <button onClick={(e) => { e.stopPropagation(); setView("zoom"); }}
            className={"px-2.5 py-1 rounded-full transition min-h-[28px] " +
              (view === "zoom" ? "bg-ink-900 text-white" : "text-slate-700")}>
            {c.zoomFactor ? `${c.zoomFactor.toFixed(1)}×` : "zoom"}
          </button>
        </div>
      )}

      {/* Feedback overlay — ALWAYS visible, not hover-gated: opacity-0 +
          group-hover used to hide these entirely on phones (no mouse, so no
          hover state ever fires), even though the footer told people to tap
          them. Buttons sized for a thumb (44px+), not a cursor. */}
      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/60
                      via-black/10 to-transparent
                      flex items-end justify-center gap-3 p-2.5">
        <button onClick={onKeep}
                className="w-11 h-11 rounded-full bg-white/95 active:bg-emerald-500
                           active:text-white text-emerald-500 grid place-items-center
                           shadow transition">
          <Heart className="w-5 h-5" />
        </button>
        <button onClick={onTrash}
                className="w-11 h-11 rounded-full bg-white/95 active:bg-rose-500
                           active:text-white text-rose-500 grid place-items-center
                           shadow transition">
          <X className="w-5 h-5" />
        </button>
      </div>
    </motion.div>
  );
}
