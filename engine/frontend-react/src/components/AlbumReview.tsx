// frontend-react/src/components/AlbumReview.tsx
//
// Post-album review grid. Shown after the album is generated so the user can
// react to the actual curated photos. Every photo is NUMBERED (#1, #2 …) so the
// user can reference them in the overall review ("photo 3 was blurry").
//
// Per photo:
//   • emoji reactions   ❤️ 👍 👎   (love / keep / discard)
//   • quality tags      Blurry · Wrong moment · Bad framing
//                       ("right moment, but not captured the way I wanted")
//   • a note box + 🎤   (typed or dictated review for THAT photo)
//
// Overall:
//   • a 1-5 star rating
//   • a review box + 🎤  (typed or dictated review for the WHOLE album)
//
// Every reaction/note/review is sent to the server, which feeds the learner so
// the next session captures these moments better (or stops over-capturing the
// ones the user dislikes).

import React, { useRef, useState } from "react";
import { motion } from "motion/react";
import { Sessions } from "../api/client";
import { savePhotos, isMobile } from "../lib/savePhotos";
import { PhotoLightbox, type LightboxPhoto } from "./PhotoLightbox";
import { useReducedMotion, withViewTransition, SPRING } from "../lib/motionUtils";

// ── Types ─────────────────────────────────────────────────────────────────────

interface AlbumPhoto {
  url: string;
  moment?: string;
  score?: number;
  tags?: string[];
}

interface Props {
  sessionId: string;
  photos: AlbumPhoto[];
}

// reaction id → { emoji/label, kind }
const REACTIONS: Array<{ id: string; label: string; emoji: string; kind: "vote" | "issue" }> = [
  { id: "love", label: "Love",         emoji: "❤️", kind: "vote" },
  { id: "up",   label: "Good",         emoji: "👍", kind: "vote" },
  { id: "down", label: "Discard",      emoji: "👎", kind: "vote" },
  { id: "blurry",       label: "Blurry",       emoji: "😵‍💫", kind: "issue" },
  { id: "wrong_moment", label: "Wrong moment", emoji: "🎯", kind: "issue" },
  { id: "bad_framing",  label: "Bad framing",  emoji: "🖼️", kind: "issue" },
];

// ── Voice recorder hook ─────────────────────────────────────────────────────────
function useRecorder(sessionId: string) {
  const [recordingFor, setRecordingFor] = useState<string | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);

  const start = async (key: string, onText: (t: string) => void) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => e.data.size && chunksRef.current.push(e.data);
      mr.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        try {
          const { transcript } = await Sessions.transcribe(sessionId, blob);
          if (transcript) onText(transcript);
        } catch (e) {
          console.error("transcribe failed", e);
        }
      };
      mr.start();
      mediaRef.current = mr;
      setRecordingFor(key);
    } catch (e) {
      console.error("mic error", e);
      alert("Microphone unavailable — type your review instead.");
    }
  };

  const stop = () => {
    mediaRef.current?.stop();
    mediaRef.current = null;
    setRecordingFor(null);
  };

  return { recordingFor, start, stop };
}

// ── Mic button ──────────────────────────────────────────────────────────────────
const MicButton: React.FC<{
  active: boolean; onStart: () => void; onStop: () => void;
}> = ({ active, onStart, onStop }) => (
  <button
    type="button"
    onClick={active ? onStop : onStart}
    title={active ? "Stop recording" : "Dictate review"}
    style={{
      ...styles.mic,
      background: active ? "#dc2626" : "#1f2937",
      borderColor: active ? "#dc2626" : "#374151",
    }}
  >
    {active ? "⏹" : "🎤"}
  </button>
);

// ── Component ─────────────────────────────────────────────────────────────────
export const AlbumReview: React.FC<Props> = ({ sessionId, photos }) => {
  const [reactions, setReactions] = useState<Record<string, string>>({});
  const [notes, setNotes]         = useState<Record<string, string>>({});
  const [savedNote, setSavedNote] = useState<Record<string, boolean>>({});
  const [overall, setOverall]     = useState("");
  const [rating, setRating]       = useState(0);
  const [overallMsg, setOverallMsg] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState("");
  const [lightbox, setLightbox] = useState<LightboxPhoto | null>(null);
  const reduced = useReducedMotion();
  const rec = useRecorder(sessionId);

  // Get the album onto the USER'S device — share sheet on a phone (→ Photos),
  // download on desktop. Albums otherwise only ever live on the server.
  const saveAll = async (urls: string[]) => {
    setSaving(true); setSaveMsg("");
    let result: Awaited<ReturnType<typeof savePhotos>> | "error" = "error";
    try {
      result = await savePhotos(urls);
      setSaveMsg(
        result === "shared"     ? "Opened share sheet — choose “Save to Photos”."
      : result === "downloaded" ? `Downloading ${urls.length} photo${urls.length > 1 ? "s" : ""}…`
      : result === "manual"     ? "This browser can't save automatically — press and hold any photo below, then choose “Save Image”."
      :                           "Couldn't load those photos — check your connection and try again.");
    } catch {
      setSaveMsg("Couldn't save those photos.");
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMsg(""), result === "manual" ? 9000 : 6000);
    }
  };

  if (!photos || photos.length === 0) return null;

  const react = async (url: string, reaction: string) => {
    const next = reactions[url] === reaction ? "" : reaction;  // toggle off
    setReactions((r) => ({ ...r, [url]: next }));
    try {
      await Sessions.photoReaction(sessionId, url, next, notes[url] || "");
    } catch (e) { console.error(e); }
  };

  const saveNote = async (url: string) => {
    try {
      await Sessions.photoReaction(sessionId, url, reactions[url] || "", notes[url] || "");
      setSavedNote((s) => ({ ...s, [url]: true }));
      setTimeout(() => setSavedNote((s) => ({ ...s, [url]: false })), 1500);
    } catch (e) { console.error(e); }
  };

  const submitOverall = async () => {
    if (!overall.trim() && !rating) return;
    setSubmitting(true); setOverallMsg("");
    try {
      const res = await Sessions.albumReview(sessionId, overall.trim(), rating);
      setOverallMsg(
        res.learned?.length
          ? `Thanks — I'll work on: ${res.learned.join(", ").replace(/_/g, " ")}.`
          : "Thanks — your review was saved and will improve future albums."
      );
    } catch (e: any) {
      setOverallMsg(`Error: ${e.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={styles.container}>
      <h3 style={styles.heading}>Review this album</h3>
      <p style={styles.sub}>
        React to each shot and tell us what to improve. Photos are numbered so you
        can mention them in your overall review below.
      </p>

      {/* Save the album to the user's own device (phone gallery / downloads) */}
      <div style={styles.saveRow}>
        <button onClick={() => saveAll(photos.map((p) => p.url))}
                disabled={saving} style={{ ...styles.saveBtn, opacity: saving ? 0.6 : 1 }}>
          {saving ? "Preparing…" : `⬇︎ Save all ${photos.length} to my phone`}
        </button>
        {saveMsg && <span style={styles.saveMsg}>{saveMsg}</span>}
      </div>

      {/* GUARANTEED path — always visible on mobile, not hidden behind a
          toast. The button above tries a one-tap save (Share Sheet, which
          some browsers refuse silently); long-pressing a photo below works
          on every phone with zero JS involved, so it's never a dead end. */}
      {isMobile() && (
        <p style={styles.mobileHint}>
          📱 The button above should open your phone's share sheet — pick
          “Save to Photos” / “Save Image” there. If nothing opens, <b>press
          and hold any photo below</b> and choose “Save Image” instead —
          that always works.
        </p>
      )}

      {/* ── Numbered photo grid ─────────────────────────────────────────── */}
      <div style={styles.grid}>
        {photos.map((p, i) => {
          const sel = reactions[p.url];
          const open = lightbox?.url === p.url;
          return (
            <motion.div key={p.url} style={styles.card}
              layout={!reduced}
              // Staggered "developing" entrance — the album is the payoff
              // screen, so the shots resolve in rather than snapping on.
              initial={reduced ? false : { opacity: 0, y: 12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={reduced ? { duration: 0 }
                                  : { delay: Math.min(i * 0.05, 0.5), ...SPRING.photo }}>
              <div style={styles.thumbWrap}>
                <span style={styles.numBadge}>#{i + 1}</span>
                {/* Opens the in-app lightbox (shared-element morph) instead of
                    dumping the raw file into a new browser tab. Kept as an
                    <a href> so middle-click / "open in new tab" still work
                    and the URL stays copyable. */}
                <a href={p.url} target="_blank" rel="noreferrer"
                   onClick={(e) => {
                     if (e.metaKey || e.ctrlKey || e.shiftKey || e.button !== 0) return;
                     e.preventDefault();
                     withViewTransition(() =>
                       setLightbox({ url: p.url, moment: p.moment, index: i }));
                   }}>
                  <img src={p.url} alt={p.moment || `photo ${i + 1}`}
                       loading="lazy"
                       style={{
                         ...styles.thumb,
                         // Only the photo being opened carries the morph name;
                         // duplicate names on one page would break the
                         // transition, so it's applied to exactly one element.
                         ...(open ? { viewTransitionName: "photo-morph" } : null),
                         ...(reduced ? null
                             : { animation: "photo-develop 600ms ease-out both" }),
                       }} />
                </a>
                {p.moment && (
                  <span style={styles.momentTag}>{p.moment.replace(/_/g, " ")}</span>
                )}
                {/* Save just this one to the phone */}
                <button onClick={(e) => { e.preventDefault(); saveAll([p.url]); }}
                        title="Save this photo to my phone"
                        style={styles.savePhotoBtn}>⬇︎</button>
              </div>

              {/* reactions */}
              <div style={styles.reactRow}>
                {REACTIONS.map((rx) => (
                  <button
                    key={rx.id}
                    onClick={() => react(p.url, rx.id)}
                    title={rx.label}
                    style={{
                      ...styles.reactBtn,
                      ...(rx.kind === "issue" ? styles.reactIssue : {}),
                      ...(sel === rx.id ? styles.reactSel : {}),
                    }}
                  >
                    <span style={{ fontSize: 15 }}>{rx.emoji}</span>
                  </button>
                ))}
              </div>

              {/* per-photo note + mic */}
              <div style={styles.noteRow}>
                <input
                  style={styles.noteInput}
                  placeholder={`Note for #${i + 1}…`}
                  value={notes[p.url] || ""}
                  onChange={(e) => setNotes((n) => ({ ...n, [p.url]: e.target.value }))}
                  onBlur={() => (notes[p.url] ? saveNote(p.url) : null)}
                />
                <MicButton
                  active={rec.recordingFor === p.url}
                  onStart={() => rec.start(p.url, (t) =>
                    setNotes((n) => ({ ...n, [p.url]: ((n[p.url] || "") + " " + t).trim() })))}
                  onStop={() => { rec.stop(); setTimeout(() => saveNote(p.url), 400); }}
                />
              </div>
              {savedNote[p.url] && <span style={styles.savedTag}>✓ saved</span>}
            </motion.div>
          );
        })}
      </div>

      <PhotoLightbox photo={lightbox}
                     onClose={() => withViewTransition(() => setLightbox(null))} />

      {/* ── Overall album review ────────────────────────────────────────── */}
      <div style={styles.overall}>
        <h4 style={styles.sectionTitle}>Overall album review</h4>

        {/* star rating */}
        <div style={styles.stars}>
          {[1, 2, 3, 4, 5].map((n) => (
            <span key={n} onClick={() => setRating(n)}
                  style={{ ...styles.star, color: n <= rating ? "#facc15" : "#475569" }}>
              ★
            </span>
          ))}
          {rating > 0 && <span style={styles.ratingNum}>{rating}/5</span>}
        </div>

        <div style={styles.overallRow}>
          <textarea
            style={styles.overallBox}
            rows={3}
            placeholder='Review the whole album. e.g. "Great candle shots, but photo 4 is blurry and you missed the group hug."'
            value={overall}
            onChange={(e) => setOverall(e.target.value)}
          />
          <MicButton
            active={rec.recordingFor === "__overall__"}
            onStart={() => rec.start("__overall__", (t) =>
              setOverall((o) => (o + " " + t).trim()))}
            onStop={rec.stop}
          />
        </div>

        <button
          style={{ ...styles.submit, opacity: submitting || (!overall.trim() && !rating) ? 0.5 : 1 }}
          disabled={submitting || (!overall.trim() && !rating)}
          onClick={submitOverall}
        >
          {submitting ? "Sending…" : "Submit overall review"}
        </button>
        {overallMsg && <p style={styles.okMsg}>{overallMsg}</p>}
      </div>
    </div>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────────
const styles: Record<string, React.CSSProperties> = {
  container: { background: "#0f172a", border: "1px solid #334155", borderRadius: 12,
               padding: "20px 22px", marginTop: 18 },
  heading: { margin: "0 0 4px", fontSize: 18, fontWeight: 700, color: "#f8fafc" },
  sub: { margin: "0 0 16px", fontSize: 13, color: "#94a3b8" },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
          gap: 14 },
  card: { background: "#1e293b", border: "1px solid #334155", borderRadius: 10,
          padding: 8, display: "flex", flexDirection: "column", gap: 6 },
  thumbWrap: { position: "relative" },
  thumb: { width: "100%", aspectRatio: "16/9", objectFit: "cover", borderRadius: 6,
           display: "block" },
  numBadge: { position: "absolute", top: 6, left: 6, background: "rgba(15,23,42,0.85)",
              color: "#fff", fontWeight: 700, fontSize: 12, padding: "2px 7px",
              borderRadius: 12, zIndex: 2 },
  momentTag: { position: "absolute", bottom: 6, right: 6, background: "rgba(37,99,235,0.85)",
               color: "#fff", fontSize: 10, padding: "2px 6px", borderRadius: 8,
               textTransform: "capitalize" },
  reactRow: { display: "flex", gap: 4, flexWrap: "wrap" },
  reactBtn: { flex: "1 1 auto", minWidth: 30, padding: "5px 0", borderRadius: 7,
              border: "1px solid #334155", background: "#0f172a", cursor: "pointer",
              transition: "all 0.12s" },
  reactIssue: { border: "1px solid #422006" },
  reactSel: { background: "#2563eb", borderColor: "#2563eb", transform: "scale(1.05)" },
  noteRow: { display: "flex", gap: 6, alignItems: "center" },
  noteInput: { flex: 1, background: "#0f172a", border: "1px solid #334155",
               borderRadius: 6, padding: "6px 8px", color: "#e2e8f0", fontSize: 12,
               outline: "none" },
  savedTag: { fontSize: 10, color: "#86efac" },
  mic: { width: 32, height: 32, borderRadius: 8, border: "1px solid #374151",
         cursor: "pointer", fontSize: 14, flexShrink: 0, color: "#fff" },
  overall: { marginTop: 22, paddingTop: 18, borderTop: "1px solid #1e293b" },
  sectionTitle: { margin: "0 0 10px", fontSize: 13, fontWeight: 600, color: "#cbd5e1",
                  textTransform: "uppercase", letterSpacing: "0.05em" },
  stars: { display: "flex", alignItems: "center", gap: 2, marginBottom: 10 },
  star: { fontSize: 26, cursor: "pointer", lineHeight: 1 },
  ratingNum: { marginLeft: 8, color: "#94a3b8", fontSize: 13 },
  overallRow: { display: "flex", gap: 8, alignItems: "flex-start" },
  overallBox: { flex: 1, background: "#1e293b", border: "1px solid #334155",
                borderRadius: 8, padding: "10px 12px", color: "#f8fafc", fontSize: 14,
                resize: "vertical", outline: "none", boxSizing: "border-box" },
  submit: { marginTop: 10, padding: "9px 22px", background: "#2563eb", color: "#fff",
            border: "none", borderRadius: 8, cursor: "pointer", fontSize: 14,
            fontWeight: 600 },
  okMsg: { marginTop: 8, color: "#86efac", fontSize: 13 },
  saveRow: { display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
             margin: "10px 0 16px" },
  saveBtn: { padding: "10px 18px", background: "#059669", color: "#fff",
             border: "none", borderRadius: 10, cursor: "pointer", fontSize: 14,
             fontWeight: 600 },
  saveMsg: { color: "#a7f3d0", fontSize: 12.5 },
  mobileHint: { margin: "0 0 16px", padding: "10px 14px", background: "#1e293b",
                border: "1px solid #334155", borderRadius: 10, color: "#cbd5e1",
                fontSize: 13, lineHeight: 1.5 },
  savePhotoBtn: { position: "absolute", right: 6, bottom: 6, width: 30, height: 30,
                  borderRadius: 15, border: "none", cursor: "pointer",
                  background: "rgba(0,0,0,0.62)", color: "#fff", fontSize: 14,
                  lineHeight: "30px", padding: 0 },
};

export default AlbumReview;
