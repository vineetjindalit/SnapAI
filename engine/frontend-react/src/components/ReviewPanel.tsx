// frontend-react/src/components/ReviewPanel.tsx
//
// Post-album review panel shown after a video finishes processing.
//
// Two sections:
//   1. Required-shots checklist  — shows which minimum birthday shots were
//      captured (✓) and which were missed (✗).
//   2. "Tell us what we missed"  — user types a description, picks a severity
//      (Critical / Important / Casual) and submits.  The server feeds the
//      feedback to the OnlineLearner so the model captures that moment more
//      reliably in future sessions.

import React, { useEffect, useState } from "react";
import { Sessions } from "../api/client";

// ── Types ─────────────────────────────────────────────────────────────────────

interface RequiredShot {
  shot_id: string;
  label: string;
  captured: boolean;
  capture_url: string | null;
}

interface MissedMoment {
  id: number;
  description: string;
  severity: string;
  moment_hint: string;
  learned: boolean;
}

interface ReviewSummary {
  required_shots: {
    total_required: number;
    captured: number;
    missed: number;
    shots: RequiredShot[];
  };
  missed_moments: MissedMoment[];
  total_captures: number;
}

interface Props {
  sessionId: string;
  eventName: string;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const SEVERITY_LABELS = {
  critical:  { label: "Critical miss",  color: "#ef4444", desc: "Key moment — cannot be without it" },
  important: { label: "Important miss", color: "#f97316", desc: "Would have made the album much better" },
  casual:    { label: "Casual miss",    color: "#eab308", desc: "Nice to have but not essential" },
} as const;

type Severity = keyof typeof SEVERITY_LABELS;

// ── Component ─────────────────────────────────────────────────────────────────

export const ReviewPanel: React.FC<Props> = ({ sessionId, eventName }) => {
  const [summary, setSummary]           = useState<ReviewSummary | null>(null);
  const [loading, setLoading]           = useState(true);
  const [missDesc, setMissDesc]         = useState("");
  const [severity, setSeverity]         = useState<Severity>("important");
  const [submitting, setSubmitting]     = useState(false);
  const [submitMsg, setSubmitMsg]       = useState("");
  const [missedList, setMissedList]     = useState<MissedMoment[]>([]);

  useEffect(() => {
    if (!sessionId) return;
    Sessions.reviewSummary(sessionId)
      .then((d) => {
        setSummary(d);
        setMissedList(d.missed_moments || []);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId]);

  const handleSubmit = async () => {
    if (!missDesc.trim()) return;
    setSubmitting(true);
    setSubmitMsg("");
    try {
      const res = await Sessions.reportMissedMoment(
        sessionId, missDesc.trim(), severity
      );
      setSubmitMsg(res.message);
      setMissDesc("");
      // Append to the local list for immediate feedback
      setMissedList((prev) => [
        ...prev,
        {
          id: res.row_id,
          description: missDesc.trim(),
          severity,
          moment_hint: res.moment_hint,
          learned: res.learned,
        },
      ]);
    } catch (e: any) {
      setSubmitMsg(`Error: ${e.message}`);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) {
    return (
      <div style={styles.container}>
        <p style={styles.muted}>Loading review…</p>
      </div>
    );
  }

  if (!summary) return null;

  const shots      = summary.required_shots?.shots ?? [];
  const captured   = shots.filter((s) => s.captured).length;
  const missed     = shots.filter((s) => !s.captured).length;
  const totalReq   = shots.length;
  const scoreColor = missed === 0 ? "#22c55e" : missed <= 2 ? "#f97316" : "#ef4444";

  return (
    <div style={styles.container}>
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <h3 style={styles.heading}>Post-capture review — {eventName}</h3>
      <p style={styles.subheading}>
        {summary.total_captures} photos captured &nbsp;·&nbsp;
        <span style={{ color: scoreColor }}>
          {captured}/{totalReq} required shots covered
        </span>
      </p>

      {/* ── Required-shots checklist ────────────────────────────────────── */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Minimum required shots</h4>
        <ul style={styles.shotList}>
          {shots.map((s) => (
            <li key={s.shot_id} style={styles.shotItem}>
              <span style={{ fontSize: 18, marginRight: 8 }}>
                {s.captured ? "✅" : "❌"}
              </span>
              <span style={{ color: s.captured ? "#e2e8f0" : "#fca5a5" }}>
                {s.label}
              </span>
              {s.captured && s.capture_url && (
                <a
                  href={s.capture_url}
                  target="_blank"
                  rel="noreferrer"
                  style={styles.viewLink}
                >
                  view
                </a>
              )}
            </li>
          ))}
        </ul>
      </section>

      {/* ── Report a miss ───────────────────────────────────────────────── */}
      <section style={styles.section}>
        <h4 style={styles.sectionTitle}>Tell us what we missed</h4>
        <p style={styles.muted}>
          Describe a moment you noticed was not captured. We'll learn from it
          and do better next time.
        </p>

        <textarea
          style={styles.textarea}
          placeholder="e.g. You missed the moment when the child blew out the candles"
          value={missDesc}
          onChange={(e) => setMissDesc(e.target.value)}
          rows={3}
        />

        {/* Severity selector */}
        <div style={styles.severityRow}>
          {(Object.entries(SEVERITY_LABELS) as [Severity, typeof SEVERITY_LABELS[Severity]][])
            .map(([key, info]) => (
              <button
                key={key}
                onClick={() => setSeverity(key)}
                style={{
                  ...styles.severityBtn,
                  borderColor: severity === key ? info.color : "#374151",
                  color:       severity === key ? info.color : "#9ca3af",
                  fontWeight:  severity === key ? 700 : 400,
                }}
                title={info.desc}
              >
                {info.label}
              </button>
            ))}
        </div>

        <button
          style={{
            ...styles.submitBtn,
            opacity: submitting || !missDesc.trim() ? 0.5 : 1,
          }}
          disabled={submitting || !missDesc.trim()}
          onClick={handleSubmit}
        >
          {submitting ? "Sending…" : "Submit feedback"}
        </button>

        {submitMsg && (
          <p style={{ marginTop: 8, color: "#86efac", fontSize: 13 }}>
            {submitMsg}
          </p>
        )}
      </section>

      {/* ── Previously filed misses ─────────────────────────────────────── */}
      {missedList.length > 0 && (
        <section style={styles.section}>
          <h4 style={styles.sectionTitle}>Your feedback for this session</h4>
          <ul style={{ listStyle: "none", margin: 0, padding: 0 }}>
            {missedList.map((m, i) => {
              const sev = SEVERITY_LABELS[m.severity as Severity]
                ?? SEVERITY_LABELS.important;
              return (
                <li key={m.id ?? i} style={styles.feedbackItem}>
                  <span style={{
                    ...styles.severityTag,
                    backgroundColor: sev.color + "33",
                    color: sev.color,
                  }}>
                    {sev.label}
                  </span>
                  <span style={{ color: "#e2e8f0", flex: 1 }}>
                    {m.description}
                  </span>
                  {m.learned && (
                    <span style={styles.learnedTag}>✓ learned</span>
                  )}
                </li>
              );
            })}
          </ul>
        </section>
      )}
    </div>
  );
};

// ── Styles ────────────────────────────────────────────────────────────────────

const styles: Record<string, React.CSSProperties> = {
  container: {
    background: "#111827",
    border: "1px solid #374151",
    borderRadius: 12,
    padding: "20px 24px",
    marginTop: 20,
  },
  heading: {
    margin: "0 0 4px",
    fontSize: 18,
    fontWeight: 700,
    color: "#f9fafb",
  },
  subheading: {
    margin: "0 0 20px",
    fontSize: 14,
    color: "#9ca3af",
  },
  section: {
    marginBottom: 24,
    paddingBottom: 24,
    borderBottom: "1px solid #1f2937",
  },
  sectionTitle: {
    margin: "0 0 12px",
    fontSize: 14,
    fontWeight: 600,
    color: "#d1d5db",
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  shotList: {
    listStyle: "none",
    margin: 0,
    padding: 0,
    display: "grid",
    gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
    gap: "8px 16px",
  },
  shotItem: {
    display: "flex",
    alignItems: "center",
    fontSize: 14,
  },
  viewLink: {
    marginLeft: 8,
    fontSize: 12,
    color: "#60a5fa",
    textDecoration: "none",
  },
  muted: {
    color: "#6b7280",
    fontSize: 13,
    margin: "0 0 10px",
  },
  textarea: {
    width: "100%",
    background: "#1f2937",
    border: "1px solid #374151",
    borderRadius: 8,
    padding: "10px 12px",
    color: "#f9fafb",
    fontSize: 14,
    resize: "vertical",
    boxSizing: "border-box",
    outline: "none",
    marginBottom: 10,
  },
  severityRow: {
    display: "flex",
    gap: 8,
    flexWrap: "wrap",
    marginBottom: 12,
  },
  severityBtn: {
    padding: "6px 14px",
    borderRadius: 20,
    border: "1.5px solid",
    background: "transparent",
    cursor: "pointer",
    fontSize: 13,
    transition: "all 0.15s",
  },
  submitBtn: {
    padding: "9px 22px",
    background: "#2563eb",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
    fontSize: 14,
    fontWeight: 600,
    transition: "opacity 0.15s",
  },
  feedbackItem: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "8px 0",
    borderBottom: "1px solid #1f2937",
    fontSize: 13,
  },
  severityTag: {
    padding: "2px 8px",
    borderRadius: 10,
    fontSize: 11,
    fontWeight: 600,
    whiteSpace: "nowrap",
  },
  learnedTag: {
    fontSize: 11,
    color: "#86efac",
    whiteSpace: "nowrap",
  },
};

export default ReviewPanel;
