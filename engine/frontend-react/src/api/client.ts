// frontend-react/src/api/client.ts
// Typed REST client for the SnapAI server. Auth-token aware.

import type { HealthResponse, SessionResponse, AdminOverviewResponse } from "./types";

const TOKEN_KEY = "snappy.auth_token";

// In dev, vite proxies "/sessions" etc. to :8765. In prod (built bundle
// served by the SnapAI backend), same-origin requests work natively.
const BASE = "";

function authHeader(): Record<string, string> {
  const t = localStorage.getItem(TOKEN_KEY);
  return t ? { Authorization: `Bearer ${t}` } : {};
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...authHeader() },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  const text = await r.text();
  if (!r.ok) {
    let err: any = text;
    try { err = JSON.parse(text); } catch {}
    throw new Error(err?.error || `HTTP ${r.status}`);
  }
  return text ? (JSON.parse(text) as T) : ({} as T);
}

// ── Health ────────────────────────────────────────────────────────────
export const Health = {
  get: () => req<HealthResponse>("GET", "/health"),
};

// ── Admin (owner-only Mac dashboard) ────────────────────────────────────
export const Admin = {
  overview: () => req<AdminOverviewResponse>("GET", "/admin/overview"),
};

// ── Sessions ──────────────────────────────────────────────────────────
export const Sessions = {
  create: (event_name: string, event_type: string, prompt: string) =>
    req<SessionResponse>("POST", "/sessions",
      { event_name, event_type, prompt }),

  list:    () => req<{ sessions: any[] }>("GET", "/sessions"),
  end:     (sid: string) => req<{ status: string }>("DELETE", `/sessions/${sid}`),
  stats:   (sid: string) => req<any>("GET", `/sessions/${sid}/stats`),
  photos:  (sid: string) => req<{ total: number; photos: any[] }>(
    "GET", `/sessions/${sid}/photos`),
  album:   (sid: string) => req<any>("POST", `/sessions/${sid}/album`),

  feedback: (sid: string, photo_url: string, kept: boolean, moment: string) =>
    req<any>("POST", `/sessions/${sid}/feedback`, { photo_url, kept, moment }),

  updatePrompt: (sid: string, text: string,
                 source: "text" | "voice" = "text") =>
    req<any>("POST", `/sessions/${sid}/prompt`, { text, source }),

  learning: (sid: string) => req<any>("GET", `/sessions/${sid}/learning`),
  discoveries: (sid: string) => req<any>("GET", `/sessions/${sid}/discoveries`),
  nameDiscovery: (sid: string, cid: string, name: string) =>
    req<any>("POST", `/sessions/${sid}/discoveries/${cid}/name`, { name }),

  // ── Video upload (multipart) ───────────────────────────────────────
  uploadVideo: async (
    sid: string, file: File, sampleFps: number = 2.0,
    onProgress?: (uploadedBytes: number, totalBytes: number) => void,
  ): Promise<{
    status: string; session_id: string;
    filename: string; size_bytes: number; sample_fps: number;
  }> => {
    const fd = new FormData();
    fd.append("video", file, file.name);
    fd.append("sample_fps", String(sampleFps));

    // Use XHR (not fetch) so we get upload progress events.
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${BASE}/sessions/${sid}/upload_video`);
      const tok = localStorage.getItem(TOKEN_KEY);
      if (tok) xhr.setRequestHeader("Authorization", `Bearer ${tok}`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress)
          onProgress(e.loaded, e.total);
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try { resolve(JSON.parse(xhr.responseText)); }
          catch (e) { reject(new Error("Bad JSON from server")); }
        } else {
          let msg = xhr.responseText;
          try { msg = JSON.parse(xhr.responseText).error || msg; } catch {}
          reject(new Error(`${xhr.status}: ${msg}`));
        }
      };
      xhr.onerror = () => reject(new Error("Network error"));
      xhr.send(fd);
    });
  },

  videoProgress: (sid: string) =>
    req<{
      status: "idle" | "queued" | "processing" | "done" | "error";
      progress: number; total_frames: number; processed_frames: number;
      captures: number; error: string | null;
      album: {
        event_name: string; total_selected: number;
        total_captured: number; cover: string; stats?: any;
      } | null;
      active: boolean;
    }>("GET", `/sessions/${sid}/video_progress`),

  // ── Category requirements / post-capture review ───────────────────────

  requiredShots: (sid: string) =>
    req<{
      total_required: number; captured: number; missed: number;
      shots: Array<{
        shot_id: string; label: string; moment_class: string;
        captured: boolean; capture_url: string | null;
      }>;
    }>("GET", `/sessions/${sid}/required_shots`),

  reportMissedMoment: (
    sid: string,
    description: string,
    severity: "critical" | "important" | "casual",
    moment_hint?: string,
  ) =>
    req<{ row_id: number; moment_hint: string; severity: string;
          learned: boolean; message: string }>(
      "POST", `/sessions/${sid}/missed_moment`,
      { description, severity, moment_hint }),

  reviewSummary: (sid: string) =>
    req<{
      session_id: string; event_name: string; event_type: string;
      required_shots: {
        total_required: number; captured: number; missed: number;
        shots: Array<{ shot_id: string; label: string;
                       captured: boolean; capture_url: string | null }>;
      };
      missed_moments: Array<{
        id: number; description: string; severity: string;
        moment_hint: string; ts: number; learned: boolean;
      }>;
      photo_reactions: Array<{
        photo_url: string; reaction: string; note: string; moment: string;
      }>;
      album_reviews: Array<{ id: number; text: string; rating: number; ts: number }>;
      album: any | null;
      total_captures: number;
    }>("GET", `/sessions/${sid}/review_summary`),

  /** One reaction (+ optional note) for one album photo. */
  photoReaction: (sid: string, photo_url: string, reaction: string, note = "") =>
    req<{ ok: boolean; reaction: string; moment: string }>(
      "POST", `/sessions/${sid}/photo_reaction`,
      { photo_url, reaction, note }),

  /** Overall album review (text and/or 1-5 rating). */
  albumReview: (sid: string, text: string, rating = 0) =>
    req<{ ok: boolean; rating: number; learned: string[] }>(
      "POST", `/sessions/${sid}/album_review`, { text, rating }),

  /** Pure speech-to-text for review dictation (does NOT change the prompt). */
  transcribe: async (sid: string, audio: Blob): Promise<{ transcript: string }> => {
    const tok = localStorage.getItem(TOKEN_KEY);
    const r = await fetch(`${BASE}/sessions/${sid}/transcribe`, {
      method: "POST",
      headers: tok ? { Authorization: `Bearer ${tok}` } : {},
      body: audio,
    });
    if (!r.ok) throw new Error(`Transcribe failed: HTTP ${r.status}`);
    return r.json();
  },
};

// ── Auth ──────────────────────────────────────────────────────────────
export const Auth = {
  register: (email: string, password: string) =>
    req<any>("POST", "/auth/register", { email, password }),

  login: async (email: string, password: string) => {
    const r = await req<{ token: string; user: any }>("POST", "/auth/login",
      { email, password });
    localStorage.setItem(TOKEN_KEY, r.token);
    return r;
  },

  me:      () => req<any>("GET",  "/auth/me"),

  logout: async () => {
    try { await req<any>("POST", "/auth/logout"); } catch {/* idempotent */}
    localStorage.removeItem(TOKEN_KEY);
  },

  refresh: async () => {
    const r = await req<{ token: string }>("POST", "/auth/refresh");
    if (r.token) localStorage.setItem(TOKEN_KEY, r.token);
    return r;
  },

  verifyEmail:    (token: string) =>
    req<{ user_id: number; email: string }>("POST", "/auth/verify", { token }),

  forgotPassword: (email: string) =>
    req<{ sent: boolean }>("POST", "/auth/forgot", { email }),

  resetPassword:  (token: string, password: string) =>
    req<any>("POST", "/auth/reset", { token, password }),

  /** Read raw token (used by WS connect, which can't set Authorization). */
  token: () => localStorage.getItem(TOKEN_KEY),
};

// ── Billing ───────────────────────────────────────────────────────────
export const Billing = {
  tiers:    () => req<any>("GET", "/billing/tiers"),
  checkout: (tier: string, success_url: string, cancel_url: string) =>
    req<{ checkout_url: string }>("POST", "/billing/checkout",
      { tier, success_url, cancel_url }),
};
