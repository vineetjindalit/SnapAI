// mobile/src/api.ts — typed client for the Snappy server.
//
// Used by the React Native app. Same routes as the web frontend, just
// typed and with auth header injection.

import AsyncStorage from "@react-native-async-storage/async-storage";

const TOKEN_KEY = "@snappy/auth_token";
let _baseUrl = "http://localhost:8765";

export function setBaseUrl(url: string) {
  _baseUrl = url.replace(/\/$/, "");
}

async function authHeader(): Promise<Record<string, string>> {
  const tok = await AsyncStorage.getItem(TOKEN_KEY);
  return tok ? { Authorization: `Bearer ${tok}` } : {};
}

async function req<T>(method: string, path: string, body?: unknown): Promise<T> {
  const auth = await authHeader();
  const r = await fetch(`${_baseUrl}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...auth },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const text = await r.text();
    throw new Error(`${r.status}: ${text}`);
  }
  return (await r.json()) as T;
}

// ── Auth ────────────────────────────────────────────────────────────────
export interface LoginResponse {
  ok: boolean;
  token: string;
  user: { id: number; email: string; role: string };
}

export const Auth = {
  register: (email: string, password: string) =>
    req<{ ok: boolean; user_id: number }>(
      "POST", "/auth/register", { email, password }),

  login: async (email: string, password: string): Promise<LoginResponse> => {
    const r = await req<LoginResponse>("POST", "/auth/login", { email, password });
    await AsyncStorage.setItem(TOKEN_KEY, r.token);
    return r;
  },

  me:     () => req<{ ok: boolean; user: any }>("GET", "/auth/me"),

  logout: () => AsyncStorage.removeItem(TOKEN_KEY),
};

// ── Sessions ────────────────────────────────────────────────────────────
export interface Session {
  session_id: string;
  event_name: string;
  event_type: string;
  prompt: string;
  ws_url: string;
}

export const Sessions = {
  create: (event_name: string, event_type: string, prompt: string) =>
    req<Session>("POST", "/sessions", { event_name, event_type, prompt }),

  list:   () => req<{ sessions: any[] }>("GET", "/sessions"),
  end:    (sid: string) => req<{ status: string }>("DELETE", `/sessions/${sid}`),
  photos: (sid: string) => req<{ total: number; photos: any[] }>(
    "GET", `/sessions/${sid}/photos`),
  album:  (sid: string) => req<any>("POST", `/sessions/${sid}/album`),

  feedback: (sid: string, photo_url: string, kept: boolean, moment: string) =>
    req<any>("POST", `/sessions/${sid}/feedback`, { photo_url, kept, moment }),

  updatePrompt: (sid: string, text: string, source: "text" | "voice" = "text") =>
    req<any>("POST", `/sessions/${sid}/prompt`, { text, source }),
};

// ── WebSocket helpers ──────────────────────────────────────────────────
export function frameStreamUrl(sid: string): string {
  return _baseUrl.replace(/^http/, "ws") + `/ws/${sid}`;
}

// ── Billing ────────────────────────────────────────────────────────────
export const Billing = {
  tiers:    () => req<any>("GET", "/billing/tiers"),
  checkout: (tier: string, success_url: string, cancel_url: string) =>
    req<{ checkout_url: string }>(
      "POST", "/billing/checkout", { tier, success_url, cancel_url }),
};

// ── Health ─────────────────────────────────────────────────────────────
export const health = () => req<any>("GET", "/health");
