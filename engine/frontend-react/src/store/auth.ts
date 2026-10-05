// frontend-react/src/store/auth.ts
// Auth state — current user + token + login/logout actions.

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { Auth, Health } from "@/api/client";

export interface CurrentUser {
  id: number;
  email: string;
  role: string;
  anonymous: boolean;
  verified: boolean;
  tier: string;
}

interface AuthState {
  authEnabled: boolean | null;       // null = not yet probed via /health
  user: CurrentUser | null;
  token: string | null;
  loading: boolean;
  error: string | null;

  // Actions
  init:           () => Promise<void>;
  login:          (email: string, password: string) => Promise<void>;
  register:       (email: string, password: string) => Promise<void>;
  logout:         () => Promise<void>;
  refresh:        () => Promise<void>;
  forgotPassword: (email: string) => Promise<{ sent: boolean }>;
  resetPassword:  (token: string, password: string) => Promise<void>;
  verifyEmail:    (token: string) => Promise<{ user_id: number; email: string }>;
  refetchMe:      () => Promise<void>;
}

const TOKEN_KEY = "snappy.auth_token";

export const useAuth = create<AuthState>()(
  persist(
    (set, get) => ({
      authEnabled: null,
      user: null,
      token: null,
      loading: false,
      error: null,

      // ── Boot: probe /health to see if auth is enforced, hydrate /me ──
      init: async () => {
        try {
          // /health now reports auth_enabled directly — the authoritative
          // signal. (Probing /auth/me alone was a bug: with auth ON and no
          // token it 401s, which read as "auth off" and skipped the gate.)
          const h = await Health.get();
          const enabled = !!(h as any)?.auth_enabled;
          // /me hydrates the logged-in user when we hold a valid token;
          // a 401 here just means "not logged in yet", not "auth off".
          const r = enabled ? await Auth.me().catch(() => null) : null;
          set({
            authEnabled: enabled,
            user: r?.user ?? null,
            // If auth is off, no token needed; clear stale one
            token: enabled ? get().token : null,
          });
        } catch {
          set({ authEnabled: false });
        }
      },

      login: async (email, password) => {
        set({ loading: true, error: null });
        try {
          const r = await Auth.login(email, password);
          localStorage.setItem(TOKEN_KEY, r.token);
          set({ user: r.user as CurrentUser, token: r.token, loading: false });
          await get().refetchMe();   // pick up tier + verified flags
        } catch (e: any) {
          set({ loading: false, error: e?.message ?? String(e) });
          throw e;
        }
      },

      register: async (email, password) => {
        set({ loading: true, error: null });
        try {
          await Auth.register(email, password);
          // Auto-login so the user lands inside the app
          await get().login(email, password);
        } catch (e: any) {
          set({ loading: false, error: e?.message ?? String(e) });
          throw e;
        }
      },

      logout: async () => {
        try { await Auth.logout(); } catch { /* idempotent */ }
        localStorage.removeItem(TOKEN_KEY);
        set({ user: null, token: null });
      },

      refresh: async () => {
        try {
          const r = await Auth.refresh();
          if (r?.token) {
            localStorage.setItem(TOKEN_KEY, r.token);
            set({ token: r.token });
          }
        } catch {
          // refresh failed — sign out cleanly
          localStorage.removeItem(TOKEN_KEY);
          set({ user: null, token: null });
        }
      },

      forgotPassword: async (email) => {
        const r = await Auth.forgotPassword(email);
        return r;
      },

      resetPassword: async (token, password) => {
        set({ loading: true, error: null });
        try {
          await Auth.resetPassword(token, password);
          set({ loading: false });
        } catch (e: any) {
          set({ loading: false, error: e?.message ?? String(e) });
          throw e;
        }
      },

      verifyEmail: async (token) => {
        const r = await Auth.verifyEmail(token);
        // If a user is currently logged in, refresh their `verified` flag
        await get().refetchMe();
        return r;
      },

      refetchMe: async () => {
        try {
          const r = await Auth.me();
          if (r?.user) set({ user: r.user as CurrentUser });
        } catch {/* token may be invalid; the gate will redirect */}
      },
    }),
    {
      name: "snappy-auth",
      partialize: (s) => ({ token: s.token, user: s.user }),
    }
  )
);
