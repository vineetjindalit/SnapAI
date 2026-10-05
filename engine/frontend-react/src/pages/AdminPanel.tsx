// frontend-react/src/pages/AdminPanel.tsx
// Owner-only Mac dashboard: every friend who's used SnapAI, every session
// they've run, live capture counts, and anything that looks broken — so the
// person running the party doesn't have to SSH in or grep logs to find out
// "did it actually work for everyone?"
//
// Reachable only when the backend reports dev_ui (SNAPPY_DEV_UI=1, set only
// by the Mac launchers) — the cloud deploy 404s the underlying endpoint, so
// friends can never see this even by guessing the URL (see App.tsx / ModeSwitcher).
import { useEffect, useState, useCallback } from "react";
import { AlertTriangle, RefreshCw, Users, Film, Clock } from "lucide-react";
import { Admin } from "@/api/client";
import type { AdminOverviewResponse } from "@/api/types";

function timeAgo(ts: number | null): string {
  if (!ts) return "—";
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function AdminPanel() {
  const [data, setData]       = useState<AdminOverviewResponse | null>(null);
  const [error, setError]     = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const r = await Admin.overview();
      setData(r); setError(null);
    } catch (e: any) {
      setError(e?.message || "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  // Poll every 5s so the owner can watch friends show up live during a party.
  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  const sessions = data?.sessions ?? [];
  const users    = data?.users ?? [];
  const withIssues = sessions.filter((s) => s.issues.length > 0);
  const activeNow   = sessions.filter((s) => s.active);
  const totalCaptures = sessions.reduce((n, s) => n + s.captures, 0);

  return (
    <div className="max-w-6xl mx-auto px-5 py-8 text-slate-100">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-xl font-bold flex items-center gap-2">
            <Users className="w-5 h-5 text-brand-400" /> Admin — who's using SnapAI
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Owner-only. Auto-refreshes every 5s. Never shown to friends.
          </p>
        </div>
        <button
          onClick={() => { setLoading(true); load(); }}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-ink-800 border border-ink-700 text-sm text-slate-300 hover:text-white hover:border-ink-600 transition"
        >
          <RefreshCw className="w-3.5 h-3.5" /> Refresh
        </button>
      </div>

      {error && (
        <div className="mb-5 px-4 py-3 rounded-lg bg-red-950/50 border border-red-800 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Summary tiles */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
        {[
          { label: "Friends registered", value: users.length, icon: Users },
          { label: "Sessions total",     value: sessions.length, icon: Film },
          { label: "Active right now",   value: activeNow.length, icon: Clock },
          { label: "Photos captured",    value: totalCaptures, icon: Film },
        ].map(({ label, value, icon: Icon }) => (
          <div key={label} className="rounded-xl bg-ink-800/60 border border-ink-700 px-4 py-3">
            <div className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-slate-400">
              <Icon className="w-3 h-3" /> {label}
            </div>
            <div className="text-2xl font-bold mt-1">{value}</div>
          </div>
        ))}
      </div>

      {/* Issues banner */}
      {withIssues.length > 0 && (
        <div className="mb-6 rounded-xl bg-amber-950/40 border border-amber-800 px-4 py-3">
          <div className="flex items-center gap-2 text-amber-300 font-semibold text-sm mb-2">
            <AlertTriangle className="w-4 h-4" /> {withIssues.length} session(s) flagged
          </div>
          <ul className="space-y-1 text-sm text-amber-200/90">
            {withIssues.map((s) => (
              <li key={s.sid} className="font-mono text-xs">
                {s.owner_email || "guest"} · {s.sid} — {s.issues.join("; ")}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Users table */}
      <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-2">
        Friends ({users.length})
      </h2>
      <div className="overflow-x-auto rounded-xl border border-ink-700 mb-8">
        <table className="w-full text-sm">
          <thead className="bg-ink-800 text-slate-400 text-left">
            <tr>
              <th className="px-3 py-2 font-medium">Email</th>
              <th className="px-3 py-2 font-medium">Tier</th>
              <th className="px-3 py-2 font-medium">Role</th>
              <th className="px-3 py-2 font-medium">Joined</th>
              <th className="px-3 py-2 font-medium">Last login</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id} className="border-t border-ink-800 hover:bg-ink-800/40">
                <td className="px-3 py-2 font-mono text-slate-200">{u.email}</td>
                <td className="px-3 py-2 text-slate-400">{u.tier}</td>
                <td className="px-3 py-2 text-slate-400">{u.role}</td>
                <td className="px-3 py-2 text-slate-400">{timeAgo(u.created)}</td>
                <td className="px-3 py-2 text-slate-400">{timeAgo(u.last_login)}</td>
              </tr>
            ))}
            {!loading && users.length === 0 && (
              <tr><td colSpan={5} className="px-3 py-6 text-center text-slate-500">
                No registered friends yet.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Sessions table */}
      <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wide mb-2">
        Sessions ({sessions.length})
      </h2>
      <div className="overflow-x-auto rounded-xl border border-ink-700">
        <table className="w-full text-sm">
          <thead className="bg-ink-800 text-slate-400 text-left">
            <tr>
              <th className="px-3 py-2 font-medium">Friend</th>
              <th className="px-3 py-2 font-medium">Event</th>
              <th className="px-3 py-2 font-medium">Started</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Captures</th>
              <th className="px-3 py-2 font-medium">Issues</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((s) => (
              <tr key={s.sid} className={"border-t border-ink-800 hover:bg-ink-800/40 " +
                                          (s.issues.length ? "bg-red-950/20" : "")}>
                <td className="px-3 py-2 text-slate-200">{s.owner_email || "guest"}</td>
                <td className="px-3 py-2 text-slate-400">
                  {s.event_name} <span className="text-slate-600">({s.event_type})</span>
                </td>
                <td className="px-3 py-2 text-slate-400">{timeAgo(s.created)}</td>
                <td className="px-3 py-2">
                  {s.active ? (
                    <span className="inline-flex items-center gap-1 text-emerald-400">
                      <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" /> active
                    </span>
                  ) : s.ended ? (
                    <span className="text-slate-500">ended</span>
                  ) : (
                    <span className="text-slate-500">idle</span>
                  )}
                  {s.video_status && (
                    <span className="ml-2 text-[11px] text-slate-500 font-mono">
                      {s.video_status}
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 font-mono text-slate-200">{s.captures}</td>
                <td className="px-3 py-2">
                  {s.issues.length > 0 ? (
                    <span className="text-red-400 text-xs">{s.issues.join("; ")}</span>
                  ) : (
                    <span className="text-slate-600 text-xs">—</span>
                  )}
                </td>
              </tr>
            ))}
            {!loading && sessions.length === 0 && (
              <tr><td colSpan={6} className="px-3 py-6 text-center text-slate-500">
                No sessions yet.
              </td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
