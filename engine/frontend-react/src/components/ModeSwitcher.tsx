import { useEffect } from "react";
import { Sparkles, Code2, Users } from "lucide-react";
import { useSession } from "@/store/session";
import { UserMenu } from "@/components/UserMenu";

export function ModeSwitcher() {
  const mode    = useSession((s) => s.mode);
  const setMode = useSession((s) => s.setMode);
  const sid     = useSession((s) => s.sid);
  const health  = useSession((s) => s.health);
  // Developer mode is owner-only — shown solely where the backend sets
  // SNAPPY_DEV_UI=1 (your Mac). Friends on the cloud never see the toggle.
  const devAllowed = !!health?.dev_ui;
  const dev = mode === "developer" || mode === "admin";

  // A persisted "developer"/"admin" mode must not leak through on a build
  // where dev is off (e.g. a friend who once toggled it, then opens the
  // cloud app). Admin is owner-only for the same reason developer is.
  useEffect(() => {
    if (!devAllowed && (mode === "developer" || mode === "admin")) setMode("user");
  }, [devAllowed, mode, setMode]);

  return (
    <header className={
      "sticky top-0 z-30 flex items-center justify-between px-5 py-3 backdrop-blur " +
      (dev ? "bg-ink-900/80 border-b border-ink-700"
           : "bg-cream-50/80 border-b border-slate-200")
    }>
      {/* Brand */}
      <div className="flex items-center gap-2">
        <span className="text-2xl">📸</span>
        <div>
          <div className={"font-bold text-lg leading-none " +
                          (dev ? "text-slate-100" : "text-ink-900")}>SnapAI</div>
          <div className={"text-[10px] uppercase tracking-widest " +
                          (dev ? "text-slate-400" : "text-slate-500")}>
            ai event photographer
          </div>
        </div>
        {sid && (
          <div className={"ml-3 px-2 py-0.5 rounded text-[10px] font-mono " +
                          (dev ? "bg-ink-700 text-slate-300"
                               : "bg-slate-100 text-slate-600")}>
            {sid}
          </div>
        )}
      </div>

      {/* Right cluster: mode toggle (owner-only) + user menu */}
      <div className="flex items-center gap-3">
        {devAllowed && (
        <div className={"inline-flex p-1 rounded-full text-sm " +
                        (dev ? "bg-ink-800 border border-ink-700"
                             : "bg-white border border-slate-200 shadow-sm")}>
          <button
            onClick={() => setMode("user")}
            className={"flex items-center gap-1.5 px-3.5 py-1.5 rounded-full transition " +
                       (mode === "user"
                         ? "bg-gradient-to-br from-brand-500 to-brand-700 text-white shadow"
                         : (dev ? "text-slate-400 hover:text-slate-200"
                                : "text-slate-600 hover:text-slate-900"))}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span className="font-medium">User</span>
          </button>
          <button
            onClick={() => setMode("developer")}
            className={"flex items-center gap-1.5 px-3.5 py-1.5 rounded-full transition " +
                       (mode === "developer"
                         ? "bg-ink-700 text-white shadow"
                         : (dev ? "text-slate-400 hover:text-slate-200"
                                : "text-slate-600 hover:text-slate-900"))}
          >
            <Code2 className="w-3.5 h-3.5" />
            <span className="font-medium">Developer</span>
          </button>
          <button
            onClick={() => setMode("admin")}
            title="Who's using SnapAI right now, and any issues"
            className={"flex items-center gap-1.5 px-3.5 py-1.5 rounded-full transition " +
                       (mode === "admin"
                         ? "bg-ink-700 text-white shadow"
                         : (dev ? "text-slate-400 hover:text-slate-200"
                                : "text-slate-600 hover:text-slate-900"))}
          >
            <Users className="w-3.5 h-3.5" />
            <span className="font-medium">Admin</span>
          </button>
        </div>
        )}

        <UserMenu dark={dev} />
      </div>
    </header>
  );
}
