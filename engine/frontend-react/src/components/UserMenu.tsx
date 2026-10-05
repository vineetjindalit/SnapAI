import { useEffect, useRef, useState } from "react";
import { ChevronDown, LogOut, Mail, User as UserIcon } from "lucide-react";
import { useAuth } from "@/store/auth";

export function UserMenu({ dark }: { dark: boolean }) {
  const user        = useAuth((s) => s.user);
  const authEnabled = useAuth((s) => s.authEnabled);
  const logout      = useAuth((s) => s.logout);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onClick = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, []);

  if (!authEnabled || !user || user.anonymous) return null;

  const initials = user.email.slice(0, 2).toUpperCase();

  return (
    <div ref={ref} className="relative">
      <button onClick={() => setOpen(!open)}
        className={"flex items-center gap-2 px-2 py-1 rounded-full transition " +
                   (dark ? "hover:bg-ink-800" : "hover:bg-slate-100")}>
        <span className={"w-7 h-7 rounded-full grid place-items-center " +
                         "bg-gradient-to-br from-brand-500 to-rose-500 " +
                         "text-white text-[11px] font-bold"}>
          {initials}
        </span>
        <ChevronDown className={"w-3.5 h-3.5 " +
                                (dark ? "text-slate-400" : "text-slate-500")} />
      </button>

      {open && (
        <div className={"absolute right-0 mt-2 w-64 rounded-xl shadow-xl " +
                        "border z-50 overflow-hidden " +
                        (dark ? "bg-ink-800 border-ink-700"
                              : "bg-white border-slate-200")}>
          <div className={"px-4 py-3 border-b " +
                          (dark ? "border-ink-700" : "border-slate-100")}>
            <div className="flex items-center gap-2 min-w-0">
              <UserIcon className={"w-4 h-4 shrink-0 " +
                                   (dark ? "text-slate-400" : "text-slate-500")} />
              <div className={"text-sm font-medium truncate " +
                              (dark ? "text-slate-100" : "text-ink-900")}>
                {user.email}
              </div>
            </div>
            <div className="flex items-center gap-2 mt-1.5">
              <span className={"px-2 py-0.5 rounded text-[10px] font-mono " +
                               "uppercase tracking-wider " +
                               (dark ? "bg-ink-700 text-slate-300"
                                     : "bg-slate-100 text-slate-600")}>
                {user.tier}
              </span>
              {!user.verified && (
                <span className="flex items-center gap-1 px-2 py-0.5 rounded
                                 text-[10px] bg-amber-100 text-amber-700">
                  <Mail className="w-3 h-3" /> verify email
                </span>
              )}
            </div>
          </div>

          <button onClick={() => { setOpen(false); logout(); }}
            className={"w-full flex items-center gap-2 px-4 py-2.5 text-sm " +
                       "transition " +
                       (dark ? "hover:bg-ink-700 text-slate-200"
                             : "hover:bg-slate-50 text-slate-700")}>
            <LogOut className="w-4 h-4" />
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}
