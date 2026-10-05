import { useEffect } from "react";
import { Health } from "@/api/client";
import { useSession } from "@/store/session";
import { ModeSwitcher } from "@/components/ModeSwitcher";
import { AuthGate }     from "@/components/AuthGate";
import { Setup }      from "@/pages/Setup";
import { UserMode }   from "@/pages/UserMode";
import { DevMode }    from "@/pages/DevMode";
import { AdminPanel } from "@/pages/AdminPanel";

export default function App() {
  const mode    = useSession((s) => s.mode);
  const sid     = useSession((s) => s.sid);
  const health  = useSession((s) => s.health);
  const setHealth = useSession((s) => s.setHealth);
  // Developer mode renders only where the backend allows it (owner's Mac).
  const devAllowed = !!health?.dev_ui;

  // Toggle <body class="dev-mode"> for the dark theme.
  useEffect(() => {
    document.body.classList.toggle("dev-mode", mode === "developer" || mode === "admin");
  }, [mode]);

  // Health poll every 8 seconds — drives the "models available" indicator
  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      try {
        const h = await Health.get();
        if (!cancelled) setHealth(h);
      } catch {} /* server may be down briefly during dev restart */
    };
    tick();
    const id = setInterval(tick, 8000);
    return () => { cancelled = true; clearInterval(id); };
  }, [setHealth]);

  // Page routing — pure state, no router (fewer deps, instant nav).
  // Admin is checked BEFORE the `!sid` gate: the owner wants to see who's
  // using SnapAI right now without first starting their own capture session.
  let body: React.ReactNode;
  if (mode === "admin" && devAllowed)           body = <AdminPanel />;
  else if (!sid)                                body = <Setup />;
  else if (mode === "developer" && devAllowed)  body = <DevMode />;
  else                                          body = <UserMode />;

  return (
    <AuthGate>
      <div className="min-h-screen flex flex-col">
        <ModeSwitcher />
        <main className="flex-1">{body}</main>
      </div>
    </AuthGate>
  );
}
