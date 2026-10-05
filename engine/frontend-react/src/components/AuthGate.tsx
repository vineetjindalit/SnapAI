import { useEffect, useState } from "react";
import { useAuth } from "@/store/auth";
import { Login }          from "@/pages/Login";
import { Register }       from "@/pages/Register";
import { ForgotPassword } from "@/pages/ForgotPassword";
import { ResetPassword }  from "@/pages/ResetPassword";
import { VerifyEmail }    from "@/pages/VerifyEmail";

type Screen = "login" | "register" | "forgot" | "reset" | "verify";

interface Props { children: React.ReactNode; }

/** Detects ?token= deeplinks for /reset and /verify on first paint. */
function deeplinkScreen(): { screen: Screen | null; token: string | null } {
  const path  = location.pathname;
  const url   = new URL(location.href);
  const token = url.searchParams.get("token");
  if (path === "/reset"  && token) return { screen: "reset",  token };
  if (path === "/verify" && token) return { screen: "verify", token };
  return { screen: null, token: null };
}


/**
 * AuthGate — when SNAPPY_AUTH=1 and the user isn't logged in, show the
 * auth screens. When AUTH is off (dev mode), let the children render.
 *
 * Also handles email-link deeplinks: /reset?token=... and /verify?token=...
 */
export function AuthGate({ children }: Props) {
  const init        = useAuth((s) => s.init);
  const authEnabled = useAuth((s) => s.authEnabled);
  const user        = useAuth((s) => s.user);

  const [booted, setBooted] = useState(false);
  const [screen, setScreen] = useState<Screen>("login");
  const [deepToken, setDeepToken] = useState<string | null>(null);

  // Boot — probe /auth/me to learn auth_enabled + load user
  useEffect(() => {
    const dl = deeplinkScreen();
    if (dl.screen) {
      setScreen(dl.screen); setDeepToken(dl.token);
    }
    init().finally(() => setBooted(true));
  }, [init]);

  if (!booted) {
    return (
      <div className="min-h-screen grid place-items-center text-slate-500">
        <span className="animate-pulse">Loading SnapAI…</span>
      </div>
    );
  }

  // Email verify deeplink — show even if logged in / auth disabled
  if (screen === "verify" && deepToken) {
    return <VerifyEmail token={deepToken}
                        onContinue={() => { history.replaceState(null, "", "/"); window.location.reload(); }} />;
  }
  if (screen === "reset" && deepToken) {
    return <ResetPassword token={deepToken}
                          onSuccess={() => { history.replaceState(null, "", "/"); setScreen("login"); setDeepToken(null); }} />;
  }

  // Auth not enforced → just pass through
  if (!authEnabled) return <>{children}</>;

  // Logged in → app
  if (user && !user.anonymous) return <>{children}</>;

  // Otherwise show the auth screens
  if (screen === "register")
    return <Register onSwitchToLogin={() => setScreen("login")} />;
  if (screen === "forgot")
    return <ForgotPassword onBack={() => setScreen("login")} />;
  return (
    <Login
      onSwitchToRegister={() => setScreen("register")}
      onSwitchToForgot={() => setScreen("forgot")} />
  );
}
