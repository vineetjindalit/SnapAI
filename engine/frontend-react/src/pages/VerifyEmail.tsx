import { useEffect, useState } from "react";
import { CheckCircle2, XCircle } from "lucide-react";
import { useAuth } from "@/store/auth";
import { AuthShell } from "@/components/AuthShell";

interface Props { token: string; onContinue: () => void; }

export function VerifyEmail({ token, onContinue }: Props) {
  const verifyEmail = useAuth((s) => s.verifyEmail);
  const [state, setState] = useState<"loading" | "ok" | "error">("loading");
  const [msg, setMsg]     = useState("");

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const r = await verifyEmail(token);
        if (cancelled) return;
        setMsg(`Verified ${r.email}`); setState("ok");
      } catch (e: any) {
        if (cancelled) return;
        setMsg(e?.message ?? "Token invalid or expired"); setState("error");
      }
    })();
    return () => { cancelled = true; };
  }, [token, verifyEmail]);

  return (
    <AuthShell title="Email verification">
      {state === "loading" && (
        <div className="text-center text-slate-600 py-4">
          Confirming your email…
        </div>
      )}
      {state === "ok" && (
        <div className="text-center py-2">
          <CheckCircle2 className="w-12 h-12 text-emerald-500 mx-auto mb-3" />
          <p className="text-emerald-700 font-medium mb-4">{msg}</p>
          <button onClick={onContinue} className="btn-primary">
            Continue to SnapAI
          </button>
        </div>
      )}
      {state === "error" && (
        <div className="text-center py-2">
          <XCircle className="w-12 h-12 text-rose-500 mx-auto mb-3" />
          <p className="text-rose-700 font-medium mb-4">{msg}</p>
          <button onClick={onContinue} className="btn-ghost">
            Back to SnapAI
          </button>
        </div>
      )}
    </AuthShell>
  );
}
