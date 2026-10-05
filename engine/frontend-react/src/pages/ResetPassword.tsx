import { useState } from "react";
import { KeyRound } from "lucide-react";
import { useAuth } from "@/store/auth";
import { AuthShell, FieldLabel, TextInput,
         ErrorBanner, SuccessBanner } from "@/components/AuthShell";

interface Props {
  token: string;
  onSuccess: () => void;
}

export function ResetPassword({ token, onSuccess }: Props) {
  const reset = useAuth((s) => s.resetPassword);
  const [pw,    setPw]    = useState("");
  const [pw2,   setPw2]   = useState("");
  const [busy,  setBusy]  = useState(false);
  const [err,   setErr]   = useState<string | null>(null);
  const [done,  setDone]  = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (pw !== pw2) { setErr("Passwords don't match"); return; }
    setBusy(true); setErr(null);
    try {
      await reset(token, pw); setDone(true);
      setTimeout(onSuccess, 1500);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally { setBusy(false); }
  };

  return (
    <AuthShell title="Set a new password"
      subtitle="Make it strong — 8+ chars, letters and numbers.">
      {done ? (
        <SuccessBanner msg="Password updated! Redirecting to sign-in…" />
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <div>
            <FieldLabel>New password</FieldLabel>
            <TextInput type="password" required autoFocus
              placeholder="•••••••• "
              value={pw} onChange={(e) => setPw(e.target.value)} />
          </div>
          <div>
            <FieldLabel>Confirm new password</FieldLabel>
            <TextInput type="password" required
              placeholder="Type it again"
              value={pw2} onChange={(e) => setPw2(e.target.value)} />
          </div>
          <ErrorBanner msg={err} />
          <button type="submit" disabled={busy}
            className="btn-primary w-full justify-center text-base py-3">
            <KeyRound className="w-5 h-5" />
            {busy ? "Updating…" : "Update password"}
          </button>
        </form>
      )}
    </AuthShell>
  );
}
