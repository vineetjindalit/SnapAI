import { useState } from "react";
import { Mail } from "lucide-react";
import { useAuth } from "@/store/auth";
import { AuthShell, FieldLabel, TextInput, ErrorBanner,
         SuccessBanner, MagneticSubmit } from "@/components/AuthShell";

interface Props { onBack: () => void; }

export function ForgotPassword({ onBack }: Props) {
  const forgot = useAuth((s) => s.forgotPassword);
  const [email, setEmail] = useState("");
  const [busy,  setBusy]  = useState(false);
  const [err,   setErr]   = useState<string | null>(null);
  const [done,  setDone]  = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true); setErr(null);
    try { await forgot(email.trim()); setDone(true); }
    catch (e: any) { setErr(e?.message ?? String(e)); }
    finally { setBusy(false); }
  };

  return (
    <AuthShell
      title="Forgot your password?"
      subtitle="No worries — we'll email you a reset link."
      footer={
        <button onClick={onBack}
          className="text-brand-600 font-medium hover:underline">
          ← Back to sign in
        </button>
      }>
      {done ? (
        <SuccessBanner msg={
          `If an account exists for "${email}", we just sent a reset link. ` +
          "Check your inbox (and spam) — link expires in an hour."
        } />
      ) : (
        <form onSubmit={submit} className="space-y-4">
          <div>
            <FieldLabel>Email</FieldLabel>
            <TextInput type="email" required autoFocus
              placeholder="you@example.com"
              value={email} onChange={(e) => setEmail(e.target.value)} />
          </div>
          <ErrorBanner msg={err} />
          <MagneticSubmit loading={busy} success={done}>
            <Mail className="w-5 h-5" />
            Send reset link
          </MagneticSubmit>
        </form>
      )}
    </AuthShell>
  );
}
