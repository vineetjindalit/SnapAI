import { useState } from "react";
import { LogIn } from "lucide-react";
import { useAuth } from "@/store/auth";
import { AuthShell, FieldLabel, TextInput, ErrorBanner,
         MagneticSubmit } from "@/components/AuthShell";

interface Props {
  onSuccess?: () => void;
  onSwitchToRegister: () => void;
  onSwitchToForgot:   () => void;
}

export function Login({ onSuccess, onSwitchToRegister, onSwitchToForgot }: Props) {
  const login   = useAuth((s) => s.login);
  const loading = useAuth((s) => s.loading);
  const error   = useAuth((s) => s.error);
  const [email, setEmail] = useState("");
  const [pw,    setPw]    = useState("");
  // Drives the button's checkmark morph — set ONLY after login actually
  // resolved, so the success state can never appear for a failed sign-in.
  const [signedIn, setSignedIn] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await login(email.trim(), pw);
      setSignedIn(true);
      onSuccess?.();
    } catch {/* error in store */}
  };

  return (
    <AuthShell
      title="Welcome back"
      subtitle="Sign in to continue capturing"
      footer={
        <>
          New to SnapAI?{" "}
          <button onClick={onSwitchToRegister}
            className="text-brand-600 font-medium hover:underline">
            Create an account
          </button>
        </>
      }>
      <form onSubmit={submit} className="space-y-4">
        <div>
          <FieldLabel>Email</FieldLabel>
          <TextInput type="email" required autoComplete="email"
            placeholder="you@example.com"
            value={email} onChange={(e) => setEmail(e.target.value)} />
        </div>
        <div>
          <div className="flex items-baseline justify-between mb-1.5">
            <FieldLabel>Password</FieldLabel>
            <button type="button" onClick={onSwitchToForgot}
              className="text-xs text-brand-600 hover:underline">
              Forgot?
            </button>
          </div>
          <TextInput type="password" required autoComplete="current-password"
            placeholder="•••••••• "
            value={pw} onChange={(e) => setPw(e.target.value)} />
        </div>

        <ErrorBanner msg={error} />

        <MagneticSubmit loading={loading} success={signedIn}>
          <LogIn className="w-5 h-5" />
          Sign in
        </MagneticSubmit>
      </form>
    </AuthShell>
  );
}
