import { useMemo, useState } from "react";
import { UserPlus } from "lucide-react";
import { useAuth } from "@/store/auth";
import { AuthShell, FieldLabel, TextInput, ErrorBanner,
         MagneticSubmit } from "@/components/AuthShell";

interface Props {
  onSuccess?: () => void;
  onSwitchToLogin: () => void;
}

const COMMON = ["password", "12345678", "qwerty12", "snappy123"];

function passwordStrength(pw: string): { score: 0|1|2|3|4; label: string } {
  if (!pw) return { score: 0, label: "" };
  if (COMMON.includes(pw.toLowerCase()))
    return { score: 1, label: "Too common — pick something unique" };
  let score = 0;
  if (pw.length >= 8)  score++;
  if (pw.length >= 12) score++;
  if (/[A-Z]/.test(pw) && /[a-z]/.test(pw)) score++;
  if (/\d/.test(pw))   score++;
  if (/[^A-Za-z0-9]/.test(pw)) score++;
  const label = ["Too short", "Weak", "Okay", "Good", "Strong"][Math.min(score, 4)];
  return { score: Math.min(score, 4) as any, label };
}

export function Register({ onSuccess, onSwitchToLogin }: Props) {
  const register = useAuth((s) => s.register);
  const loading  = useAuth((s) => s.loading);
  const error    = useAuth((s) => s.error);
  const [email, setEmail] = useState("");
  const [pw,    setPw]    = useState("");
  const strength = useMemo(() => passwordStrength(pw), [pw]);

  const valid = email.includes("@") && pw.length >= 8 &&
                /[A-Za-z]/.test(pw) && /\d/.test(pw);

  // Set ONLY after registration actually resolved — the checkmark morph must
  // never appear for a request that failed.
  const [created, setCreated] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!valid) return;
    try {
      await register(email.trim(), pw);
      setCreated(true);
      onSuccess?.();
    } catch {}
  };

  return (
    <AuthShell
      title="Create your account"
      subtitle="It takes 30 seconds. We never share your data."
      footer={
        <>
          Already have an account?{" "}
          <button onClick={onSwitchToLogin}
            className="text-brand-600 font-medium hover:underline">
            Sign in
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
          <FieldLabel>Password</FieldLabel>
          <TextInput type="password" required autoComplete="new-password"
            placeholder="At least 8 chars, letters + numbers"
            value={pw} onChange={(e) => setPw(e.target.value)} />
          {pw && (
            <div className="mt-2">
              <div className="flex gap-1 h-1">
                {[1,2,3,4].map((i) => (
                  <div key={i}
                    className={"flex-1 rounded-full transition " +
                      (strength.score >= i
                        ? (strength.score < 2 ? "bg-rose-400"
                         : strength.score < 3 ? "bg-amber-400"
                         : strength.score < 4 ? "bg-lime-400"
                         :                       "bg-emerald-500")
                        : "bg-slate-200")} />
                ))}
              </div>
              <p className="mt-1 text-xs text-slate-500">{strength.label}</p>
            </div>
          )}
        </div>

        <ErrorBanner msg={error} />

        <MagneticSubmit loading={loading} success={created} disabled={!valid}>
          <UserPlus className="w-5 h-5" />
          Create account
        </MagneticSubmit>

        <p className="text-center text-xs text-slate-500">
          By creating an account you agree to our Terms and Privacy Policy.
        </p>
      </form>
    </AuthShell>
  );
}
