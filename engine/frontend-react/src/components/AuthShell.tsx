// frontend-react/src/components/AuthShell.tsx
// Shared visual frame for all auth screens. Centred card, subtle gradient.

import { ReactNode, useRef, useState } from "react";
import { motion } from "motion/react";
import { Check, Loader2 } from "lucide-react";
import { useReducedMotion, SPRING } from "@/lib/motionUtils";

interface Props {
  title: string;
  subtitle?: string;
  children: ReactNode;
  footer?: ReactNode;
}

export function AuthShell({ title, subtitle, children, footer }: Props) {
  return (
    <div className="min-h-screen flex flex-col items-center justify-center
                    px-5 py-12 bg-gradient-to-br from-cream-100 via-cream-50
                    to-rose-100/30">
      <div className="w-full max-w-md">
        <div className="text-center mb-7">
          <div className="inline-block text-4xl mb-2">📸</div>
          <h1 className="text-3xl font-bold tracking-tight">{title}</h1>
          {subtitle && (
            <p className="mt-2 text-slate-600 text-sm">{subtitle}</p>
          )}
        </div>
        <div className="card-soft">
          {children}
        </div>
        {footer && (
          <div className="mt-6 text-center text-sm text-slate-600">
            {footer}
          </div>
        )}
      </div>
    </div>
  );
}


export function FieldLabel({ children }: { children: ReactNode }) {
  return (
    <label className="text-sm font-medium text-slate-700 block mb-1.5">
      {children}
    </label>
  );
}


export function TextInput(props: React.InputHTMLAttributes<HTMLInputElement>) {
  return (
    <input {...props}
      className={"w-full px-4 py-3 rounded-xl border border-slate-200 " +
                 "focus:border-brand-500 focus:ring-2 focus:ring-brand-500/20 " +
                 // Focus lifts the field slightly and deepens its shadow, so
                 // the active field is obvious without a jumpy layout shift.
                 "focus:shadow-md focus:shadow-brand-500/5 " +
                 "outline-none transition-all duration-200 " + (props.className ?? "")} />
  );
}


/**
 * Submit button that morphs in place rather than swapping in a separate
 * spinner element: label → spinner while the request is genuinely in flight
 * → checkmark once it actually succeeded.
 *
 * `loading` and `success` are passed in from real request state by the
 * caller; this component never fakes either. "Magnetic" = it leans a few
 * pixels toward the cursor on hover (pointer devices only, and never under
 * reduced motion).
 */
export function MagneticSubmit({
  children, loading = false, success = false, disabled = false, className = "",
}: {
  children: ReactNode; loading?: boolean; success?: boolean;
  disabled?: boolean; className?: string;
}) {
  const reduced = useReducedMotion();
  const ref = useRef<HTMLButtonElement>(null);
  const [pull, setPull] = useState({ x: 0, y: 0 });

  const onMove = (e: React.MouseEvent) => {
    if (reduced || loading || success) return;
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    // Small, clamped offset toward the pointer — enough to feel responsive,
    // never enough to move the hit target out from under the cursor.
    setPull({
      x: Math.max(-8, Math.min(8, (e.clientX - (r.left + r.width / 2)) * 0.12)),
      y: Math.max(-5, Math.min(5, (e.clientY - (r.top + r.height / 2)) * 0.18)),
    });
  };

  return (
    <motion.button
      ref={ref}
      type="submit"
      disabled={disabled || loading || success}
      onMouseMove={onMove}
      onMouseLeave={() => setPull({ x: 0, y: 0 })}
      animate={reduced ? { x: 0, y: 0 } : { x: pull.x, y: pull.y }}
      transition={reduced ? { duration: 0 } : SPRING.ui}
      whileTap={reduced ? undefined : { scale: 0.98 }}
      className={"btn-primary w-full justify-center text-base py-3 " +
                 "disabled:opacity-100 " +   // the morph conveys busy, not fading
                 (success ? "!from-emerald-500 !to-emerald-600 " : "") + className}>
      <motion.span layout={!reduced} className="inline-flex items-center gap-2">
        {success ? <Check className="w-5 h-5" />
         : loading ? <Loader2 className="w-5 h-5 animate-spin" />
         : children}
      </motion.span>
    </motion.button>
  );
}


export function ErrorBanner({ msg }: { msg: string | null }) {
  if (!msg) return null;
  return (
    <div className="p-3 rounded-lg bg-rose-50 border border-rose-200
                    text-rose-700 text-sm">
      {msg}
    </div>
  );
}


export function SuccessBanner({ msg }: { msg: string | null }) {
  if (!msg) return null;
  return (
    <div className="p-3 rounded-lg bg-emerald-50 border border-emerald-200
                    text-emerald-700 text-sm">
      {msg}
    </div>
  );
}
