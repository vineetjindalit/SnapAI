// frontend-react/src/lib/motionUtils.ts
// Shared motion primitives: reduced-motion awareness and the native View
// Transitions API with a plain instant fallback.
//
// Design rule for this app: motion is FEEDBACK, not decoration. Every helper
// here is driven by a real application state value (socket phase, capture
// confidence, an actual capture event). Nothing animates on a timer that
// isn't backed by something the backend actually did.

import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

/** True when the user has asked the OS for reduced motion. Live-updates if
 *  they change the setting mid-session. */
export function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined"
      && typeof window.matchMedia === "function"
      && window.matchMedia(QUERY).matches,
  );

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mq = window.matchMedia(QUERY);
    const onChange = (e: MediaQueryListEvent) => setReduced(e.matches);
    // Safari < 14 only has the deprecated addListener signature.
    if (mq.addEventListener) mq.addEventListener("change", onChange);
    else mq.addListener(onChange);
    return () => {
      if (mq.removeEventListener) mq.removeEventListener("change", onChange);
      else mq.removeListener(onChange);
    };
  }, []);

  return reduced;
}

/** Non-hook read, for call sites outside React render (event handlers). */
export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined"
    && typeof window.matchMedia === "function"
    && window.matchMedia(QUERY).matches;
}

type ViewTransitionCapableDocument = Document & {
  startViewTransition?: (cb: () => void | Promise<void>) => {
    finished: Promise<void>;
    ready: Promise<void>;
    updateCallbackDone: Promise<void>;
  };
};

/**
 * Run a DOM-mutating update inside a native View Transition when the browser
 * supports it AND the user hasn't asked for reduced motion. Otherwise the
 * update is applied immediately and identically — the fallback is "no
 * animation", never "no update", so state changes can never be lost or
 * delayed by this helper.
 */
export function withViewTransition(update: () => void): void {
  const doc = document as ViewTransitionCapableDocument;
  if (typeof doc.startViewTransition !== "function" || prefersReducedMotion()) {
    update();
    return;
  }
  try {
    const transition = doc.startViewTransition(update);
    // startViewTransition's promises (ready/finished) REJECT when this
    // transition is legitimately superseded by a newer one, or the document
    // becomes hidden mid-transition — both are normal, expected outcomes
    // (e.g. a user clicking through two capture cards quickly), not errors.
    // Left unhandled, each rejection surfaces as an uncaught
    // InvalidStateError — the update itself already committed via the
    // callback regardless, so there's nothing to recover; this just
    // prevents a harmless abort from being reported as a real failure.
    transition.ready.catch(() => {});
    transition.finished.catch(() => {});
    transition.updateCallbackDone.catch(() => {});
  } catch {
    // Any synchronous failure in the transition machinery must not swallow
    // the update — apply it directly.
    update();
  }
}

export const supportsViewTransitions = (): boolean =>
  typeof document !== "undefined"
  && typeof (document as ViewTransitionCapableDocument).startViewTransition === "function";

/** Spring presets, so timing is consistent across components. */
export const SPRING = {
  /** UI chrome: snappy, minimal overshoot. */
  ui:     { type: "spring", stiffness: 380, damping: 32, mass: 0.7 } as const,
  /** Photos settling into a grid: softer, slightly heavier. */
  photo:  { type: "spring", stiffness: 260, damping: 28, mass: 0.9 } as const,
  /** Large shared-element morphs. */
  morph:  { type: "spring", stiffness: 300, damping: 34, mass: 0.9 } as const,
};
