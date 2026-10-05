// frontend-react/src/components/PhotoLightbox.tsx
//
// Full-size photo view, opened by morphing FROM the grid thumbnail rather
// than cross-fading to a separate element.
//
// Uses the native View Transitions API where available: the thumbnail and the
// full image share a `view-transition-name`, so the browser animates the SAME
// element growing. Where it's unsupported (or reduced-motion is on) the
// lightbox simply appears — the fallback is "no animation", never "no view".

import { useEffect } from "react";
import { X } from "lucide-react";
import { useReducedMotion } from "@/lib/motionUtils";

export interface LightboxPhoto {
  url: string;
  moment?: string;
  index?: number;
}

interface Props {
  photo: LightboxPhoto | null;
  onClose: () => void;
}

export function PhotoLightbox({ photo, onClose }: Props) {
  const reduced = useReducedMotion();

  // Esc to close + lock background scroll while open.
  useEffect(() => {
    if (!photo) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [photo, onClose]);

  if (!photo) return null;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={photo.moment ? `Photo: ${photo.moment}` : "Photo"}
      onClick={onClose}
      className="fixed inset-0 z-50 grid place-items-center bg-black/85 backdrop-blur-sm p-4"
      style={reduced ? undefined : { animation: "fadeIn 0.18s ease-out" }}
    >
      <button
        onClick={onClose}
        aria-label="Close photo"
        className="absolute top-4 right-4 w-11 h-11 rounded-full bg-white/15
                   hover:bg-white/25 active:bg-white/30 backdrop-blur
                   grid place-items-center text-white transition">
        <X className="w-5 h-5" />
      </button>

      <figure className="max-w-5xl w-full" onClick={(e) => e.stopPropagation()}>
        <img
          src={photo.url}
          alt={photo.moment ?? "Captured moment"}
          // Same view-transition-name as the originating thumbnail — this is
          // what makes the browser morph one into the other instead of
          // fading between two unrelated elements.
          style={{ viewTransitionName: "photo-morph" }}
          className="w-full max-h-[80vh] object-contain rounded-xl shadow-2xl"
        />
        {photo.moment && (
          <figcaption className="mt-3 text-center text-sm text-white/80 capitalize">
            {photo.moment.replace(/_/g, " ")}
            {typeof photo.index === "number" && (
              <span className="ml-2 text-white/50">#{photo.index + 1}</span>
            )}
          </figcaption>
        )}
      </figure>
    </div>
  );
}
