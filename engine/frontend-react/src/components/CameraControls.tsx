// frontend-react/src/components/CameraControls.tsx
//
// Hardware camera controls — best-effort via MediaStreamTrack.applyConstraints.
//
// Each camera device exposes different capabilities. A modern phone camera
// might give us zoom + focus + exposure + ISO. A laptop FaceTime camera
// gives us almost nothing. We probe getCapabilities() and only render the
// sliders that the active device actually supports.
//
// Software enhancement (brightness, contrast, saturation, sharpness) is
// always applied server-side — see backend/models/enhance.py. THIS file
// is for the small subset of HARDWARE knobs the browser exposes; they
// affect the raw frames we send to the server before processing.

import { useEffect, useState } from "react";
import { Camera, ZoomIn, Sun, Aperture, Focus, ChevronDown, ChevronUp } from "lucide-react";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  dark?: boolean;
}

interface CapInfo {
  zoom?:     { min: number; max: number; step: number };
  focusDistance?: { min: number; max: number; step: number };
  exposureCompensation?: { min: number; max: number; step: number };
  iso?:      { min: number; max: number; step: number };
  brightness?: { min: number; max: number; step: number };
  contrast?:   { min: number; max: number; step: number };
  saturation?: { min: number; max: number; step: number };
  sharpness?:  { min: number; max: number; step: number };
  whiteBalanceMode?: string[];
  focusMode?: string[];
  exposureMode?: string[];
}

export function CameraControls({ videoRef, dark = false }: Props) {
  const [caps, setCaps] = useState<CapInfo | null>(null);
  const [track, setTrack] = useState<MediaStreamTrack | null>(null);
  const [values, setValues] = useState<Record<string, number | string>>({});
  const [collapsed, setCollapsed] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Re-probe whenever the video element acquires a srcObject
  useEffect(() => {
    const tryProbe = () => {
      const v = videoRef.current;
      if (!v || !v.srcObject) return;
      const t = (v.srcObject as MediaStream).getVideoTracks()[0];
      if (!t) return;
      try {
        const c = (t.getCapabilities?.() ?? {}) as any;
        const s = (t.getSettings?.()    ?? {}) as any;
        setTrack(t); setCaps(c); setValues(s);
      } catch (e: any) { setError(e.message); }
    };
    const id = setInterval(tryProbe, 1000);
    tryProbe();
    return () => clearInterval(id);
  }, [videoRef]);

  // Apply a constraint
  const apply = async (key: string, value: number | string) => {
    if (!track) return;
    try {
      // Most settings go inside `advanced` — that's the WICG-spec form for
      // device-specific overrides. Plain top-level constraints are
      // sometimes silently ignored by browsers.
      await track.applyConstraints({ advanced: [{ [key]: value } as any] });
      setValues((v) => ({ ...v, [key]: value }));
    } catch (e: any) {
      setError(`${key} not supported: ${e.message}`);
      setTimeout(() => setError(null), 3000);
    }
  };

  // Nothing to show
  const supports = caps ? Object.keys(caps).filter((k) => {
    const v = (caps as any)[k];
    return v && (typeof v === "object" || Array.isArray(v));
  }) : [];

  // Render
  const cardCls = dark
    ? "rounded-lg bg-ink-800 border border-ink-700 text-slate-200"
    : "rounded-2xl bg-white border border-slate-100 shadow-sm";
  const subText = dark ? "text-slate-400" : "text-slate-500";

  if (!track) return null;
  if (supports.length === 0) {
    // Nothing the browser can adjust — surface this fact so user knows.
    return (
      <div className={"p-3 text-xs " + cardCls + " " + subText}>
        <div className="flex items-center gap-2">
          <Camera className="w-3.5 h-3.5" />
          Camera offers no hardware controls (most laptop webcams don't).
          Software enhancement is always on — see capture metadata.
        </div>
      </div>
    );
  }

  if (collapsed) {
    return (
      <button onClick={() => setCollapsed(false)}
        className={"w-full text-left px-4 py-3 flex items-center gap-2 " +
                   "transition hover:opacity-90 " + cardCls}>
        <Camera className="w-4 h-4" />
        <span className="text-sm font-medium">Camera controls</span>
        <span className={"ml-auto text-xs " + subText}>
          {supports.length} available
        </span>
        <ChevronDown className="w-4 h-4" />
      </button>
    );
  }

  return (
    <div className={"p-4 " + cardCls}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 font-medium">
          <Camera className="w-4 h-4" />
          Camera controls
        </div>
        <button onClick={() => setCollapsed(true)}
          className={"text-xs " + subText + " hover:underline flex items-center gap-1"}>
          collapse <ChevronUp className="w-3 h-3" />
        </button>
      </div>

      {/* Zoom */}
      {caps?.zoom && (
        <SliderRow icon={<ZoomIn className="w-3.5 h-3.5"/>}
          label="Zoom" cap={caps.zoom} value={Number(values.zoom) || caps.zoom.min}
          onChange={(v) => apply("zoom", v)} dark={dark}
          format={(v) => `${v.toFixed(1)}×`} />
      )}

      {/* Focus distance */}
      {caps?.focusDistance && (
        <SliderRow icon={<Focus className="w-3.5 h-3.5"/>}
          label="Focus" cap={caps.focusDistance}
          value={Number(values.focusDistance) || caps.focusDistance.min}
          onChange={(v) => apply("focusDistance", v)} dark={dark} />
      )}

      {/* Exposure */}
      {caps?.exposureCompensation && (
        <SliderRow icon={<Sun className="w-3.5 h-3.5"/>}
          label="Exposure" cap={caps.exposureCompensation}
          value={Number(values.exposureCompensation) || 0}
          onChange={(v) => apply("exposureCompensation", v)} dark={dark}
          format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)} EV`} />
      )}

      {/* ISO */}
      {caps?.iso && (
        <SliderRow icon={<Aperture className="w-3.5 h-3.5"/>}
          label="ISO" cap={caps.iso}
          value={Number(values.iso) || caps.iso.min}
          onChange={(v) => apply("iso", v)} dark={dark}
          format={(v) => `${Math.round(v)}`} />
      )}

      {/* Brightness / contrast / saturation / sharpness — phone-only */}
      {(["brightness", "contrast", "saturation", "sharpness"] as const).map((k) =>
        (caps as any)?.[k] ? (
          <SliderRow key={k} icon={null} label={k.charAt(0).toUpperCase() + k.slice(1)}
            cap={(caps as any)[k]} value={Number((values as any)[k]) || 0}
            onChange={(v) => apply(k, v)} dark={dark} />
        ) : null
      )}

      {/* White-balance mode (enum) */}
      {caps?.whiteBalanceMode && (
        <ModeRow label="White balance" options={caps.whiteBalanceMode}
          value={values.whiteBalanceMode as string}
          onChange={(v) => apply("whiteBalanceMode", v)} dark={dark} />
      )}

      {/* Focus mode (enum) */}
      {caps?.focusMode && (
        <ModeRow label="Focus mode" options={caps.focusMode}
          value={values.focusMode as string}
          onChange={(v) => apply("focusMode", v)} dark={dark} />
      )}

      {error && (
        <div className="mt-2 text-xs text-rose-500">{error}</div>
      )}
    </div>
  );
}


function SliderRow({ icon, label, cap, value, onChange, dark, format }: {
  icon: React.ReactNode; label: string;
  cap: { min: number; max: number; step: number };
  value: number; onChange: (v: number) => void; dark: boolean;
  format?: (v: number) => string;
}) {
  const subText = dark ? "text-slate-400" : "text-slate-500";
  return (
    <div className="mb-3">
      <div className="flex items-center justify-between mb-1 text-xs">
        <span className="flex items-center gap-1.5 font-medium">{icon}{label}</span>
        <span className={"font-mono " + subText}>
          {format ? format(value) : value.toFixed(2)}
        </span>
      </div>
      <input type="range"
        min={cap.min} max={cap.max} step={cap.step || (cap.max - cap.min) / 100}
        value={value} onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-brand-500" />
    </div>
  );
}


function ModeRow({ label, options, value, onChange, dark }: {
  label: string; options: string[]; value: string;
  onChange: (v: string) => void; dark: boolean;
}) {
  const inputCls = dark
    ? "bg-ink-700 border-ink-700 text-slate-100"
    : "bg-white border-slate-200 text-ink-900";
  return (
    <div className="mb-3">
      <div className="text-xs font-medium mb-1">{label}</div>
      <div className="flex gap-1 flex-wrap">
        {options.map((o) => (
          <button key={o} onClick={() => onChange(o)}
            className={"px-2.5 py-1 rounded-md text-xs border transition " +
              (value === o
                ? "bg-brand-500 text-white border-brand-500"
                : inputCls + " hover:border-brand-500")}>
            {o}
          </button>
        ))}
      </div>
    </div>
  );
}
