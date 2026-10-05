import { useMemo } from "react";

export function ScoreTimeline({ values }: { values: number[] }) {
  const path = useMemo(() => {
    if (values.length < 2) return "";
    const w = 600, h = 60;
    const max = Math.max(0.5, ...values);
    const dx  = w / (values.length - 1);
    return values.map((v, i) =>
      `${i === 0 ? "M" : "L"} ${i * dx},${h - (v / max) * h}`).join(" ");
  }, [values]);

  return (
    <svg viewBox="0 0 600 60" className="w-full h-16">
      <line x1="0" y1="30" x2="600" y2="30" stroke="rgba(100,116,139,.3)" strokeDasharray="2 4" />
      <path d={path} stroke="#a78bfa" strokeWidth="2" fill="none"
            strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}
