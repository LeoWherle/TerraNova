import type { KnifeOverlayProps } from "@/hooks/useKnifeTool";

export function KnifeOverlay({ active, points }: KnifeOverlayProps) {
  if (!active || points.length < 2) return null;

  const polylineStr = points.map((p) => `${p.x},${p.y}`).join(" ");

  return (
    <svg className="knife-overlay">
      {/* Glow layer */}
      <polyline
        points={polylineStr}
        fill="none"
        stroke="#ff4444"
        strokeWidth={4}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.3}
      />
      {/* Main line */}
      <polyline
        points={polylineStr}
        fill="none"
        stroke="#ff4444"
        strokeWidth={2}
        strokeDasharray="6 4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
