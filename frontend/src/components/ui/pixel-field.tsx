"use client";

import { useMemo } from "react";

/**
 * Ambient "tech whimsical" background: small glowing pixels drifting upward and
 * fading, in the Xcelsior brand triad. CSS-only (no canvas, no JS animation
 * loop). Sits behind glass content — parent must clip via `overflow-hidden`.
 * All motion is gated behind `prefers-reduced-motion: no-preference`.
 */

const BRAND_COLORS = ["#38dbff", "#9b6dff", "#ff8f8f"] as const;

type Pixel = {
  left: number;
  top: number;
  size: number;
  color: string;
  duration: number;
  delay: number;
  drift: number;
};

// Deterministic pseudo-random so SSR and client agree (no hydration mismatch).
function seeded(i: number, salt: number): number {
  const x = Math.sin(i * 928.371 + salt * 41.117) * 43758.5453;
  return x - Math.floor(x);
}

function buildPixels(count: number): Pixel[] {
  return Array.from({ length: count }, (_, i) => ({
    left: seeded(i, 1) * 100,
    top: seeded(i, 2) * 100,
    size: 2 + Math.round(seeded(i, 3) * 2), // 2–4px
    color: BRAND_COLORS[Math.floor(seeded(i, 4) * BRAND_COLORS.length)],
    duration: 6 + seeded(i, 5) * 8, // 6–14s
    delay: seeded(i, 6) * -14, // negative → staggered, already-in-flight on load
    drift: 24 + seeded(i, 7) * 40, // px of upward travel
  }));
}

export function PixelField({ count = 16, className }: { count?: number; className?: string }) {
  const pixels = useMemo(() => buildPixels(count), [count]);
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute inset-0 overflow-hidden ${className ?? ""}`}
    >
      {pixels.map((p, i) => (
        <span
          key={i}
          className="pixel-field-dot"
          style={
            {
              left: `${p.left}%`,
              top: `${p.top}%`,
              width: `${p.size}px`,
              height: `${p.size}px`,
              background: p.color,
              boxShadow: `0 0 ${p.size * 2}px ${p.color}`,
              // consumed by the `pixel-float` keyframe (globals.css)
              ["--pf-duration" as string]: `${p.duration}s`,
              ["--pf-delay" as string]: `${p.delay}s`,
              ["--pf-drift" as string]: `-${p.drift}px`,
            } as React.CSSProperties
          }
        />
      ))}
    </div>
  );
}
