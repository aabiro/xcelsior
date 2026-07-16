"use client";

import { useMemo } from "react";

/**
 * Ambient "tech whimsical" background: premium glowing orbs drifting upward and
 * fading, in the Xcelsior brand triad (cyan/purple weighted over emerald).
 * Uses the design-bundle orb sprites (public/particles/*), CSS-only motion (no
 * canvas / JS loop). Sits behind glass content — parent must clip via
 * `overflow-hidden`. Motion is gated behind `prefers-reduced-motion`.
 */

// Weighted like the design bundle's particle field: cyan/purple dominate, emerald accents.
const SPRITES = [
  { src: "/particles/particle-cyan.svg", weight: 0.42 },
  { src: "/particles/particle-purple.svg", weight: 0.42 },
  { src: "/particles/particle-emerald.svg", weight: 0.16 },
] as const;

type Orb = {
  left: number;
  top: number;
  size: number;
  src: string;
  opacity: number;
  duration: number;
  delay: number;
  drift: number;
};

// Deterministic pseudo-random so SSR and client agree (no hydration mismatch).
function seeded(i: number, salt: number): number {
  const x = Math.sin(i * 928.371 + salt * 41.117) * 43758.5453;
  return x - Math.floor(x);
}

function pickSprite(r: number): string {
  let acc = 0;
  for (const s of SPRITES) {
    acc += s.weight;
    if (r <= acc) return s.src;
  }
  return SPRITES[0].src;
}

function buildOrbs(count: number): Orb[] {
  return Array.from({ length: count }, (_, i) => {
    const depth = seeded(i, 3); // 0 far … 1 near
    return {
      left: seeded(i, 1) * 100,
      top: seeded(i, 2) * 100,
      size: Math.round((10 + seeded(i, 8) * 28) * (0.55 + depth * 0.6)), // ~10–40px, depth-scaled
      src: pickSprite(seeded(i, 4)),
      opacity: 0.35 + depth * 0.5,
      duration: 7 + seeded(i, 5) * 7, // 7–14s
      delay: seeded(i, 6) * -14, // negative → staggered, already in-flight on load
      drift: 24 + seeded(i, 7) * 46, // px of upward travel
    };
  });
}

export function PixelField({ count = 16, className }: { count?: number; className?: string }) {
  const orbs = useMemo(() => buildOrbs(count), [count]);
  return (
    <div
      aria-hidden
      className={`pointer-events-none absolute inset-0 overflow-hidden ${className ?? ""}`}
    >
      {orbs.map((p, i) => (
        <img
          key={i}
          src={p.src}
          alt=""
          className="pixel-field-dot"
          style={
            {
              left: `${p.left}%`,
              top: `${p.top}%`,
              width: `${p.size}px`,
              height: `${p.size}px`,
              // orbs carry their own radial glow; opacity is animated by the keyframe,
              // scaled by this per-orb ceiling via the CSS var below.
              ["--pf-max-opacity" as string]: p.opacity.toFixed(2),
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
