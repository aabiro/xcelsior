"use client";

/* eslint-disable @next/next/no-img-element */

/**
 * Canada map hero using the high-fidelity external SVG with real provincial
 * boundaries, masked edge fades, connection arcs, and city dots.
 */
export function CanadaMapHero({
  hostCount,
  className,
}: {
  hostCount: number;
  className?: string;
}) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl border border-border bg-surface ${className ?? ""}`}
      style={{ minHeight: 220 }}
    >
      {/* Aurora glow layers */}
      <div className="pointer-events-none absolute inset-0" style={{ filter: "blur(70px)" }}>
        <div className="absolute left-[20%] -top-8 h-48 w-72 rounded-full bg-accent-cyan/12" style={{ animation: "aurora-drift 6s ease-in-out infinite" }} />
        <div className="absolute left-[50%] top-2 h-36 w-56 rounded-full bg-accent-violet/10" style={{ animation: "aurora-drift 8s ease-in-out infinite 2s" }} />
        <div className="absolute right-[10%] top-12 h-28 w-44 rounded-full bg-accent-red/6" style={{ animation: "aurora-drift 7s ease-in-out infinite 4s" }} />
      </div>

      {/* ── Canada Map SVG (external) ────────────────────────────── */}
      <img
        src="/canada-map-arc.svg"
        alt=""
        aria-hidden
        className="pointer-events-none absolute top-0 right-0 hidden h-full object-cover object-right dark:block"
        loading="eager"
      />
      <img
        src="/canada-map-arc-light.svg"
        alt=""
        aria-hidden
        className="pointer-events-none absolute top-0 right-0 block h-full object-cover object-right dark:hidden"
        loading="eager"
      />

      {/* ── Text overlay ─────────────────────────────────────────── */}
      <div className="relative z-10 flex flex-col items-start justify-center p-6 sm:p-8 h-full" style={{ minHeight: 220 }}>
        <div className="flex items-center gap-2 mb-3">
          <span className="inline-block h-2 w-2 rounded-full bg-emerald animate-pulse" />
          <span className="text-xs font-medium uppercase tracking-wider text-emerald">Online</span>
        </div>
        <h2 className="text-3xl sm:text-4xl font-bold tracking-tight">
          <span className="text-accent-cyan">{hostCount}</span>{" "}
          <span className="text-text-primary">Canadian Host{hostCount === 1 ? "" : "s"}</span>
        </h2>
        <p className="mt-2 text-sm text-text-secondary max-w-md">
          GPU compute nodes distributed across Canada&apos;s provinces — connected, sovereign, and ready&nbsp;for&nbsp;AI&nbsp;workloads
        </p>
        <div className="mt-4 flex flex-wrap gap-3 text-[11px] text-text-muted">
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-cyan" />
            Primary Hub
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent-violet" />
            West Coast
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block h-1.5 w-1.5 rounded-full bg-emerald" />
            Atlantic
          </span>
        </div>
      </div>
    </div>
  );
}
