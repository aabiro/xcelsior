"use client";

import { cn } from "@/lib/utils";
import { AlertTriangle, TrendingDown, Zap } from "lucide-react";
import type { ReactNode } from "react";

/** Shared spot accent surface — use across dashboard & marketing spot UI. */
export function SpotSurface({
  children,
  className,
  variant = "default",
}: {
  children: ReactNode;
  className?: string;
  variant?: "default" | "warning" | "hero";
}) {
  return (
    <div
      className={cn(
        "rounded-xl border transition-colors",
        variant === "hero" &&
          "border-emerald/25 bg-gradient-to-br from-emerald/[0.08] via-surface/80 to-accent-cyan/[0.04] shadow-[0_0_40px_rgba(16,185,129,0.06)]",
        variant === "warning" && "border-amber-500/30 bg-amber-500/[0.07]",
        variant === "default" && "border-emerald/20 bg-emerald/[0.04]",
        className,
      )}
    >
      {children}
    </div>
  );
}

export function SpotBadge({
  children,
  className,
  size = "sm",
}: {
  children: ReactNode;
  className?: string;
  size?: "sm" | "md";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border border-emerald/35 bg-emerald/12 font-semibold uppercase tracking-wide text-emerald",
        size === "sm" ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-0.5 text-[11px]",
        className,
      )}
    >
      <TrendingDown className={size === "sm" ? "h-2.5 w-2.5" : "h-3 w-3"} />
      {children}
    </span>
  );
}

export function SpotSavingsPill({ pct, className }: { pct: number; className?: string }) {
  if (pct <= 0) return null;
  return (
    <span
      className={cn(
        "rounded-full bg-emerald/15 px-2 py-0.5 text-[10px] font-semibold text-emerald tabular-nums",
        className,
      )}
    >
      −{pct}%
    </span>
  );
}

export function SpotRateDisplay({
  rateCad,
  onDemandCad,
  savingsPct,
  loading,
  size = "md",
}: {
  rateCad?: number | null;
  onDemandCad?: number | null;
  savingsPct?: number | null;
  loading?: boolean;
  size?: "sm" | "md" | "lg";
}) {
  const sizeClass =
    size === "lg" ? "text-2xl" : size === "sm" ? "text-base" : "text-xl";

  if (loading) {
    return <p className="text-xs text-text-muted animate-pulse">Loading spot rate…</p>;
  }
  if (rateCad == null) {
    return <p className="text-xs text-text-muted">—</p>;
  }

  return (
    <div className="flex flex-wrap items-baseline gap-2">
      <span className={cn("font-bold font-mono text-emerald tabular-nums", sizeClass)}>
        ${rateCad.toFixed(2)}
        <span className="text-sm font-normal text-text-muted">/hr CAD</span>
      </span>
      {savingsPct != null && savingsPct > 0 && <SpotSavingsPill pct={savingsPct} />}
      {onDemandCad != null && onDemandCad > rateCad && (
        <span className="text-xs text-text-muted line-through font-mono tabular-nums">
          ${onDemandCad.toFixed(2)}/hr on-demand
        </span>
      )}
    </div>
  );
}

export function SpotInterruptWarning({
  title = "Interruptible spot instance",
  children,
  className,
}: {
  title?: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <SpotSurface variant="warning" className={cn("px-3.5 py-3 space-y-2", className)}>
      <div className="flex items-center gap-2 text-sm font-medium text-amber-200">
        <AlertTriangle className="h-4 w-4 shrink-0" />
        {title}
      </div>
      {children}
    </SpotSurface>
  );
}

export function SpotKillSwitchBanner({ message }: { message?: string | null }) {
  return (
    <SpotSurface variant="warning" className="px-3 py-2.5">
      <p className="text-xs text-amber-100/90 leading-relaxed">
        {message || "Spot instances are temporarily unavailable. On-demand launches are unaffected."}
      </p>
    </SpotSurface>
  );
}

export function SpotSupplyIndicator({
  supply,
  demand,
  className,
}: {
  supply?: number;
  demand?: number;
  className?: string;
}) {
  const supplyN = supply ?? 0;
  const demandN = demand ?? 0;
  let label = "Balanced";
  let tone = "text-accent-cyan";
  if (supplyN <= 0) {
    label = "Tight supply";
    tone = "text-amber-400";
  } else {
    const ratio = demandN / supplyN;
    if (ratio < 0.5) {
      label = "High availability";
      tone = "text-emerald";
    } else if (ratio >= 1) {
      label = "Elevated demand";
      tone = "text-amber-400";
    }
  }

  return (
    <p className={cn("text-[11px]", tone, className)}>
      <Zap className="inline h-3 w-3 mr-0.5" />
      {label}
      {supply != null && demand != null && (
        <span className="text-text-muted ml-1">
          · {supply} GPUs · {demand} queued
        </span>
      )}
    </p>
  );
}