"use client";

import { cn } from "@/lib/utils";
import type { Instance } from "@/lib/api";

export type PricingMode = "on_demand" | "spot";

/** Resolve pricing mode from enriched instance fields or payload. */
export function resolveInstancePricingMode(inst: Instance): PricingMode {
  if (inst.pricing_mode === "spot") return "spot";
  if (inst.preemptible) return "spot";
  const payloadMode = inst.payload?.pricing_mode;
  if (payloadMode === "spot") return "spot";
  return "on_demand";
}

interface PricingModeBadgeProps {
  mode: PricingMode;
  className?: string;
  compact?: boolean;
}

export function PricingModeBadge({ mode, className, compact }: PricingModeBadgeProps) {
  const isSpot = mode === "spot";
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border font-medium uppercase tracking-wide",
        compact ? "px-1.5 py-0 text-[9px]" : "px-2 py-0.5 text-[10px]",
        isSpot
          ? "border-emerald/40 bg-emerald/10 text-emerald"
          : "border-border/80 bg-surface text-text-muted",
        className,
      )}
    >
      {isSpot ? "Spot" : "On-Demand"}
    </span>
  );
}

export function InstancePricingModeBadge({
  instance,
  className,
  compact,
}: {
  instance: Instance;
  className?: string;
  compact?: boolean;
}) {
  return (
    <PricingModeBadge
      mode={resolveInstancePricingMode(instance)}
      className={className}
      compact={compact}
    />
  );
}