"use client";

import { cn } from "@/lib/utils";
import type { Instance } from "@/lib/api";
import { TrendingDown } from "lucide-react";

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
        "inline-flex items-center gap-0.5 rounded-full border font-semibold uppercase tracking-wide",
        compact ? "px-1.5 py-0 text-[9px]" : "px-2 py-0.5 text-[10px]",
        isSpot
          ? "border-emerald/40 bg-emerald/12 text-emerald shadow-[0_0_12px_rgba(16,185,129,0.12)]"
          : "border-border/80 bg-surface/90 text-text-muted",
        className,
      )}
    >
      {isSpot && <TrendingDown className={compact ? "h-2 w-2" : "h-2.5 w-2.5"} />}
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