import { cn } from "@/lib/utils";
import {
  LucideIcon,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";
import type { ReactNode } from "react";

interface StatCardProps {
  label: string;
  value: ReactNode;
  icon?: LucideIcon;
  trend?: "up" | "down" | "flat";
  trendValue?: string;
  className?: string;
  glow?: "cyan" | "violet" | "emerald" | "gold";
}

export function StatCard({
  label,
  value,
  icon: Icon,
  trend,
  trendValue,
  className,
  glow,
}: StatCardProps) {
  const TrendIcon =
    trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  const trendColor =
    trend === "up"
      ? "text-emerald"
      : trend === "down"
        ? "text-accent-red"
        : "text-text-muted";

  const glowClass = glow ? `stat-glow-${glow}` : "";
  const iconColorMap = {
    cyan: "text-accent-cyan",
    violet: "text-accent-violet",
    emerald: "text-emerald",
    gold: "text-accent-gold",
  };
  const iconColor = glow ? iconColorMap[glow] : "text-text-muted";

  return (
    <div
      className={cn(
        "glow-card rounded-xl border border-border bg-surface p-5",
        glowClass,
        className,
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm text-text-secondary">{label}</span>
        {Icon && (
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-surface-hover">
            <Icon className={cn("h-4 w-4", iconColor)} />
          </div>
        )}
      </div>
      <div className="flex items-end gap-2">
        <span className="text-3xl font-bold font-mono tracking-tight">{value}</span>
        {trend && trendValue && (
          <span className={cn("flex items-center text-xs gap-0.5 mb-1", trendColor)}>
            <TrendIcon className="h-3 w-3" />
            {trendValue}
          </span>
        )}
      </div>
    </div>
  );
}
