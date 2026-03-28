import { cn } from "@/lib/utils";
import {
  LucideIcon,
  TrendingUp,
  TrendingDown,
  Minus,
} from "lucide-react";

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: LucideIcon;
  trend?: "up" | "down" | "flat";
  trendValue?: string;
  className?: string;
}

export function StatCard({
  label,
  value,
  icon: Icon,
  trend,
  trendValue,
  className,
}: StatCardProps) {
  const TrendIcon =
    trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  const trendColor =
    trend === "up"
      ? "text-emerald"
      : trend === "down"
        ? "text-accent-red"
        : "text-text-muted";

  return (
    <div
      className={cn(
        "rounded-xl border border-border bg-surface p-5 card-hover",
        className,
      )}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-text-secondary">{label}</span>
        {Icon && <Icon className="h-4 w-4 text-text-muted" />}
      </div>
      <div className="flex items-end gap-2">
        <span className="text-2xl font-bold font-mono">{value}</span>
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
