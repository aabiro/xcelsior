import { Circle, Crown, Gem, Medal, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

const TIERS: Record<string, { label: string; icon: LucideIcon; className: string }> = {
  new_user: { label: "New", icon: Circle, className: "bg-navy-lighter text-text-muted" },
  bronze: { label: "Bronze", icon: Medal, className: "bg-amber-900/30 text-amber-400 border-amber-700" },
  silver: { label: "Silver", icon: Medal, className: "bg-gray-600/30 text-gray-300 border-gray-500" },
  gold: { label: "Gold", icon: Medal, className: "bg-yellow-600/30 text-yellow-300 border-yellow-600" },
  platinum: { label: "Platinum", icon: Gem, className: "bg-cyan-900/30 text-cyan-300 border-cyan-600" },
  diamond: { label: "Diamond", icon: Crown, className: "badge-diamond text-white border-purple-500" },
};

export function ReputationBadge({
  tier,
  score,
  size = "sm",
  className,
}: {
  tier: string;
  score?: number;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const config = TIERS[tier?.toLowerCase()] || TIERS.new_user;
  const sizeClasses = {
    sm: "text-xs px-2 py-0.5",
    md: "text-sm px-3 py-1",
    lg: "text-base px-4 py-1.5",
  };

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border font-medium",
        config.className,
        sizeClasses[size],
        className,
      )}
    >
      <config.icon className="h-[1em] w-[1em] shrink-0" aria-hidden />
      <span>{config.label}</span>
      {score !== undefined && (
        <span className="font-mono opacity-75">({score})</span>
      )}
    </span>
  );
}
