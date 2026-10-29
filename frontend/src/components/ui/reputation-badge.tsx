import { cn } from "@/lib/utils";

const TIERS: Record<string, { label: string; icon: string; className: string }> = {
  new_user: { label: "New", icon: "🔘", className: "bg-navy-lighter text-text-muted" },
  bronze: { label: "Bronze", icon: "🥉", className: "bg-amber-900/30 text-amber-400 border-amber-700" },
  silver: { label: "Silver", icon: "🥈", className: "bg-gray-600/30 text-gray-300 border-gray-500" },
  gold: { label: "Gold", icon: "🥇", className: "bg-yellow-600/30 text-yellow-300 border-yellow-600" },
  platinum: { label: "Platinum", icon: "💎", className: "bg-cyan-900/30 text-cyan-300 border-cyan-600" },
  diamond: { label: "Diamond", icon: "👑", className: "badge-diamond text-white border-purple-500" },
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
      <span>{config.icon}</span>
      <span>{config.label}</span>
      {score !== undefined && (
        <span className="font-mono opacity-75">({score})</span>
      )}
    </span>
  );
}
