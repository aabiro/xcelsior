import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-semibold shadow-sm",
  {
    variants: {
      variant: {
        default: "border-accent-cyan/30 bg-accent-cyan/15 text-white",
        active: "border-emerald/30 bg-emerald/15 text-emerald",
        dead: "border-accent-red/30 bg-accent-red/15 text-accent-red",
        queued: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue",
        running: "border-emerald/30 bg-emerald/15 text-emerald",
        completed: "border-accent-cyan/30 bg-accent-cyan/15 text-white",
        failed: "border-accent-red/30 bg-accent-red/15 text-accent-red",
        cancelled: "border-accent-gold/30 bg-accent-gold/15 text-accent-gold",
        warning: "border-accent-gold/30 bg-accent-gold/15 text-accent-gold",
        info: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue",
        // Instance lifecycle states
        starting: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue animate-status-restarting",
        stopping: "border-accent-gold/30 bg-accent-gold/15 text-accent-gold animate-status-stopping",
        stopped: "border-[#6366f1]/30 bg-[#6366f1]/15 text-[#818cf8]",
        restarting: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue animate-status-restarting",
        terminated: "border-accent-red/30 bg-accent-red/15 text-accent-red",
      },
    },
    defaultVariants: { variant: "default" },
  },
);

interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return (
    <span
      className={cn(badgeVariants({ variant }), className)}
      style={{
        textShadow: '0 1px 4px rgba(0,0,0,0.18), 0 0px 1px #fff',
        letterSpacing: '0.18em',
        lineHeight: 1.2,
      }}
      {...props}
    />
  );
}

export function StatusBadge({ status }: { status: string }) {
  const variant = (
    {
      active: "active",
      pending: "queued",
      dead: "dead",
      draining: "warning",
      maintenance: "warning",
      queued: "queued",
      assigned: "queued",
      starting: "starting",
      running: "running",
      completed: "completed",
      failed: "failed",
      cancelled: "cancelled",
      // Instance lifecycle
      stopping: "stopping",
      stopped: "stopped",
      restarting: "restarting",
      terminated: "terminated",
      // Legacy pause states
      user_paused: "stopped",
      paused_low_balance: "warning",
    } as Record<string, BadgeProps["variant"]>
  )[status] || "default";

  const label =
    {
      user_paused: "Stopped",
      paused_low_balance: "Low Balance",
    }[status] ?? status.charAt(0).toUpperCase() + status.slice(1);

  return <Badge variant={variant}>{label}</Badge>;
}
