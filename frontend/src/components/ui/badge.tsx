import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-bold border border-accent-cyan/40 bg-gradient-to-r from-accent-cyan/80 to-accent-cyan/40 shadow-sm",
  {
    variants: {
      variant: {
        default: "bg-navy-lighter text-text-secondary",
        active: "bg-emerald/20 text-emerald",
        dead: "bg-accent-red/20 text-accent-red",
        queued: "bg-ice-blue/20 text-ice-blue",
        running: "bg-emerald/20 text-emerald",
        completed: "bg-navy-lighter text-text-secondary",
        failed: "bg-accent-red/20 text-accent-red",
        cancelled: "bg-accent-gold/20 text-accent-gold",
        warning: "bg-accent-gold/20 text-accent-gold",
        info: "bg-ice-blue/20 text-ice-blue",
        // Instance lifecycle states
        starting: "bg-ice-blue/15 text-ice-blue animate-status-restarting",
        stopping: "bg-accent-gold/15 text-accent-gold animate-status-stopping",
        stopped: "bg-[#6366f1]/20 text-[#818cf8]",
        restarting: "bg-ice-blue/15 text-ice-blue animate-status-restarting",
        terminated: "bg-accent-red/20 text-accent-red",
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
