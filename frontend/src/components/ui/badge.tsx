import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";

const badgeVariants = cva(
  "inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-sm font-medium normal-case tracking-normal shadow-sm",
  {
    variants: {
      variant: {
        default: "border-accent-cyan/30 bg-accent-cyan/15 text-accent-cyan dark:text-white",
        active: "border-emerald/30 bg-emerald/15 text-emerald",
        dead: "border-accent-red/30 bg-accent-red/15 text-accent-red",
        queued: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue",
        running: "border-emerald/30 bg-emerald/15 text-emerald",
        completed: "border-accent-cyan/30 bg-accent-cyan/15 text-accent-cyan dark:text-white",
        failed: "border-accent-red/30 bg-accent-red/15 text-accent-red",
        cancelled: "border-accent-gold/30 bg-accent-gold/15 text-accent-gold",
        warning: "border-accent-gold/30 bg-accent-gold/15 text-accent-gold",
        info: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue",
        // Instance lifecycle states
        starting: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue animate-status-restarting",
        stopping: "border-accent-gold/30 bg-accent-gold/15 text-accent-gold animate-status-stopping",
        stopped: "border-[#6366f1]/30 bg-[#6366f1]/15 text-[#4338ca] dark:text-[#818cf8]",
        restarting: "border-ice-blue/30 bg-ice-blue/15 text-ice-blue animate-status-restarting",
        terminated: "border-accent-red/30 bg-accent-red/15 text-accent-red",
        preempted: "border-amber-500/30 bg-amber-500/15 text-amber-400",
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
      preempted: "preempted",
      // Legacy pause states
      user_paused: "stopped",
      paused_low_balance: "warning",
    } as Record<string, BadgeProps["variant"]>
  )[status] || "default";

  const label =
    {
      user_paused: "Stopped",
      paused_low_balance: "Low Balance",
      preempted: "Preempted",
    }[status] ?? status.charAt(0).toUpperCase() + status.slice(1);

  return <Badge variant={variant}>{label}</Badge>;
}
