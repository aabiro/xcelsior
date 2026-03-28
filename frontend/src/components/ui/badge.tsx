import { cn } from "@/lib/utils";
import { cva, type VariantProps } from "class-variance-authority";
import type { HTMLAttributes } from "react";

const badgeVariants = cva(
  "inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium",
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
      },
    },
    defaultVariants: { variant: "default" },
  },
);

interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
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
      running: "running",
      completed: "completed",
      failed: "failed",
      cancelled: "cancelled",
    } as Record<string, BadgeProps["variant"]>
  )[status] || "default";

  return <Badge variant={variant}>{status}</Badge>;
}
