"use client";

import { Wallet, Eye, Server } from "lucide-react";
import { useLocale } from "@/lib/locale";
import type { TeamContext, InstanceConcurrency } from "@/lib/team-context";
import { cn } from "@/lib/utils";

interface TeamContextBannerProps {
  team: TeamContext;
  variant?: "billing" | "instances" | "volumes" | "analytics" | "artifacts" | "general";
  concurrency?: InstanceConcurrency | null;
  className?: string;
}

/**
 * What this workspace does not let you do here, and nothing else.
 *
 * It used to open with a "{team} workspace" card on every page. The top bar's
 * workspace switcher already says which workspace is active, so the card only
 * repeated it (owner review, 2026-11-22). It now renders only a restriction
 * (read-only billing or instances) or the team's instance concurrency, and
 * nothing when there is neither.
 */
export function TeamContextBanner({
  team,
  variant = "general",
  concurrency,
  className,
}: TeamContextBannerProps) {
  const { t } = useLocale();

  if (!team.isTeamMember) return null;

  const billingReadOnly = variant === "billing" && !team.canManageBilling;
  const instancesReadOnly =
    (variant === "instances" || variant === "volumes" || variant === "artifacts") && !team.canWriteInstances;
  if (!billingReadOnly && !instancesReadOnly && !concurrency) return null;

  const atCap = concurrency ? concurrency.active >= concurrency.cap : false;

  return (
    <div className={cn("flex flex-wrap items-center gap-2", className)} role="status">
      {billingReadOnly && (
        <span className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 bg-surface/60 px-2.5 py-1 text-xs text-text-muted">
          <Wallet className="h-3.5 w-3.5" />
          {t("dash.team.billing_read_only")}
        </span>
      )}
      {instancesReadOnly && (
        <span className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 bg-surface/60 px-2.5 py-1 text-xs text-text-muted">
          <Eye className="h-3.5 w-3.5" />
          {variant === "volumes" ? t("dash.volumes.viewer_blocked") : t("dash.team.instances_read_only")}
        </span>
      )}
      {concurrency && (
        <span
          className={cn(
            "inline-flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs font-mono",
            atCap
              ? "border-accent-red/30 bg-accent-red/10 text-accent-red"
              : "border-border/60 bg-surface/60 text-text-secondary",
          )}
        >
          <Server className="h-3.5 w-3.5" />
          {t("dash.team.concurrency_usage", {
            active: String(concurrency.active),
            cap: String(concurrency.cap),
          })}
          {concurrency.shared && (
            <span className="text-text-muted">· {t("dash.team.concurrency_shared")}</span>
          )}
        </span>
      )}
    </div>
  );
}
