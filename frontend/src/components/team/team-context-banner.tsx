"use client";

import { Users, Wallet, Eye, Server } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useLocale } from "@/lib/locale";
import type { TeamContext, InstanceConcurrency } from "@/lib/team-context";
import { formatPlanLabel, formatTeamRoleLabel } from "@/lib/team-context";
import { cn } from "@/lib/utils";

interface TeamContextBannerProps {
  team: TeamContext;
  variant?: "billing" | "instances" | "volumes" | "analytics" | "artifacts" | "general";
  concurrency?: InstanceConcurrency | null;
  className?: string;
}

export function TeamContextBanner({
  team,
  variant = "general",
  concurrency,
  className,
}: TeamContextBannerProps) {
  const { t } = useLocale();

  if (!team.isTeamMember) return null;

  const roleLabel = formatTeamRoleLabel(team.teamRole);
  const atCap = concurrency ? concurrency.active >= concurrency.cap : false;

  return (
    <div
      className={cn(
        "rounded-xl border border-accent-cyan/20 bg-accent-cyan/5 px-4 py-3",
        className,
      )}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent-cyan/15">
            <Users className="h-4 w-4 text-accent-cyan" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-semibold text-text-primary">
                {t("dash.team.banner_title", { name: team.teamName || team.teamId || "Team" })}
              </p>
              {roleLabel && (
                <Badge variant="info">
                  {roleLabel}
                </Badge>
              )}
              {team.teamPlan && (
                <Badge variant="default">
                  {formatPlanLabel(team.teamPlan)}
                </Badge>
              )}
            </div>
            <p className="mt-0.5 text-xs text-text-muted">
              {variant === "billing"
                ? t("dash.team.banner_billing_desc")
                : variant === "instances"
                  ? t("dash.team.banner_instances_desc")
                  : variant === "volumes"
                    ? t("dash.team.banner_volumes_desc")
                    : variant === "analytics"
                      ? t("dash.team.banner_analytics_desc")
                      : variant === "artifacts"
                        ? t("dash.team.banner_artifacts_desc")
                        : t("dash.team.banner_desc")}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 sm:justify-end">
          {variant === "billing" && !team.canManageBilling && (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 bg-surface/60 px-2.5 py-1 text-xs text-text-muted">
              <Wallet className="h-3.5 w-3.5" />
              {t("dash.team.billing_read_only")}
            </span>
          )}
          {(variant === "instances" || variant === "volumes" || variant === "artifacts") && !team.canWriteInstances && (
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-border/60 bg-surface/60 px-2.5 py-1 text-xs text-text-muted">
              <Eye className="h-3.5 w-3.5" />
              {variant === "volumes"
                ? t("dash.volumes.viewer_blocked")
                : t("dash.team.instances_read_only")}
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
      </div>
    </div>
  );
}