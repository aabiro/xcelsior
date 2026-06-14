"use client";

import { motion } from "framer-motion";
import {
  Building2,
  Eye,
  KeyRound,
  Lock,
  Shield,
  Sparkles,
  User,
  Users,
  Zap,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useLocale } from "@/lib/locale";
import type { TeamContext } from "@/lib/team-context";
import { formatTeamRoleLabel } from "@/lib/team-context";

export type OAuthScopeValue =
  | "instances:read"
  | "instances:write"
  | "billing:read"
  | "billing:write"
  | "hosts:read"
  | "hosts:write"
  | "profile"
  | "email"
  | "offline_access"
  | "api";

const SCOPE_META: Record<
  string,
  { labelKey: string; tone: "cyan" | "gold" | "emerald" | "violet" | "muted" }
> = {
  "instances:read": { labelKey: "dash.settings.oauth.scope_instances_read", tone: "cyan" },
  "instances:write": { labelKey: "dash.settings.oauth.scope_instances_write", tone: "cyan" },
  "billing:read": { labelKey: "dash.settings.oauth.scope_billing_read", tone: "gold" },
  "billing:write": { labelKey: "dash.settings.oauth.scope_billing_write", tone: "gold" },
  "hosts:read": { labelKey: "dash.settings.oauth.scope_hosts_read", tone: "emerald" },
  "hosts:write": { labelKey: "dash.settings.oauth.scope_hosts_write", tone: "emerald" },
  api: { labelKey: "dash.settings.oauth.scope_full_api", tone: "violet" },
  profile: { labelKey: "dash.settings.oauth.scope_profile", tone: "muted" },
  email: { labelKey: "dash.settings.oauth.scope_email", tone: "muted" },
  offline_access: { labelKey: "dash.settings.oauth.scope_offline", tone: "muted" },
};

const TONE_CLASS: Record<string, string> = {
  cyan: "bg-accent-cyan/12 text-accent-cyan ring-accent-cyan/25",
  gold: "bg-accent-gold/12 text-accent-gold ring-accent-gold/25",
  emerald: "bg-emerald/12 text-emerald ring-emerald/25",
  violet: "bg-violet-400/12 text-violet-300 ring-violet-400/25",
  muted: "bg-surface-hover text-text-secondary ring-border/60",
};

interface CredentialScopePanelProps {
  team: TeamContext;
  clientCount: number;
  className?: string;
}

export function CredentialScopePanel({ team, clientCount, className }: CredentialScopePanelProps) {
  const { t } = useLocale();
  const isTeam = team.isTeamMember;
  const roleLabel = formatTeamRoleLabel(team.teamRole);
  const workspaceName = isTeam
    ? team.teamName || team.teamId || t("dash.settings.credentials.team_workspace")
    : t("dash.settings.credentials.personal_workspace");

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: "easeOut" }}
      className={cn("relative overflow-hidden rounded-2xl", className)}
    >
      <div
        className="absolute inset-0 rounded-2xl bg-gradient-to-br from-accent-cyan/20 via-transparent to-accent-gold/15 opacity-80"
        aria-hidden
      />
      <div
        className="absolute -right-16 -top-16 h-48 w-48 rounded-full bg-accent-cyan/10 blur-3xl"
        aria-hidden
      />
      <div
        className="absolute -bottom-20 -left-10 h-40 w-40 rounded-full bg-accent-gold/10 blur-3xl"
        aria-hidden
      />

      <div className="relative rounded-2xl border border-white/10 bg-navy-light/40 p-4 sm:p-5 backdrop-blur-sm">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-3.5 min-w-0">
            <div
              className={cn(
                "flex h-11 w-11 shrink-0 items-center justify-center rounded-xl ring-1",
                isTeam
                  ? "bg-accent-cyan/15 text-accent-cyan ring-accent-cyan/30"
                  : "bg-accent-gold/15 text-accent-gold ring-accent-gold/30",
              )}
            >
              {isTeam ? <Users className="h-5 w-5" /> : <User className="h-5 w-5" />}
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <p className="text-sm font-semibold text-text-primary truncate">
                  {t("dash.settings.credentials.workspace_title", { name: workspaceName })}
                </p>
                {roleLabel && (
                  <Badge variant="info" className="text-[10px] uppercase tracking-wide">
                    {roleLabel}
                  </Badge>
                )}
                {team.teamPlan && (
                  <Badge variant="default" className="text-[10px] capitalize">
                    {team.teamPlan}
                  </Badge>
                )}
              </div>
              <p className="mt-1 text-xs text-text-muted leading-relaxed max-w-xl">
                {isTeam
                  ? t("dash.settings.credentials.workspace_team_desc")
                  : t("dash.settings.credentials.workspace_personal_desc")}
              </p>
              <code className="mt-2 inline-block max-w-full truncate rounded-md bg-background/50 px-2 py-0.5 text-[10px] font-mono text-text-muted ring-1 ring-border/40">
                {team.billingCustomerId}
              </code>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2 lg:justify-end shrink-0">
            <span className="inline-flex items-center gap-1.5 rounded-lg border border-border/50 bg-surface/50 px-2.5 py-1.5 text-xs text-text-secondary">
              <KeyRound className="h-3.5 w-3.5 text-accent-cyan" />
              {t("dash.settings.credentials.clients_in_workspace", { count: String(clientCount) })}
            </span>
            {isTeam && !team.canWriteInstances && (
              <span className="inline-flex items-center gap-1.5 rounded-lg border border-accent-gold/30 bg-accent-gold/10 px-2.5 py-1.5 text-xs text-accent-gold">
                <Eye className="h-3.5 w-3.5" />
                {t("dash.settings.credentials.viewer_read_only")}
              </span>
            )}
          </div>
        </div>

        <div className="mt-4 grid gap-2 sm:grid-cols-3">
          <ScopeLegendCard
            icon={Building2}
            title={t("dash.settings.credentials.legend_wallet")}
            detail={t("dash.settings.credentials.legend_wallet_desc")}
            tone="gold"
          />
          <ScopeLegendCard
            icon={Zap}
            title={t("dash.settings.credentials.legend_instances")}
            detail={t("dash.settings.credentials.legend_instances_desc")}
            tone="cyan"
          />
          <ScopeLegendCard
            icon={Shield}
            title={t("dash.settings.credentials.legend_scopes")}
            detail={t("dash.settings.credentials.legend_scopes_desc")}
            tone="violet"
          />
        </div>
      </div>
    </motion.div>
  );
}

function ScopeLegendCard({
  icon: Icon,
  title,
  detail,
  tone,
}: {
  icon: typeof Building2;
  title: string;
  detail: string;
  tone: "cyan" | "gold" | "violet";
}) {
  const toneRing =
    tone === "cyan"
      ? "ring-accent-cyan/20 bg-accent-cyan/5"
      : tone === "gold"
        ? "ring-accent-gold/20 bg-accent-gold/5"
        : "ring-violet-400/20 bg-violet-400/5";
  const toneIcon =
    tone === "cyan" ? "text-accent-cyan" : tone === "gold" ? "text-accent-gold" : "text-violet-300";

  return (
    <div className={cn("rounded-xl border border-border/40 px-3 py-2.5 ring-1", toneRing)}>
      <div className="flex items-center gap-2">
        <Icon className={cn("h-3.5 w-3.5", toneIcon)} />
        <p className="text-[11px] font-semibold text-text-primary">{title}</p>
      </div>
      <p className="mt-1 text-[10px] text-text-muted leading-snug">{detail}</p>
    </div>
  );
}

export function WorkspaceScopeBadge({
  workspaceLabel,
  teamName,
  compact,
}: {
  workspaceLabel?: string;
  teamName?: string;
  compact?: boolean;
}) {
  const { t } = useLocale();
  const isTeam = workspaceLabel === "team";

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ring-1",
        isTeam
          ? "bg-accent-cyan/10 text-accent-cyan ring-accent-cyan/25"
          : "bg-accent-gold/10 text-accent-gold ring-accent-gold/25",
        compact && "px-1.5",
      )}
    >
      {isTeam ? <Users className="h-3 w-3" /> : <User className="h-3 w-3" />}
      {isTeam
        ? teamName || t("dash.settings.credentials.team_workspace")
        : t("dash.settings.credentials.personal_workspace")}
    </span>
  );
}

export function ScopeChipRow({
  scopes,
  maxVisible = 5,
  size = "sm",
}: {
  scopes: string[];
  maxVisible?: number;
  size?: "sm" | "md";
}) {
  const { t } = useLocale();
  if (!scopes?.length) {
    return (
      <span className="text-[11px] text-text-muted italic">
        {t("dash.settings.credentials.no_scopes")}
      </span>
    );
  }

  const visible = scopes.slice(0, maxVisible);
  const overflow = scopes.length - visible.length;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {visible.map((scope) => {
        const meta = SCOPE_META[scope] || { labelKey: scope, tone: "muted" as const };
        const label = SCOPE_META[scope]
          ? t(meta.labelKey)
          : scope;
        return (
          <span
            key={scope}
            className={cn(
              "inline-flex items-center gap-1 rounded-md font-mono ring-1",
              TONE_CLASS[meta.tone],
              size === "sm" ? "px-1.5 py-0.5 text-[10px]" : "px-2 py-0.5 text-[11px]",
            )}
          >
            {scope === "api" && <Sparkles className="h-2.5 w-2.5" />}
            {scope.includes(":write") && <Lock className="h-2.5 w-2.5 opacity-70" />}
            {label}
          </span>
        );
      })}
      {overflow > 0 && (
        <span className="rounded-md bg-surface-hover px-1.5 py-0.5 text-[10px] font-mono text-text-muted ring-1 ring-border/50">
          +{overflow}
        </span>
      )}
    </div>
  );
}