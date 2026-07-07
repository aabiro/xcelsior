"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { FadeIn } from "@/components/ui/motion";
import { Zap, Users, Cpu, Plus } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { FirstRunCard } from "@/components/onboarding/first-run-card";
import type { Host, Instance, ReputationEntry } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { CanadaMapHero } from "@/components/ui/canada-hero";
import { openLaunchModal } from "@/lib/launch-modal";
import { cn } from "@/lib/utils";
import { siteIcon } from "@/lib/brand-assets";

function ThemeAssetIcon({ name }: { name: string }) {
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "dark")} className="site-theme-dark" alt="" aria-hidden />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "light")} className="site-theme-light" alt="" aria-hidden />
    </>
  );
}

function OverviewActionVisual({
  accent,
  iconName,
}: {
  accent: "launch" | "provider";
  iconName: string;
}) {
  return (
    <div className="dashboard-overview-visual-shell">
      <div className="dashboard-overview-visual-icon">
        <ThemeAssetIcon name={iconName} />
      </div>
      <div className="site-live-badge absolute right-4 top-4">
        <span className="site-live-dot" />
        Live
      </div>
      <div className="dashboard-overview-visual-grid" aria-hidden />
    </div>
  );
}

function OverviewActionCard({
  title,
  description,
  href,
  onClick,
  buttonLabel,
  accent,
  iconName,
  reverse = false,
}: {
  title: string;
  description: string;
  href?: string;
  onClick?: () => void;
  buttonLabel: string;
  accent: "launch" | "provider";
  iconName: string;
  reverse?: boolean;
}) {
  return (
    <div className="dashboard-overview-action" data-accent={accent} data-reverse={reverse ? "true" : "false"}>
      <div className="dashboard-overview-action-copy">
        <div className="dashboard-overview-action-badges">
          <span className={cn("dashboard-overview-chip", accent === "launch" ? "dashboard-overview-chip--violet" : "dashboard-overview-chip--accent")}>
            {accent === "launch" ? "Launch" : "Supply"}
          </span>
          <span className="dashboard-overview-chip">{accent === "launch" ? "GPU Ready" : "Hosts"}</span>
        </div>

        <h2 className="dashboard-overview-action-title">{title}</h2>
        <p className="site-card-copy">{description}</p>

        <div className="dashboard-overview-action-cta">
          {onClick ? (
            <button
              type="button"
              onClick={onClick}
              className={cn("site-button gap-2 px-5 py-3 text-sm", accent === "launch" ? "site-button-primary" : "site-button-ghost")}
            >
              <Plus className="h-4 w-4" />
              {buttonLabel}
            </button>
          ) : (
            <Link
              href={href ?? "#"}
              className={cn("site-button gap-2 px-5 py-3 text-sm", accent === "launch" ? "site-button-primary" : "site-button-ghost")}
            >
              <Plus className="h-4 w-4" />
              {buttonLabel}
            </Link>
          )}
        </div>
      </div>

      <div className="dashboard-overview-action-visual">
        <OverviewActionVisual accent={accent} iconName={iconName} />
      </div>
    </div>
  );
}

function SectionMarker({ code, label }: { code: string; label: string }) {
  return (
    <div className="site-marker">
      <span className="site-marker-code">[ {code} ]</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
  );
}

function DashboardSection({
  code,
  label,
  children,
}: {
  code: string;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <section className="dashboard-overview-section site-rails">
      <SectionMarker code={code} label={label} />
      {children}
    </section>
  );
}

function rankClass(index: number) {
  if (index === 0) return "dashboard-rank dashboard-rank-gold";
  if (index === 1) return "dashboard-rank dashboard-rank-silver";
  if (index === 2) return "dashboard-rank dashboard-rank-coral";
  return "dashboard-rank dashboard-rank-default";
}

function tierClass(tierKey: string) {
  return `dashboard-chip dashboard-tier-${tierKey}`;
}

export default function DashboardOverview() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [instances, setInstances] = useState<Instance[]>([]);
  const [leaderboard, setLeaderboard] = useState<ReputationEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useApi();
  const { t } = useLocale();
  const { user } = useAuth();

  const load = useCallback(() => {
    Promise.allSettled([
      api.fetchHosts(),
      api.fetchInstances(),
      api.fetchLeaderboard(),
    ]).then(([h, j, l]) => {
      if (h.status === "fulfilled") setHosts(h.value.hosts || []);
      if (j.status === "fulfilled") setInstances(j.value.instances || []);
      if (l.status === "fulfilled") setLeaderboard(l.value.leaderboard || []);
      setLoading(false);
    });
  }, [api]);

  useEffect(() => { load(); }, [load]);

  useEventStream({
    eventTypes: ["job_status", "job_submitted", "host_registered", "host_removed"],
    onEvent: () => { load(); },
  });

  const admittedHosts = hosts.filter((h) => h.admitted === true || String(h.admitted) === "true");
  const activeHosts = admittedHosts.filter((h) => h.status === "active").length;
  const runningInstances = instances.filter((j) => j.status === "running").length;
  const queuedInstances = instances.filter((j) => j.status === "queued").length;

  if (loading) {
    return (
      <div className="dashboard-overview">
        <div className="dashboard-overview-loading-grid">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="dashboard-overview-loading-card skeleton-pulse" />
          ))}
        </div>
      </div>
    );
  }

  const heroStats = [
    { label: t("dash.overview.active_hosts"), value: activeHosts, tone: "cyan" },
    { label: t("dash.overview.running_instances"), value: runningInstances, tone: "violet" },
    { label: t("dash.overview.total_hosts"), value: admittedHosts.length, tone: "emerald" },
    { label: t("dash.overview.queued"), value: queuedInstances, tone: "gold" },
  ] as const;

  return (
    <div className="dashboard-overview">
      <FadeIn className="dashboard-overview-hero">
        <div className="dashboard-overview-hero-head">
          <div>
            <div className="dashboard-overview-eyebrow">Dashboard</div>
            <SectionMarker code="01" label={t("dash.overview")} />
            <p className="dashboard-overview-intro">
              Track live capacity, running workloads, and verified supply across your Xcelsior workspace.
            </p>
          </div>
          <button type="button" onClick={() => openLaunchModal()} className="site-button site-button-primary dashboard-overview-hero-cta">
            <Plus className="h-4 w-4" />
            Launch Instance
          </button>
        </div>

        <div className="dashboard-overview-hero-grid">
          <div className="dashboard-overview-hero-stats">
            {heroStats.map((stat) => (
              <div key={stat.label} className="dashboard-overview-hero-stat">
                <div className="dashboard-overview-hero-stat-label">{stat.label}</div>
                <div className="dashboard-overview-hero-stat-value-row">
                  <div className="dashboard-overview-hero-stat-value">{stat.value}</div>
                  <span className={`dashboard-overview-hero-stat-dot dashboard-overview-hero-stat-dot--${stat.tone}`} />
                </div>
              </div>
            ))}
          </div>
          <CanadaMapHero hostCount={admittedHosts.length} className="dashboard-overview-map" />
        </div>
      </FadeIn>

      <FirstRunCard
        customerId={user?.customer_id || user?.user_id}
        show={instances.length === 0}
      />

      <DashboardSection code="02" label="Launch + Supply">
        <FadeIn delay={0.12} className="dashboard-overview-actions">
          <OverviewActionCard
            title="Launch your next instance"
            description="Bring up compute fast, then tune workloads and containers from Instances."
            onClick={() => openLaunchModal()}
            buttonLabel="Launch Instance"
            accent="launch"
            iconName="bolt"
          />
          <OverviewActionCard
            title="Become a provider"
            description="Bring spare GPUs online, register capacity, and start listing from Hosts."
            href="/dashboard/hosts"
            buttonLabel="Register Host"
            accent="provider"
            iconName="server"
            reverse
          />
        </FadeIn>
      </DashboardSection>

      <DashboardSection code="03" label={t("dash.overview.recent_instances")}>
        <FadeIn delay={0.18} className="dashboard-overview-data-grid">
          <div className="dashboard-data-card glow-card">
            <div className="dashboard-data-card-header">
              <Zap className="h-4 w-4 text-[var(--cyan)]" />
              <h3 className="text-sm font-semibold text-[var(--text)]">{t("dash.overview.recent_instances")}</h3>
            </div>
            <div className="dashboard-data-card-body">
              {instances.length === 0 ? (
                <p className="text-sm text-[var(--text-4)]">{t("dash.overview.no_instances")}</p>
              ) : (
                <div className="dashboard-data-list">
                  {instances.slice(0, 5).map((inst) => (
                    <div key={inst.job_id} className="dashboard-data-row">
                      <div className="flex items-center gap-3">
                        <div className="dashboard-data-row-icon">
                          <Zap className="h-3.5 w-3.5 text-[var(--violet)]" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-[var(--text)]">{inst.name || inst.job_id}</p>
                          <p className="font-mono text-xs text-[var(--text-4)]">{inst.gpu_type || inst.gpu_model}</p>
                        </div>
                      </div>
                      <Badge variant={inst.status === "running" ? "active" : inst.status === "completed" ? "completed" : inst.status === "queued" ? "queued" : "default"}>
                        {inst.status}
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="dashboard-data-card glow-card">
            <div className="dashboard-data-card-header">
              <Users className="h-4 w-4 text-[var(--gold)]" />
              <h3 className="text-sm font-semibold text-[var(--text)]">{t("dash.overview.top_providers")}</h3>
            </div>
            <div className="dashboard-data-card-body">
              {leaderboard.length === 0 ? (
                <p className="text-sm text-[var(--text-4)]">{t("dash.overview.no_leaderboard")}</p>
              ) : (
                <div className="dashboard-data-list">
                  {leaderboard.slice(0, 5).map((entry, i) => {
                    const tierLabels: Record<string, string> = {
                      diamond: "Diamond", platinum: "Platinum", gold: "Gold",
                      silver: "Silver", bronze: "Bronze", new_user: "New",
                    };
                    const tierKey = (entry.tier || "new_user").toLowerCase();
                    const gpuShort = entry.gpu_model
                      ? entry.gpu_model.replace(/NVIDIA\s*/i, "").replace(/GeForce\s*/i, "")
                      : null;
                    return (
                      <div key={entry.entity_id || i} className="dashboard-data-row">
                        <div className="flex min-w-0 items-center gap-3">
                          <span className={rankClass(i)}>{i + 1}</span>
                          <div className="min-w-0">
                            <span className="block truncate text-sm font-medium text-[var(--text)]">{entry.user_id || entry.entity_id}</span>
                            {entry.jobs_completed != null && entry.jobs_completed > 0 && (
                              <span className="text-[10px] text-[var(--text-4)]">{entry.jobs_completed} job{entry.jobs_completed !== 1 ? "s" : ""}</span>
                            )}
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1.5">
                          {gpuShort && (
                            <span className="dashboard-chip dashboard-chip-emerald">
                              <Cpu className="h-2.5 w-2.5" />
                              {gpuShort}
                            </span>
                          )}
                          <span className={tierClass(tierKey)}>{tierLabels[tierKey] || tierKey}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </FadeIn>
      </DashboardSection>
    </div>
  );
}
