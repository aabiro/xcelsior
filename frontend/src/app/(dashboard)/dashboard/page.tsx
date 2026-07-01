"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { Server, Activity, Zap, Users, Cpu, Plus } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { FirstRunCard } from "@/components/onboarding/first-run-card";
import type { Host, Instance, ReputationEntry } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { CanadaMapHero } from "@/components/ui/canada-hero";
import { openLaunchModal } from "@/lib/launch-modal";
import { cn } from "@/lib/utils";

function MapRocketIcon({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} style={style}>
      <defs>
        <linearGradient id="rocketGrad" x1="2" y1="22" x2="22" y2="2" gradientUnits="userSpaceOnUse">
          <stop stopColor="currentColor" stopOpacity="0.4" />
          <stop offset="1" stopColor="currentColor" />
        </linearGradient>
      </defs>
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M13.5 2C13.5 2 17.5 3.5 19.5 7.5C20.4674 9.43478 20.8 11.6 20 13.5C21.5 13.5 22 15 22 15C22 15 19.5 16.5 17 17L19 22C19 22 17 21 15.5 19C12.5 18.5 7 19.5 5 22C5 22 7 17 5 15C4.5 12.5 5.5 7 8 5C8 5 9 3 13.5 2ZM14.1213 11.1213C15.2929 9.94975 17.1924 9.94975 18.364 11.1213C17.1924 12.2929 15.2929 12.2929 14.1213 11.1213Z"
        fill="url(#rocketGrad)"
      />
      <path d="M4 21C4 21 6 18 5 16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function MapServerIcon({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} style={style}>
      <defs>
        <linearGradient id="serverGrad" x1="2" y1="2" x2="22" y2="22" gradientUnits="userSpaceOnUse">
          <stop stopColor="currentColor" stopOpacity="0.8" />
          <stop offset="1" stopColor="currentColor" stopOpacity="0.2" />
        </linearGradient>
      </defs>
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M4 6C4 4.89543 4.89543 4 6 4H18C19.1046 4 20 4.89543 20 6V9C20 10.1046 19.1046 11 18 11H6C4.89543 11 4 10.1046 4 9V6ZM7 6.5C6.44772 6.5 6 6.94772 6 7.5C6 8.05228 6.44772 8.5 7 8.5H9C9.55228 8.5 10 8.05228 10 7.5C10 6.94772 9.55228 6.5 9 6.5H7ZM16 8.5C16.5523 8.5 17 8.05228 17 7.5C17 6.94772 16.5523 6.5 16 6.5C15.4477 6.5 15 6.94772 15 7.5C15 8.05228 15.4477 8.5 16 8.5Z"
        fill="url(#serverGrad)"
      />
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M4 15C4 13.8954 4.89543 13 6 13H18C19.1046 13 20 13.8954 20 15V18C20 19.1046 19.1046 20 18 20H6C4.89543 20 4 19.1046 4 18V15ZM7 15.5C6.44772 15.5 6 15.9477 6 16.5C6 17.0523 6.44772 17.5 7 17.5H9C9.55228 17.5 10 17.0523 10 16.5C10 15.9477 9.55228 15.5 9 15.5H7ZM16 17.5C16.5523 17.5 17 17.0523 17 16.5C17 15.9477 16.5523 15.5 16 15.5C15.4477 15.5 15 15.9477 15 16.5C15 17.0523 15.4477 17.5 16 17.5Z"
        fill="url(#serverGrad)"
      />
      <path d="M12 11V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function OverviewActionVisual({
  accent,
  icon: Icon,
}: {
  accent: "launch" | "provider";
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
}) {
  return (
    <div className="dashboard-overview-visual-shell">
      <Icon
        className="dashboard-overview-visual-icon"
        style={{ transform: accent === "launch" ? "translateY(-2px)" : undefined }}
      />
      <div className="site-live-badge absolute right-4 top-4">
        <span className="site-live-dot" />
        Live
      </div>
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
  icon: Icon,
  reverse = false,
}: {
  title: string;
  description: string;
  href?: string;
  onClick?: () => void;
  buttonLabel: string;
  accent: "launch" | "provider";
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
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
        <OverviewActionVisual accent={accent} icon={Icon} />
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

  return (
    <div className="dashboard-overview">
      <link rel="preload" href="/rocket.svg?v=6" as="image" type="image/svg+xml" />
      <link rel="preload" href="/gpu.svg?v=2" as="image" type="image/svg+xml" />
      <link rel="preload" href="/rocket-light.svg?v=6" as="image" type="image/svg+xml" />
      <link rel="preload" href="/gpu-light.svg?v=2" as="image" type="image/svg+xml" />
      <AuroraBackground className="z-0" />

      <FadeIn className="dashboard-overview-hero">
        <div className="dashboard-overview-hero-grid">
          <div style={{ animation: "heroUp .7s ease both" }}>
            <SectionMarker code="01" label={t("dash.overview.title")} />
            <h1 className="dashboard-overview-title">{t("dash.overview.title")}</h1>
          </div>
          <CanadaMapHero hostCount={admittedHosts.length} className="dashboard-overview-map" />
        </div>
      </FadeIn>

      <FirstRunCard
        customerId={user?.customer_id || user?.user_id}
        show={instances.length === 0}
      />

      <DashboardSection code="02" label={t("dash.overview.title")}>
        <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StaggerItem><StatCard label={t("dash.overview.active_hosts")} value={activeHosts} icon={Server} glow="cyan" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.running_instances")} value={runningInstances} icon={Zap} glow="violet" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.total_hosts")} value={admittedHosts.length} icon={Cpu} glow="emerald" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.queued")} value={queuedInstances} icon={Activity} glow="gold" /></StaggerItem>
        </StaggerList>
      </DashboardSection>

      <DashboardSection code="03" label="Launch your next instance">
        <FadeIn delay={0.18} className="dashboard-overview-actions">
          <OverviewActionCard
            title="Launch your next instance"
            description="Bring up compute fast, then tune workloads and containers from Instances."
            onClick={() => openLaunchModal()}
            buttonLabel="Launch Instance"
            accent="launch"
            icon={MapRocketIcon}
          />
          <OverviewActionCard
            title="Become a provider"
            description="Bring spare GPUs online, register capacity, and start listing from Hosts."
            href="/dashboard/hosts"
            buttonLabel="Register Host"
            accent="provider"
            icon={MapServerIcon}
            reverse
          />
        </FadeIn>
      </DashboardSection>

      <DashboardSection code="04" label={t("dash.overview.recent_instances")}>
        <FadeIn delay={0.25} className="dashboard-overview-data-grid">
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
