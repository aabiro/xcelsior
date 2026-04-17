"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { Server, Activity, Zap, Users, Cpu, Rocket, ArrowUpRight, Plus, type LucideIcon } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host, Instance, ReputationEntry } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { CanadaMapHero } from "@/components/ui/canada-hero";
import { cn } from "@/lib/utils";

// Custom Canada-map-inspired SVG icons for the action cards
function MapRocketIcon({ className, style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className={className} style={style}>
      <defs>
        <linearGradient id="rocketGrad" x1="2" y1="22" x2="22" y2="2" gradientUnits="userSpaceOnUse">
          <stop stopColor="currentColor" stopOpacity="0.4" />
          <stop offset="1" stopColor="currentColor" />
        </linearGradient>
      </defs>
      {/* Abstract Canada Maple Leaf + Rocket hybrid silhouette */}
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
      {/* Abstract Server + Nodes mapping */}
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
      {/* Node connecting line */}
      <path d="M12 11V13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

function OverviewActionVisual({
  accent,
  icon: Icon,
  mirrored = false,
}: {
  accent: "launch" | "provider";
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  mirrored?: boolean;
}) {
  const tones = {
    launch: {
      shell: "border-accent-red/20 bg-[radial-gradient(circle_at_0%_0%,rgba(8,145,178,0.18),transparent_24%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(245,249,255,0.94))] dark:bg-[radial-gradient(circle_at_0%_0%,rgba(0,212,255,0.16),transparent_26%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      panel: "bg-white/[0.72] dark:bg-[#091120]/85",
      ring: "text-accent-red shadow-[0_18px_52px_rgba(234,88,12,0.14)] dark:shadow-[0_18px_52px_rgba(255,82,82,0.18)]",
      cornerA: "bg-[radial-gradient(circle_at_0%_0%,rgba(8,145,178,0.22),transparent_55%)]",
      cornerB: "bg-[radial-gradient(circle_at_100%_100%,rgba(91,33,182,0.18),transparent_55%)]",
      img: "/rocket.svg?v=4",
      imgLight: "/rocket-light.svg?v=4",
    },
    provider: {
      shell: "border-accent-cyan/20 bg-[radial-gradient(circle_at_100%_0%,rgba(8,145,178,0.18),transparent_24%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(244,249,255,0.94))] dark:bg-[radial-gradient(circle_at_100%_0%,rgba(0,212,255,0.16),transparent_26%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      panel: "bg-white/[0.72] dark:bg-[#091120]/85",
      ring: "text-accent-cyan shadow-[0_18px_52px_rgba(8,145,178,0.14)] dark:shadow-[0_18px_52px_rgba(0,212,255,0.16)]",
      cornerA: "bg-[radial-gradient(circle_at_100%_0%,rgba(234,88,12,0.22),transparent_55%)]",
      cornerB: "bg-[radial-gradient(circle_at_0%_100%,rgba(91,33,182,0.18),transparent_55%)]",
      img: "/gpu.svg?v=2",
      imgLight: "/gpu-light.svg?v=2",
    },
  } as const;
  const tone = tones[accent];

  return (
    <div className={cn("relative h-[180px] w-full max-w-[220px] overflow-hidden rounded-[30px] border p-4 backdrop-blur-sm", tone.shell)}>
      <div className={cn("absolute inset-[18px] rounded-[24px] border border-white/[0.45] dark:border-white/[0.08]", tone.panel)} />
      <div className={cn("pointer-events-none absolute inset-0 rounded-[30px]", tone.cornerA)} />
      <div className={cn("pointer-events-none absolute inset-0 rounded-[30px]", tone.cornerB)} />
      <div className="relative flex h-full items-center justify-center rounded-[26px] p-[1.5px] bg-gradient-to-br from-accent-cyan/30 via-accent-violet/20 to-accent-red/30">
        <div className={cn("flex h-full w-full items-center justify-center rounded-[24.5px] bg-white dark:bg-[#080c18] overflow-hidden", tone.ring)}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={tone.imgLight} alt="" aria-hidden className="relative z-10 h-[85%] w-[85%] object-contain dark:hidden" loading="eager" fetchPriority="high" />
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={tone.img} alt="" aria-hidden className="relative z-10 hidden h-[85%] w-[85%] object-contain dark:block" loading="eager" fetchPriority="high" />
        </div>
      </div>
    </div>
  );
}

function OverviewActionCard({
  title,
  description,
  href,
  buttonLabel,
  accent,
  icon: Icon,
  buttonIcon: ButtonIcon,
  reverse = false,
  mirrorVisual = false,
}: {
  title: string;
  description: string;
  href: string;
  buttonLabel: string;
  accent: "launch" | "provider";
  icon: React.ComponentType<{ className?: string; style?: React.CSSProperties }>;
  buttonIcon: LucideIcon;
  reverse?: boolean;
  mirrorVisual?: boolean;
}) {
  const styles = {
    launch: {
      shell:
        "border-accent-red/20 bg-[radial-gradient(circle_at_0%_0%,rgba(234,88,12,0.15),transparent_22%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(246,250,255,0.95))] dark:bg-[radial-gradient(circle_at_0%_0%,rgba(255,82,82,0.16),transparent_24%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      chip: "border-accent-red/20 bg-accent-red/10 text-accent-red dark:border-accent-red/25",
      subchip: "border-accent-cyan/[0.18] bg-accent-cyan/[0.08] text-accent-cyan dark:border-accent-cyan/20",
      button: "bg-accent-red text-white hover:bg-accent-red-hover",
    },
    provider: {
      shell:
        "border-accent-cyan/20 bg-[radial-gradient(circle_at_100%_0%,rgba(8,145,178,0.16),transparent_22%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(244,249,255,0.94))] dark:bg-[radial-gradient(circle_at_100%_0%,rgba(0,212,255,0.16),transparent_24%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      chip: "border-accent-cyan/20 bg-accent-cyan/10 text-accent-cyan dark:border-accent-cyan/25",
      subchip: "border-accent-violet/[0.18] bg-accent-violet/[0.08] text-accent-violet dark:border-accent-violet/20",
      button: "border border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/15",
    },
  } as const;

  const tone = styles[accent];
  const alignRight = accent === "launch";

  return (
    <div className={cn("relative overflow-hidden rounded-[30px] border px-6 py-6 md:px-8 md:py-7", tone.shell)}>
      <div className="relative flex flex-col gap-8 lg:min-h-[260px] lg:justify-between">
        <div className={cn("flex flex-col gap-6 lg:items-start", reverse ? "lg:flex-row-reverse" : "lg:flex-row")}>
          <OverviewActionVisual accent={accent} icon={Icon} mirrored={mirrorVisual} />

          <div className={cn("flex flex-1 flex-col justify-start lg:self-start", alignRight ? "lg:items-end lg:text-right" : "lg:items-start lg:text-left")}>
            <div className={cn("flex flex-wrap items-center gap-2", alignRight ? "lg:justify-end" : "lg:justify-start")}>
              <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em]", tone.chip)}>
                {accent === "launch" ? "Launch" : "Supply"}
              </span>
              <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-medium uppercase tracking-[0.2em]", tone.subchip)}>
                {accent === "launch" ? "GPU Ready" : "Hosts"}
              </span>
            </div>

            <div className="mt-1 max-w-xl space-y-2">
              <h2 className="text-xl font-semibold text-text-primary md:text-[1.45rem]">{title}</h2>
              <p className="text-sm leading-relaxed text-text-secondary">{description}</p>
            </div>
          </div>
        </div>

        <div className={cn("flex", alignRight ? "lg:justify-end" : "lg:justify-start")}>
          <Link
            href={href}
            className={cn(
              "inline-flex h-10 w-fit items-center justify-center gap-2 rounded-full px-4 text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ice-blue",
              tone.button,
            )}
          >
            <ButtonIcon className="h-4 w-4" />
            {buttonLabel}
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function DashboardOverview() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [instances, setInstances] = useState<Instance[]>([]);
  const [leaderboard, setLeaderboard] = useState<ReputationEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useApi();
  const { t } = useLocale();

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

  // Live updates — re-fetch on job/host status changes
  useEventStream({
    eventTypes: ["job_status", "job_submitted", "host_registered", "host_removed"],
    onEvent: () => { load(); },
  });

  const admittedHosts = hosts.filter((h) => h.admitted !== false && String(h.admitted) !== "false");
  const activeHosts = admittedHosts.filter((h) => h.status === "active").length;
  const runningInstances = instances.filter((j) => j.status === "running").length;
  const queuedInstances = instances.filter((j) => j.status === "queued").length;

  if (loading) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold">{t("dash.overview.title")}</h1>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 relative">
      {/* Preload hero SVGs — scoped to this page so other routes don't get unused-preload warnings */}
      <link rel="preload" href="/rocket.svg?v=4" as="image" type="image/svg+xml" />
      <link rel="preload" href="/gpu.svg?v=2" as="image" type="image/svg+xml" />
      <link rel="preload" href="/rocket-light.svg?v=4" as="image" type="image/svg+xml" />
      <link rel="preload" href="/gpu-light.svg?v=2" as="image" type="image/svg+xml" />
      <AuroraBackground className="z-0" />

      <div className="relative z-10 space-y-6">
        <h1 className="text-2xl font-bold">{t("dash.overview.title")}</h1>

        {/* Canada Map Hero */}
        <FadeIn>
          <CanadaMapHero hostCount={admittedHosts.length} />
        </FadeIn>

        {/* Stats Row */}
        <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StaggerItem><StatCard label={t("dash.overview.active_hosts")} value={activeHosts} icon={Server} glow="cyan" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.running_instances")} value={runningInstances} icon={Zap} glow="violet" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.total_hosts")} value={admittedHosts.length} icon={Cpu} glow="emerald" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.queued")} value={queuedInstances} icon={Activity} glow="gold" /></StaggerItem>
        </StaggerList>

          <FadeIn delay={0.18} className="space-y-4">
            <OverviewActionCard
              title="Launch your next instance"
              description="Bring up compute fast, then tune workloads and containers from Instances."
              href="/dashboard/instances?launch=true"
              buttonLabel="Launch Instance"
              accent="launch"
              icon={MapRocketIcon}
              buttonIcon={Plus}
              reverse={false}
              mirrorVisual
            />
            <OverviewActionCard
              title="Become a provider"
              description="Bring spare GPUs online, register capacity, and start listing from Hosts."
              href="/dashboard/hosts"
              buttonLabel="Register Host"
              accent="provider"
              icon={MapServerIcon}
              buttonIcon={Plus}
              reverse
              mirrorVisual
            />
          </FadeIn>

        <FadeIn delay={0.25} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Recent Instances */}
          <div className="glow-card rounded-xl border border-border bg-surface">
            <div className="border-b border-border/60 px-5 py-3.5">
              <h3 className="flex items-center gap-2 text-sm font-semibold">
                <Zap className="h-4 w-4 text-accent-cyan" /> {t("dash.overview.recent_instances")}
              </h3>
            </div>
            <div className="p-5">
              {instances.length === 0 ? (
                <p className="text-sm text-text-muted">{t("dash.overview.no_instances")}</p>
              ) : (
                <div className="space-y-2.5">
                  {instances.slice(0, 5).map((inst) => (
                    <div key={inst.job_id} className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/50 p-3 transition-colors hover:bg-surface-hover">
                      <div className="flex items-center gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-violet/10">
                          <Zap className="h-3.5 w-3.5 text-accent-violet" />
                        </div>
                        <div>
                          <p className="text-sm font-medium">{inst.name || inst.job_id}</p>
                          <p className="text-xs text-text-muted font-mono">{inst.gpu_type || inst.gpu_model}</p>
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

          {/* Top Providers / Leaderboard */}
          <div className="glow-card rounded-xl border border-border bg-surface">
            <div className="border-b border-border/60 px-5 py-3.5">
              <h3 className="flex items-center gap-2 text-sm font-semibold">
                <Users className="h-4 w-4 text-accent-gold" /> {t("dash.overview.top_providers")}
              </h3>
            </div>
            <div className="p-5">
              {leaderboard.length === 0 ? (
                <p className="text-sm text-text-muted">{t("dash.overview.no_leaderboard")}</p>
              ) : (
                <div className="space-y-2.5">
                  {leaderboard.slice(0, 5).map((entry, i) => {
                    const rankColors = ["text-accent-gold bg-accent-gold/15", "text-text-secondary bg-surface-hover", "text-accent-red bg-accent-red/10"];
                    const rankClass = rankColors[i] || "text-text-muted bg-surface-hover";
                    const tierStyles: Record<string, string> = {
                      diamond: "bg-ice-blue/15 text-ice-blue border-ice-blue/30",
                      platinum: "bg-accent-violet/15 text-accent-violet border-accent-violet/30",
                      gold: "bg-accent-gold/15 text-accent-gold border-accent-gold/30",
                      silver: "bg-text-secondary/15 text-text-secondary border-text-secondary/30",
                      bronze: "bg-accent-red/15 text-accent-red border-accent-red/30",
                      new_user: "bg-surface-hover text-text-muted border-border",
                    };
                    const tierLabels: Record<string, string> = {
                      diamond: "Diamond", platinum: "Platinum", gold: "Gold",
                      silver: "Silver", bronze: "Bronze", new_user: "New",
                    };
                    const tierKey = (entry.tier || "new_user").toLowerCase();
                    const tierClass = tierStyles[tierKey] || "bg-surface-hover text-text-muted border-border";
                    const gpuShort = entry.gpu_model
                      ? entry.gpu_model.replace(/NVIDIA\s*/i, "").replace(/GeForce\s*/i, "")
                      : null;
                    return (
                      <div key={entry.entity_id || i} className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/50 p-3 transition-colors hover:bg-surface-hover">
                        <div className="flex items-center gap-3">
                          <span className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${rankClass}`}>
                            {i + 1}
                          </span>
                          <div className="min-w-0">
                            <span className="text-sm font-medium truncate block">{entry.user_id || entry.entity_id}</span>
                            {entry.jobs_completed != null && entry.jobs_completed > 0 && (
                              <span className="text-[10px] text-text-muted">{entry.jobs_completed} job{entry.jobs_completed !== 1 ? "s" : ""}</span>
                            )}
                          </div>
                        </div>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {gpuShort && (
                            <span className="inline-flex items-center gap-1 rounded-full border border-emerald/30 bg-emerald/10 px-2 py-0.5 text-[10px] font-semibold text-emerald">
                              <Cpu className="h-2.5 w-2.5" />{gpuShort}
                            </span>
                          )}
                          <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${tierClass}`}>
                            {tierLabels[tierKey] || tierKey}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </FadeIn>
      </div>
    </div>
  );
}
