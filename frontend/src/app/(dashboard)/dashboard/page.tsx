"use client";

import { useEffect, useState, useCallback } from "react";
import Link from "next/link";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { Server, Activity, Zap, Users, Cpu, Rocket, ArrowUpRight, ArrowUpLeft, ArrowUp, Plus, type LucideIcon } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host, Instance, ReputationEntry } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { CanadaMapHero } from "@/components/ui/canada-hero";
import { cn } from "@/lib/utils";

function LaunchRocketGlyph() {
  return (
    <svg
      aria-hidden
      viewBox="0 0 48 48"
      className="relative z-10 h-9 w-9 md:h-10 md:w-10"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path
        d="M11 24L18 16H30C35 16 38 20 40 24C38 28 35 32 30 32H18L11 24Z"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinejoin="round"
      />
      <path
        d="M18 16L23 10L24.5 17.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18 32L23 38L24.5 30.5"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="24" cy="24" r="4.4" stroke="currentColor" strokeWidth="1.8" />
      <path
        d="M40 20.5C44.5 22 45.7 24.2 45.7 24.2C45.7 24.2 44.5 26.4 40 27.9L34 24.2L40 20.5Z"
        fill="currentColor"
        fillOpacity="0.72"
      />
      <path
        d="M11 24L7.2 21.2L7.2 26.8L11 24Z"
        fill="currentColor"
        fillOpacity="0.24"
      />
    </svg>
  );
}

function OverviewActionVisual({
  accent,
  icon: Icon,
  mirrored = false,
}: {
  accent: "launch" | "provider";
  icon: LucideIcon;
  mirrored?: boolean;
}) {
  const tones = {
    launch: {
      shell:
        "border-accent-red/20 bg-[radial-gradient(circle_at_top_left,rgba(220,38,38,0.16),transparent_36%),radial-gradient(circle_at_top_right,rgba(14,165,233,0.14),transparent_42%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(245,249,255,0.94))] dark:bg-[radial-gradient(circle_at_top_left,rgba(255,82,82,0.18),transparent_42%),radial-gradient(circle_at_top_right,rgba(0,212,255,0.12),transparent_42%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      panel: "bg-white/[0.72] dark:bg-[#091120]/85",
      ring: "border-accent-red/20 text-accent-red shadow-[0_18px_52px_rgba(220,38,38,0.14)] dark:shadow-[0_18px_52px_rgba(255,82,82,0.18)]",
      lineA: "from-accent-red/0 via-accent-red/[0.55] to-accent-red/0",
      lineB: "from-accent-cyan/0 via-accent-cyan/50 to-accent-cyan/0",
      dotA: "bg-accent-red/[0.70]",
      dotB: "bg-accent-cyan/[0.55]",
      arrow: "text-white/90 dark:text-white/[0.72]",
    },
    provider: {
      shell:
        "border-accent-cyan/20 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.18),transparent_38%),radial-gradient(circle_at_top_left,rgba(124,58,237,0.16),transparent_44%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(244,249,255,0.94))] dark:bg-[radial-gradient(circle_at_top_right,rgba(0,212,255,0.16),transparent_42%),radial-gradient(circle_at_top_left,rgba(124,58,237,0.14),transparent_44%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      panel: "bg-white/[0.72] dark:bg-[#091120]/85",
      ring: "border-accent-cyan/20 text-accent-cyan shadow-[0_18px_52px_rgba(14,165,233,0.14)] dark:shadow-[0_18px_52px_rgba(0,212,255,0.16)]",
      lineA: "from-accent-cyan/0 via-accent-cyan/[0.60] to-accent-cyan/0",
      lineB: "from-accent-violet/0 via-accent-violet/[0.48] to-accent-violet/0",
      dotA: "bg-accent-cyan/[0.70]",
      dotB: "bg-accent-violet/[0.58]",
      arrow: "text-white/88 dark:text-white/[0.68]",
    },
  } as const;

  const tone = tones[accent];
  const CornerArrow = accent === "launch" ? ArrowUpLeft : ArrowUpRight;
  const iconClassName = "relative z-10 h-9 w-9 md:h-10 md:w-10";

  return (
    <div className={cn("relative h-[180px] w-full max-w-[220px] overflow-hidden rounded-[30px] border p-4 backdrop-blur-sm", tone.shell)}>
      <div className={cn("absolute inset-[18px] rounded-[24px] border border-white/[0.45] dark:border-white/[0.08]", tone.panel)} />
      <div
        className={cn(
          "absolute left-6 right-6 top-[38%] h-px bg-gradient-to-r",
          mirrored ? "-scale-x-100" : "",
          tone.lineA,
        )}
      />
      <div
        className={cn(
          "absolute left-10 right-10 top-[58%] h-px bg-gradient-to-r",
          mirrored ? "-scale-x-100" : "",
          tone.lineB,
        )}
      />
      <div className={cn("absolute h-3.5 w-3.5 rounded-full blur-[1px]", mirrored ? "left-7 top-7" : "right-7 top-7", tone.dotA)} />
      <div className={cn("absolute h-4 w-4 rounded-full blur-[1px]", mirrored ? "right-7 bottom-7" : "left-7 bottom-7", tone.dotB)} />
      <CornerArrow
        className={cn(
          "absolute z-20 h-4 w-4",
          accent === "launch" ? "left-6 top-6" : "right-6 top-6",
          tone.arrow,
        )}
      />
      <div className={cn("relative flex h-full items-center justify-center rounded-[26px] border", tone.ring)}>
        <div className="absolute inset-5 rounded-[18px] border border-white/30 bg-gradient-to-br from-white/30 to-transparent dark:border-white/[0.06] dark:from-white/[0.06]" />
        {accent === "provider" && (
          <>
            <div className="absolute left-1/2 top-[44%] z-20 h-px w-16 -translate-x-1/2 rounded-full bg-accent-cyan/70 dark:bg-accent-cyan/75" />
            <div className="absolute left-1/2 top-[54%] z-20 h-px w-14 -translate-x-1/2 rounded-full bg-accent-cyan/70 dark:bg-accent-cyan/75" />
          </>
        )}
        {accent === "launch" ? <LaunchRocketGlyph /> : <Icon className={iconClassName} />}
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
  icon: LucideIcon;
  buttonIcon: LucideIcon;
  reverse?: boolean;
  mirrorVisual?: boolean;
}) {
  const styles = {
    launch: {
      shell:
        "border-accent-red/20 bg-[radial-gradient(circle_at_top_left,rgba(220,38,38,0.15),transparent_44%),radial-gradient(circle_at_top_right,rgba(14,165,233,0.12),transparent_40%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(246,250,255,0.95))] dark:bg-[radial-gradient(circle_at_top_left,rgba(255,82,82,0.16),transparent_42%),radial-gradient(circle_at_top_right,rgba(0,212,255,0.1),transparent_40%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      chip: "border-accent-red/20 bg-accent-red/10 text-accent-red dark:border-accent-red/25",
      subchip: "border-accent-cyan/[0.18] bg-accent-cyan/[0.08] text-accent-cyan dark:border-accent-cyan/20",
      button: "bg-accent-red text-white hover:bg-accent-red-hover",
    },
    provider: {
      shell:
        "border-accent-cyan/20 bg-[radial-gradient(circle_at_top_right,rgba(14,165,233,0.16),transparent_42%),radial-gradient(circle_at_top_left,rgba(124,58,237,0.14),transparent_44%),linear-gradient(180deg,rgba(255,255,255,0.98),rgba(244,249,255,0.94))] dark:bg-[radial-gradient(circle_at_top_right,rgba(0,212,255,0.16),transparent_42%),radial-gradient(circle_at_top_left,rgba(124,58,237,0.12),transparent_44%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
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
        <div className={cn("flex flex-col gap-6 lg:items-center", reverse ? "lg:flex-row-reverse" : "lg:flex-row")}>
          <OverviewActionVisual accent={accent} icon={Icon} mirrored={mirrorVisual} />

          <div className={cn("flex flex-1 flex-col", alignRight ? "lg:items-end lg:text-right" : "lg:items-start lg:text-left")}>
            <div className={cn("flex flex-wrap items-center gap-2", alignRight ? "lg:justify-end" : "lg:justify-start")}>
              <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em]", tone.chip)}>
                {accent === "launch" ? "Launch" : "Supply"}
              </span>
              <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-medium uppercase tracking-[0.2em]", tone.subchip)}>
                {accent === "launch" ? "GPU Ready" : "Hosts"}
              </span>
            </div>

            <div className="mt-4 max-w-xl space-y-2">
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

  const activeHosts = hosts.filter((h) => h.status === "active").length;
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
      <AuroraBackground className="z-0" />

      <div className="relative z-10 space-y-6">
        <h1 className="text-2xl font-bold">{t("dash.overview.title")}</h1>

        {/* Canada Map Hero */}
        <FadeIn>
          <CanadaMapHero hostCount={hosts.length} />
        </FadeIn>

        {/* Stats Row */}
        <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StaggerItem><StatCard label={t("dash.overview.active_hosts")} value={activeHosts} icon={Server} glow="cyan" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.running_instances")} value={runningInstances} icon={Zap} glow="violet" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.total_hosts")} value={hosts.length} icon={Cpu} glow="emerald" /></StaggerItem>
          <StaggerItem><StatCard label={t("dash.overview.queued")} value={queuedInstances} icon={Activity} glow="gold" /></StaggerItem>
        </StaggerList>

        <FadeIn delay={0.18} className="space-y-4">
          <OverviewActionCard
            title="Launch your next instance"
            description="Bring up compute fast, then tune workloads and containers from Instances."
            href="/dashboard/instances?launch=true"
            buttonLabel="Launch Instance"
            accent="launch"
            icon={Rocket}
            buttonIcon={Plus}
            reverse={false}
            mirrorVisual
          />
          <OverviewActionCard
            title="Become a provider"
            description="Bring spare GPUs online, register capacity, and start listing from Hosts."
            href="/dashboard/hosts"
            buttonLabel="Open Hosts"
            accent="provider"
            icon={Server}
            buttonIcon={ArrowUpRight}
            reverse
            mirrorVisual={false}
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
