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

function OverviewActionCard({
  title,
  description,
  href,
  buttonLabel,
  accent,
  icon: Icon,
  buttonIcon: ButtonIcon,
}: {
  title: string;
  description: string;
  href: string;
  buttonLabel: string;
  accent: "launch" | "provider";
  icon: LucideIcon;
  buttonIcon: LucideIcon;
}) {
  const styles = {
    launch: {
      shell: "border-accent-red/20 bg-[radial-gradient(circle_at_top_right,rgba(255,82,82,0.16),transparent_42%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      ring: "border-accent-red/25 text-accent-red shadow-[0_18px_48px_rgba(255,82,82,0.18)]",
      chip: "border-accent-red/25 bg-accent-red/10 text-accent-red",
      orbA: "bg-accent-red/20",
      orbB: "bg-accent-gold/15",
      button: "bg-accent-red text-white hover:bg-accent-red-hover",
    },
    provider: {
      shell: "border-accent-cyan/20 bg-[radial-gradient(circle_at_top_right,rgba(0,212,255,0.16),transparent_42%),linear-gradient(180deg,rgba(8,12,24,0.96),rgba(7,11,20,0.92))]",
      ring: "border-accent-cyan/25 text-accent-cyan shadow-[0_18px_48px_rgba(0,212,255,0.16)]",
      chip: "border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan",
      orbA: "bg-accent-cyan/20",
      orbB: "bg-accent-violet/15",
      button: "border border-accent-cyan/25 bg-accent-cyan/10 text-accent-cyan hover:bg-accent-cyan/15",
    },
  } as const;

  const tone = styles[accent];

  return (
    <div className={cn("relative overflow-hidden rounded-[28px] border p-6", tone.shell)}>
      <div className={cn("absolute -right-10 top-0 h-40 w-40 rounded-full blur-3xl", tone.orbA)} />
      <div className={cn("absolute right-14 top-20 h-24 w-24 rounded-full blur-3xl", tone.orbB)} />

      <div className="relative flex h-full flex-col justify-between gap-8">
        <div className="flex items-start justify-between gap-6">
          <div className="space-y-3">
            <span className={cn("inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.24em]", tone.chip)}>
              {accent === "launch" ? "Start" : "Supply"}
            </span>
            <div className="max-w-sm space-y-2">
              <h2 className="text-xl font-semibold text-text-primary">{title}</h2>
              <p className="text-sm leading-relaxed text-text-secondary">{description}</p>
            </div>
          </div>

          <div className={cn("relative hidden h-36 w-36 shrink-0 items-center justify-center rounded-[2rem] border bg-[#091120]/85 backdrop-blur md:flex", tone.ring)}>
            <div className="absolute inset-5 rounded-[1.5rem] border border-white/5 bg-gradient-to-br from-white/5 to-transparent" />
            <div className={cn("absolute left-6 top-6 h-3 w-3 rounded-full", tone.orbB)} />
            <div className={cn("absolute bottom-6 right-6 h-4 w-4 rounded-full", tone.orbA)} />
            <Icon className="relative z-10 h-12 w-12" />
            <ArrowUpRight className="absolute right-6 top-6 h-4 w-4 text-white/55" />
          </div>
        </div>

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

        <FadeIn delay={0.18} className="grid grid-cols-1 gap-6 xl:grid-cols-2">
          <OverviewActionCard
            title="Launch your next instance"
            description="Spin up GPU compute fast, then keep tuning from the Instances view."
            href="/dashboard/instances?launch=true"
            buttonLabel="Launch Instance"
            accent="launch"
            icon={Rocket}
            buttonIcon={Plus}
          />
          <OverviewActionCard
            title="Become a provider"
            description="Bring spare GPUs online and start listing capacity from the Hosts tab."
            href="/dashboard/hosts"
            buttonLabel="Open Hosts"
            accent="provider"
            icon={Server}
            buttonIcon={ArrowUpRight}
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
