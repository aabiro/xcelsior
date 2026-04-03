"use client";

import { useEffect, useState, useCallback } from "react";
import { StatCard } from "@/components/ui/stat-card";
import { Badge } from "@/components/ui/badge";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { Server, Briefcase, DollarSign, Activity, Zap, Users, Cpu } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host, Instance, ReputationEntry } from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { AuroraBackground } from "@/components/ui/aurora-bg";
import { CanadaMapHero } from "@/components/ui/canada-hero";

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

        <FadeIn delay={0.25} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Recent Instances */}
          <div className="glow-card rounded-xl border border-border bg-surface">
            <div className="border-b border-border/60 px-5 py-3.5">
              <h3 className="flex items-center gap-2 text-sm font-semibold">
                <Briefcase className="h-4 w-4 text-accent-cyan" /> {t("dash.overview.recent_instances")}
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
                    };
                    const tierClass = tierStyles[(entry.tier || "").toLowerCase()] || "bg-surface-hover text-text-muted border-border";
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
                            {entry.tier || "new"}
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
