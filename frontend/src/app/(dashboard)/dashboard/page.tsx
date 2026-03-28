"use client";

import { useEffect, useState } from "react";
import { StatCard } from "@/components/ui/stat-card";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { Server, Briefcase, DollarSign, Activity, Zap, Users } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Host, Job, ReputationEntry } from "@/lib/api";

export default function DashboardOverview() {
  const [hosts, setHosts] = useState<Host[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [leaderboard, setLeaderboard] = useState<ReputationEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useApi();
  const { t } = useLocale();

  useEffect(() => {
    Promise.allSettled([
      api.fetchHosts(),
      api.fetchJobs(),
      api.fetchLeaderboard(),
    ]).then(([h, j, l]) => {
      if (h.status === "fulfilled") setHosts(h.value.hosts || []);
      if (j.status === "fulfilled") setJobs(j.value.jobs || []);
      if (l.status === "fulfilled") setLeaderboard(l.value.leaderboard || []);
      setLoading(false);
    });
  }, [api]);

  const activeHosts = hosts.filter((h) => h.status === "active").length;
  const runningJobs = jobs.filter((j) => j.status === "running").length;
  const queuedJobs = jobs.filter((j) => j.status === "queued").length;

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
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">{t("dash.overview.title")}</h1>

      {/* Stats Row */}
      <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StaggerItem><StatCard label={t("dash.overview.active_hosts")} value={activeHosts} icon={Server} /></StaggerItem>
        <StaggerItem><StatCard label={t("dash.overview.running_jobs")} value={runningJobs} icon={Briefcase} /></StaggerItem>
        <StaggerItem><StatCard label={t("dash.overview.total_hosts")} value={hosts.length} icon={DollarSign} /></StaggerItem>
        <StaggerItem><StatCard label={t("dash.overview.queued")} value={queuedJobs} icon={Activity} /></StaggerItem>
      </StaggerList>

      <FadeIn delay={0.25} className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Jobs */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Briefcase className="h-4 w-4" /> {t("dash.overview.recent_jobs")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {jobs.length === 0 ? (
              <p className="text-sm text-text-muted">{t("dash.overview.no_jobs")}</p>
            ) : (
              <div className="space-y-3">
                {jobs.slice(0, 5).map((job) => (
                  <div key={job.job_id} className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div>
                      <p className="text-sm font-medium">{job.name || job.job_id}</p>
                      <p className="text-xs text-text-muted">{job.gpu_type || job.gpu_model}</p>
                    </div>
                    <Badge variant={job.status === "running" ? "active" : job.status === "completed" ? "completed" : job.status === "queued" ? "queued" : "default"}>
                      {job.status}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Top Hosts / Leaderboard */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Users className="h-4 w-4" /> {t("dash.overview.top_providers")}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {leaderboard.length === 0 ? (
              <p className="text-sm text-text-muted">{t("dash.overview.no_leaderboard")}</p>
            ) : (
              <div className="space-y-3">
                {leaderboard.slice(0, 5).map((entry, i) => (
                  <div key={entry.entity_id || i} className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div className="flex items-center gap-3">
                      <span className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-gold/20 text-xs font-bold text-accent-gold">
                        {i + 1}
                      </span>
                      <span className="text-sm font-medium">{entry.user_id || entry.entity_id}</span>
                    </div>
                    <span className="text-sm font-mono text-text-secondary">
                      {entry.score?.toFixed(1) || "—"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  );
}
