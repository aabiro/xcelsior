"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, Clock, Lock, Play, RefreshCw, Server, Activity, Layers } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/ui/stat-card";
import * as api from "@/lib/api";
import type { HpcCluster, HpcJob, HpcJobScope } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { attentionItems, formatDuration, summarise } from "./hpc-format";
import { ClustersPanel } from "./clusters-panel";
import { JobsPanel } from "./jobs-panel";
import { SubmitPanel } from "./submit-panel";

/** The read model is the reconciler's copy, so polling it costs a cluster nothing. */
const AUTO_REFRESH_MS = 30_000;

type Tab = "clusters" | "jobs" | "submit";

export default function HpcPage() {
  const { user } = useAuth();
  const router = useRouter();
  // Internal operator console: every route behind it is platform-admin only.
  const hasHpcAccess = !!user?.is_admin;

  useEffect(() => {
    if (user && !hasHpcAccess) router.replace("/dashboard");
  }, [user, hasHpcAccess, router]);

  if (!user || !hasHpcAccess) return null;

  return <HpcConsole />;
}

function errorMessage(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

function HpcConsole() {
  const { t } = useLocale();
  const [tab, setTab] = useState<Tab>("clusters");
  const [clusters, setClusters] = useState<HpcCluster[]>([]);
  const [clustersError, setClustersError] = useState<string | null>(null);
  // Open jobs feed the summary whatever the Jobs tab is filtered to.
  const [openJobs, setOpenJobs] = useState<HpcJob[]>([]);
  const [openHasMore, setOpenHasMore] = useState(false);
  const [jobs, setJobs] = useState<HpcJob[]>([]);
  const [jobsError, setJobsError] = useState<string | null>(null);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [scope, setScope] = useState<HpcJobScope>("open");
  const [clusterFilter, setClusterFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [now, setNow] = useState(() => Date.now());
  const [loadedAt, setLoadedAt] = useState<number | null>(null);
  // A slow response for an old filter must not overwrite a newer one.
  const request = useRef(0);

  const load = useCallback(async () => {
    const id = ++request.current;
    setRefreshing(true);
    const filtered = scope !== "open" || clusterFilter !== "";
    const [c, open, list] = await Promise.allSettled([
      api.fetchHpcClusters(),
      api.fetchHpcJobs({ state: "open", limit: 200 }),
      filtered
        ? api.fetchHpcJobs({ state: scope, clusterId: clusterFilter || undefined })
        : Promise.resolve(null),
    ]);
    if (id !== request.current) return;

    if (c.status === "fulfilled") {
      setClusters(c.value.clusters ?? []);
      setClustersError(null);
    } else {
      setClustersError(errorMessage(c.reason, "Could not load clusters"));
    }

    if (open.status === "fulfilled") {
      setOpenJobs(open.value.jobs ?? []);
      setOpenHasMore(!!open.value.next_cursor);
    }

    const shown = filtered ? list : open;
    if (shown.status === "fulfilled" && shown.value) {
      setJobs(shown.value.jobs ?? []);
      setNextCursor(shown.value.next_cursor ?? null);
      setJobsError(null);
    } else if (shown.status === "rejected") {
      setJobsError(errorMessage(shown.reason, "Could not load jobs"));
    }

    const at = Date.now();
    setNow(at);
    setLoadedAt(at);
    setLoading(false);
    setRefreshing(false);
  }, [scope, clusterFilter]);

  useEffect(() => { void load(); }, [load]);

  useEffect(() => {
    if (!autoRefresh) return;
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") void load();
    }, AUTO_REFRESH_MS);
    return () => clearInterval(timer);
  }, [autoRefresh, load]);

  // Ages on screen keep counting between loads.
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 5_000);
    return () => clearInterval(timer);
  }, []);

  const loadMore = useCallback(async () => {
    if (!nextCursor) return;
    setLoadingMore(true);
    try {
      const page = await api.fetchHpcJobs({
        state: scope,
        clusterId: clusterFilter || undefined,
        cursor: nextCursor,
      });
      setJobs((prev) => {
        const seen = new Set(prev.map((j) => j.mapping_id));
        return [...prev, ...(page.jobs ?? []).filter((j) => !seen.has(j.mapping_id))];
      });
      setNextCursor(page.next_cursor ?? null);
    } catch (err) {
      setJobsError(errorMessage(err, "Could not load more jobs"));
    } finally {
      setLoadingMore(false);
    }
  }, [nextCursor, scope, clusterFilter]);

  const summary = useMemo(() => summarise(clusters, openJobs, now), [clusters, openJobs, now]);
  const attention = useMemo(() => attentionItems(clusters, openJobs, now), [clusters, openJobs, now]);
  const plus = openHasMore ? "+" : "";

  const tabs: { id: Tab; label: string; icon: typeof Server; count?: number }[] = [
    { id: "clusters", label: t("dash.hpc.tab_clusters"), icon: Server, count: clusters.length },
    { id: "jobs", label: t("dash.hpc.tab_jobs"), icon: Clock, count: summary.open },
    { id: "submit", label: t("dash.hpc.tab_submit"), icon: Play },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold">{t("dash.hpc.title")}</h1>
            <Badge variant="warning" className="text-xs">
              <Lock className="h-3 w-3" /> {t("dash.hpc.internal_badge")}
            </Badge>
          </div>
          <p className="mt-1 max-w-2xl text-sm text-text-secondary">{t("dash.hpc.subtitle")}</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-xs text-text-muted" aria-live="polite">
            {loadedAt ? `Updated ${formatDuration(Math.round((now - loadedAt) / 1000))} ago` : "Loading…"}
          </span>
          <label className="flex cursor-pointer items-center gap-1.5 text-xs text-text-secondary">
            <input
              type="checkbox"
              className="h-3.5 w-3.5 accent-[var(--color-accent-cyan,#00d4ff)]"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh
          </label>
          <Button variant="outline" size="sm" onClick={() => void load()} disabled={refreshing}>
            <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} /> {t("common.refresh")}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <StatCard
          label="Clusters healthy"
          value={loading ? "—" : `${summary.clustersHealthy}/${summary.clustersEnabled}`}
          icon={Server}
          glow={summary.clustersHealthy === summary.clustersEnabled ? "emerald" : "gold"}
        />
        <StatCard label="Open jobs" value={loading ? "—" : `${summary.open}${plus}`} icon={Layers} glow="cyan" />
        <StatCard label="Running" value={loading ? "—" : `${summary.running}${plus}`} icon={Activity} glow="violet" />
        <StatCard
          label="Needs attention"
          value={loading ? "—" : attention.length}
          icon={AlertTriangle}
          glow={attention.length > 0 ? "gold" : "emerald"}
        />
      </div>

      {attention.length > 0 && (
        <div role="status" className="rounded-xl border border-accent-gold/30 bg-accent-gold/10 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-accent-gold">
            <AlertTriangle className="h-4 w-4" /> Needs attention
          </div>
          <ul className="space-y-1 text-sm text-text-primary">
            {attention.map((item) => (
              <li key={item} className="flex gap-2">
                <span aria-hidden className="text-accent-gold">•</span>
                <span className="min-w-0 break-words">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div role="tablist" aria-label="HPC console sections" className="flex gap-1 overflow-x-auto rounded-lg bg-surface p-1">
        {tabs.map((tb) => (
          <button
            key={tb.id}
            role="tab"
            aria-selected={tab === tb.id}
            onClick={() => setTab(tb.id)}
            className={cn(
              "flex items-center gap-1.5 whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
              tab === tb.id ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary",
            )}
          >
            <tb.icon className="h-3.5 w-3.5" /> {tb.label}
            {tb.count ? (
              <span className="ml-1 rounded-full bg-ice-blue px-1.5 py-0.5 text-[10px] text-white">{tb.count}</span>
            ) : null}
          </button>
        ))}
      </div>

      {tab === "clusters" && (
        <ClustersPanel
          clusters={clusters}
          loading={loading}
          error={clustersError}
          now={now}
          onChanged={() => void load()}
        />
      )}

      {tab === "jobs" && (
        <JobsPanel
          jobs={jobs}
          clusters={clusters}
          loading={loading}
          error={jobsError}
          now={now}
          scope={scope}
          onScope={setScope}
          clusterFilter={clusterFilter}
          onClusterFilter={setClusterFilter}
          hasMore={!!nextCursor}
          loadingMore={loadingMore}
          onLoadMore={() => void loadMore()}
          onChanged={() => void load()}
        />
      )}

      {tab === "submit" && (
        <SubmitPanel
          onSubmitted={() => {
            setScope("open");
            setClusterFilter("");
            setTab("jobs");
            void load();
          }}
        />
      )}
    </div>
  );
}
