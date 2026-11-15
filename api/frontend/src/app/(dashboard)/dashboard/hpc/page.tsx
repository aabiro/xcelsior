"use client";

import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import {
  AlertTriangle,
  ArrowRight,
  CircleAlert,
  Clock,
  Cpu,
  Inbox,
  Lock,
  Play,
  ReceiptText,
  RefreshCw,
  Server,
  ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import * as api from "@/lib/api";
import type { HpcCluster, HpcJob, HpcJobScope } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { attentionItems, formatDuration, summarise, type AttentionItem, type HpcTab } from "./hpc-format";
import { FleetPanel } from "./fleet-panel";
import { JobsPanel } from "./jobs-panel";
import { LedgerPanel } from "./ledger-panel";
import { SubmissionsPanel } from "./submissions-panel";
import { SubmitPanel } from "./submit-panel";

/** The read model is the reconciler's copy, so polling it costs a cluster nothing. */
const AUTO_REFRESH_MS = 30_000;

const TABS: HpcTab[] = ["fleet", "jobs", "ledger", "submissions", "submit"];

export default function HpcPage() {
  const { user } = useAuth();
  const router = useRouter();
  // Internal operator console: every route behind it is platform-admin only.
  const hasHpcAccess = !!user?.is_admin;

  useEffect(() => {
    if (user && !hasHpcAccess) router.replace("/dashboard");
  }, [user, hasHpcAccess, router]);

  if (!user || !hasHpcAccess) return null;

  return (
    <Suspense fallback={null}>
      <HpcConsole />
    </Suspense>
  );
}

function errorMessage(err: unknown, fallback: string): string {
  return err instanceof Error && err.message ? err.message : fallback;
}

function HpcConsole() {
  const { t } = useLocale();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  // The tab and the open job live in the address bar, so a view can be linked
  // to and the back button closes a job instead of leaving the console.
  const tabParam = searchParams.get("tab");
  const openJobId = searchParams.get("job");
  const tab: HpcTab = TABS.includes(tabParam as HpcTab) ? (tabParam as HpcTab) : openJobId ? "jobs" : "fleet";

  const setQuery = useCallback(
    (changes: Record<string, string | null>) => {
      const next = new URLSearchParams(searchParams.toString());
      for (const [key, value] of Object.entries(changes)) {
        if (value === null || value === "") next.delete(key);
        else next.set(key, value);
      }
      const query = next.toString();
      router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
    },
    [router, pathname, searchParams],
  );
  const setTab = useCallback((next: HpcTab) => setQuery({ tab: next === "fleet" ? null : next, job: null }), [setQuery]);
  const openJob = useCallback(
    (id: string | null) => setQuery(id ? { tab: "jobs", job: id } : { job: null }),
    [setQuery],
  );

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
  const [focusCluster, setFocusCluster] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [now, setNow] = useState(() => Date.now());
  const [loadedAt, setLoadedAt] = useState<number | null>(null);
  // The ledger and the submit log reload on a deliberate refresh only: they
  // change hourly, and a timer would throw away the pages already loaded.
  const [refreshKey, setRefreshKey] = useState(0);
  // A slow response for an old filter must not overwrite a newer one.
  const request = useRef(0);

  const load = useCallback(async () => {
    const id = ++request.current;
    setRefreshing(true);
    setLoadingMore(false);
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

  const refresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
    void load();
  }, [load]);

  const loadMore = useCallback(async () => {
    if (!nextCursor) return;
    const id = request.current;
    setLoadingMore(true);
    try {
      const page = await api.fetchHpcJobs({
        state: scope,
        clusterId: clusterFilter || undefined,
        cursor: nextCursor,
      });
      if (id !== request.current) return;
      setJobs((prev) => {
        const seen = new Set(prev.map((j) => j.mapping_id));
        return [...prev, ...(page.jobs ?? []).filter((j) => !seen.has(j.mapping_id))];
      });
      setNextCursor(page.next_cursor ?? null);
    } catch (err) {
      if (id === request.current) setJobsError(errorMessage(err, "Could not load more jobs"));
    } finally {
      if (id === request.current) setLoadingMore(false);
    }
  }, [nextCursor, scope, clusterFilter]);

  const summary = useMemo(() => summarise(clusters, openJobs, now), [clusters, openJobs, now]);
  const attention = useMemo(() => attentionItems(clusters, openJobs, now), [clusters, openJobs, now]);
  // Counts from the loaded page are a floor when more jobs exist.
  const plus = !summary.exact && openHasMore ? "+" : "";
  const critical = attention.some((item) => item.severity === "critical");

  const viewClusterJobs = useCallback(
    (clusterId: string) => {
      setScope("open");
      setClusterFilter(clusterId);
      setTab("jobs");
    },
    [setTab],
  );

  const act = (item: AttentionItem) => {
    if (item.tab === "fleet") setFocusCluster(item.clusterId ?? null);
    setTab(item.tab);
  };

  const tabs: { id: HpcTab; label: string; icon: typeof Server; count?: number }[] = [
    { id: "fleet", label: t("dash.hpc.tab_clusters"), icon: Server, count: clusters.length },
    { id: "jobs", label: t("dash.hpc.tab_jobs"), icon: Clock, count: summary.open },
    { id: "ledger", label: t("dash.hpc.tab_ledger"), icon: ReceiptText },
    { id: "submissions", label: t("dash.hpc.tab_submissions"), icon: Inbox },
    { id: "submit", label: t("dash.hpc.tab_submit"), icon: Play },
  ];

  const status = loading
    ? { label: "Loading…", dot: "bg-text-muted", text: "text-text-muted" }
    : clustersError
      ? { label: "Could not reach the API", dot: "bg-accent-red", text: "text-accent-red" }
      : critical
        ? { label: "Needs attention now", dot: "bg-accent-red", text: "text-accent-red" }
        : attention.length > 0
          ? { label: `${attention.length} to look at`, dot: "bg-accent-gold", text: "text-accent-gold" }
          : summary.clustersEnabled === 0
            ? { label: "No cluster is polled", dot: "bg-text-muted", text: "text-text-muted" }
            : { label: "All clear", dot: "bg-emerald", text: "text-emerald" };

  const figures = [
    {
      label: "Clusters healthy",
      value: loading ? "—" : `${summary.clustersHealthy}/${summary.clustersEnabled}`,
      note: loading ? "" : `${clusters.length} registered`,
      tone: !loading && summary.clustersHealthy < summary.clustersEnabled ? "text-accent-gold" : "",
    },
    {
      label: "Running",
      value: loading ? "—" : `${summary.running}${plus}`,
      note: loading ? "" : `of ${summary.open}${plus} open`,
      tone: "",
    },
    {
      label: "Pending",
      value: loading ? "—" : `${summary.pending}${plus}`,
      note: loading ? "" : summary.awaitingAccounting > 0 ? `${summary.awaitingAccounting} awaiting accounting` : "in the queue",
      tone: "",
    },
    {
      label: "Needs attention",
      value: loading ? "—" : String(attention.length),
      note: loading ? "" : attention.length === 0 ? "nothing to look at" : `${summary.stale} stale · ${summary.orphans} unmapped`,
      tone: !loading && attention.length > 0 ? (critical ? "text-accent-red" : "text-accent-gold") : "",
    },
  ];

  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-2xl border border-accent-cyan/20 bg-gradient-to-br from-accent-cyan/10 via-surface to-accent-violet/10">
        <div aria-hidden className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-accent-violet/15 blur-3xl" />
        <div aria-hidden className="absolute -bottom-28 -left-16 h-56 w-56 rounded-full bg-accent-cyan/10 blur-3xl" />

        <div className="relative flex flex-col gap-5 p-5 sm:p-7 lg:flex-row lg:items-start lg:justify-between">
          <div className="min-w-0">
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-cyan/20 text-accent-cyan">
                <Cpu className="h-5 w-5" />
              </div>
              <Badge variant="warning" className="gap-1 text-[10px] uppercase tracking-widest">
                <Lock className="h-3 w-3" /> {t("dash.hpc.internal_badge")}
              </Badge>
            </div>
            <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">{t("dash.hpc.title")}</h1>
            <p className="mt-2 flex max-w-2xl items-start gap-2 text-sm text-text-secondary">
              <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald" />
              <span>{t("dash.hpc.subtitle")}</span>
            </p>
          </div>

          <div className="flex shrink-0 flex-col gap-3 lg:items-end">
            <div className="flex flex-wrap items-center gap-2">
              <Button onClick={() => setTab("submit")}>
                <Play className="h-4 w-4" /> {t("dash.hpc.tab_submit")}
              </Button>
              <Button variant="outline" onClick={refresh} disabled={refreshing}>
                <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin")} /> {t("common.refresh")}
              </Button>
            </div>
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
              <span className={cn("flex items-center gap-1.5 font-medium", status.text)} role="status">
                <span aria-hidden className={cn("h-2 w-2 rounded-full", status.dot, status.dot === "bg-emerald" && "animate-pulse")} />
                {status.label}
              </span>
              <span className="text-text-muted" aria-live="polite">
                {loadedAt ? `Updated ${formatDuration(Math.round((now - loadedAt) / 1000))} ago` : ""}
              </span>
              <label className="flex cursor-pointer items-center gap-1.5 text-text-secondary">
                <input
                  type="checkbox"
                  className="h-3.5 w-3.5 accent-[var(--color-accent-cyan,#00d4ff)]"
                  checked={autoRefresh}
                  onChange={(e) => setAutoRefresh(e.target.checked)}
                />
                Auto-refresh
              </label>
            </div>
          </div>
        </div>

        <dl className="relative grid grid-cols-2 border-t border-border/60 lg:grid-cols-4">
          {figures.map((figure, i) => (
            <div
              key={figure.label}
              className={cn(
                "px-5 py-4 sm:px-7",
                i % 2 === 1 && "border-l border-border/60",
                i >= 2 && "border-t border-border/60 lg:border-t-0",
                i === 2 && "lg:border-l",
              )}
            >
              <dt className="text-xs text-text-secondary">{figure.label}</dt>
              <dd className={cn("mt-1 font-mono text-3xl font-bold tracking-tight", figure.tone)}>{figure.value}</dd>
              <dd className="mt-0.5 min-h-4 text-[11px] text-text-muted">{figure.note}</dd>
            </div>
          ))}
        </dl>
      </section>

      {attention.length > 0 && (
        <section
          aria-label="Needs attention"
          className={cn(
            "rounded-xl border p-4",
            critical ? "border-accent-red/30 bg-accent-red/10" : "border-accent-gold/30 bg-accent-gold/10",
          )}
        >
          <h2 className={cn("mb-2 flex items-center gap-2 text-sm font-semibold", critical ? "text-accent-red" : "text-accent-gold")}>
            <AlertTriangle className="h-4 w-4" /> Needs attention
          </h2>
          <ul className="divide-y divide-border/50">
            {attention.map((item) => (
              <li key={item.text} className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 py-2 text-sm">
                <span className="flex min-w-0 items-start gap-2">
                  {item.severity === "critical"
                    ? <CircleAlert className="mt-0.5 h-4 w-4 shrink-0 text-accent-red" />
                    : <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-accent-gold" />}
                  <span className="min-w-0 break-words text-text-primary">{item.text}</span>
                </span>
                <button
                  type="button"
                  onClick={() => act(item)}
                  className="flex shrink-0 items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-text-secondary hover:bg-surface-hover hover:text-text-primary"
                >
                  {item.tab === "jobs" ? "Show jobs" : "Show cluster"} <ArrowRight className="h-3 w-3" />
                </button>
              </li>
            ))}
          </ul>
        </section>
      )}

      <div role="tablist" aria-label="HPC console sections" className="flex gap-1 overflow-x-auto rounded-xl border border-border bg-surface p-1">
        {tabs.map((tb) => (
          <button
            key={tb.id}
            role="tab"
            id={`hpc-tab-${tb.id}`}
            aria-selected={tab === tb.id}
            aria-controls="hpc-tabpanel"
            onClick={() => setTab(tb.id)}
            className={cn(
              "flex items-center gap-2 whitespace-nowrap rounded-lg px-3.5 py-2 text-sm font-medium transition-colors",
              tab === tb.id
                ? "bg-card text-text-primary shadow-sm ring-1 ring-border"
                : "text-text-muted hover:bg-surface-hover hover:text-text-primary",
            )}
          >
            <tb.icon className={cn("h-4 w-4", tab === tb.id && "text-accent-cyan")} /> {tb.label}
            {tb.count ? (
              <span className="rounded-full bg-surface-hover px-1.5 py-0.5 text-[10px] tabular-nums text-text-secondary">
                {tb.count}
              </span>
            ) : null}
          </button>
        ))}
      </div>

      <div role="tabpanel" id="hpc-tabpanel" aria-labelledby={`hpc-tab-${tab}`}>
        {tab === "fleet" && (
          <FleetPanel
            clusters={clusters}
            loading={loading}
            error={clustersError}
            now={now}
            focusClusterId={focusCluster}
            onViewJobs={viewClusterJobs}
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
            openJobId={openJobId}
            onOpenJob={openJob}
          />
        )}

        {tab === "ledger" && (
          <LedgerPanel clusters={clusters} now={now} refreshKey={refreshKey} onOpenJob={openJob} />
        )}

        {tab === "submissions" && (
          <SubmissionsPanel clusters={clusters} now={now} refreshKey={refreshKey} onOpenJob={openJob} />
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
    </div>
  );
}
