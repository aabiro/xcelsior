"use client";

import { useState, useCallback, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  RefreshCw, Trash2, Loader2, Activity, Server, ScrollText,
  Key, Terminal, BarChart3,
} from "lucide-react";
import { FadeIn } from "@/components/ui/motion";
import { useLocale } from "@/lib/locale";
import { useEventStream } from "@/hooks/useEventStream";
import * as api from "@/lib/api";
import type {
  ServerlessEndpoint, ServerlessEndpointMetrics, ServerlessWorker, ServerlessJob, ServerlessApiKey,
} from "@/lib/api";
import { toast } from "sonner";

import { formatTokenRateFromPricing } from "./constants";
import { CopyableText } from "./copyable-text";
import { MetricsPanel } from "./metrics-panel";
import { WorkersPanel } from "./workers-panel";
import { LogsPanel } from "./logs-panel";
import { KeysPanel } from "./keys-panel";
import { TryItConsole } from "./try-it-console";
import { CostUsagePanel } from "./cost-usage-panel";
import type { DetailTab } from "./types";
import {
  ApiUrlCard, EngineBadge, ServerlessBackLink, ServerlessHero, ServerlessSegmentedTabs,
} from "./serverless-ui";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

const TABS: { id: DetailTab; icon: typeof Activity; labelKey: string }[] = [
  { id: "overview", icon: BarChart3, labelKey: "dash.serverless.tab_overview" },
  { id: "workers", icon: Server, labelKey: "dash.serverless.tab_workers" },
  { id: "jobs", icon: ScrollText, labelKey: "dash.serverless.tab_jobs" },
  { id: "tryit", icon: Terminal, labelKey: "dash.serverless.tab_tryit" },
  { id: "keys", icon: Key, labelKey: "dash.serverless.tab_keys" },
];

interface EndpointDetailProps {
  endpointId: string;
  canWrite: boolean;
}

export function EndpointDetail({ endpointId, canWrite }: EndpointDetailProps) {
  const { t } = useLocale();
  const [tab, setTab] = useState<DetailTab>("overview");
  const [endpoint, setEndpoint] = useState<ServerlessEndpoint | null>(null);
  const [metrics, setMetrics] = useState<ServerlessEndpointMetrics | null>(null);
  const [workers, setWorkers] = useState<ServerlessWorker[]>([]);
  const [jobs, setJobs] = useState<ServerlessJob[]>([]);
  const [keys, setKeys] = useState<ServerlessApiKey[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      const [epRes, metricsRes, workersRes, jobsRes, keysRes] = await Promise.all([
        api.getServerlessEndpoint(endpointId),
        api.getServerlessEndpointMetrics(endpointId, 24).catch(() => ({ metrics: null })),
        api.listServerlessWorkers(endpointId).catch(() => ({ workers: [] })),
        api.listServerlessJobs(endpointId).catch(() => ({ jobs: [] })),
        api.listServerlessKeys(endpointId).catch(() => ({ keys: [] })),
      ]);
      setEndpoint(epRes.endpoint);
      setMetrics(metricsRes.metrics ?? null);
      setWorkers(workersRes.workers || []);
      setJobs(jobsRes.jobs || []);
      setKeys(keysRes.keys || []);
      setNotFound(false);
      setLoadError(false);
    } catch (err) {
      // Only treat a real 404 as "not found". Transient errors (auth refresh,
      // 5xx, network) should offer a retry instead of masquerading as a deleted
      // endpoint, otherwise a flaky load looks like "endpoint not found".
      const status = err instanceof api.ApiError ? err.status : 0;
      if (status === 404) {
        setNotFound(true);
        setLoadError(false);
      } else {
        setLoadError(true);
        toast.error(t("dash.serverless.load_failed"));
      }
    } finally {
      setLoading(false);
    }
  }, [endpointId, t]);

  useEffect(() => { void load(false); }, [load]);

  useEffect(() => {
    const onTeamChanged = () => { void load(false); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [load]);

  useEventStream({
    eventTypes: [
      "serverless_job.completed",
      "serverless_job.failed",
      "serverless_job.queued",
      "serverless_worker.scaled",
      "serverless_worker.ready",
      "serverless_endpoint.updated",
    ],
    onEvent: (_type, data) => {
      if (data?.endpoint_id === endpointId || !data?.endpoint_id) void load(false);
    },
  });

  // Live polling, worker heartbeats, boot timers, and utilization metrics
  // don't always emit SSE events, so refresh on a steady cadence while the
  // endpoint is spinning workers. Skips ticks when the tab is backgrounded and
  // refreshes immediately when it returns to the foreground.
  useEffect(() => {
    const status = endpoint?.status;
    const live = status === "active" || status === "provisioning";
    if (!live) return;
    const timer = setInterval(() => {
      if (document.visibilityState === "visible") void load(false);
    }, 5000);
    const onVisible = () => {
      if (document.visibilityState === "visible") void load(false);
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [endpoint?.status, load]);

  const handleDelete = async () => {
    setConfirmDelete(false);
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    setDeleting(true);
    try {
      await api.deleteServerlessEndpoint(endpointId);
      toast.success(t("dash.serverless.deleted"));
      window.location.href = "/dashboard/inference";
    } catch {
      toast.error(t("dash.serverless.delete_failed"));
    } finally {
      setDeleting(false);
    }
  };

  if (loading && !endpoint) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="h-8 w-8 animate-spin text-text-muted" />
      </div>
    );
  }

  // Transient failure (auth refresh, 5xx, network), offer a retry rather than
  // claiming the endpoint doesn't exist.
  if (!endpoint && loadError && !notFound) {
    return (
      <div className="flex flex-col items-center gap-3 py-24 text-center text-text-muted">
        <p>{t("dash.serverless.load_failed")}</p>
        <div className="flex items-center gap-3">
          <Button variant="outline" size="sm" onClick={() => void load(true)}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("gear.retry")}
          </Button>
          <ServerlessBackLink href="/dashboard/inference">
            {t("dash.serverless.back_list")}
          </ServerlessBackLink>
        </div>
      </div>
    );
  }

  if (!endpoint) {
    return (
      <div className="flex flex-col items-center gap-2 py-24 text-center text-text-muted">
        <p>{t("dash.serverless.not_found")}</p>
        <ServerlessBackLink href="/dashboard/inference">
          {t("dash.serverless.back_list")}
        </ServerlessBackLink>
      </div>
    );
  }

  const title = endpoint.name || endpoint.model_name || endpoint.model_id || endpoint.endpoint_id;
  const tokenRate =
    endpoint.pricing?.token_billing
      ? formatTokenRateFromPricing(endpoint.model_ref || endpoint.model_name || "", endpoint.pricing)
      : null;
  const heroDescription = [
    endpoint.model_ref || endpoint.model_name || endpoint.model_id,
    tokenRate,
  ].filter(Boolean).join(" · ");

  return (
    <FadeIn className="space-y-6">
      <ServerlessBackLink href="/dashboard/inference">{t("dash.serverless.back_list")}</ServerlessBackLink>

      <ServerlessHero
        icon={Activity}
        badge={endpoint.mode}
        title={title}
        description={heroDescription}
        accent="violet"
        compact
      >
        <Badge variant={endpoint.status === "active" ? "active" : "warning"}>
          {["active", "provisioning", "scaled_down", "error", "deleted"].includes(endpoint.status)
            ? t(`dash.serverless.status_${endpoint.status}`)
            : endpoint.status}
        </Badge>
        {endpoint.mode === "preset" && <EngineBadge engine={endpoint.managed_engine} />}
        <Button variant="outline" size="sm" onClick={() => void load(true)}>
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
        {canWrite && (
          <Button variant="ghost" size="sm" className="text-red-500" onClick={() => setConfirmDelete(true)} disabled={deleting}>
            {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
          </Button>
        )}
      </ServerlessHero>

      <ConfirmDialog
        open={confirmDelete}
        title={t("dash.serverless.delete_title")}
        description={t("dash.serverless.delete_confirm")}
        confirmLabel={t("dash.serverless.delete_cta")}
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setConfirmDelete(false)}
      />

      <div className="flex flex-wrap items-center gap-3 text-xs text-text-muted -mt-2">
        <CopyableText text={endpoint.endpoint_id} />
        {endpoint.vanity_slug && (
          <span className="font-mono text-text-secondary">/{endpoint.vanity_slug}</span>
        )}
      </div>

      {endpoint.openai_base_url && endpoint.mode === "preset" && (
        <ApiUrlCard
          title={t("dash.serverless.openai_base")}
          url={endpoint.openai_base_url}
          slug={endpoint.vanity_slug}
          invokePath={endpoint.invoke_path}
        />
      )}

      <ServerlessSegmentedTabs tabs={TABS} value={tab} onChange={setTab} label={t} />

      {tab === "overview" && (
        <div className="space-y-6">
          <MetricsPanel metrics={metrics} jobs={jobs} loading={loading} />
          <CostUsagePanel endpoint={endpoint} metrics={metrics} />
        </div>
      )}
      {tab === "workers" && <WorkersPanel workers={workers} jobs={jobs} loading={loading} />}
      {tab === "jobs" && <LogsPanel jobs={jobs} loading={loading} />}
      {tab === "tryit" && <TryItConsole endpoint={endpoint} canWrite={canWrite} />}
      {tab === "keys" && (
        <KeysPanel endpointId={endpointId} keys={keys} canWrite={canWrite} onRefresh={() => void load(false)} />
      )}
    </FadeIn>
  );
}