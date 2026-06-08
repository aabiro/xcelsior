"use client";

import { useState, useCallback, useEffect } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ArrowLeft, RefreshCw, Trash2, Loader2, Activity, Server, ScrollText,
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
import { cn } from "@/lib/utils";
import { CopyableText } from "./copyable-text";
import { MetricsPanel } from "./metrics-panel";
import { WorkersPanel } from "./workers-panel";
import { LogsPanel } from "./logs-panel";
import { KeysPanel } from "./keys-panel";
import { TryItConsole } from "./try-it-console";
import { CostUsagePanel } from "./cost-usage-panel";
import type { DetailTab } from "./types";

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
  const [deleting, setDeleting] = useState(false);

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
    } catch {
      toast.error(t("dash.serverless.load_failed"));
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

  const handleDelete = async () => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    if (!confirm(t("dash.serverless.delete_confirm"))) return;
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

  if (!endpoint) {
    return (
      <div className="text-center py-24 text-text-muted">
        <p>{t("dash.serverless.not_found")}</p>
        <Link href="/dashboard/inference" className="text-accent-violet text-sm mt-2 inline-block">
          {t("dash.serverless.back_list")}
        </Link>
      </div>
    );
  }

  const title = endpoint.name || endpoint.model_name || endpoint.model_id || endpoint.endpoint_id;

  return (
    <FadeIn className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-2">
          <Link
            href="/dashboard/inference"
            className="inline-flex items-center gap-1 text-xs text-text-muted hover:text-text-primary"
          >
            <ArrowLeft className="h-3 w-3" /> {t("dash.serverless.back_list")}
          </Link>
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-bold">{title}</h1>
            <Badge variant={endpoint.status === "active" ? "active" : "warning"}>{endpoint.status}</Badge>
            <Badge variant="default">{endpoint.mode}</Badge>
          </div>
          <CopyableText text={endpoint.endpoint_id} />
        </div>
        <div className="flex gap-2 shrink-0">
          <Button variant="outline" size="sm" onClick={() => void load(true)}>
            <RefreshCw className="h-3.5 w-3.5" />
          </Button>
          {canWrite && (
            <Button variant="ghost" size="sm" className="text-red-500" onClick={handleDelete} disabled={deleting}>
              {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            </Button>
          )}
        </div>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 overflow-x-auto border-b border-border pb-px scrollbar-none">
        {TABS.map((item) => (
          <button
            key={item.id}
            type="button"
            onClick={() => setTab(item.id)}
            className={cn(
              "flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium whitespace-nowrap border-b-2 -mb-px transition-colors",
              tab === item.id
                ? "border-accent-violet text-accent-violet"
                : "border-transparent text-text-muted hover:text-text-primary",
            )}
          >
            <item.icon className="h-4 w-4" />
            {t(item.labelKey)}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <div className="space-y-6">
          <MetricsPanel metrics={metrics} jobs={jobs} loading={loading} />
          <CostUsagePanel endpoint={endpoint} metrics={metrics} />
        </div>
      )}
      {tab === "workers" && <WorkersPanel workers={workers} loading={loading} />}
      {tab === "jobs" && <LogsPanel jobs={jobs} loading={loading} />}
      {tab === "tryit" && <TryItConsole endpoint={endpoint} canWrite={canWrite} />}
      {tab === "keys" && (
        <KeysPanel endpointId={endpointId} keys={keys} canWrite={canWrite} onRefresh={() => void load(false)} />
      )}
    </FadeIn>
  );
}