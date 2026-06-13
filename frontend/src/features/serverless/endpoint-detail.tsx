"use client";

import { useState, useCallback, useEffect } from "react";
import Link from "next/link";
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
      <ServerlessBackLink href="/dashboard/inference">{t("dash.serverless.back_list")}</ServerlessBackLink>

      <ServerlessHero
        icon={Activity}
        badge={endpoint.mode}
        title={title}
        description={endpoint.model_ref || endpoint.model_name || endpoint.model_id}
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
      {tab === "workers" && <WorkersPanel workers={workers} loading={loading} />}
      {tab === "jobs" && <LogsPanel jobs={jobs} loading={loading} />}
      {tab === "tryit" && <TryItConsole endpoint={endpoint} canWrite={canWrite} />}
      {tab === "keys" && (
        <KeysPanel endpointId={endpointId} keys={keys} canWrite={canWrite} onRefresh={() => void load(false)} />
      )}
    </FadeIn>
  );
}