"use client";

import { useState, useCallback, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  RefreshCw, Trash2, Loader2, Activity, Server, ScrollText,
  Key, Terminal, BarChart3, Rocket, Settings,
} from "lucide-react";
import Link from "next/link";
import { FadeIn } from "@/components/ui/motion";
import { useLocale } from "@/lib/locale";
import { useEventStream } from "@/hooks/useEventStream";
import * as api from "@/lib/api";
import type {
  ServerlessEndpoint, ServerlessEndpointMetrics, ServerlessWorker, ServerlessJob, ServerlessApiKey,
  GpuAvailability,
} from "@/lib/api";
import { toast } from "sonner";

import { formatTokenRateFromPricing } from "./constants";
import { formatModelDisplayName, formatServerlessChip } from "./format";
import { CopyableText } from "./copyable-text";
import { WorkersPanel } from "./workers-panel";
import { LogsPanel } from "./logs-panel";
import { KeysPanel } from "./keys-panel";
import { TryItConsole } from "./try-it-console";
import { ServerlessOverview } from "./serverless-overview";
import type { DetailTab } from "./types";
import {
  ApiUrlCard, EngineBadge, ServerlessBackLink, ServerlessHero, ServerlessSegmentedTabs,
} from "./serverless-ui";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Dialog } from "@/components/ui/dialog";
import { DeployStudio } from "./deploy-studio";

const TABS: { id: DetailTab; icon: typeof Activity; labelKey: string }[] = [
  { id: "overview", icon: BarChart3, labelKey: "dash.serverless.tab_overview" },
  { id: "workers", icon: Server, labelKey: "dash.serverless.tab_workers" },
  { id: "jobs", icon: ScrollText, labelKey: "dash.serverless.tab_jobs" },
  { id: "tryit", icon: Terminal, labelKey: "dash.serverless.tab_tryit" },
  { id: "keys", icon: Key, labelKey: "dash.serverless.tab_keys" },
];

type EndpointWorkerEditForm = {
  min_workers: number;
  max_workers: number;
  idle_timeout_sec: number;
  max_concurrency: number;
  execution_mode: "sync" | "async";
  queue_timeout_sec: number;
  scaling_policy_type: string;
  scaling_policy_value: number;
};

function editFormFromEndpoint(endpoint: ServerlessEndpoint): EndpointWorkerEditForm {
  return {
    min_workers: endpoint.min_workers ?? 0,
    max_workers: endpoint.max_workers ?? 3,
    idle_timeout_sec: endpoint.idle_timeout_sec ?? 60,
    max_concurrency: endpoint.max_concurrency ?? 1,
    execution_mode: endpoint.execution_mode || "sync",
    queue_timeout_sec: endpoint.queue_timeout_sec ?? 120,
    scaling_policy_type: endpoint.scaling_policy_type || "queue_delay",
    scaling_policy_value: endpoint.scaling_policy_value ?? 1,
  };
}

function buildWorkerEditPatch(endpoint: ServerlessEndpoint, form: EndpointWorkerEditForm) {
  const current = editFormFromEndpoint(endpoint);
  const patch: Partial<EndpointWorkerEditForm> = {};
  (Object.keys(form) as (keyof EndpointWorkerEditForm)[]).forEach((key) => {
    if (form[key] !== current[key]) {
      patch[key] = form[key] as never;
    }
  });
  return patch;
}

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
  const [allEndpoints, setAllEndpoints] = useState<ServerlessEndpoint[]>([]);
  const [gpus, setGpus] = useState<GpuAvailability[]>([]);
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [loadError, setLoadError] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [createOpen, setCreateOpen] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [editForm, setEditForm] = useState<EndpointWorkerEditForm | null>(null);
  const [confirmEdit, setConfirmEdit] = useState(false);
  const [savingEdit, setSavingEdit] = useState(false);

  const load = useCallback(async (showSpinner = false) => {
    if (showSpinner) setLoading(true);
    try {
      const listEndpoints =
        typeof api.listServerlessEndpoints === "function"
          ? api.listServerlessEndpoints().catch(() => ({ endpoints: [] }))
          : Promise.resolve({ endpoints: [] });
      const availableGpus =
        typeof api.fetchAvailableGPUs === "function"
          ? api.fetchAvailableGPUs().catch(() => ({ gpus: [] }))
          : Promise.resolve({ gpus: [] });
      const [epRes, metricsRes, workersRes, jobsRes, keysRes, listRes, gpuRes] = await Promise.all([
        api.getServerlessEndpoint(endpointId),
        api.getServerlessEndpointMetrics(endpointId, 24).catch(() => ({ metrics: null })),
        api.listServerlessWorkers(endpointId).catch(() => ({ workers: [] })),
        api.listServerlessJobs(endpointId).catch(() => ({ jobs: [] })),
        api.listServerlessKeys(endpointId).catch(() => ({ keys: [] })),
        listEndpoints,
        availableGpus,
      ]);
      setEndpoint(epRes.endpoint);
      setMetrics(metricsRes.metrics ?? null);
      setWorkers(workersRes.workers || []);
      setJobs(jobsRes.jobs || []);
      setKeys(keysRes.keys || []);
      setAllEndpoints(listRes.endpoints || []);
      setGpus(gpuRes.gpus || []);
      setNotFound(false);
      setLoadError(false);
    } catch (err) {
      // Only treat a real 404 as "not found". Transient errors (auth refresh,
      // 5xx, network) should offer a retry instead of masquerading as a deleted
      // endpoint, otherwise a flaky load looks like "endpoint not found".
      const status = typeof err === "object" && err && "status" in err
        ? Number((err as { status?: number }).status || 0)
        : 0;
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

  const openWorkerEdit = () => {
    if (!endpoint) return;
    setEditForm(editFormFromEndpoint(endpoint));
    setEditOpen(true);
  };

  const updateEditForm = <K extends keyof EndpointWorkerEditForm>(field: K, value: EndpointWorkerEditForm[K]) => {
    setEditForm((prev) => (prev ? { ...prev, [field]: value } : prev));
  };

  const requestWorkerEditSave = () => {
    if (!endpoint || !editForm) return;
    if (!canWrite) {
      toast.error(t("dash.serverless.viewer_blocked"));
      return;
    }
    const patch = buildWorkerEditPatch(endpoint, editForm);
    if (Object.keys(patch).length === 0) {
      setEditOpen(false);
      return;
    }
    setConfirmEdit(true);
  };

  const confirmWorkerEditSave = async () => {
    if (!endpoint || !editForm) return;
    const patch = buildWorkerEditPatch(endpoint, editForm);
    setConfirmEdit(false);
    setSavingEdit(true);
    try {
      await api.patchServerlessEndpoint(endpointId, patch);
      toast.success("Endpoint settings updated");
      setEditOpen(false);
      await load(false);
    } catch {
      toast.error("Failed to update endpoint settings");
    } finally {
      setSavingEdit(false);
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

  const rawModel = endpoint.model_ref || endpoint.model_name || endpoint.model_id;
  const title = endpoint.name || formatModelDisplayName(rawModel) || endpoint.endpoint_id;
  const tokenRate =
    endpoint.pricing?.token_billing
      ? formatTokenRateFromPricing(endpoint.model_ref || endpoint.model_name || "", endpoint.pricing)
      : null;
  const heroDescription = [
    formatModelDisplayName(rawModel),
    tokenRate,
  ].filter(Boolean).join(" · ");

  return (
    <FadeIn className="space-y-6">
      <ServerlessBackLink href="/dashboard/inference">{t("dash.serverless.back_list")}</ServerlessBackLink>

      <ServerlessHero
        icon={Activity}
	        badge={formatServerlessChip(endpoint.mode)}
        title={title}
        description={heroDescription}
        accent="violet"
        compact
      >
        <Badge variant={endpoint.status === "active" ? "active" : "warning"}>
	          {["active", "provisioning", "scaled_down", "error", "deleted"].includes(endpoint.status)
	            ? t(`dash.serverless.status_${endpoint.status}`)
	            : formatServerlessChip(endpoint.status)}
        </Badge>
        {endpoint.mode === "preset" && <EngineBadge engine={endpoint.managed_engine} />}
        <Button variant="outline" size="sm" onClick={() => void load(true)}>
          <RefreshCw className="h-3.5 w-3.5" />
        </Button>
        {canWrite && (
          <>
            <Button variant="outline" size="sm" onClick={openWorkerEdit}>
              <Settings className="h-3.5 w-3.5" /> Edit workers
            </Button>
            <Button variant="ghost" size="sm" className="text-red-500" onClick={() => setConfirmDelete(true)} disabled={deleting}>
              {deleting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
            </Button>
          </>
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

      <ConfirmDialog
        open={confirmEdit}
        title="Confirm changes"
        description="Xcelsior may start a new worker before releasing the old one. During this update, running workers can briefly exceed the configured maximum and startup usually takes about a minute."
        confirmLabel="Apply changes"
        onConfirm={confirmWorkerEditSave}
        onCancel={() => setConfirmEdit(false)}
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

	      <div className="grid gap-5 xl:grid-cols-[260px_minmax(0,1fr)]">
	        <aside className="space-y-3">
	          <div className="rounded-xl border border-border bg-surface p-3">
	            <div className="mb-3 flex items-center justify-between gap-2">
	              <p className="text-sm font-medium">My Endpoints</p>
	              {canWrite && (
	                <Button size="sm" variant="outline" onClick={() => setCreateOpen(true)}>
	                  <Rocket className="h-3.5 w-3.5" /> Create
	                </Button>
	              )}
	            </div>
	            <div className="space-y-1">
              {allEndpoints.map((ep) => {
                const active = ep.endpoint_id === endpointId;
                return (
                  <Link
                    key={ep.endpoint_id}
                    href={`/dashboard/inference/${ep.endpoint_id}`}
                    className={`block rounded-lg px-2.5 py-2.5 text-left text-sm transition-all border border-transparent ${active ? "bg-accent-violet/12 text-accent-violet border-accent-violet/20" : "text-text-muted hover:bg-surface-hover hover:text-text-primary"}`}
                  >
                    <span className={`block truncate text-[10px] font-mono tracking-tight ${active ? "text-accent-violet/90" : "text-text-muted"}`}>
                      {ep.openai_base_url || ep.invoke_path || "/run"}
                    </span>
                    <span className={`block truncate text-sm font-semibold mt-0.5 ${active ? "text-accent-violet" : "text-text-primary"}`}>
                      {ep.name || "Untitled Endpoint"}
                    </span>
                    <span className="block truncate text-[11px] opacity-70 mt-0.5">
                      {formatServerlessChip(ep.status)} · Workers {ep.min_workers}-{ep.max_workers}
                    </span>
                  </Link>
                );
              })}
            </div>
	          </div>
	        </aside>
	        <div className="min-w-0 space-y-5">
	          <ServerlessSegmentedTabs tabs={TABS} value={tab} onChange={setTab} label={t} />

	          {tab === "overview" && <ServerlessOverview endpoint={endpoint} metrics={metrics} />}
	          {tab === "workers" && <WorkersPanel endpointId={endpointId} workers={workers} jobs={jobs} loading={loading} />}
	          {tab === "jobs" && <LogsPanel endpoint={endpoint} jobs={jobs} loading={loading} canWrite={canWrite} />}
	          {tab === "tryit" && <TryItConsole endpoint={endpoint} canWrite={canWrite} />}
	          {tab === "keys" && (
	            <KeysPanel endpointId={endpointId} keys={keys} canWrite={canWrite} onRefresh={() => void load(false)} />
	          )}
	        </div>
	      </div>

	      <Dialog
	        open={editOpen}
	        onClose={() => setEditOpen(false)}
	        title="Edit worker settings"
	        description="Changes here can start, replace, or release endpoint workers."
	        maxWidth="max-w-2xl"
	      >
	        {editForm && (
	          <div className="space-y-5">
	            <div className="grid gap-4 sm:grid-cols-2">
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Workers min</span>
	                <input
	                  type="number"
	                  min={0}
	                  max={editForm.max_workers}
	                  value={editForm.min_workers}
	                  onChange={(e) => updateEditForm("min_workers", Math.max(0, Number(e.target.value) || 0))}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                />
	              </label>
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Workers max</span>
	                <input
	                  type="number"
	                  min={Math.max(1, editForm.min_workers)}
	                  value={editForm.max_workers}
	                  onChange={(e) => updateEditForm("max_workers", Math.max(1, Number(e.target.value) || 1))}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                />
	              </label>
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Idle timeout(s)</span>
	                <input
	                  type="number"
	                  min={1}
	                  max={3600}
	                  value={editForm.idle_timeout_sec}
	                  onChange={(e) => updateEditForm("idle_timeout_sec", Math.max(1, Number(e.target.value) || 1))}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                />
	              </label>
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Max concurrency</span>
	                <input
	                  type="number"
	                  min={1}
	                  max={256}
	                  value={editForm.max_concurrency}
	                  onChange={(e) => updateEditForm("max_concurrency", Math.max(1, Number(e.target.value) || 1))}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                />
	              </label>
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Execution mode</span>
	                <select
	                  value={editForm.execution_mode}
	                  onChange={(e) => updateEditForm("execution_mode", e.target.value === "async" ? "async" : "sync")}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                >
	                  <option value="sync">Sync</option>
	                  <option value="async">Async</option>
	                </select>
	              </label>
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Queue timeout(s)</span>
	                <input
	                  type="number"
	                  min={1}
	                  max={3600}
	                  value={editForm.queue_timeout_sec}
	                  onChange={(e) => updateEditForm("queue_timeout_sec", Math.max(1, Number(e.target.value) || 1))}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                />
	              </label>
	            </div>

	            <div className="grid gap-4 sm:grid-cols-2">
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">Scale policy</span>
	                <select
	                  value={editForm.scaling_policy_type}
	                  onChange={(e) => updateEditForm("scaling_policy_type", e.target.value)}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                >
	                  <option value="queue_delay">Queue delay</option>
	                  <option value="queue_request_count">Request count</option>
	                </select>
	              </label>
	              <label className="space-y-1.5 text-sm">
	                <span className="font-medium">{editForm.scaling_policy_type === "queue_delay" ? "Queue delay time(s)" : "Request count"}</span>
	                <input
	                  type="number"
	                  min={1}
	                  max={1000}
	                  value={editForm.scaling_policy_value}
	                  onChange={(e) => updateEditForm("scaling_policy_value", Math.max(1, Number(e.target.value) || 1))}
	                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
	                />
	              </label>
	            </div>

	            <div className="rounded-lg border border-border bg-surface-hover/40 px-3 py-2 text-xs text-text-muted">
	              Setting min workers to 0 removes idle worker charge after the endpoint scales down. Cold starts and active running time are still billed per running worker.
	            </div>

	            <div className="flex justify-end gap-3">
	              <Button variant="outline" size="sm" onClick={() => setEditOpen(false)} disabled={savingEdit}>
	                Cancel
	              </Button>
	              <Button size="sm" onClick={requestWorkerEditSave} disabled={savingEdit}>
	                {savingEdit ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
	                Save changes
	              </Button>
	            </div>
	          </div>
	        )}
	      </Dialog>

	      <Dialog open={createOpen} onClose={() => setCreateOpen(false)} title="Create endpoint" maxWidth="max-w-5xl" bodyClassName="px-6 pb-6 overflow-y-auto">
	        <DeployStudio gpus={gpus} canWrite={canWrite} />
	      </Dialog>
	    </FadeIn>
	  );
	}
