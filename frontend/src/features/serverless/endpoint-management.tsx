"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { Dialog } from "@/components/ui/dialog";
import {
  Cpu,
  Globe,
  Key,
  Loader2,
  RefreshCw,
  Rocket,
  ScrollText,
  Settings,
  Terminal,
  Trash2,
} from "lucide-react";
import type {
  ServerlessApiKey,
  ServerlessEndpoint,
  ServerlessJob,
  ServerlessWorker,
} from "@/lib/api";
import * as api from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { CopyableText } from "./copyable-text";
import { formatModelDisplayName, formatServerlessChip, formatWorkerSecondPrice } from "./format";
import { EndpointWorkerMonitor } from "./endpoint-worker-monitor";
import { ServerlessEmptyState, ServerlessSegmentedTabs } from "./serverless-ui";
import { TryItConsole } from "./try-it-console";
import { LogsPanel } from "./logs-panel";
import { KeysPanel } from "./keys-panel";

type EndpointTab = "tryit" | "jobs" | "keys";

type EndpointWorkerEditForm = {
  min_workers: number;
  max_workers: number;
  idle_timeout_sec: number;
  max_concurrency: number;
  execution_mode: "sync" | "async";
  queue_timeout_sec: number;
  scaling_policy_type: "queue_request_count" | "queue_delay";
  scaling_policy_value: number;
};

const TABS = [
  { id: "tryit" as const, icon: Terminal, labelKey: "dash.serverless.tab_tryit" },
  { id: "jobs" as const, icon: ScrollText, labelKey: "dash.serverless.tab_jobs" },
  { id: "keys" as const, icon: Key, labelKey: "dash.serverless.tab_keys" },
];

function endpointLabel(endpoint: ServerlessEndpoint): string {
  const name = String(endpoint.name || "").trim();
  if (name && !name.includes("/")) return name;
  return endpoint.endpoint_id.slice(0, 18);
}

function editFormFromEndpoint(endpoint: ServerlessEndpoint): EndpointWorkerEditForm {
  const mode = endpoint.execution_mode || "sync";
  return {
    min_workers: endpoint.min_workers ?? 1,
    max_workers: endpoint.max_workers ?? 3,
    idle_timeout_sec: endpoint.idle_timeout_sec ?? 60,
    max_concurrency: endpoint.max_concurrency ?? 1,
    execution_mode: mode,
    queue_timeout_sec: endpoint.queue_timeout_sec ?? 120,
    scaling_policy_type: mode === "async" ? "queue_request_count" : (endpoint.scaling_policy_type as EndpointWorkerEditForm["scaling_policy_type"]) || "queue_delay",
    scaling_policy_value: endpoint.scaling_policy_value ?? (mode === "async" ? 1 : 4),
  };
}

function workerPatch(endpoint: ServerlessEndpoint, form: EndpointWorkerEditForm) {
  const current = editFormFromEndpoint(endpoint);
  const patch: Partial<EndpointWorkerEditForm> = {};
  (Object.keys(form) as (keyof EndpointWorkerEditForm)[]).forEach((key) => {
    if (form[key] !== current[key]) patch[key] = form[key] as never;
  });
  return patch;
}

export function ServerlessEndpointManagement({
  endpoints,
  selectedEndpointId,
  canWrite,
  loading,
  onSelectEndpoint,
  onRefresh,
  onCreateEndpoint,
  onDeleteEndpoint,
}: {
  endpoints: ServerlessEndpoint[];
  selectedEndpointId?: string;
  canWrite: boolean;
  loading?: boolean;
  onSelectEndpoint: (endpointId: string) => void;
  onRefresh: () => void;
  onCreateEndpoint: () => void;
  onDeleteEndpoint: (endpointId: string) => void;
}) {
  const { t } = useLocale();
  const [tab, setTab] = useState<EndpointTab>("tryit");
  const [workers, setWorkers] = useState<ServerlessWorker[]>([]);
  const [jobs, setJobs] = useState<ServerlessJob[]>([]);
  const [keys, setKeys] = useState<ServerlessApiKey[]>([]);
  const [detailLoading, setDetailLoading] = useState(false);
  const [editOpen, setEditOpen] = useState(false);
  const [editForm, setEditForm] = useState<EndpointWorkerEditForm | null>(null);
  const [confirmEdit, setConfirmEdit] = useState(false);
  const [savingEdit, setSavingEdit] = useState(false);

  const selectedEndpoint = useMemo(() => {
    if (!endpoints.length) return null;
    return endpoints.find((endpoint) => endpoint.endpoint_id === selectedEndpointId) || endpoints[0];
  }, [endpoints, selectedEndpointId]);

  const endpointId = selectedEndpoint?.endpoint_id || "";

  const loadDetail = useCallback(async () => {
    if (!endpointId) return;
    setDetailLoading(true);
    try {
      const [workersRes, jobsRes, keysRes] = await Promise.all([
        api.listServerlessWorkers(endpointId).catch(() => ({ workers: [] })),
        api.listServerlessJobs(endpointId).catch(() => ({ jobs: [] })),
        api.listServerlessKeys(endpointId).catch(() => ({ keys: [] })),
      ]);
      setWorkers(workersRes.workers || []);
      setJobs(jobsRes.jobs || []);
      setKeys(keysRes.keys || []);
    } finally {
      setDetailLoading(false);
    }
  }, [endpointId]);

  useEffect(() => {
    void loadDetail();
  }, [loadDetail]);

  useEffect(() => {
    if (!endpointId) return;
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") void loadDetail();
    }, 5000);
    return () => window.clearInterval(id);
  }, [endpointId, loadDetail]);

  const updateEditForm = <K extends keyof EndpointWorkerEditForm>(field: K, value: EndpointWorkerEditForm[K]) => {
    setEditForm((prev) => {
      if (!prev) return prev;
      if (field === "execution_mode") {
        const mode = value as EndpointWorkerEditForm["execution_mode"];
        if (mode === "async") {
          return {
            ...prev,
            execution_mode: "async",
            scaling_policy_type: "queue_request_count",
            scaling_policy_value: 1,
          };
        }
        return {
          ...prev,
          execution_mode: "sync",
          scaling_policy_type: prev.scaling_policy_type === "queue_request_count" ? prev.scaling_policy_type : "queue_delay",
          scaling_policy_value: prev.scaling_policy_type === "queue_request_count" ? prev.scaling_policy_value : 4,
        };
      }
      if (field === "scaling_policy_type") {
        const policy = value as EndpointWorkerEditForm["scaling_policy_type"];
        return {
          ...prev,
          scaling_policy_type: policy,
          scaling_policy_value: policy === "queue_delay" ? 4 : 1,
        };
      }
      return { ...prev, [field]: value };
    });
  };

  const openEdit = () => {
    if (!selectedEndpoint) return;
    setEditForm(editFormFromEndpoint(selectedEndpoint));
    setEditOpen(true);
  };

  const requestSaveEdit = () => {
    if (!selectedEndpoint || !editForm) return;
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    const patch = workerPatch(selectedEndpoint, editForm);
    if (Object.keys(patch).length === 0) {
      setEditOpen(false);
      return;
    }
    setConfirmEdit(true);
  };

  const saveEdit = async () => {
    if (!selectedEndpoint || !editForm) return;
    setSavingEdit(true);
    setConfirmEdit(false);
    try {
      await api.patchServerlessEndpoint(selectedEndpoint.endpoint_id, workerPatch(selectedEndpoint, editForm));
      toast.success("Endpoint settings updated");
      setEditOpen(false);
      onRefresh();
      await loadDetail();
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to update endpoint settings");
    } finally {
      setSavingEdit(false);
    }
  };

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Loader2 className="h-7 w-7 animate-spin text-text-muted" />
      </div>
    );
  }

  if (!selectedEndpoint) {
    return (
      <ServerlessEmptyState icon={Cpu} title={t("dash.serverless.empty")} accent="cyan">
        {canWrite && (
          <Button className="mt-2" onClick={onCreateEndpoint}>
            <Rocket className="h-4 w-4" /> Create endpoint
          </Button>
        )}
      </ServerlessEmptyState>
    );
  }

  const rawModel = selectedEndpoint.model_ref || selectedEndpoint.model_name || selectedEndpoint.model_id || "";
  const fullUrl = typeof window !== "undefined"
    ? `${window.location.origin}${selectedEndpoint.openai_base_url || selectedEndpoint.invoke_path || ""}`
    : selectedEndpoint.openai_base_url || selectedEndpoint.invoke_path || "";

  return (
    <div className="grid gap-5 xl:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="rounded-xl border border-border bg-surface p-3">
        <div className="mb-3 flex items-center justify-between gap-2">
          <p className="text-sm font-medium">My Endpoints</p>
          {canWrite && (
            <Button size="sm" onClick={onCreateEndpoint}>
              <Rocket className="h-3.5 w-3.5" /> Create
            </Button>
          )}
        </div>
        <div className="space-y-1">
          {endpoints.map((endpoint) => {
            const active = endpoint.endpoint_id === selectedEndpoint.endpoint_id;
            return (
              <button
                key={endpoint.endpoint_id}
                type="button"
                onClick={() => onSelectEndpoint(endpoint.endpoint_id)}
                className={cn(
                  "block w-full rounded-lg px-2 py-2 text-left text-sm transition-colors",
                  active ? "bg-accent-violet/12 text-accent-violet" : "text-text-muted hover:bg-surface-hover hover:text-text-primary",
                )}
              >
                <span className="block truncate font-medium">{endpointLabel(endpoint)}</span>
                <span className="block truncate text-[11px] opacity-75">
                  {formatServerlessChip(endpoint.status)} · Workers {endpoint.min_workers}-{endpoint.max_workers}
                </span>
              </button>
            );
          })}
        </div>
      </aside>

      <div className="min-w-0 space-y-5">
        <section className="glow-card brand-top-accent rounded-xl border border-border bg-surface p-4">
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0 space-y-3">
              <div className="rounded-lg border border-border bg-background px-3 py-2.5">
                <p className="mb-1 text-[11px] uppercase tracking-wide text-text-muted">OpenAI base URL</p>
                <CopyableText text={fullUrl || "-"} className="font-mono text-sm text-text-primary" />
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-xl font-semibold">{endpointLabel(selectedEndpoint)}</h2>
                <Badge variant={selectedEndpoint.status === "active" ? "active" : "warning"}>
                  {formatServerlessChip(selectedEndpoint.status)}
                </Badge>
                <Badge variant="default">{formatServerlessChip(selectedEndpoint.execution_mode || "sync")}</Badge>
              </div>
              <div className="grid gap-x-6 gap-y-2 text-xs text-text-muted sm:grid-cols-2 xl:grid-cols-3">
                <span>Endpoint ID <CopyableText text={selectedEndpoint.endpoint_id} /></span>
                <span>Model {formatModelDisplayName(rawModel) || "-"}</span>
                <span className="inline-flex items-center gap-1"><Globe className="h-3 w-3" /> {selectedEndpoint.region}</span>
                <span>Image {selectedEndpoint.image_ref || selectedEndpoint.docker_image || "-"}</span>
                <span>{selectedEndpoint.gpu_count || 1} * {selectedEndpoint.gpu_type || selectedEndpoint.gpu_tier || "Auto"}</span>
                <span>Workers Min {selectedEndpoint.min_workers} - Max {selectedEndpoint.max_workers}</span>
                <span>{formatWorkerSecondPrice(selectedEndpoint.pricing)}</span>
                <span>Created {selectedEndpoint.created_at ? new Date(selectedEndpoint.created_at * 1000).toLocaleString() : "-"}</span>
              </div>
            </div>
            <div className="flex shrink-0 flex-wrap gap-2">
              <Button variant="outline" size="sm" onClick={() => { onRefresh(); void loadDetail(); }}>
                <RefreshCw className="h-3.5 w-3.5" />
              </Button>
              {canWrite && (
                <>
                  <Button variant="outline" size="sm" onClick={openEdit}>
                    <Settings className="h-3.5 w-3.5" /> Settings
                  </Button>
                  <Button variant="ghost" size="sm" className="text-red-500" onClick={() => onDeleteEndpoint(selectedEndpoint.endpoint_id)}>
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </>
              )}
            </div>
          </div>
          <EndpointWorkerMonitor endpointId={selectedEndpoint.endpoint_id} workers={workers} loading={detailLoading} />
        </section>

        <ServerlessSegmentedTabs tabs={TABS} value={tab} onChange={setTab} label={t} />
        {tab === "tryit" && <TryItConsole endpoint={selectedEndpoint} canWrite={canWrite} />}
        {tab === "jobs" && (
          <LogsPanel
            endpoint={selectedEndpoint}
            jobs={jobs}
            loading={detailLoading}
            canWrite={canWrite}
            onOpenTryIt={() => setTab("tryit")}
          />
        )}
        {tab === "keys" && (
          <KeysPanel endpointId={selectedEndpoint.endpoint_id} keys={keys} canWrite={canWrite} onRefresh={loadDetail} />
        )}
      </div>

      <ConfirmDialog
        open={confirmEdit}
        title="Confirm changes"
        description="Xcelsior may start a new worker before releasing the old one. During this update, running workers can briefly exceed the configured maximum and startup usually takes about a minute."
        confirmLabel="Apply changes"
        onConfirm={saveEdit}
        onCancel={() => setConfirmEdit(false)}
      />

      <Dialog
        open={editOpen}
        onClose={() => setEditOpen(false)}
        title="Endpoint settings"
        description="Worker-affecting changes can start or replace endpoint workers."
        maxWidth="max-w-2xl"
      >
        {editForm && (
          <div className="space-y-5">
            <div className="grid gap-4 sm:grid-cols-2">
              <NumberField label="Workers min" value={editForm.min_workers} min={0} max={editForm.max_workers} onChange={(value) => updateEditForm("min_workers", value)} />
              <NumberField label="Workers max" value={editForm.max_workers} min={Math.max(1, editForm.min_workers)} max={32} onChange={(value) => updateEditForm("max_workers", value)} />
              <NumberField label="Idle timeout(s)" value={editForm.idle_timeout_sec} min={60} max={3600} onChange={(value) => updateEditForm("idle_timeout_sec", value)} />
              <NumberField label="Max concurrency" value={editForm.max_concurrency} min={1} max={256} onChange={(value) => updateEditForm("max_concurrency", value)} />
              <label className="space-y-1.5 text-sm">
                <span className="font-medium">Execution mode</span>
                <select
                  value={editForm.execution_mode}
                  onChange={(event) => updateEditForm("execution_mode", event.target.value === "async" ? "async" : "sync")}
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
                >
                  <option value="sync">Sync</option>
                  <option value="async">Async</option>
                </select>
              </label>
              <NumberField label="Queue timeout(s)" value={editForm.queue_timeout_sec} min={1} max={3600} onChange={(value) => updateEditForm("queue_timeout_sec", value)} />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <label className="space-y-1.5 text-sm">
                <span className="font-medium">Scale policy</span>
                <select
                  value={editForm.scaling_policy_type}
                  onChange={(event) => updateEditForm("scaling_policy_type", event.target.value === "queue_delay" ? "queue_delay" : "queue_request_count")}
                  disabled={editForm.execution_mode === "async"}
                  className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet disabled:opacity-70"
                >
                  {editForm.execution_mode === "sync" && <option value="queue_delay">Queue delay</option>}
                  <option value="queue_request_count">Request count</option>
                </select>
              </label>
              <NumberField
                label={
                  editForm.execution_mode === "async"
                    ? "Worker Max Request Count"
                    : editForm.scaling_policy_type === "queue_delay"
                      ? "Queue Delay Time(s)"
                      : "Request Count"
                }
                value={editForm.scaling_policy_value}
                min={1}
                max={1000}
                onChange={(value) => updateEditForm("scaling_policy_value", value)}
              />
            </div>

            <div className="rounded-lg border border-border bg-surface-hover/40 px-3 py-2 text-xs text-text-muted">
              Min workers 0 removes idle worker charge after scale-down. Active cold-start and running time are billed per running worker, except dashboard Try It tests.
            </div>

            <div className="flex justify-end gap-3">
              <Button variant="outline" size="sm" onClick={() => setEditOpen(false)} disabled={savingEdit}>
                Cancel
              </Button>
              <Button size="sm" onClick={requestSaveEdit} disabled={savingEdit}>
                {savingEdit ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                Save changes
              </Button>
            </div>
          </div>
        )}
      </Dialog>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="space-y-1.5 text-sm">
      <span className="font-medium">{label}</span>
      <input
        type="number"
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Math.max(min, Math.min(max, Number(event.target.value) || min)))}
        className="w-full rounded-lg border border-border bg-surface px-3 py-2 text-sm outline-none focus:border-accent-violet"
      />
    </label>
  );
}
