"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { LaunchInstanceModal } from "@/components/instances/launch-instance-modal";
import {
  Briefcase, Plus, Search, RefreshCw, XCircle, ArrowUpDown, ArrowUp, ArrowDown,
  Square, Play, RotateCcw, Zap, Camera, Lock, Unlock, Pencil, RotateCw,
} from "lucide-react";
import { RefreshCw as Restart } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Instance } from "@/lib/api";
import {
  stopInstance, startInstance, restartInstance, terminateInstance,
  lockInstance, unlockInstance, resetInstance, renameInstance,
} from "@/lib/api";
import { toast } from "sonner";
import { useEventStream } from "@/hooks/useEventStream";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { SaveAsTemplateDialog } from "@/components/instances/save-as-template-dialog";

type SortKey = "name" | "gpu_type" | "status" | "created_at";
type SortDir = "asc" | "desc";
type InstanceAction =
  | "stop" | "start" | "restart" | "reset" | "terminate"
  | "cancel" | "requeue" | "lock" | "unlock";
type ActionPending = { id: string; action: InstanceAction } | null;

const ACTION_CONFIRM: Record<InstanceAction, {
  title: string; description: string; confirmLabel: string; variant: "danger" | "default";
}> = {
  stop: {
    title: "Stop this instance?",
    description: "You won't be charged for GPU time while it's stopped, but storage keeps running. Start it again whenever you're ready.",
    confirmLabel: "Stop",
    variant: "default",
  },
  start: {
    title: "Start this instance?",
    description: "The container is restored from its stopped state. Compute billing resumes immediately.",
    confirmLabel: "Start",
    variant: "default",
  },
  restart: {
    title: "Restart this instance?",
    description: "The container is stopped and restarted. Data is preserved and billing is continuous.",
    confirmLabel: "Restart",
    variant: "default",
  },
  reset: {
    title: "Reset the container?",
    description: "Restarts your pod with a fresh /workspace. Mounted volumes are preserved; ephemeral scratch data is wiped.",
    confirmLabel: "Reset",
    variant: "default",
  },
  terminate: {
    title: "Terminate this instance permanently?",
    description: "All data in /workspace will be deleted and can't be recovered. Named volumes are preserved.",
    confirmLabel: "Terminate",
    variant: "danger",
  },
  cancel: {
    title: "Cancel instance?",
    description: "The queued or provisioning instance will be removed from the queue.",
    confirmLabel: "Cancel",
    variant: "danger",
  },
  requeue: {
    title: "Requeue instance?",
    description: "Returns to the queue and is reassigned to a fresh host. Job definition and volumes are preserved; container state is reset.",
    confirmLabel: "Requeue",
    variant: "default",
  },
  lock: {
    title: "Lock this instance?",
    description: "While locked, every lifecycle action (stop / start / restart / reset / terminate / rename) is blocked. Useful when you've configured something you don't want to wipe by accident.",
    confirmLabel: "Lock",
    variant: "default",
  },
  unlock: {
    title: "Unlock this instance?",
    description: "Re-enables all lifecycle actions.",
    confirmLabel: "Unlock",
    variant: "default",
  },
};

function IconButton({
  title,
  onClick,
  disabled,
  tone = "default",
  children,
}: {
  title: string;
  onClick: () => void;
  disabled?: boolean;
  tone?: "default" | "danger" | "warn" | "success" | "accent";
  children: React.ReactNode;
}) {
  const toneClass = {
    default: "text-text-secondary hover:text-text-primary hover:bg-surface-hover",
    danger: "text-accent-red/80 hover:text-accent-red hover:bg-accent-red/10",
    warn: "text-accent-gold/80 hover:text-accent-gold hover:bg-accent-gold/10",
    success: "text-emerald/80 hover:text-emerald hover:bg-emerald/10",
    accent: "text-ice-blue/80 hover:text-ice-blue hover:bg-ice-blue/10",
  }[tone];
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
      className={`inline-flex h-7 w-7 items-center justify-center rounded-md transition-colors disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:bg-transparent ${disabled ? "text-text-tertiary" : toneClass}`}
    >
      {children}
    </button>
  );
}

function RowActions({
  inst,
  onAction,
  onSnapshot,
  onRename,
}: {
  inst: Instance;
  onAction: (id: string, action: InstanceAction) => void;
  onSnapshot: (inst: Instance) => void;
  onRename: (inst: Instance) => void;
}) {
  const { status } = inst;
  const isLocked = (inst as unknown as { locked?: boolean; payload?: { locked?: boolean } }).locked === true
    || (inst as unknown as { payload?: { locked?: boolean } }).payload?.locked === true;
  const isRunning = status === "running";
  const isStopped = status === "stopped";
  const isQueued = ["queued", "assigned", "leased"].includes(status);
  const isTerminal = ["completed", "failed", "cancelled", "terminated", "preempted"].includes(status);

  // Locked: only Unlock works. Everything else is visibly disabled so the
  // user understands *why* actions are ghosted rather than silently failing.
  if (isLocked) {
    return (
      <div className="inline-flex items-center gap-0.5">
        <IconButton
          title="Unlock instance"
          tone="warn"
          onClick={() => onAction(inst.job_id, "unlock")}
        >
          <Unlock className="h-3.5 w-3.5" />
        </IconButton>
        <IconButton title="Locked — unlock to edit" onClick={() => {}} disabled>
          <Pencil className="h-3.5 w-3.5" />
        </IconButton>
        <IconButton title="Locked — unlock to restart" onClick={() => {}} disabled>
          <Restart className="h-3.5 w-3.5" />
        </IconButton>
        <IconButton title="Locked — unlock to stop/terminate" onClick={() => {}} disabled>
          <Square className="h-3.5 w-3.5" />
        </IconButton>
      </div>
    );
  }

  // Compose the inline row. The intent is RunPod-style: see all actions at a
  // glance, click directly without opening a dropdown. Unlike the old
  // ⋮-menu layout, this lets users build muscle-memory around icon positions.
  return (
    <div className="inline-flex items-center gap-0.5">
      {(isRunning || isStopped) && (
        <IconButton
          title="Lock instance"
          onClick={() => onAction(inst.job_id, "lock")}
        >
          <Lock className="h-3.5 w-3.5" />
        </IconButton>
      )}
      {(isRunning || isStopped) && (
        <IconButton title="Rename" onClick={() => onRename(inst)}>
          <Pencil className="h-3.5 w-3.5" />
        </IconButton>
      )}
      {isRunning && (
        <>
          <IconButton
            title="Restart container"
            tone="accent"
            onClick={() => onAction(inst.job_id, "restart")}
          >
            <RotateCw className="h-3.5 w-3.5" />
          </IconButton>
          <IconButton
            title="Reset /workspace (preserves volumes)"
            tone="accent"
            onClick={() => onAction(inst.job_id, "reset")}
          >
            <RefreshCw className="h-3.5 w-3.5" />
          </IconButton>
          <IconButton
            title="Save as template"
            tone="accent"
            onClick={() => onSnapshot(inst)}
          >
            <Camera className="h-3.5 w-3.5" />
          </IconButton>
          <IconButton
            title="Stop instance"
            tone="warn"
            onClick={() => onAction(inst.job_id, "stop")}
          >
            <Square className="h-3.5 w-3.5" />
          </IconButton>
        </>
      )}
      {isStopped && (
        <>
          <IconButton
            title="Start instance"
            tone="success"
            onClick={() => onAction(inst.job_id, "start")}
          >
            <Play className="h-3.5 w-3.5" />
          </IconButton>
          <IconButton
            title="Terminate instance"
            tone="danger"
            onClick={() => onAction(inst.job_id, "terminate")}
          >
            <Zap className="h-3.5 w-3.5" />
          </IconButton>
        </>
      )}
      {isQueued && (
        <IconButton
          title="Cancel queued instance"
          tone="danger"
          onClick={() => onAction(inst.job_id, "cancel")}
        >
          <XCircle className="h-3.5 w-3.5" />
        </IconButton>
      )}
      {isTerminal && (
        <IconButton
          title="Requeue"
          onClick={() => onAction(inst.job_id, "requeue")}
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </IconButton>
      )}
    </div>
  );
}

export default function InstancesPage() {
  const [instances, setInstances] = useState<Instance[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [sortKey, setSortKey] = useState<SortKey>("created_at");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);
  const [pendingAction, setPendingAction] = useState<ActionPending>(null);
  const [snapshotTarget, setSnapshotTarget] = useState<Instance | null>(null);
  const api = useApi();
  const { t } = useLocale();
  const router = useRouter();
  const searchParams = useSearchParams();

  const load = useCallback(() => {
    setLoading(true);
    api.fetchInstances()
      .then((res) => setInstances(res.instances || []))
      .catch(() => toast.error("Failed to load instances"))
      .finally(() => setLoading(false));
  }, [api]);

  useEffect(() => { load(); }, [load]);

  useEventStream({
    eventTypes: ["job_status", "job_submitted", "job_error"],
    onEvent: (_type, data) => {
      if (data.error) toast.warning(String(data.message || "Your instance is waiting for a GPU"));
      else load();
    },
  });

  const launchOpen = searchParams.get("launch") === "true";

  const openLaunchModal = useCallback(() => {
    const next = new URLSearchParams(searchParams.toString());
    next.set("launch", "true");
    router.push(`/dashboard/instances?${next.toString()}`, { scroll: false });
  }, [router, searchParams]);

  const closeLaunchModal = useCallback(() => {
    const next = new URLSearchParams(searchParams.toString());
    next.delete("launch");
    const query = next.toString();
    router.replace(query ? `/dashboard/instances?${query}` : "/dashboard/instances", { scroll: false });
  }, [router, searchParams]);

  function requestAction(id: string, action: NonNullable<ActionPending>["action"]) {
    setPendingAction({ id, action });
  }

  async function executeAction() {
    if (!pendingAction) return;
    const { id, action } = pendingAction;
    setPendingAction(null);
    // Optimistic UI — immediately reflect the intended terminal/transitional
    // state so the row visually responds to the click rather than waiting on
    // the server round-trip (the terminate endpoint is 202 accepted + async
    // SSH kill, so ground truth can lag several seconds). If the API call
    // fails, `load()` in the catch path corrects the optimistic state.
    const optimisticStatus: Partial<Record<InstanceAction, string>> = {
      stop: "stopping",
      start: "starting",
      restart: "starting",
      terminate: "terminated",
      cancel: "cancelled",
      requeue: "queued",
      // reset/lock/unlock don't change status — omit so the row stays put
    };
    const prevInstances = instances;
    const target = optimisticStatus[action];
    if (target) {
      setInstances((curr) =>
        curr.map((inst) =>
          inst.job_id === id ? { ...inst, status: target } : inst,
        ),
      );
    }
    try {
      switch (action) {
        case "stop":      await stopInstance(id);        toast.success("Instance stopping…"); break;
        case "start":     await startInstance(id);       toast.success("Instance starting…"); break;
        case "restart":   await restartInstance(id);     toast.success("Instance restarting…"); break;
        case "reset":     await resetInstance(id);       toast.success("Container resetting…"); break;
        case "terminate": await terminateInstance(id);   toast.success("Instance terminating…"); break;
        case "cancel":    await api.cancelInstance(id);  toast.success("Instance cancelled"); break;
        case "requeue":   await api.requeueInstance(id); toast.success("Instance requeued"); break;
        case "lock":      await lockInstance(id);        toast.success("Instance locked"); break;
        case "unlock":    await unlockInstance(id);      toast.success("Instance unlocked"); break;
      }
      load();
    } catch (err) {
      // Rollback optimistic state so the user sees the real status on failure
      setInstances(prevInstances);
      toast.error(err instanceof Error ? err.message : `${action} failed`);
    }
  }

  const filtered = instances
    .filter((j) => {
      if (statusFilter !== "all" && j.status !== statusFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!j.name?.toLowerCase().includes(q) && !j.job_id?.toLowerCase().includes(q)) return false;
      }
      return true;
    })
    .sort((a, b) => {
      let av: string | number | undefined, bv: string | number | undefined;
      if (sortKey === "created_at") {
        av = a.created_at || a.submitted_at || "";
        bv = b.created_at || b.submitted_at || "";
      } else if (sortKey === "gpu_type") {
        av = a.gpu_type || a.gpu_model || "";
        bv = b.gpu_type || b.gpu_model || "";
      } else {
        av = (a as unknown as Record<string, unknown>)[sortKey] as string ?? "";
        bv = (b as unknown as Record<string, unknown>)[sortKey] as string ?? "";
      }
      const cmp =
        typeof av === "number" && typeof bv === "number"
          ? av - bv
          : String(av).localeCompare(String(bv));
      return sortDir === "asc" ? cmp : -cmp;
    });

  const { paginate, totalPages } = usePagination(filtered, 10);
  const pageItems = paginate(page);

  useEffect(() => { setPage(1); }, [search, statusFilter, sortKey, sortDir]);

  function toggleSort(key: SortKey) {
    if (sortKey === key) setSortDir(sortDir === "asc" ? "desc" : "asc");
    else { setSortKey(key); setSortDir("asc"); }
  }

  function SortIcon({ col }: { col: SortKey }) {
    if (sortKey !== col) return <ArrowUpDown className="h-3 w-3 opacity-40" />;
    return sortDir === "asc" ? <ArrowUp className="h-3 w-3" /> : <ArrowDown className="h-3 w-3" />;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.instances.title")}</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
          <Button size="sm" onClick={openLaunchModal}>
            <Plus className="h-3.5 w-3.5" /> {t("dash.instances.submit")}
          </Button>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-col gap-3 sm:flex-row">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
          <Input
            className="pl-9"
            placeholder={t("dash.instances.search")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
          <option value="all">{t("dash.instances.all_status")}</option>
          <option value="queued">{t("dash.instances.queued")}</option>
          <option value="running">{t("dash.instances.running")}</option>
          <option value="stopping">{t("dash.instances.stopping")}</option>
          <option value="stopped">{t("dash.instances.stopped")}</option>
          <option value="restarting">{t("dash.instances.restarting")}</option>
          <option value="completed">{t("dash.instances.completed")}</option>
          <option value="failed">{t("dash.instances.failed")}</option>
          <option value="cancelled">{t("dash.instances.cancelled")}</option>
          <option value="terminated">{t("dash.instances.terminated")}</option>
        </Select>
      </div>

      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="h-16 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : filtered.length === 0 ? (
        <Card className="p-12 text-center">
          <Briefcase className="mx-auto h-12 w-12 text-text-muted mb-4" />
          <h3 className="text-lg font-semibold mb-1">{t("dash.instances.empty")}</h3>
          <p className="text-sm text-text-secondary">{t("dash.instances.empty_desc")}</p>
        </Card>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-text-secondary">
                <th
                  className="py-3 pr-4 text-left font-medium cursor-pointer select-none"
                  onClick={() => toggleSort("name")}
                >
                  <span className="inline-flex items-center gap-1">
                    {t("dash.instances.col_job")} <SortIcon col="name" />
                  </span>
                </th>
                <th
                  className="py-3 px-4 text-left font-medium cursor-pointer select-none"
                  onClick={() => toggleSort("gpu_type")}
                >
                  <span className="inline-flex items-center gap-1">
                    {t("dash.instances.col_gpu")} <SortIcon col="gpu_type" />
                  </span>
                </th>
                <th
                  className="py-3 px-4 text-center font-medium cursor-pointer select-none"
                  onClick={() => toggleSort("status")}
                >
                  <span className="inline-flex items-center gap-1 justify-center">
                    {t("dash.instances.col_status")} <SortIcon col="status" />
                  </span>
                </th>
                <th
                  className="py-3 px-4 text-center font-medium cursor-pointer select-none"
                  onClick={() => toggleSort("created_at")}
                >
                  <span className="inline-flex items-center gap-1 justify-center">
                    {t("dash.instances.col_created")} <SortIcon col="created_at" />
                  </span>
                </th>
                <th className="py-3 px-4 text-right font-medium">{t("dash.instances.col_actions")}</th>
              </tr>
            </thead>
            <tbody>
              {pageItems.map((inst) => (
                <tr
                  key={inst.job_id}
                  className="border-b border-border/50 hover:bg-surface-hover transition-colors"
                >
                  <td className="py-3 pr-4">
                    <Link
                      href={`/dashboard/instances/${inst.job_id}`}
                      className="font-medium text-ice-blue hover:underline"
                    >
                      {inst.name || inst.job_id}
                    </Link>
                  </td>
                  <td className="py-3 px-4 text-text-secondary">{inst.gpu_type || inst.host_gpu || inst.gpu_model || (inst.host_id ? "Auto" : "Pending")}</td>
                  <td className="py-3 px-4 text-center">
                    <StatusBadge status={inst.status} />
                  </td>
                  <td className="py-3 px-4 text-center text-text-muted">
                    {(inst.created_at || inst.submitted_at)
                      ? new Date((inst.created_at || inst.submitted_at) * 1000).toLocaleDateString()
                      : "—"}
                  </td>
                  <td className="py-3 px-4 text-right">
                    <div className="flex justify-end items-center gap-1">
                      <Link href={`/dashboard/instances/${inst.job_id}`}>
                        <Button variant="ghost" size="sm">View</Button>
                      </Link>
                      <RowActions
                        inst={inst}
                        onAction={requestAction}
                        onSnapshot={setSnapshotTarget}
                        onRename={async (i) => {
                          const next = window.prompt("Rename instance:", i.name || "");
                          if (!next || next.trim() === i.name) return;
                          try {
                            await renameInstance(i.job_id, next.trim());
                            toast.success("Renamed");
                            load();
                          } catch (err) {
                            toast.error(err instanceof Error ? err.message : "Rename failed");
                          }
                        }}
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="mt-4 flex items-center justify-between">
            <span className="text-xs text-text-muted">{t("dash.instances.count", { count: filtered.length })}</span>
            <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />
          </div>
        </div>
      )}

      <SaveAsTemplateDialog
        open={snapshotTarget !== null}
        instanceId={snapshotTarget?.job_id ?? ""}
        defaultName={snapshotTarget?.name || undefined}
        onClose={() => setSnapshotTarget(null)}
      />

      {pendingAction && (
        <ConfirmDialog
          open
          title={ACTION_CONFIRM[pendingAction.action].title}
          description={ACTION_CONFIRM[pendingAction.action].description}
          confirmLabel={ACTION_CONFIRM[pendingAction.action].confirmLabel}
          cancelLabel={t("common.cancel")}
          variant={ACTION_CONFIRM[pendingAction.action].variant}
          onConfirm={executeAction}
          onCancel={() => setPendingAction(null)}
        />
      )}

      <LaunchInstanceModal
        open={launchOpen}
        onClose={closeLaunchModal}
        onLaunched={(jobId, inst) => {
          closeLaunchModal();
          // Optimistically add so the user can navigate to it immediately
          setInstances((prev) => {
            if (prev.some((i) => i.job_id === jobId)) return prev;
            const stub: Instance = inst ?? {
              job_id: jobId,
              name: "",
              status: "queued",
              docker_image: "",
              created_at: Date.now() / 1000,
              submitted_at: Date.now() / 1000,
            } as Instance;
            return [stub, ...prev];
          });
          load();
        }}
      />
    </div>
  );
}
