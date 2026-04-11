"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";
import { Pagination, usePagination } from "@/components/ui/pagination";
import { Dialog } from "@/components/ui/dialog";
import { LaunchInstanceForm } from "@/components/instances/launch-instance-form";
import {
  Briefcase, Plus, Search, RefreshCw, XCircle, ArrowUpDown, ArrowUp, ArrowDown,
  MoreVertical, Square, Play, RotateCcw, Zap,
} from "lucide-react";
import { RefreshCw as Restart } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { Instance } from "@/lib/api";
import { stopInstance, startInstance, restartInstance, terminateInstance } from "@/lib/api";
import { toast } from "sonner";
import { useEventStream } from "@/hooks/useEventStream";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

type SortKey = "name" | "gpu_type" | "status" | "created_at";
type SortDir = "asc" | "desc";
type ActionPending = { id: string; action: "stop" | "start" | "restart" | "terminate" | "cancel" } | null;

const ACTION_CONFIRM: Record<NonNullable<ActionPending>["action"], {
  title: string; description: string; confirmLabel: string; variant: "danger" | "default";
}> = {
  stop: {
    title: "Stop instance?",
    description: "The container is gracefully stopped (SIGTERM). Data and volumes are preserved. Storage billing continues.",
    confirmLabel: "Stop",
    variant: "default",
  },
  start: {
    title: "Start instance?",
    description: "The container is restored from its stopped state. Compute billing resumes immediately.",
    confirmLabel: "Start",
    variant: "default",
  },
  restart: {
    title: "Restart instance?",
    description: "The container is stopped and restarted. All data is preserved. Billing is continuous — no gap.",
    confirmLabel: "Restart",
    variant: "default",
  },
  terminate: {
    title: "Terminate instance?",
    description: "Hard-kills and permanently removes the container. Named volumes are preserved, but all other container data is lost. This cannot be undone.",
    confirmLabel: "Terminate",
    variant: "danger",
  },
  cancel: {
    title: "Cancel instance?",
    description: "The queued or provisioning instance will be removed from the queue.",
    confirmLabel: "Cancel",
    variant: "danger",
  },
};

function RowActions({
  inst,
  onAction,
}: {
  inst: Instance;
  onAction: (id: string, action: NonNullable<ActionPending>["action"]) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const { status } = inst;
  const isRunning = status === "running";
  const isStopped = ["stopped", "user_paused", "paused_low_balance"].includes(status);
  const isQueued = ["queued", "assigned", "leased"].includes(status);
  const isFailed = status === "failed";
  const isTerminal = ["completed", "failed", "cancelled", "terminated", "preempted"].includes(status);

  useEffect(() => {
    if (!open) return;
    function close(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  type ActionItem = {
    label: string;
    action: NonNullable<ActionPending>["action"];
    icon: React.ReactNode;
    className?: string;
  };
  const actions: ActionItem[] = [
    ...(isRunning
      ? [
          { label: "Stop", action: "stop" as const, icon: <Square className="h-3.5 w-3.5" />, className: "text-accent-gold" },
          { label: "Restart", action: "restart" as const, icon: <Restart className="h-3.5 w-3.5" />, className: "text-ice-blue" },
        ]
      : []),
    ...(isStopped
      ? [
          { label: "Start", action: "start" as const, icon: <Play className="h-3.5 w-3.5" />, className: "text-emerald" },
          { label: "Restart", action: "restart" as const, icon: <Restart className="h-3.5 w-3.5" />, className: "text-ice-blue" },
        ]
      : []),
    ...(isFailed ? [{ label: "Requeue", action: "restart" as const, icon: <RotateCcw className="h-3.5 w-3.5" /> }] : []),
    ...(isQueued ? [{ label: "Cancel", action: "cancel" as const, icon: <XCircle className="h-3.5 w-3.5" />, className: "text-accent-red" }] : []),
    ...(!isTerminal ? [{ label: "Terminate", action: "terminate" as const, icon: <Zap className="h-3.5 w-3.5" />, className: "text-accent-red" }] : []),
  ];

  if (actions.length === 0) return null;

  return (
    <div className="relative" ref={ref}>
      <Button variant="ghost" size="sm" onClick={() => setOpen(!open)}>
        <MoreVertical className="h-3.5 w-3.5" />
      </Button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-50 w-36 rounded-lg border border-border bg-surface shadow-xl">
          {actions.map((a) => (
            <button
              key={a.action + a.label}
              onClick={() => { setOpen(false); onAction(inst.job_id, a.action); }}
              className={`flex w-full items-center gap-2 px-3 py-2 text-sm hover:bg-surface-hover transition-colors ${a.className ?? "text-text-secondary"}`}
            >
              {a.icon} {a.label}
            </button>
          ))}
        </div>
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
    eventTypes: ["job_status", "job_submitted"],
    onEvent: () => { load(); },
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
    try {
      switch (action) {
        case "stop":      await stopInstance(id);        toast.success("Instance stopping…"); break;
        case "start":     await startInstance(id);       toast.success("Instance starting…"); break;
        case "restart":   await restartInstance(id);     toast.success("Instance restarting…"); break;
        case "terminate": await terminateInstance(id);   toast.success("Instance terminated"); break;
        case "cancel":    await api.cancelInstance(id);  toast.success("Instance cancelled"); break;
      }
      load();
    } catch (err) {
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
                  <td className="py-3 px-4 text-text-secondary">{inst.gpu_type || inst.gpu_model || "—"}</td>
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
                      <RowActions inst={inst} onAction={requestAction} />
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

      <Dialog
        open={launchOpen}
        onClose={closeLaunchModal}
        title={t("dash.newinstance.title")}
        description={t("dash.newinstance.subtitle")}
        maxWidth="max-w-5xl"
        className="h-[88vh]"
        bodyClassName="overflow-y-auto px-6 pb-6"
      >
        <LaunchInstanceForm
          className="pt-6"
          onCancel={closeLaunchModal}
          onSubmitted={() => {
            closeLaunchModal();
            load();
          }}
        />
      </Dialog>
    </div>
  );
}
