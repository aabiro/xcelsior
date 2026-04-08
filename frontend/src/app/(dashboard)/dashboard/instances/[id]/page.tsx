"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import dynamic from "next/dynamic";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge, Badge } from "@/components/ui/badge";
import { LogViewer } from "@/components/ui/log-viewer";
import {
  ArrowLeft, Clock, Cpu, DollarSign, Server, RotateCcw, XCircle, Terminal, Wifi, WifiOff,
  Copy, Globe, Container, Square, Loader2, AlertTriangle, Info, ChevronDown, ChevronUp,
  Play, RefreshCw, Zap, MoreVertical, Link2,
} from "lucide-react";
import {
  fetchInstance, cancelInstance, requeueInstance,
  stopInstance, startInstance, restartInstance, terminateInstance,
} from "@/lib/api";
import type { Instance } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useInstanceWebSocket } from "@/hooks/useInstanceWebSocket";

const WebTerminal = dynamic(
  () => import("@/components/terminal/WebTerminal").then((m) => m.WebTerminal),
  { ssr: false },
);

const STATUS_STEPS = ["queued", "assigned", "starting", "running", "completed"] as const;

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400);
  const h = Math.floor((seconds % 86400) / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function DirectAccessSection({ instance }: { instance: Instance }) {
  const [open, setOpen] = useState(false);
  const containerRef = instance.container_name || `xcl-${instance.job_id}`;
  return (
    <div className="rounded-lg border border-border/50 overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-3 py-2 text-xs text-text-muted hover:text-text-secondary transition-colors"
      >
        {open ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        <span>Direct access</span>
      </button>
      {open && (
        <div className="px-3 pb-3 space-y-2">
          {instance.host_ip && (
            <div>
              <p className="text-xs text-text-muted mb-1">Direct SSH (requires mesh network):</p>
              <div className="flex items-center gap-2">
                <code className="flex-1 text-xs font-mono text-text-secondary bg-background rounded px-2 py-1.5 select-all border border-border">
                  ssh -p {instance.ssh_port || 22} root@{instance.host_ip}
                </code>
                <button
                  onClick={() => { navigator.clipboard.writeText(`ssh -p ${instance.ssh_port || 22} root@${instance.host_ip}`); toast.success("Copied"); }}
                  className="text-text-muted hover:text-text-primary transition-colors shrink-0"
                  title="Copy"
                >
                  <Copy className="h-3.5 w-3.5" />
                </button>
              </div>
            </div>
          )}
          <div>
            <p className="text-xs text-text-muted mb-1">Docker exec (on host machine):</p>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-xs font-mono text-text-secondary bg-background rounded px-2 py-1.5 select-all border border-border">
                docker exec -it {containerRef} /bin/bash
              </code>
              <button
                onClick={() => { navigator.clipboard.writeText(`docker exec -it ${containerRef} /bin/bash`); toast.success("Copied"); }}
                className="text-text-muted hover:text-text-primary transition-colors shrink-0"
                title="Copy"
              >
                <Copy className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

type ConfirmAction = "stop" | "start" | "restart" | "terminate" | "cancel" | null;

const CONFIRM_CONFIGS: Record<NonNullable<ConfirmAction>, {
  title: string;
  description: string;
  confirmLabel: string;
  variant: "danger" | "default";
}> = {
  stop: {
    title: "Stop instance?",
    description: "The container will be gracefully stopped (SIGTERM). Your data and volumes are preserved. Storage billing continues. You can start it again at any time.",
    confirmLabel: "Stop",
    variant: "default",
  },
  start: {
    title: "Start instance?",
    description: "The container will be restored from its stopped state. Compute billing resumes immediately.",
    confirmLabel: "Start",
    variant: "default",
  },
  restart: {
    title: "Restart instance?",
    description: "The container will be stopped and immediately restarted. All data is preserved.",
    confirmLabel: "Restart",
    variant: "default",
  },
  terminate: {
    title: "Terminate instance?",
    description: "This will hard-kill and permanently destroy the container. This action cannot be undone. Named volumes are preserved, but all other container data will be lost.",
    confirmLabel: "Terminate",
    variant: "danger",
  },
  cancel: {
    title: "Cancel instance?",
    description: "The queued or provisioning instance will be cancelled and removed from the queue.",
    confirmLabel: "Cancel",
    variant: "danger",
  },
};

export default function InstanceDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { t } = useLocale();
  const [instance, setInstance] = useState<Instance | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionPending, setActionPending] = useState(false);
  const [confirmAction, setConfirmAction] = useState<ConfirmAction>(null);
  const [showTerminal, setShowTerminal] = useState(false);
  const [terminalMounted, setTerminalMounted] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);
  const prevStatusRef = useRef<string | null>(null);
  const [wsLogs, setWsLogs] = useState<{ timestamp: number | string; level?: string; message: string }[]>([]);
  const [uptickKey, setUptickKey] = useState(0);

  // Track status transitions (don't auto-open terminal — avoids scroll jumps)
  useEffect(() => {
    prevStatusRef.current = instance?.status ?? null;
  }, [instance?.status]);

  // Tick uptime every 30s
  useEffect(() => {
    if (instance?.status !== "running" || !instance?.started_at) return;
    const id = setInterval(() => setUptickKey((k) => k + 1), 30_000);
    return () => clearInterval(id);
  }, [instance?.status, instance?.started_at]);

  const load = () => {
    setLoading(true);
    fetchInstance(id)
      .then((r) => setInstance(r.instance))
      .catch(() => toast.error("Failed to load instance"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { setJobError(null); load(); }, [id]);

  const isLive = instance?.status === "queued" || instance?.status === "assigned"
    || instance?.status === "starting" || instance?.status === "running" || instance?.status === "stopping"
    || instance?.status === "restarting";
  const onWsInstance = useCallback((i: Instance) => {
    // WS payloads may have 'image' instead of 'docker_image'
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const raw = i as any;
    if (!i.docker_image && raw.image) {
      i.docker_image = raw.image as string;
    }
    setInstance(i);
  }, []);
  const onWsJobError = useCallback((err: { job_id: string; error: string; message: string }) => {
    setJobError(err.message);
  }, []);
  const onWsLog = useCallback((log: { job_id: string; timestamp: number; line: string; level: string }) => {
    setWsLogs((prev) => [...prev, { timestamp: log.timestamp, level: log.level, message: log.line }].slice(-3000));
  }, []);
  const wsState = useInstanceWebSocket(id, {
    onInstance: onWsInstance,
    onJobError: onWsJobError,
    onLog: onWsLog,
    enabled: isLive,
  });

  async function executeAction(action: ConfirmAction) {
    if (!action) return;
    setConfirmAction(null);
    setActionPending(true);
    try {
      switch (action) {
        case "stop":      await stopInstance(id);      toast.success("Instance stopping…"); break;
        case "start":     await startInstance(id);     toast.success("Instance starting…"); break;
        case "restart":   await restartInstance(id);   toast.success("Instance restarting…"); break;
        case "terminate": await terminateInstance(id); toast.success("Instance terminated"); break;
        case "cancel":    await cancelInstance(id);    toast.success("Instance cancelled"); break;
      }
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : `${action} failed`);
    } finally {
      setActionPending(false);
    }
  }

  async function handleRequeue() {
    try {
      const res = await requeueInstance(id);
      toast.success("Instance requeued");
      const newId = res.instance?.job_id;
      if (newId && newId !== id) {
        router.push(`/dashboard/instances/${newId}`);
      } else {
        load();
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Requeue failed");
    }
  }

  const [showConnectModal, setShowConnectModal] = useState(false);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="h-8 w-48 rounded bg-surface skeleton-pulse" />
        <div className="h-40 rounded-xl bg-surface skeleton-pulse" />
        <div className="h-64 rounded-xl bg-surface skeleton-pulse" />
      </div>
    );
  }

  if (!instance) {
    return (
      <Card className="p-12 text-center">
        <h2 className="text-xl font-semibold mb-2">{t("dash.instances.not_found")}</h2>
        <p className="text-text-secondary mb-4">{t("dash.instances.not_found_desc")}</p>
        <Link href="/dashboard/instances"><Button variant="outline">{t("dash.instances.back")}</Button></Link>
      </Card>
    );
  }

  const status = instance.status;
  const isRunning = status === "running";
  const isStopped = status === "stopped" || status === "user_paused" || status === "paused_low_balance";
  const isQueued = status === "queued" || status === "assigned" || status === "leased";
  const isTransitional = status === "stopping" || status === "restarting" || status === "starting";
  const isFailed = status === "failed";
  const isTerminal = ["completed", "failed", "cancelled", "terminated", "preempted"].includes(status);

  const currentStepIdx = STATUS_STEPS.indexOf(
    (status === "failed" || status === "cancelled") ? "running" : (status as typeof STATUS_STEPS[number]),
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-center gap-3">
          <Link href="/dashboard/instances">
            <Button variant="ghost" size="icon" className="h-8 w-8">
              <ArrowLeft className="h-4 w-4" />
            </Button>
          </Link>
          <div>
            <h1 className="text-2xl font-bold">{instance.name || instance.job_id}</h1>
            <p className="text-sm font-mono text-text-muted">{instance.job_id}</p>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap justify-end">
          {/* Live / reconnecting / disconnected indicator */}
          {isLive && (
            <span
              className={`flex items-center gap-1.5 text-xs font-medium ${
                wsState.connected ? "text-emerald" : wsState.reconnecting ? "text-accent-gold" : "text-accent-red"
              }`}
              title={wsState.connected ? "Live" : wsState.reconnecting ? "Reconnecting…" : "Disconnected"}
            >
              <span className={`inline-block h-2 w-2 rounded-full ${
                wsState.connected ? "bg-emerald shadow-[0_0_6px_rgba(16,185,129,0.6)]" : wsState.reconnecting ? "bg-accent-gold animate-pulse shadow-[0_0_6px_rgba(234,179,8,0.5)]" : "bg-accent-red shadow-[0_0_6px_rgba(239,68,68,0.5)]"
              }`} />
              {wsState.connected ? "Live" : wsState.reconnecting ? "Reconnecting…" : "Disconnected"}
            </span>
          )}
          {/* Transitional state — gradient sweep text */}
          {isTransitional && (
            <span className="flex items-center gap-1.5 text-sm font-medium">
              <Loader2 className="h-4 w-4 brand-gradient-spinner" />
              <span className="brand-gradient-text">
                {status === "stopping" ? "Stopping…" : status === "starting" ? "Starting…" : "Restarting…"}
              </span>
            </span>
          )}
          {/* Queued/provisioning cancel */}
          {isQueued && (
            <Button
              variant="outline" size="sm"
              onClick={() => setConfirmAction("cancel")}
              disabled={actionPending}
              className="text-accent-red border-accent-red/30 hover:bg-accent-red/10"
            >
              <XCircle className="h-3.5 w-3.5" /> {t("dash.instances.cancel")}
            </Button>
          )}
          {/* Failed requeue */}
          {isFailed && (
            <Button variant="outline" size="sm" onClick={handleRequeue} disabled={actionPending}>
              <RotateCcw className="h-3.5 w-3.5" /> {t("dash.instances.requeue")}
            </Button>
          )}
        </div>
      </div>

      {/* Status Timeline */}
      <Card>
        <h2 className="text-sm font-semibold text-text-secondary mb-4">{t("dash.instances.timeline")}</h2>
        <div className="flex items-center gap-1">
          {STATUS_STEPS.map((step, i) => {
            const reached = i <= currentStepIdx;
            const isCurrent = i === currentStepIdx;
            return (
              <div key={step} className="flex items-center flex-1">
                <div className="flex flex-col items-center flex-1">
                  <div
                    className={`h-3 w-3 rounded-full border-2 ${
                      reached
                        ? isCurrent && (status === "failed" || status === "cancelled")
                          ? "border-accent-red bg-accent-red"
                          : "border-emerald bg-emerald"
                        : "border-text-muted/50 bg-text-muted/20"
                    }`}
                  />
                  <span className={`mt-1.5 text-xs capitalize ${reached ? "text-text-primary" : "text-text-muted"}`}>
                    {step}
                  </span>
                </div>
                {i < STATUS_STEPS.length - 1 && (
                  <div className={`h-0.5 flex-1 ${i < currentStepIdx ? "bg-emerald" : "bg-text-muted/30"}`} />
                )}
              </div>
            );
          })}
        </div>
        {(status === "failed" || status === "cancelled" || status === "terminated") && (
          <div className="mt-3 flex items-center gap-2">
            <Badge variant={status === "failed" ? "failed" : status === "terminated" ? "terminated" : "cancelled"}>
              {status}
            </Badge>
            <span className="text-xs text-text-muted">{t("dash.instances.during_exec", { status })}</span>
          </div>
        )}
        {(isStopped || isTransitional) && (
          <div className="mt-3 flex items-center gap-2">
            <StatusBadge status={status} />
            <span className="text-xs text-text-muted">
              {isStopped ? "Instance is stopped. Storage billing continues." : ""}
              {status === "stopping" ? "Gracefully stopping container…" : ""}
              {status === "restarting" ? "Restarting container, data preserved…" : ""}
            </span>
          </div>
        )}
      </Card>

      {/* Job Error Banner */}
      {jobError && (
        <div className="rounded-lg border border-accent-red/30 bg-accent-red/10 p-4 flex items-start gap-3">
          <AlertTriangle className="h-5 w-5 text-accent-red mt-0.5 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-accent-red">Error</p>
            <p className="text-sm text-text-secondary mt-1">{jobError}</p>
          </div>
          <button onClick={() => setJobError(null)} className="text-text-muted hover:text-text-primary">
            <XCircle className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-ice-blue/10">
            <Cpu className="h-5 w-5 text-ice-blue" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.instances.label_gpu")}</p>
            <p className="font-medium">{instance.gpu_type || instance.gpu_model || t("dash.instances.auto")}</p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-emerald/10">
            <Clock className="h-5 w-5 text-emerald" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.instances.label_duration")}</p>
            <p className="font-medium font-mono">
              {instance.duration_sec
                ? `${(instance.duration_sec / 3600).toFixed(1)}h`
                : instance.elapsed_sec
                  ? `${(instance.elapsed_sec / 60).toFixed(0)}m`
                  : "—"}
            </p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-gold/10">
            <DollarSign className="h-5 w-5 text-accent-gold" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.instances.label_cost")}</p>
            <p className="font-medium font-mono">
              {instance.cost_cad != null ? `$${instance.cost_cad.toFixed(2)} CAD` : "—"}
            </p>
          </div>
        </Card>

        <Card className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-accent-red/10">
            <Server className="h-5 w-5 text-accent-red" />
          </div>
          <div>
            <p className="text-xs text-text-muted">{t("dash.instances.label_host")}</p>
            <p className="font-medium">
              {instance.host_id ? (
                <Link href={`/dashboard/hosts/${instance.host_id}`} className="text-ice-blue hover:underline">
                  {instance.host_id.slice(0, 12)}
                </Link>
              ) : (
                t("dash.instances.unassigned")
              )}
            </p>
          </div>
        </Card>
      </div>

      {/* Logs — shown first, above terminal */}
      <Card>
        <div className="flex items-center gap-2 mb-3">
          <Terminal className="h-4 w-4 text-text-muted" />
          <h2 className="text-sm font-semibold text-text-secondary">{t("dash.instances.logs")}</h2>
          {status === "queued" && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-accent-gold">
              <Loader2 className="h-3 w-3 animate-spin" />
              Waiting for assignment…
            </span>
          )}
          {status === "assigned" && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-ice-blue">
              <Loader2 className="h-3 w-3 animate-spin" />
              Host assigned, preparing…
            </span>
          )}
          {status === "starting" && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-ice-blue">
              <Loader2 className="h-3 w-3 animate-spin" />
              Pulling image &amp; starting container…
            </span>
          )}
        </div>
        <LogViewer
          jobId={id}
          live={isRunning || status === "starting" || status === "assigned" || status === "queued"}
          wsLogs={wsLogs}
          wsConnected={wsState.connected}
        />
      </Card>

      {/* Terminal Card — controls in card header, terminal below */}
      {(isRunning || status === "starting" || isStopped) && (
        <Card>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Terminal className="h-4 w-4 text-text-muted" />
              <h2 className="text-sm font-semibold text-text-secondary">Terminal</h2>
              {(isRunning || status === "starting") && (
                <Button size="sm" variant="outline" onClick={() => { setShowTerminal(!showTerminal); if (!terminalMounted) setTerminalMounted(true); }} className="ml-2 h-7 text-xs">
                  {showTerminal ? "Hide" : "Open"}
                </Button>
              )}
              {isRunning && (
                <Button size="sm" variant="outline" onClick={() => setShowConnectModal(true)} className="h-7 text-xs text-ice-blue border-ice-blue/30 hover:bg-ice-blue/10">
                  <Info className="h-3 w-3" /> Connection Info
                </Button>
              )}
            </div>
            <div className="flex items-center gap-2">
              {isRunning && (
                <>
                  <Button size="sm" variant="outline" onClick={() => setConfirmAction("restart")} disabled={actionPending} className="h-7 text-xs">
                    <RefreshCw className="h-3 w-3" /> Restart
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setConfirmAction("stop")} disabled={actionPending} className="h-7 text-xs text-accent-gold border-accent-gold/30 hover:bg-accent-gold/10">
                    <Square className="h-3 w-3" /> Stop
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setConfirmAction("terminate")} disabled={actionPending} className="h-7 text-xs text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                    <Zap className="h-3 w-3" /> Terminate
                  </Button>
                </>
              )}
              {isStopped && (
                <>
                  <Button size="sm" onClick={() => setConfirmAction("start")} disabled={actionPending} className="h-7 text-xs bg-emerald hover:bg-emerald/80 text-white">
                    <Play className="h-3 w-3" /> Start
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setConfirmAction("restart")} disabled={actionPending} className="h-7 text-xs">
                    <RefreshCw className="h-3 w-3" /> Restart
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setConfirmAction("terminate")} disabled={actionPending} className="h-7 text-xs text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                    <Zap className="h-3 w-3" /> Terminate
                  </Button>
                </>
              )}
            </div>
          </div>
          {terminalMounted && (isRunning || status === "starting") && (
            <div className="h-[500px] rounded-lg overflow-hidden border border-border" style={{ display: showTerminal ? undefined : "none" }}>
              <WebTerminal instanceId={id} onClose={() => setShowTerminal(false)} />
            </div>
          )}
        </Card>
      )}

      {/* Details — at bottom */}
      <Card>
        <h2 className="text-sm font-semibold text-text-secondary mb-3">{t("dash.instances.details")}</h2>
        <dl className="grid gap-y-2 gap-x-6 text-sm sm:grid-cols-2">
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.instances.docker_image")}</dt>
            <dd className="font-mono">{instance.docker_image || "—"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.instances.pricing_tier")}</dt>
            <dd className="capitalize">{instance.tier || "on-demand"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.instances.submitted")}</dt>
            <dd>{(instance.submitted_at || instance.created_at) ? new Date((instance.submitted_at || instance.created_at!) * 1000).toLocaleString() : "—"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.instances.job_id")}</dt>
            <dd className="font-mono text-text-muted">{instance.job_id}</dd>
          </div>
          {instance.host_gpu && (
            <div className="flex justify-between sm:block">
              <dt className="text-text-muted">GPU</dt>
              <dd className="font-medium">{instance.host_gpu}{instance.host_vram_gb ? ` · ${instance.host_vram_gb} GB VRAM` : ""}</dd>
            </div>
          )}
          {instance.started_at && (
            <div className="flex justify-between sm:block">
              <dt className="text-text-muted">Started</dt>
              <dd>{new Date(Number(instance.started_at) * 1000).toLocaleString()}</dd>
            </div>
          )}
          {instance.started_at && isRunning && (
            <div className="flex justify-between sm:block">
              <dt className="text-text-muted">Uptime</dt>
              <dd key={uptickKey}>{formatUptime(Date.now() / 1000 - Number(instance.started_at))}</dd>
            </div>
          )}
          {instance.completed_at && (
            <div className="flex justify-between sm:block">
              <dt className="text-text-muted">Completed</dt>
              <dd>{new Date(Number(instance.completed_at) * 1000).toLocaleString()}</dd>
            </div>
          )}
        </dl>
      </Card>

      {/* Connect Modal */}
      {showConnectModal && isRunning && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={() => setShowConnectModal(false)}>
          <div className="w-full max-w-lg mx-4 rounded-xl border border-border bg-surface p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Globe className="h-5 w-5 text-ice-blue" />
                <h2 className="text-lg font-semibold">Connection Details</h2>
              </div>
              <button onClick={() => setShowConnectModal(false)} className="text-text-muted hover:text-text-primary">
                <XCircle className="h-5 w-5" />
              </button>
            </div>

            {instance.host_ip ? (
              <div className="space-y-4">
                <div className="rounded-lg p-3 border bg-ice-blue/5 border-ice-blue/30">
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <p className="text-xs font-medium text-text-secondary">SSH Connect</p>
                    <div className="group relative">
                      <Info className="h-3 w-3 text-text-muted cursor-help" />
                      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10 w-56 rounded-md bg-surface-overlay border border-border p-2 text-xs text-text-muted shadow-lg">
                        Connect via Xcelsior&apos;s SSH proxy. Your SSH keys from Settings are automatically injected.
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-sm font-mono text-ice-blue bg-background rounded px-2.5 py-2 select-all border border-border">
                      ssh root@connect.xcelsior.ca -p {instance.ssh_port || 22}
                    </code>
                    <button
                      onClick={() => { navigator.clipboard.writeText(`ssh root@connect.xcelsior.ca -p ${instance.ssh_port || 22}`); toast.success("Copied"); }}
                      className="text-text-muted hover:text-text-primary transition-colors shrink-0"
                      title="Copy"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
                <DirectAccessSection instance={instance} />
                <dl className="grid gap-y-2 gap-x-6 text-sm sm:grid-cols-2 pt-2 border-t border-border">
                  {instance.host_gpu && (
                    <div>
                      <dt className="text-text-muted text-xs">GPU</dt>
                      <dd className="font-medium">{instance.host_gpu}{instance.host_vram_gb ? ` · ${instance.host_vram_gb} GB` : ""}</dd>
                    </div>
                  )}
                  {instance.host_id && (
                    <div>
                      <dt className="text-text-muted text-xs">Host</dt>
                      <dd className="font-mono text-xs">{instance.host_id}</dd>
                    </div>
                  )}
                  {instance.ssh_port && (
                    <div>
                      <dt className="text-text-muted text-xs">SSH Port</dt>
                      <dd className="font-mono">{instance.ssh_port}</dd>
                    </div>
                  )}
                  {instance.container_name && (
                    <div>
                      <dt className="text-text-muted text-xs">Container</dt>
                      <dd className="font-mono text-xs">{instance.container_name}</dd>
                    </div>
                  )}
                </dl>
              </div>
            ) : (
              <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-4">
                <div className="flex items-start gap-2">
                  <Clock className="h-4 w-4 text-accent-gold shrink-0 mt-0.5" />
                  <div>
                    <p className="text-sm font-medium text-accent-gold">Waiting for host assignment</p>
                    <p className="text-xs text-text-muted mt-1">Connection details will appear once the scheduler assigns a GPU host.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Confirm dialog */}
      {confirmAction && (
        <ConfirmDialog
          open
          title={CONFIRM_CONFIGS[confirmAction].title}
          description={CONFIRM_CONFIGS[confirmAction].description}
          confirmLabel={CONFIRM_CONFIGS[confirmAction].confirmLabel}
          cancelLabel={t("common.cancel")}
          variant={CONFIRM_CONFIGS[confirmAction].variant}
          onConfirm={() => executeAction(confirmAction)}
          onCancel={() => setConfirmAction(null)}
        />
      )}
    </div>
  );
}

