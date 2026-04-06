"use client";

import { useEffect, useState, useCallback } from "react";
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
} from "lucide-react";
import { fetchInstance, cancelInstance, requeueInstance } from "@/lib/api";
import type { Instance } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { useInstanceWebSocket } from "@/hooks/useInstanceWebSocket";

const WebTerminal = dynamic(
  () => import("@/components/terminal/WebTerminal").then((m) => m.WebTerminal),
  { ssr: false },
);

const STATUS_STEPS = ["queued", "assigned", "running", "completed"] as const;

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
        <span>Direct access (advanced)</span>
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

export default function InstanceDetailPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { t } = useLocale();
  const [instance, setInstance] = useState<Instance | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [showTerminal, setShowTerminal] = useState(false);
  const [jobError, setJobError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    fetchInstance(id)
      .then((r) => setInstance(r.instance))
      .catch(() => toast.error("Failed to load instance"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { setJobError(null); load(); }, [id]);

  // Live WebSocket updates for active instances
  const isLive = instance?.status === "queued" || instance?.status === "assigned" || instance?.status === "running";
  const onWsInstance = useCallback((i: Instance) => setInstance(i), []);
  const onWsJobError = useCallback((err: { job_id: string; error: string; message: string }) => {
    setJobError(err.message);
  }, []);
  const wsState = useInstanceWebSocket(id, {
    onInstance: onWsInstance,
    onJobError: onWsJobError,
    enabled: isLive,
  });

  async function handleCancel() {
    setConfirmCancel(false);
    try {
      await cancelInstance(id);
      toast.success("Instance cancelled");
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Cancel failed");
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

  const isActive = instance.status === "queued" || instance.status === "running" || instance.status === "assigned";
  const isFailed = instance.status === "failed";
  // Status timeline step
  const currentStepIdx = STATUS_STEPS.indexOf(
    (instance.status === "failed" || instance.status === "cancelled") ? "running" : (instance.status as typeof STATUS_STEPS[number]),
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
        <div className="flex items-center gap-2">
          <StatusBadge status={instance.status} />
          {isLive && (
            <span
              className={`flex items-center gap-1 text-xs ${wsState.connected ? "text-emerald" : "text-text-muted"}`}
              title={wsState.connected ? "Live" : wsState.reconnecting ? "Reconnecting…" : "Disconnected"}
            >
              {wsState.connected ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
              {wsState.connected ? "Live" : wsState.reconnecting ? "Reconnecting…" : ""}
            </span>
          )}
          {isActive && instance.status === "running" && (
            <Button size="sm" variant="outline" onClick={() => setShowTerminal(!showTerminal)}>
              <Terminal className="h-3.5 w-3.5" /> {showTerminal ? "Hide Terminal" : "Terminal"}
            </Button>
          )}
          {isActive && instance.status === "running" && (
            <Button size="sm" onClick={() => setConfirmCancel(true)} className="bg-accent-red hover:bg-accent-red/80 text-white">
              <Square className="h-3.5 w-3.5" /> Stop
            </Button>
          )}
          {isActive && instance.status !== "running" && (
            <Button variant="outline" size="sm" onClick={() => setConfirmCancel(true)} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
              <XCircle className="h-3.5 w-3.5" /> {t("dash.instances.cancel")}
            </Button>
          )}
          {isFailed && (
            <Button variant="outline" size="sm" onClick={handleRequeue}>
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
                        ? isCurrent && (instance.status === "failed" || instance.status === "cancelled")
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
        {(instance.status === "failed" || instance.status === "cancelled") && (
          <div className="mt-3 flex items-center gap-2">
            <Badge variant={instance.status === "failed" ? "failed" : "cancelled"}>
              {instance.status}
            </Badge>
            <span className="text-xs text-text-muted">{t("dash.instances.during_exec", { status: instance.status })}</span>
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

      {/* Details Grid */}
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

      {/* Metadata */}
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
            <dd>{(instance.submitted_at || instance.created_at) ? new Date(instance.submitted_at || instance.created_at!).toLocaleString() : "—"}</dd>
          </div>
          <div className="flex justify-between sm:block">
            <dt className="text-text-muted">{t("dash.instances.job_id")}</dt>
            <dd className="font-mono text-text-muted">{instance.job_id}</dd>
          </div>
        </dl>
      </Card>

      {/* Connection Info — shown when job is running or has run */}
      {instance.host_id && (instance.status === "running" || instance.status === "completed" || instance.status === "failed") && (
        <Card>
          <div className="flex items-center gap-2 mb-3">
            <Globe className="h-4 w-4 text-ice-blue" />
            <h2 className="text-sm font-semibold text-text-secondary">Connection Details</h2>
            {instance.status === "running" && (
              <span className="ml-auto flex items-center gap-1 text-xs text-emerald">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald animate-pulse" /> Live
              </span>
            )}
          </div>

          {/* Quick-connect commands */}
          {instance.status === "running" && instance.host_ip && (
            <div className="mb-4 space-y-3">
              {/* Primary: branded SSH */}
              <div className="rounded-lg p-3 border bg-ice-blue/5 border-ice-blue/30">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <p className="text-xs font-medium text-text-secondary">SSH Connect</p>
                  <div className="group relative">
                    <Info className="h-3 w-3 text-text-muted cursor-help" />
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block z-10 w-56 rounded-md bg-surface-overlay border border-border p-2 text-xs text-text-muted shadow-lg">
                      Connect via Xcelsior&apos;s SSH proxy. Your SSH keys from Settings are automatically injected into the instance.
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <code className="flex-1 text-xs font-mono text-ice-blue bg-background rounded px-2 py-1.5 select-all border border-border">
                    ssh root@connect.xcelsior.ca -p {instance.ssh_port || 22}
                  </code>
                  <button
                    onClick={() => { navigator.clipboard.writeText(`ssh root@connect.xcelsior.ca -p ${instance.ssh_port || 22}`); toast.success("Copied"); }}
                    className="text-text-muted hover:text-text-primary transition-colors shrink-0"
                    title="Copy SSH command"
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </button>
                </div>
              </div>

              {/* Direct access (expandable) */}
              <DirectAccessSection instance={instance} />
            </div>
          )}

          {/* Instance details grid */}
          <dl className="grid gap-y-2 gap-x-6 text-sm sm:grid-cols-2">
            {instance.host_gpu && (
              <div className="flex justify-between sm:block">
                <dt className="text-text-muted">GPU</dt>
                <dd className="font-medium">{instance.host_gpu}{instance.host_vram_gb ? ` · ${instance.host_vram_gb} GB VRAM` : ""}</dd>
              </div>
            )}
            <div className="flex justify-between sm:block">
              <dt className="text-text-muted">Host</dt>
              <dd className="font-mono text-xs">{instance.host_id}</dd>
            </div>
            {instance.docker_image && (
              <div className="flex justify-between sm:block">
                <dt className="text-text-muted">Image</dt>
                <dd className="font-mono text-xs">{instance.docker_image}</dd>
              </div>
            )}
            {instance.ssh_port && (
              <div className="flex justify-between sm:block">
                <dt className="text-text-muted">SSH Port</dt>
                <dd className="font-mono">{instance.ssh_port}</dd>
              </div>
            )}
            {instance.container_name && (
              <div className="flex justify-between sm:block">
                <dt className="text-text-muted">Container</dt>
                <dd className="font-mono text-xs">{instance.container_name}</dd>
              </div>
            )}
            {instance.started_at && (
              <div className="flex justify-between sm:block">
                <dt className="text-text-muted">Started</dt>
                <dd>{new Date(Number(instance.started_at) * 1000).toLocaleString()}</dd>
              </div>
            )}
            {instance.started_at && instance.status === "running" && (
              <div className="flex justify-between sm:block">
                <dt className="text-text-muted">Uptime</dt>
                <dd>{formatUptime(Date.now() / 1000 - Number(instance.started_at))}</dd>
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
      )}

      {/* Web Terminal */}
      {showTerminal && instance.status === "running" && (
        <WebTerminal instanceId={id} onClose={() => setShowTerminal(false)} />
      )}

      {/* Logs */}
      <Card>
        <div className="flex items-center gap-2 mb-3">
          <Terminal className="h-4 w-4 text-text-muted" />
          <h2 className="text-sm font-semibold text-text-secondary">{t("dash.instances.logs")}</h2>
          {instance.status === "queued" && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-accent-gold">
              <Loader2 className="h-3 w-3 animate-spin" />
              Waiting for assignment…
            </span>
          )}
          {instance.status === "assigned" && (
            <span className="ml-auto flex items-center gap-1.5 text-xs text-ice-blue">
              <Loader2 className="h-3 w-3 animate-spin" />
              Setting up instance…
            </span>
          )}
        </div>
        <LogViewer
          jobId={id}
          live={instance.status === "running" || instance.status === "assigned" || instance.status === "queued"}
        />
      </Card>

      <ConfirmDialog
        open={confirmCancel}
        title={t("dash.instances.cancel_title")}
        description={t("dash.instances.cancel_desc")}
        confirmLabel={t("dash.instances.cancel")}
        cancelLabel={t("common.cancel")}
        variant="danger"
        onConfirm={handleCancel}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
}
