"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  ChevronDown,
  CircleAlert,
  KeyRound,
  ListChecks,
  Loader2,
  PlugZap,
  Plus,
  Server,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import * as api from "@/lib/api";
import type { HpcCluster, HpcConnectionCheck } from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  CLUSTER_STATUS_LABEL,
  clusterStatus,
  formatAge,
  formatDuration,
  formatTimestamp,
  rateLabel,
  signalLate,
  type ClusterStatus,
} from "./hpc-format";
import { RegisterClusterDialog } from "./register-cluster-dialog";

const STATUS_BADGE: Record<ClusterStatus, "active" | "warning" | "failed" | "outline" | "info"> = {
  healthy: "active",
  degraded: "warning",
  backing_off: "failed",
  never_polled: "info",
  disabled: "outline",
};

/** The card's left edge, so a sick cluster is found before any text is read. */
const STATUS_EDGE: Record<ClusterStatus, string> = {
  healthy: "before:bg-emerald",
  degraded: "before:bg-accent-gold",
  backing_off: "before:bg-accent-red",
  never_polled: "before:bg-ice-blue",
  disabled: "before:bg-border",
};

const STATUS_DOT: Record<ClusterStatus, string> = {
  healthy: "bg-emerald",
  degraded: "bg-accent-gold",
  backing_off: "bg-accent-red",
  never_polled: "bg-ice-blue",
  disabled: "bg-text-muted",
};

interface Props {
  clusters: HpcCluster[];
  loading: boolean;
  error: string | null;
  now: number;
  /** The cluster an attention item pointed at: scrolled to and outlined. */
  focusClusterId?: string | null;
  onViewJobs: (clusterId: string) => void;
  onChanged: () => void;
}

export function FleetPanel({ clusters, loading, error, now, focusClusterId, onViewJobs, onChanged }: Props) {
  const [registering, setRegistering] = useState(false);

  if (loading) {
    return (
      <div className="grid grid-cols-1 gap-4 xl:grid-cols-2" role="status" aria-label="Loading clusters">
        {[0, 1].map((i) => (
          <div key={i} className="h-72 animate-pulse rounded-xl border border-border bg-surface" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {error && (
        <Card className="border-accent-red/30 p-4">
          <p className="flex items-center gap-2 text-sm text-accent-red">
            <CircleAlert className="h-4 w-4 shrink-0" /> {error}
          </p>
        </Card>
      )}

      {clusters.length === 0 && !error ? (
        <Card className="flex flex-col items-center px-6 py-14 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent-cyan/15 text-accent-cyan">
            <Server className="h-6 w-6" />
          </div>
          <p className="text-base font-semibold">No clusters registered</p>
          <p className="mx-auto mt-1 max-w-md text-sm text-text-muted">
            Register a Slurm cluster from one of the adapter profiles. It starts disabled, so nothing is polled
            until a connection check has proven the transport.
          </p>
          <Button className="mt-5" onClick={() => setRegistering(true)}>
            <Plus className="h-4 w-4" /> Register cluster
          </Button>
        </Card>
      ) : (
        <>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm text-text-muted">
              {clusters.length} registered · {clusters.filter((c) => c.enabled).length} polled by the reconciler
            </p>
            <Button variant="outline" size="sm" onClick={() => setRegistering(true)}>
              <Plus className="h-3.5 w-3.5" /> Register cluster
            </Button>
          </div>
          {/* One cluster takes the full width; two columns only when there is a second. */}
          <div className={cn("grid grid-cols-1 gap-4", clusters.length > 1 && "xl:grid-cols-2")}>
            {clusters.map((cluster) => (
              <ClusterCard
                key={cluster.cluster_id}
                cluster={cluster}
                now={now}
                focused={focusClusterId === cluster.cluster_id}
                onViewJobs={onViewJobs}
                onChanged={onChanged}
              />
            ))}
          </div>
        </>
      )}

      <RegisterClusterDialog
        open={registering}
        onClose={() => setRegistering(false)}
        onRegistered={() => {
          setRegistering(false);
          onChanged();
        }}
      />
    </div>
  );
}

/** One of the three things a healthy pass proves, in the order a pass reaches them. */
function Signal({
  label,
  at,
  late,
  now,
  last,
}: {
  label: string;
  at: string | null;
  late: boolean;
  now: number;
  last?: boolean;
}) {
  return (
    <li className="relative flex min-w-0 flex-1 flex-col items-center text-center">
      {!last && (
        <span
          aria-hidden
          className={cn(
            "absolute left-1/2 top-[13px] h-px w-full",
            late ? "bg-border" : "bg-gradient-to-r from-emerald/60 to-emerald/20",
          )}
        />
      )}
      <span
        className={cn(
          "relative z-10 flex h-7 w-7 items-center justify-center rounded-full border",
          late
            ? "border-accent-red/40 bg-accent-red/15 text-accent-red"
            : "border-emerald/40 bg-emerald/15 text-emerald",
        )}
      >
        {late ? <X className="h-3.5 w-3.5" /> : <Check className="h-3.5 w-3.5" />}
      </span>
      <span className="mt-1.5 text-xs font-medium text-text-primary">{label}</span>
      <span className="sr-only">{late ? "late or never answered" : "answering"}</span>
      <span className="text-[11px] tabular-nums text-text-muted" title={formatTimestamp(at)}>
        {formatAge(at, now)}
      </span>
    </li>
  );
}

/** Running and pending as one bar, so the queue's shape reads at a glance. */
function JobMix({ cluster }: { cluster: HpcCluster }) {
  const jobs = cluster.open_jobs;
  if (!jobs) return null;
  const other = Math.max(0, jobs.total - jobs.running - jobs.pending);
  const parts = [
    { key: "running", label: "running", count: jobs.running, bar: "bg-emerald", dot: "bg-emerald" },
    { key: "pending", label: "pending", count: jobs.pending, bar: "bg-ice-blue", dot: "bg-ice-blue" },
    { key: "other", label: "held or other", count: other, bar: "bg-accent-gold", dot: "bg-accent-gold" },
  ].filter((p) => p.count > 0);
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <span className="text-xs font-medium text-text-secondary">Open jobs</span>
        <span className="font-mono text-lg font-semibold leading-none tabular-nums">{jobs.total}</span>
      </div>
      <div
        className="flex h-2 overflow-hidden rounded-full bg-surface-hover"
        role="img"
        aria-label={
          jobs.total === 0
            ? "No open jobs"
            : parts.map((p) => `${p.count} ${p.label}`).join(", ")
        }
      >
        {parts.map((p) => (
          <span key={p.key} className={cn("h-full", p.bar)} style={{ width: `${(p.count / jobs.total) * 100}%` }} />
        ))}
      </div>
      <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-text-muted">
        {jobs.total === 0 ? (
          <span>Nothing queued or running</span>
        ) : (
          parts.map((p) => (
            <span key={p.key} className="flex items-center gap-1.5">
              <span aria-hidden className={cn("h-1.5 w-1.5 rounded-full", p.dot)} />
              {p.count} {p.label}
            </span>
          ))
        )}
        {jobs.stale > 0 && <span className="text-accent-gold">{jobs.stale} stale</span>}
        {jobs.awaiting_accounting > 0 && (
          <span className="text-accent-gold">{jobs.awaiting_accounting} awaiting accounting</span>
        )}
      </div>
    </div>
  );
}

function Fact({ label, children, className }: { label: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("min-w-0", className)}>
      <dt className="text-[11px] uppercase tracking-wide text-text-muted">{label}</dt>
      <dd className="mt-0.5 break-words text-sm text-text-primary">{children}</dd>
    </div>
  );
}

function ClusterCard({
  cluster,
  now,
  focused,
  onViewJobs,
  onChanged,
}: {
  cluster: HpcCluster;
  now: number;
  focused: boolean;
  onViewJobs: (clusterId: string) => void;
  onChanged: () => void;
}) {
  const [confirming, setConfirming] = useState(false);
  const [checking, setChecking] = useState(false);
  const [check, setCheck] = useState<HpcConnectionCheck | null>(null);
  const [open, setOpen] = useState(false);
  const card = useRef<HTMLDivElement>(null);
  const status = clusterStatus(cluster, now);
  const { health, transport } = cluster;
  const name = cluster.display_name || cluster.cluster_id;
  const facts = Object.entries(cluster.config_facts ?? {});
  const failures = Object.entries(cluster.failure_counts ?? {}).sort((a, b) => b[1] - a[1]);
  const failureTotal = failures.reduce((n, [, count]) => n + count, 0);
  const rate = rateLabel(cluster);
  const detailsId = `cluster-details-${cluster.cluster_id}`;

  useEffect(() => {
    if (!focused) return;
    setOpen(true);
    card.current?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, [focused]);

  const runCheck = async () => {
    setConfirming(false);
    setChecking(true);
    try {
      const res = await api.checkHpcClusterConnection(cluster.cluster_id);
      setCheck(res.check);
      if (res.check.ok) toast.success(`${name}: connection check passed`);
      else toast.error(`${name}: connection check failed (${res.check.error_code || "error"})`);
      onChanged();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Connection check failed");
    } finally {
      setChecking(false);
    }
  };

  return (
    <div ref={card} className="scroll-mt-24">
    <Card
      data-testid={`cluster-${cluster.cluster_id}`}
      className={cn(
        "relative space-y-5 overflow-hidden p-5 pl-6",
        "before:absolute before:inset-y-0 before:left-0 before:w-1",
        STATUS_EDGE[status],
        focused && "ring-2 ring-accent-cyan/60",
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-surface-hover text-ice-blue">
            <Server className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <h3 className="truncate text-base font-semibold leading-tight">{name}</h3>
            <p className="mt-0.5 truncate font-mono text-xs text-text-muted">
              {cluster.cluster_id} · profile {cluster.profile}
            </p>
          </div>
        </div>
        <Badge variant={STATUS_BADGE[status]} className="gap-1.5 text-xs">
          <span
            aria-hidden
            className={cn("h-1.5 w-1.5 rounded-full", STATUS_DOT[status], status === "healthy" && "animate-pulse")}
          />
          {CLUSTER_STATUS_LABEL[status]}
        </Badge>
      </div>

      {status === "backing_off" && (
        <p className="rounded-lg border border-accent-red/30 bg-accent-red/10 px-3 py-2 text-xs text-text-primary">
          Not polled until <span className="font-medium">{formatTimestamp(health.backing_off_until)}</span> after{" "}
          {health.consecutive_failures} failed passes
          {health.last_error_code && (
            <>
              {" "}
              (<span className="font-mono text-accent-red">{health.last_error_code}</span>)
            </>
          )}
          .
        </p>
      )}

      {cluster.enabled ? (
        <ol className="flex" aria-label="What the last pass proved">
          <Signal label="Transport" at={health.transport_ok_at} late={signalLate(health.transport_ok_at, cluster, now)} now={now} />
          <Signal label="Controller" at={health.controller_ok_at} late={signalLate(health.controller_ok_at, cluster, now)} now={now} />
          <Signal label="Accounting" at={health.accounting_ok_at} late={signalLate(health.accounting_ok_at, cluster, now)} now={now} last />
        </ol>
      ) : (
        <p className="rounded-lg border border-dashed border-border px-3 py-3 text-xs text-text-muted">
          Disabled: the reconciler does not poll it. Run a connection check, then register it again with polling
          enabled.
        </p>
      )}

      <JobMix cluster={cluster} />

      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
        <Fact label="Polled">
          every {formatDuration(cluster.poll_interval_seconds)}
          <span className="block text-xs text-text-muted" title={formatTimestamp(health.last_poll_at)}>
            last {formatAge(health.last_poll_at, now)}
          </span>
        </Fact>
        <Fact label="Rate">
          {rate ?? <span className="text-text-muted">Not priced</span>}
        </Fact>
        <Fact label="Failures so far" className="col-span-2 sm:col-span-1">
          <span className={cn("tabular-nums", failureTotal > 0 && "text-accent-gold")}>{failureTotal}</span>
          {health.last_error_code && status !== "backing_off" && (
            <span className="block truncate font-mono text-xs text-accent-red" title={health.last_error_code}>
              {health.last_error_code}
            </span>
          )}
        </Fact>
      </dl>

      <div className="flex flex-wrap gap-1.5">
        {cluster.research_only && <Badge variant="info" className="text-[11px]">Research only</Badge>}
        <Badge variant={cluster.marketplace ? "active" : "outline"} className="text-[11px]">
          Marketplace {cluster.marketplace ? "on" : "off"}
        </Badge>
        <Badge variant={cluster.agent_access ? "active" : "outline"} className="text-[11px]">
          Agent access {cluster.agent_access ? "on" : "off"}
        </Badge>
      </div>

      {cluster.orphan_slurm_job_ids.length > 0 && (
        <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/10 px-3 py-2 text-xs">
          <span className="font-semibold text-accent-gold">Unmapped Slurm jobs: </span>
          <span className="break-all font-mono text-text-primary">{cluster.orphan_slurm_job_ids.join(", ")}</span>
        </div>
      )}

      {check && (
        <div
          role="status"
          className={cn(
            "rounded-lg border px-3 py-2 text-xs",
            check.ok ? "border-emerald/30 bg-emerald/10" : "border-accent-red/30 bg-accent-red/10",
          )}
        >
          <p className={cn("font-semibold", check.ok ? "text-emerald" : "text-accent-red")}>
            {check.ok ? "Connection check passed" : `Connection check failed: ${check.error_code || "error"}`}
          </p>
          {check.error && <p className="mt-1 break-words text-text-primary">{check.error}</p>}
          {check.missing && check.missing.length > 0 && (
            <p className="mt-1 break-words text-text-primary">Missing config: {check.missing.join(", ")}</p>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-4">
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={() => onViewJobs(cluster.cluster_id)}>
            <ListChecks className="h-3.5 w-3.5" /> View jobs <ArrowRight className="h-3 w-3" />
          </Button>
          <Button variant="outline" size="sm" onClick={() => setConfirming(true)} disabled={checking}>
            {checking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}
            {checking ? "Checking…" : "Check connection"}
          </Button>
        </div>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-expanded={open}
          aria-controls={detailsId}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-text-secondary hover:bg-surface-hover hover:text-text-primary"
        >
          Details
          <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")} />
        </button>
      </div>

      {open && (
        <div id={detailsId} className="space-y-4 rounded-lg border border-border bg-background/40 p-4">
          <div>
            <h4 className="mb-2 flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-text-secondary">
              <KeyRound className="h-3.5 w-3.5" /> How it is reached
            </h4>
            {transport.kind === "registry" ? (
              <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <Fact label="Host" className="sm:col-span-2">
                  <span className="break-all font-mono text-xs">
                    {transport.remote_user ? `${transport.remote_user}@` : ""}
                    {transport.ssh_host}:{transport.ssh_port}
                  </span>
                </Fact>
                <Fact label="Credential reference" className="sm:col-span-2">
                  <span className="break-all font-mono text-xs">{transport.credential_ref || "—"}</span>
                </Fact>
                <Fact label="Pinned host key" className="sm:col-span-2">
                  <span className="break-all font-mono text-xs">{transport.host_key_fingerprint || "not pinned"}</span>
                </Fact>
              </dl>
            ) : (
              <p className="text-xs text-text-muted">
                Through the adapter profile&apos;s own settings. Nothing is registered for this cluster.
              </p>
            )}
          </div>

          <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <Fact label="Suspended time">
              {cluster.suspend_holds_gres === false
                ? "Free: a suspended job releases its GPUs"
                : "Billed: a suspended job keeps its GPUs"}
            </Fact>
            <Fact label="Backing off until">{formatTimestamp(health.backing_off_until)}</Fact>
          </dl>

          {failures.length > 0 && (
            <div>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-secondary">
                Failures by stage and code
              </h4>
              <ul className="space-y-1">
                {failures.map(([key, count]) => {
                  const [stage, code] = key.split(":");
                  return (
                    <li key={key} className="flex items-center justify-between gap-3 text-xs">
                      <span className="min-w-0 truncate">
                        <span className="text-text-muted">{stage}</span>{" "}
                        <span className="font-mono text-text-primary">{code || "error"}</span>
                      </span>
                      <span className="font-mono tabular-nums text-text-secondary">{count}</span>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          <div>
            <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-text-secondary">
              Slurm config{" "}
              <span className="font-normal normal-case tracking-normal text-text-muted">
                {facts.length === 0
                  ? "· not captured yet. Run a connection check."
                  : `· ${facts.length} values captured ${formatAge(cluster.config_captured_at, now)}`}
              </span>
            </h4>
            {facts.length > 0 && (
              <dl className="max-h-56 space-y-1 overflow-auto">
                {facts.map(([key, value]) => (
                  <div key={key} className="flex gap-3 text-xs">
                    <dt className="w-44 shrink-0 truncate font-mono text-text-muted" title={key}>{key}</dt>
                    <dd className="min-w-0 break-all font-mono text-text-primary">
                      {typeof value === "string" ? value : JSON.stringify(value)}
                    </dd>
                  </div>
                ))}
              </dl>
            )}
          </div>
        </div>
      )}

      <ConfirmDialog
        open={confirming}
        title={`Check connection to ${name}?`}
        description="This connects to the cluster the way the reconciler does, runs scontrol ping and scontrol show config, and stores the config. The check is written to the audit log."
        confirmLabel="Run check"
        onConfirm={() => void runCheck()}
        onCancel={() => setConfirming(false)}
      />
    </Card>
    </div>
  );
}
