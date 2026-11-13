"use client";

import { useState } from "react";
import { CheckCircle2, ChevronDown, CircleAlert, Loader2, PlugZap, Server, XCircle } from "lucide-react";
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
  signalLate,
  type ClusterStatus,
} from "./hpc-format";

const STATUS_BADGE: Record<ClusterStatus, "active" | "warning" | "failed" | "outline" | "info"> = {
  healthy: "active",
  degraded: "warning",
  backing_off: "failed",
  never_polled: "info",
  disabled: "outline",
};

interface Props {
  clusters: HpcCluster[];
  loading: boolean;
  error: string | null;
  now: number;
  onChanged: () => void;
}

export function ClustersPanel({ clusters, loading, error, now, onChanged }: Props) {
  if (loading) {
    return (
      <div className="flex justify-center py-12" role="status" aria-label="Loading clusters">
        <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
      </div>
    );
  }
  if (error) {
    return (
      <Card className="border-accent-red/30">
        <p className="flex items-center gap-2 text-sm text-accent-red">
          <CircleAlert className="h-4 w-4 shrink-0" /> {error}
        </p>
      </Card>
    );
  }
  if (clusters.length === 0) {
    return (
      <Card className="py-10 text-center">
        <Server className="mx-auto mb-2 h-8 w-8 text-text-muted" />
        <p className="text-sm font-medium">No clusters registered</p>
        <p className="mx-auto mt-1 max-w-md text-sm text-text-muted">
          Register one with <code className="font-mono text-xs">POST /api/v1/hpc/clusters</code>. It starts
          disabled until a connection check has proven the transport.
        </p>
      </Card>
    );
  }
  return (
    <div className={cn("grid grid-cols-1 gap-4", clusters.length > 1 && "xl:grid-cols-2")}>
      {clusters.map((cluster) => (
        <ClusterCard key={cluster.cluster_id} cluster={cluster} now={now} onChanged={onChanged} />
      ))}
    </div>
  );
}

function Signal({ label, at, late, now }: { label: string; at: string | null; late: boolean; now: number }) {
  const Icon = late ? XCircle : CheckCircle2;
  return (
    <div className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 rounded-lg border border-border bg-background/40 px-3 py-2">
      <span className="flex items-center gap-1.5 text-xs font-medium text-text-secondary">
        <Icon className={cn("h-3.5 w-3.5 shrink-0", late ? "text-accent-red" : "text-emerald")} />
        {label}
      </span>
      <span className="whitespace-nowrap text-xs tabular-nums text-text-muted" title={formatTimestamp(at)}>
        {formatAge(at, now)}
      </span>
    </div>
  );
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] uppercase tracking-wide text-text-muted">{label}</dt>
      <dd className="truncate text-sm text-text-primary">{children}</dd>
    </div>
  );
}

function ClusterCard({ cluster, now, onChanged }: { cluster: HpcCluster; now: number; onChanged: () => void }) {
  const [confirming, setConfirming] = useState(false);
  const [checking, setChecking] = useState(false);
  const [check, setCheck] = useState<HpcConnectionCheck | null>(null);
  const [showFacts, setShowFacts] = useState(false);
  const status = clusterStatus(cluster, now);
  const { health, transport } = cluster;
  const name = cluster.display_name || cluster.cluster_id;
  const facts = Object.entries(cluster.config_facts ?? {});

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
    <Card className={cn("space-y-4 p-5", !cluster.enabled && "opacity-80")} data-testid={`cluster-${cluster.cluster_id}`}>
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 shrink-0 text-ice-blue" />
            <h3 className="truncate text-base font-semibold">{name}</h3>
          </div>
          <p className="mt-0.5 truncate font-mono text-xs text-text-muted">
            {cluster.cluster_id} · profile {cluster.profile}
          </p>
        </div>
        <Badge variant={STATUS_BADGE[status]} className="text-xs">{CLUSTER_STATUS_LABEL[status]}</Badge>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {cluster.research_only && <Badge variant="info" className="text-[11px]">Research only</Badge>}
        <Badge variant={cluster.marketplace ? "active" : "outline"} className="text-[11px]">
          Marketplace {cluster.marketplace ? "on" : "off"}
        </Badge>
        <Badge variant={cluster.agent_access ? "active" : "outline"} className="text-[11px]">
          Agent access {cluster.agent_access ? "on" : "off"}
        </Badge>
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
        <Signal label="Transport" at={health.transport_ok_at} late={signalLate(health.transport_ok_at, cluster, now)} now={now} />
        <Signal label="Controller" at={health.controller_ok_at} late={signalLate(health.controller_ok_at, cluster, now)} now={now} />
        <Signal label="Accounting" at={health.accounting_ok_at} late={signalLate(health.accounting_ok_at, cluster, now)} now={now} />
      </div>

      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
        <Fact label="Last poll">
          <span title={formatTimestamp(health.last_poll_at)}>{formatAge(health.last_poll_at, now)}</span>
        </Fact>
        <Fact label="Poll interval">{formatDuration(cluster.poll_interval_seconds)}</Fact>
        <Fact label="Failed passes">
          <span className={cn(health.consecutive_failures > 0 && "text-accent-red")}>{health.consecutive_failures}</span>
        </Fact>
        <Fact label="Last error">
          {health.last_error_code ? <span className="font-mono text-xs text-accent-red">{health.last_error_code}</span> : "none"}
        </Fact>
        <Fact label="Backing off until">{formatTimestamp(health.backing_off_until)}</Fact>
        <Fact label="Reached by">
          {transport.kind === "registry" ? (
            <span className="font-mono text-xs">
              {transport.remote_user ? `${transport.remote_user}@` : ""}{transport.ssh_host}:{transport.ssh_port}
            </span>
          ) : "profile settings"}
        </Fact>
      </dl>

      {transport.kind === "registry" && (
        <dl className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <Fact label="Pinned host key">
            <span className="font-mono text-xs" title={transport.host_key_fingerprint || undefined}>
              {transport.host_key_fingerprint || "not pinned"}
            </span>
          </Fact>
          <Fact label="Credential reference">
            <span className="font-mono text-xs">{transport.credential_ref || "—"}</span>
          </Fact>
        </dl>
      )}

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

      <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border pt-3">
        <button
          type="button"
          onClick={() => setShowFacts((v) => !v)}
          disabled={facts.length === 0}
          aria-expanded={showFacts}
          className="flex items-center gap-1 text-xs font-medium text-text-secondary hover:text-text-primary disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ChevronDown className={cn("h-3.5 w-3.5 transition-transform", showFacts && "rotate-180")} />
          {facts.length === 0
            ? "No config captured yet"
            : `Slurm config (${facts.length}) · captured ${formatAge(cluster.config_captured_at, now)}`}
        </button>
        <Button variant="outline" size="sm" onClick={() => setConfirming(true)} disabled={checking}>
          {checking ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <PlugZap className="h-3.5 w-3.5" />}
          {checking ? "Checking…" : "Check connection"}
        </Button>
      </div>

      {showFacts && facts.length > 0 && (
        <dl className="max-h-64 space-y-1 overflow-auto rounded-lg border border-border bg-background/40 p-3">
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

      <ConfirmDialog
        open={confirming}
        title={`Check connection to ${name}?`}
        description="This connects to the cluster the way the reconciler does, runs scontrol ping and scontrol show config, and stores the config. The check is written to the audit log."
        confirmLabel="Run check"
        onConfirm={() => void runCheck()}
        onCancel={() => setConfirming(false)}
      />
    </Card>
  );
}
