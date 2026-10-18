"use client";

import { useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Activity, AlertTriangle, Cpu, Loader2, ScrollText, Server } from "lucide-react";
import type { ServerlessWorker } from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatServerlessChip } from "./format";
import { WorkerTelemetryStrip } from "./resource-telemetry";
import { WorkerLogsModal } from "./worker-logs-modal";

const ACTIVE_STATES = new Set(["starting", "booting", "ready", "idle", "busy", "draining", "error", "failed", "exited"]);
const HIDDEN_STATES = new Set(["terminated"]);

function stateOf(worker: ServerlessWorker): string {
  const raw = String(worker.state || "").toLowerCase();
  if ((raw === "ready" || raw === "idle") && (worker.current_concurrency ?? 0) > 0) return "busy";
  return raw || "unknown";
}

function rank(worker: ServerlessWorker): number {
  const state = stateOf(worker);
  if (state === "busy" || state === "ready" || state === "idle") return 0;
  if (state === "booting" || state === "starting") return 1;
  if (state === "error" || state === "failed") return 2;
  if (state === "draining") return 3;
  return 9;
}

function shortId(id: string): string {
  if (!id) return "-";
  return id.length > 18 ? `${id.slice(0, 10)}…${id.slice(-5)}` : id;
}

export function EndpointWorkerMonitor({
  endpointId,
  workers,
  loading,
}: {
  endpointId: string;
  workers: ServerlessWorker[];
  loading?: boolean;
}) {
  const [selectedId, setSelectedId] = useState<string>("");
  const [logsOpen, setLogsOpen] = useState(false);

  const visibleWorkers = useMemo(
    () =>
      [...workers]
        .filter((worker) => !HIDDEN_STATES.has(stateOf(worker)))
        .sort((a, b) => {
          const ar = rank(a);
          const br = rank(b);
          if (ar !== br) return ar - br;
          return (b.allocated_at ?? b.created_at ?? 0) - (a.allocated_at ?? a.created_at ?? 0);
        }),
    [workers],
  );

  const monitorableWorkers = visibleWorkers.filter((worker) => ACTIVE_STATES.has(stateOf(worker)));
  const selected = monitorableWorkers.find((worker) => worker.worker_id === selectedId) || monitorableWorkers[0] || null;

  useEffect(() => {
    if (!selected) {
      setSelectedId("");
      return;
    }
    if (selected.worker_id !== selectedId) setSelectedId(selected.worker_id);
  }, [selected, selectedId]);

  const count = monitorableWorkers.length;

  return (
    <div className="mt-4 border-t border-border/70 pt-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Activity className="h-4 w-4 text-accent-cyan" />
          Workers: {count}
          {loading && <Loader2 className="h-3.5 w-3.5 animate-spin text-text-muted" />}
        </div>
        {selected && (
          <Button type="button" variant="ghost" size="sm" onClick={() => setLogsOpen(true)}>
            <ScrollText className="h-4 w-4" /> Logs
          </Button>
        )}
      </div>

      {!selected ? null : (
        <div className="mt-3 rounded-xl border border-border bg-surface-hover/35 p-3">
          <div className="grid gap-3 lg:grid-cols-[220px_minmax(0,1fr)]">
            <div className="space-y-1">
              {monitorableWorkers.map((worker) => {
                const state = stateOf(worker);
                const active = worker.worker_id === selected.worker_id;
                return (
                  <button
                    key={worker.worker_id}
                    type="button"
                    onClick={() => setSelectedId(worker.worker_id)}
                    className={cn(
                      "flex w-full items-center justify-between gap-2 rounded-lg border px-2.5 py-2 text-left text-xs transition-colors",
                      active ? "border-accent-cyan/40 bg-accent-cyan/10" : "border-border/60 hover:bg-surface-hover",
                    )}
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-mono text-text-secondary">{shortId(worker.worker_id)}</span>
                      <span className="mt-0.5 flex items-center gap-1 text-[11px] text-text-muted">
                        <Cpu className="h-3 w-3" /> {worker.gpu_count ?? 1} GPU
                      </span>
                    </span>
                    <Badge variant={state === "error" || state === "failed" ? "failed" : state === "booting" || state === "starting" ? "warning" : "default"} className="shrink-0">
                      {formatServerlessChip(state)}
                    </Badge>
                  </button>
                );
              })}
            </div>

            <div className="min-w-0 space-y-3">
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-muted">
                <span className="font-mono text-text-secondary">{selected.worker_id}</span>
                {selected.host_id && (
                  <span className="inline-flex items-center gap-1">
                    <Server className="h-3 w-3" /> {selected.host_id}
                  </span>
                )}
                <span>{selected.current_concurrency ?? 0} concurrent</span>
              </div>
              {selected.error_message && (
                <p className="flex items-start gap-1.5 text-xs text-red-400">
                  <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" /> {selected.error_message}
                </p>
              )}
              <WorkerTelemetryStrip endpointId={endpointId} workerId={selected.worker_id} compact={false} />
            </div>
          </div>
        </div>
      )}

      <WorkerLogsModal
        endpointId={endpointId}
        workers={monitorableWorkers}
        selectedWorkerId={selected?.worker_id}
        open={logsOpen && !!selected}
        onClose={() => setLogsOpen(false)}
      />
    </div>
  );
}
