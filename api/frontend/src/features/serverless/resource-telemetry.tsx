"use client";

import { useEffect, useState } from "react";
import { Activity } from "lucide-react";
import type { ServerlessWorkerTelemetry } from "@/lib/api";
import * as api from "@/lib/api";
import { cn } from "@/lib/utils";

function clampPct(value?: number | null) {
  return Math.max(0, Math.min(100, Number(value || 0)));
}

function TelemetryBar({ label, value, detail, tone = "bg-accent-cyan" }: {
  label: string;
  value: number;
  detail?: string;
  tone?: string;
}) {
  const pct = clampPct(value);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between gap-2 text-[11px]">
        <span className="text-text-muted">{label}</span>
        <span className="font-mono text-text-secondary">{detail || `${Math.round(pct)}%`}</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-border/70">
        <div className={cn("h-full rounded-full transition-[width]", tone)} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function ResourceTelemetryBars({ telemetry, compact = false }: {
  telemetry?: ServerlessWorkerTelemetry | null;
  compact?: boolean;
}) {
  const empty = !telemetry;
  const memDetail = telemetry && telemetry.gpu_memory_total_gb > 0
    ? `${telemetry.gpu_memory_used_gb.toFixed(1)}/${telemetry.gpu_memory_total_gb.toFixed(0)} GB`
    : telemetry
      ? `${Math.round(telemetry.gpu_memory_pct)}%`
      : "Waiting";
  return (
    <div className={cn("grid gap-2", compact ? "grid-cols-2" : "sm:grid-cols-4")}>
      <TelemetryBar label="GPU" value={telemetry?.gpu_util_pct ?? 0} detail={empty ? "Waiting" : undefined} tone="bg-emerald" />
      <TelemetryBar label="GPU memory" value={telemetry?.gpu_memory_pct ?? 0} detail={memDetail} tone="bg-accent-cyan" />
      <TelemetryBar label="CPU" value={telemetry?.cpu_util_pct ?? 0} detail={empty ? "Waiting" : undefined} tone="bg-accent-violet" />
      <TelemetryBar label="RAM" value={telemetry?.system_memory_pct ?? 0} detail={empty ? "Waiting" : undefined} tone="bg-amber-400" />
      {empty && (
        <div className="col-span-full flex items-center gap-1.5 text-[11px] text-text-muted">
          <Activity className="h-3 w-3" />
          Waiting for telemetry
        </div>
      )}
      {telemetry?.stale && (
        <div className="col-span-full text-[11px] text-amber-400">Telemetry stale</div>
      )}
    </div>
  );
}

export function WorkerTelemetryStrip({ endpointId, workerId, compact = true }: {
  endpointId: string;
  workerId: string;
  compact?: boolean;
}) {
  const [telemetry, setTelemetry] = useState<ServerlessWorkerTelemetry | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      api.getServerlessWorkerTelemetry(endpointId, workerId)
        .then((res) => {
          if (!cancelled) setTelemetry(res.telemetry ?? null);
        })
        .catch(() => {
          if (!cancelled) setTelemetry(null);
        });
    };
    load();
    const id = window.setInterval(load, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [endpointId, workerId]);

  return <ResourceTelemetryBars telemetry={telemetry} compact={compact} />;
}
