"use client";

import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Server, Loader2, Cpu, Activity, Clock, ChevronDown, ChevronRight,
  CircleDot, AlertTriangle,
} from "lucide-react";
import type { ServerlessWorker, ServerlessJob } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";
import { ServerlessEmptyState } from "./serverless-ui";

const STATE_COLORS: Record<string, "active" | "info" | "warning" | "default" | "failed"> = {
  ready: "active",
  idle: "info",
  busy: "active",
  booting: "warning",
  draining: "warning",
  exited: "default",
  error: "failed",
};

/** Live dot colour per worker state. */
const STATE_DOT: Record<string, string> = {
  ready: "bg-emerald",
  idle: "bg-accent-cyan",
  busy: "bg-emerald",
  booting: "bg-amber-400",
  draining: "bg-amber-400",
  exited: "bg-text-muted",
  error: "bg-red-500",
};

function relTime(ts?: number): string {
  if (!ts) return "—";
  const diff = Date.now() / 1000 - ts;
  if (diff < 0) return "just now";
  if (diff < 60) return `${Math.floor(diff)}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

function durationSince(ts?: number): string {
  if (!ts) return "—";
  const s = Math.max(0, Math.floor(Date.now() / 1000 - ts));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
}

/** Effective display state: a ready worker with concurrency is "busy". */
function displayState(w: ServerlessWorker): string {
  const raw = (w.state || "").toLowerCase();
  if ((raw === "ready" || raw === "idle") && (w.current_concurrency ?? 0) > 0) return "busy";
  return raw || "unknown";
}

interface WorkersPanelProps {
  workers: ServerlessWorker[];
  jobs?: ServerlessJob[];
  loading?: boolean;
}

export function WorkersPanel({ workers, jobs = [], loading }: WorkersPanelProps) {
  const { t } = useLocale();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  // Recent jobs grouped per worker — this is the per-worker "log".
  const jobsByWorker = useMemo(() => {
    const map = new Map<string, ServerlessJob[]>();
    for (const j of jobs) {
      if (!j.worker_id) continue;
      const list = map.get(j.worker_id) ?? [];
      list.push(j);
      map.set(j.worker_id, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => (b.created_at ?? b.queued_at ?? 0) - (a.created_at ?? a.queued_at ?? 0));
    }
    return map;
  }, [jobs]);

  // Fleet summary counts by effective state.
  const fleet = useMemo(() => {
    const counts: Record<string, number> = { ready: 0, idle: 0, busy: 0, booting: 0, draining: 0, exited: 0, error: 0 };
    for (const w of workers) {
      const s = displayState(w);
      counts[s] = (counts[s] ?? 0) + 1;
    }
    return counts;
  }, [workers]);

  // Active (running) workers shown first, then booting, then the rest.
  const orderedWorkers = useMemo(() => {
    const rank: Record<string, number> = { busy: 0, ready: 1, idle: 2, booting: 3, draining: 4, error: 5, exited: 6 };
    return [...workers].sort((a, b) => {
      const ra = rank[displayState(a)] ?? 9;
      const rb = rank[displayState(b)] ?? 9;
      if (ra !== rb) return ra - rb;
      return (b.allocated_at ?? 0) - (a.allocated_at ?? 0);
    });
  }, [workers]);

  if (loading && workers.length === 0) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
      </div>
    );
  }

  if (workers.length === 0) {
    return (
      <ServerlessEmptyState
        icon={Server}
        title={t("dash.serverless.workers_empty")}
        description={t("dash.serverless.workers_empty_desc")}
      />
    );
  }

  const summaryItems: { key: string; label: string }[] = [
    { key: "busy", label: t("dash.serverless.wstate_busy") },
    { key: "ready", label: t("dash.serverless.wstate_ready") },
    { key: "idle", label: t("dash.serverless.wstate_idle") },
    { key: "booting", label: t("dash.serverless.wstate_booting") },
    { key: "draining", label: t("dash.serverless.wstate_draining") },
  ];

  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div className="space-y-5">
      {/* Fleet summary header */}
      <div className="glow-card rounded-xl border border-border bg-surface p-4">
        <div className="flex items-center justify-between mb-3">
          <p className="text-sm font-medium flex items-center gap-2">
            <Activity className="h-4 w-4 text-accent-violet" />
            {t("dash.serverless.fleet_title")}
          </p>
          <span className="text-xs text-text-muted">
            {t("dash.serverless.fleet_total", { count: String(workers.length) })}
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          {summaryItems.map((item) => (
            <div
              key={item.key}
              className={cn(
                "flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 text-xs",
                (fleet[item.key] ?? 0) > 0 ? "border-border bg-surface-hover" : "border-border/50 opacity-50",
              )}
            >
              <span className={cn("h-2 w-2 rounded-full", STATE_DOT[item.key], (item.key === "busy" || item.key === "booting") && (fleet[item.key] ?? 0) > 0 && "animate-pulse")} />
              <span className="font-mono font-semibold">{fleet[item.key] ?? 0}</span>
              <span className="text-text-muted">{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Per-worker cards */}
      <div className="space-y-3">
        {orderedWorkers.map((w) => {
          const s = displayState(w);
          const workerJobs = jobsByWorker.get(w.worker_id) ?? [];
          const isOpen = expanded.has(w.worker_id);
          const isBooting = s === "booting";
          return (
            <div
              key={w.worker_id}
              className={cn(
                "glow-card rounded-xl border bg-surface transition-colors",
                s === "busy" || s === "ready" ? "border-emerald/25" : s === "error" ? "border-red-500/30" : "border-border",
              )}
            >
              <button
                type="button"
                onClick={() => toggle(w.worker_id)}
                className="flex w-full items-center gap-3 p-4 text-left"
              >
                <span className={cn("h-2.5 w-2.5 rounded-full shrink-0", STATE_DOT[s] ?? "bg-text-muted", (s === "busy" || s === "booting") && "animate-pulse")} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-xs text-text-secondary truncate">{w.worker_id}</span>
                    <Badge variant={STATE_COLORS[s] ?? "default"}>{t(`dash.serverless.wstate_${s}`) || s}</Badge>
                    {(w.current_concurrency ?? 0) > 0 && (
                      <span className="text-[11px] text-text-muted">
                        {w.current_concurrency} {t("dash.serverless.concurrent")}
                      </span>
                    )}
                  </div>
                  <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted">
                    {w.host_id ? (
                      <span className="inline-flex items-center gap-1">
                        <Server className="h-3 w-3" /> {w.host_id}
                      </span>
                    ) : null}
                    <span className="inline-flex items-center gap-1">
                      <Cpu className="h-3 w-3" /> {w.gpu_count ?? 1} GPU
                    </span>
                    {isBooting ? (
                      <span className="inline-flex items-center gap-1 text-amber-500">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        {t("dash.serverless.worker_booting_for", { dur: durationSince(w.allocated_at) })}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {t("dash.serverless.worker_up_for", { dur: durationSince(w.allocated_at) })}
                      </span>
                    )}
                    {!isBooting && w.last_heartbeat_at ? (
                      <span className="inline-flex items-center gap-1">
                        <CircleDot className="h-3 w-3" />
                        {t("dash.serverless.worker_heartbeat", { ago: relTime(w.last_heartbeat_at) })}
                      </span>
                    ) : null}
                  </div>
                  {w.error_message ? (
                    <p className="mt-1.5 inline-flex items-start gap-1 text-[11px] text-red-400">
                      <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" /> {w.error_message}
                    </p>
                  ) : null}
                </div>
                {workerJobs.length > 0 && (
                  isOpen ? <ChevronDown className="h-4 w-4 text-text-muted shrink-0" /> : <ChevronRight className="h-4 w-4 text-text-muted shrink-0" />
                )}
              </button>

              {/* Per-worker recent jobs ("logs") */}
              {isOpen && workerJobs.length > 0 && (
                <div className="border-t border-border/60 px-4 py-3">
                  <p className="text-[11px] font-medium text-text-muted mb-2 uppercase tracking-wide">
                    {t("dash.serverless.worker_recent_jobs", { count: String(workerJobs.length) })}
                  </p>
                  <ul className="divide-y divide-border/40">
                    {workerJobs.slice(0, 8).map((j) => (
                      <li key={j.job_id} className="flex items-center justify-between gap-3 py-1.5 text-xs">
                        <span className="font-mono text-text-muted truncate">{j.job_id.slice(0, 16)}</span>
                        <span className="flex items-center gap-2 shrink-0">
                          {j.gpu_seconds != null && (
                            <span className="text-text-muted">{j.gpu_seconds.toFixed(1)}s GPU</span>
                          )}
                          <Badge variant={j.status === "COMPLETED" ? "active" : j.status === "FAILED" ? "failed" : "default"}>
                            {j.status}
                          </Badge>
                          <span className="text-text-muted">{relTime(j.finished_at ?? j.created_at)}</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
