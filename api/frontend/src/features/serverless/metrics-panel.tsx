"use client";

import { useMemo } from "react";
import { StatCard } from "@/components/ui/stat-card";
import {
  BarChart3, Clock, Zap, DollarSign, Layers, Activity, Gauge,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "@/lib/recharts";
import type { ServerlessEndpointMetrics, ServerlessJob } from "@/lib/api";
import { useLocale } from "@/lib/locale";

interface MetricsPanelProps {
  metrics: ServerlessEndpointMetrics | null;
  jobs: ServerlessJob[];
  loading?: boolean;
}

export function MetricsPanel({ metrics, jobs, loading }: MetricsPanelProps) {
  const { t } = useLocale();

  const chartData = useMemo(() => {
    const buckets = new Map<string, { completed: number; failed: number }>();
    for (const job of jobs) {
      if (!job.finished_at) continue;
      const d = new Date(job.finished_at * 1000);
      const key = `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:00`;
      const b = buckets.get(key) ?? { completed: 0, failed: 0 };
      if (job.status === "COMPLETED") b.completed++;
      else if (job.status === "FAILED") b.failed++;
      buckets.set(key, b);
    }
    return [...buckets.entries()]
      .slice(-24)
      .map(([time, v]) => ({ time, ...v }));
  }, [jobs]);

  if (loading && !metrics) {
    return <div className="h-48 animate-pulse rounded-xl bg-surface-hover" />;
  }

  const m = metrics;
  const successPct = m ? `${(m.success_rate * 100).toFixed(1)}%` : "-";

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label={t("dash.serverless.metric_requests")}
          value={(m?.total_requests ?? 0).toLocaleString()}
          icon={BarChart3}
          glow="violet"
        />
        <StatCard
          label={t("dash.serverless.metric_success")}
          value={successPct}
          icon={Activity}
          glow="emerald"
        />
        <StatCard
          label={t("dash.serverless.metric_queue")}
          value={m?.avg_queue_ms ? `${Math.round(m.avg_queue_ms)}ms` : "-"}
          icon={Clock}
          glow="cyan"
        />
        <StatCard
          label={t("dash.serverless.metric_cost")}
          value={`$${(m?.total_cost_cad ?? 0).toFixed(2)}`}
          icon={DollarSign}
          glow="gold"
        />
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          label={t("dash.serverless.metric_workers")}
          value={`${m?.active_workers ?? 0} active`}
          icon={Layers}
          glow="violet"
        />
        <StatCard
          label={t("dash.serverless.metric_tokens")}
          value={m?.tokens_per_sec ? `${m.tokens_per_sec}/s` : "-"}
          icon={Zap}
          glow="cyan"
        />
        <StatCard
          label={t("dash.serverless.metric_queue_depth")}
          value={m?.queue_depth ?? 0}
          icon={Gauge}
          glow="gold"
        />
      </div>

      {/* Live utilization strip, Novita-style at-a-glance fleet usage */}
      {m && (m.active_workers > 0 || (m.booting_workers ?? 0) > 0 || m.queue_depth > 0) && (
        <div className="glow-card rounded-xl border border-border bg-surface p-4">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-medium flex items-center gap-2">
              <Activity className="h-4 w-4 text-emerald" />
              {t("dash.serverless.util_title")}
              <span className="ml-1 h-2 w-2 rounded-full bg-emerald animate-pulse" />
            </p>
            <span className="text-xs text-text-muted">
              {t("dash.serverless.util_window_reqs")}: <span className="font-mono">{m.window_requests}</span>
            </span>
          </div>
          <UtilizationBar busy={m.busy_workers} idle={m.idle_workers} booting={m.booting_workers ?? 0} />
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 text-center">
            <UtilStat label={t("dash.serverless.wstate_busy")} value={m.busy_workers} tone="text-emerald" />
            <UtilStat label={t("dash.serverless.wstate_idle")} value={m.idle_workers} tone="text-accent-cyan" />
            <UtilStat label={t("dash.serverless.wstate_booting")} value={m.booting_workers ?? 0} tone="text-amber-400" />
            <UtilStat label={t("dash.serverless.util_in_flight")} value={m.queue_depth} tone="text-accent-violet" />
          </div>
        </div>
      )}

      {chartData.length > 1 && (
        <div className="glow-card brand-top-accent rounded-xl border border-border bg-surface p-4">
          <p className="text-sm font-medium mb-4">{t("dash.serverless.job_activity")}</p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
              <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                <defs>
                  <linearGradient id="slCompleted" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent-emerald)" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="var(--accent-emerald)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="slFailed" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--accent-red)" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="var(--accent-red)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.5} />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} stroke="var(--text-muted)" />
                <YAxis tick={{ fontSize: 10 }} stroke="var(--text-muted)" allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface)",
                    border: "1px solid var(--border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Area type="monotone" dataKey="completed" stackId="1" stroke="var(--accent-emerald)" fill="url(#slCompleted)" />
                <Area type="monotone" dataKey="failed" stackId="1" stroke="var(--accent-red)" fill="url(#slFailed)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
}

function UtilizationBar({ busy, idle, booting }: { busy: number; idle: number; booting: number }) {
  const total = Math.max(1, busy + idle + booting);
  const seg = (n: number) => `${(n / total) * 100}%`;
  return (
    <div className="flex h-2.5 w-full overflow-hidden rounded-full bg-border">
      {busy > 0 && <div className="bg-emerald" style={{ width: seg(busy) }} title={`${busy} busy`} />}
      {idle > 0 && <div className="bg-accent-cyan" style={{ width: seg(idle) }} title={`${idle} idle`} />}
      {booting > 0 && <div className="bg-amber-400 animate-pulse" style={{ width: seg(booting) }} title={`${booting} booting`} />}
    </div>
  );
}

function UtilStat({ label, value, tone }: { label: string; value: number; tone: string }) {
  return (
    <div className="rounded-lg border border-border/60 bg-surface-hover/40 p-2">
      <p className={`font-mono text-lg font-bold leading-none ${tone}`}>{value}</p>
      <p className="mt-1 text-[10px] uppercase tracking-wide text-text-muted">{label}</p>
    </div>
  );
}