"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Thermometer, Cpu, Zap, MemoryStick, RefreshCw, Server } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import { toast } from "sonner";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "@/lib/recharts";
import { cn } from "@/lib/utils";

// Normalised per-host telemetry entry
interface HostMetrics {
  host_id: string;
  utilization: number;      // GPU util %
  temp: number;             // °C
  memory_used_gb: number;
  memory_total_gb: number;
  power_draw_w: number;
  active_jobs: number;
  stale: boolean;
  timestamp: number;
}

function mapHost(host_id: string, raw: Record<string, unknown>): HostMetrics {
  const m = (raw.metrics as Record<string, number>) || {};
  return {
    host_id,
    utilization: Number(m.utilization ?? 0),
    temp: Number(m.temp ?? 0),
    memory_used_gb: Number(m.memory_used_gb ?? 0),
    memory_total_gb: Number(m.memory_total_gb ?? 0),
    power_draw_w: Number(m.power_draw_w ?? 0),
    active_jobs: Number(m.active_jobs ?? 0),
    stale: Boolean(raw.stale),
    timestamp: Number(raw.timestamp ?? 0),
  };
}

export default function TelemetryPage() {
  const [hosts, setHosts] = useState<HostMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useApi();
  const { t } = useLocale();

  const load = () => {
    setLoading(true);
    api.fetchTelemetry()
      .then((res) => {
        // Backend returns { ok, hosts: { [host_id]: { timestamp, metrics: {...}, stale } } }
        const raw = (res as unknown as { hosts?: Record<string, Record<string, unknown>> }).hosts || {};
        setHosts(Object.entries(raw).map(([id, v]) => mapHost(id, v)));
      })
      .catch(() => toast.error("Failed to load telemetry"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  // Aggregate across all hosts (sum utilization, avg temp, sum memory/power)
  const count = hosts.length;
  const agg = hosts.reduce(
    (acc, h) => ({
      utilization: acc.utilization + h.utilization,
      temp: acc.temp + h.temp,
      memory_used_gb: acc.memory_used_gb + h.memory_used_gb,
      memory_total_gb: acc.memory_total_gb + h.memory_total_gb,
      power_draw_w: acc.power_draw_w + h.power_draw_w,
      active_jobs: acc.active_jobs + h.active_jobs,
    }),
    { utilization: 0, temp: 0, memory_used_gb: 0, memory_total_gb: 0, power_draw_w: 0, active_jobs: 0 },
  );
  const avgUtil = count ? Math.round(agg.utilization / count) : 0;
  const avgTemp = count ? Math.round(agg.temp / count) : 0;

  // Chart data: one bar per host for each metric
  const chartData = hosts.map((h) => ({
    host: h.host_id.slice(-8),
    "GPU Util %": h.utilization,
    "Temp °C": h.temp,
    "Mem GB": parseFloat(h.memory_used_gb.toFixed(1)),
    "Power W": h.power_draw_w,
    stale: h.stale,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.telemetry.title")}</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label={t("dash.telemetry.gpu_util")} value={`${avgUtil}%`} icon={Cpu} glow="cyan" />
            <StatCard label={t("dash.telemetry.temperature")} value={`${avgTemp}°C`} icon={Thermometer} glow="gold" />
            <StatCard label={t("dash.telemetry.memory")} value={`${agg.memory_used_gb.toFixed(1)} / ${agg.memory_total_gb.toFixed(0)}GB`} icon={MemoryStick} glow="violet" />
            <StatCard label={t("dash.telemetry.power")} value={`${agg.power_draw_w.toFixed(0)}W`} icon={Zap} glow="emerald" />
          </div>

          {count === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16 text-center">
                <Server className="h-12 w-12 text-text-muted mb-3 opacity-30" />
                <p className="text-sm text-text-muted">No live telemetry — start a worker to see GPU metrics</p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <TelemetryChart title={t("dash.telemetry.chart_util")} data={chartData} dataKey="GPU Util %" color="#38bdf8" unit="%" />
              <TelemetryChart title={t("dash.telemetry.chart_temp")} data={chartData} dataKey="Temp °C" color="#f59e0b" unit="°C" />
              <TelemetryChart title={t("dash.telemetry.chart_mem")} data={chartData} dataKey="Mem GB" color="#8b5cf6" unit="GB" />
              <TelemetryChart title={t("dash.telemetry.chart_power")} data={chartData} dataKey="Power W" color="#ef4444" unit="W" />
            </div>
          )}

          {/* Per-host breakdown */}
          {count > 0 && (
            <div className="space-y-2">
              <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider">Per-Host Breakdown</h2>
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                {hosts.map((h) => (
                  <Card key={h.host_id} className={cn("border", h.stale && "opacity-60 border-warning/30")}>
                    <CardContent className="p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono text-text-muted truncate">{h.host_id}</span>
                        {h.stale && <span className="text-[10px] bg-warning/15 text-warning px-1.5 py-0.5 rounded font-medium">STALE</span>}
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div><span className="text-text-muted text-xs">GPU</span><p className="font-semibold text-accent-cyan">{h.utilization}%</p></div>
                        <div><span className="text-text-muted text-xs">Temp</span><p className="font-semibold text-accent-gold">{h.temp}°C</p></div>
                        <div><span className="text-text-muted text-xs">VRAM</span><p className="font-semibold text-accent-violet">{h.memory_used_gb.toFixed(1)}/{h.memory_total_gb.toFixed(0)}GB</p></div>
                        <div><span className="text-text-muted text-xs">Power</span><p className="font-semibold text-accent-red">{h.power_draw_w.toFixed(0)}W</p></div>
                      </div>
                      {h.active_jobs > 0 && (
                        <div className="flex items-center gap-1.5 pt-1 border-t border-border/40">
                          <span className="live-dot" />
                          <span className="text-xs text-emerald">{h.active_jobs} active job{h.active_jobs !== 1 ? "s" : ""}</span>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function TelemetryChart({ title, data, dataKey, color, unit }: {
  title: string;
  data: Record<string, unknown>[];
  dataKey: string;
  color: string;
  unit?: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id={`fill-${dataKey.replace(/[^a-z]/gi, "")}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="host" tick={{ fill: "#94a3b8", fontSize: 10 }} stroke="#475569" />
              <YAxis stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 11 }} unit={unit} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }}
                labelStyle={{ color: "#94a3b8" }}
                formatter={(v: unknown) => [`${v}${unit ?? ""}`, dataKey]}
              />
              <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#fill-${dataKey.replace(/[^a-z]/gi, "")})`} strokeWidth={2} dot={{ fill: color, r: 4 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
