"use client";

import { useCallback, useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Thermometer, Cpu, Zap, MemoryStick, RefreshCw, Server } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "@/lib/recharts";
import { cn } from "@/lib/utils";
import { readTelemetry } from "@/lib/telemetry";
import type { TelemetryData, TelemetrySnapshot } from "@/lib/api";

// Normalised per-host telemetry entry
interface HostMetrics extends TelemetryData {
  host_id: string;
}

function mapHost(host_id: string, raw: TelemetrySnapshot): HostMetrics {
  return {host_id, ...readTelemetry(raw)};
}

const formatted = (value: number | null, digits: number, unit: string) => value === null ? "Unavailable" : `${value.toFixed(digits)}${unit}`;
const memory = (used: number | null, total: number | null) => used === null || total === null ? "Unavailable" : `${used.toFixed(1)} / ${total.toFixed(0)}GB`;

export default function TelemetryPage() {
  const [hosts, setHosts] = useState<HostMetrics[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const api = useApi();
  const { t } = useLocale();

  const load = useCallback((opts?: { refresh?: boolean }) => {
    if (opts?.refresh) setLoading(true);
    setError(null);
    api.fetchTelemetry()
      .then((res) => {
        setHosts(Object.entries(res.hosts).map(([id, v]) => mapHost(id, v)));
      })
      .catch(() => setError("Telemetry could not be loaded. Please retry."))
      .finally(() => setLoading(false));
  }, [api]);

  useEffect(() => {
    let active = true;
    api.fetchTelemetry()
      .then((res) => {
        if (!active) return;
        setHosts(Object.entries(res.hosts).map(([id, v]) => mapHost(id, v)));
      })
      .catch(() => { if (active) setError("Telemetry could not be loaded. Please retry."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [api]);

  // A fleet total is unavailable if any participating sensor is unknown.
  const count = hosts.length;
  const aggregate = (key: Exclude<keyof TelemetryData, "stale">, average = false) => {
    const values = hosts.map(host => host[key]).filter((value): value is number => value !== null);
    return !count || values.length !== count ? null : values.reduce((sum, value) => sum + value, 0) / (average ? count : 1);
  };
  const avgUtil = aggregate("utilization", true);
  const avgTemp = aggregate("temp", true);

  // Chart data: one bar per host for each metric
  const chartData = hosts.map((h) => ({
    host: h.host_id.slice(-8),
    "GPU Util %": h.utilization,
    "Temp °C": h.temp,
    "Mem GB": h.memory_used_gb,
    "Power W": h.power_draw_w,
    stale: h.stale,
  }));

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.telemetry.title")}</h1>
        <Button variant="outline" size="sm" onClick={() => load({ refresh: true })}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : error ? (
        <Card><CardContent className="space-y-3 p-6" role="alert">
          <p>{error}</p>
          <Button variant="outline" onClick={() => load({refresh:true})}>Retry</Button>
        </CardContent></Card>
      ) : (
        <>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label={t("dash.telemetry.gpu_util")} value={formatted(avgUtil, 0, "%")} icon={Cpu} glow="cyan" />
            <StatCard label={t("dash.telemetry.temperature")} value={formatted(avgTemp, 0, "°C")} icon={Thermometer} glow="gold" />
            <StatCard label={t("dash.telemetry.memory")} value={memory(aggregate("memory_used_gb"), aggregate("memory_total_gb"))} icon={MemoryStick} glow="violet" />
            <StatCard label={t("dash.telemetry.power")} value={formatted(aggregate("power_draw_w"), 0, "W")} icon={Zap} glow="emerald" />
          </div>

          {count === 0 ? (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-16 text-center">
                <Server className="h-12 w-12 text-text-muted mb-3 opacity-30" />
                <p className="text-sm text-text-muted">No live telemetry, start a worker to see GPU metrics</p>
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
                        <div><span className="text-text-muted text-xs">GPU</span><p className="font-semibold text-accent-cyan">{formatted(h.utilization, 0, "%")}</p></div>
                        <div><span className="text-text-muted text-xs">Temp</span><p className="font-semibold text-accent-gold">{formatted(h.temp, 0, "°C")}</p></div>
                        <div><span className="text-text-muted text-xs">VRAM</span><p className="font-semibold text-accent-violet">{memory(h.memory_used_gb, h.memory_total_gb)}</p></div>
                        <div><span className="text-text-muted text-xs">Power</span><p className="font-semibold text-accent-red">{formatted(h.power_draw_w, 0, "W")}</p></div>
                      </div>
                      {h.active_jobs !== null && h.active_jobs > 0 && (
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
