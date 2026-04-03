"use client";

import { useEffect, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Activity, Thermometer, Cpu, Zap, MemoryStick, RefreshCw } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import type { TelemetryData } from "@/lib/api";
import { toast } from "sonner";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

export default function TelemetryPage() {
  const [data, setData] = useState<TelemetryData[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useApi();
  const { t } = useLocale();

  const load = () => {
    setLoading(true);
    api.fetchTelemetry()
      .then((res) => {
        const telemetry = res.telemetry || {};
        const flat = Object.values(telemetry).flat() as TelemetryData[];
        setData(flat);
      })
      .catch(() => toast.error("Failed to load telemetry"))
      .finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  const latest = data[data.length - 1];

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
            <StatCard label={t("dash.telemetry.gpu_util")} value={`${latest?.gpu_util ?? 0}%`} icon={Cpu} glow="cyan" />
            <StatCard label={t("dash.telemetry.temperature")} value={`${latest?.temperature ?? 0}°C`} icon={Thermometer} glow="gold" />
            <StatCard label={t("dash.telemetry.memory")} value={`${latest?.mem_used_mb ? (latest.mem_used_mb / 1024).toFixed(1) : 0}GB`} icon={MemoryStick} glow="violet" />
            <StatCard label={t("dash.telemetry.power")} value={`${latest?.power_draw_w ?? 0}W`} icon={Zap} glow="emerald" />
          </div>

          {/* Charts */}
          {data.length > 0 && (
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <TelemetryChart title={t("dash.telemetry.chart_util")} data={data} dataKey="gpu_util" color="#38bdf8" />
            <TelemetryChart title={t("dash.telemetry.chart_temp")} data={data} dataKey="temperature" color="#f59e0b" />
            <TelemetryChart title={t("dash.telemetry.chart_mem")} data={data} dataKey="mem_used_mb" color="#10b981" />
            <TelemetryChart title={t("dash.telemetry.chart_power")} data={data} dataKey="power_draw_w" color="#dc2626" />
          </div>
          )}
        </>
      )}
    </div>
  );
}

function TelemetryChart({
  title,
  data,
  dataKey,
  color,
}: {
  title: string;
  data: TelemetryData[];
  dataKey: string;
  color: string;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-48">
          <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={0} debounce={1}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id={`fill-${dataKey}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={color} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={color} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis dataKey="timestamp" tick={false} stroke="#475569" />
              <YAxis stroke="#475569" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Area type="monotone" dataKey={dataKey} stroke={color} fill={`url(#fill-${dataKey})`} strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
