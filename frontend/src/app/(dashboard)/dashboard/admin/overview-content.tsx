"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { FadeIn, StaggerList, StaggerItem, CountUp, HoverCard } from "@/components/ui/motion";
import { DesktopPushHealthCard } from "@/components/admin/DesktopPushHealthCard";
import {
  Users, Server, Activity, DollarSign, Clock, Zap,
  RefreshCw, TrendingUp, LayoutDashboard, Gauge, AlertTriangle, UserCheck,
} from "lucide-react";
import * as api from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import {
  AreaChart, Area, BarChart, Bar, Legend,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

const RANGE_PRESETS = [
  { label: "7d", days: 7 },
  { label: "14d", days: 14 },
  { label: "30d", days: 30 },
  { label: "60d", days: 60 },
  { label: "90d", days: 90 },
];

export default function AdminOverviewPage() {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.fetchAdminOverview>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);

  const load = useCallback(() => {
    setLoading(true);
    api.fetchAdminOverview(days).then(setData).finally(() => setLoading(false));
  }, [days]);

  useEffect(() => { load(); }, [load]);
  useEventStream({
    eventTypes: ["job_status", "job_submitted", "host_registered", "host_removed", "user_registered"],
    onEvent: load,
  });

  const k = data?.kpis;
  const t = data?.trends;
  const hasData = k && (k.total_users > 0 || k.total_jobs > 0 || k.active_hosts > 0);

  const trendDir = (pct: number | undefined) =>
    pct == null || pct === 0 ? undefined : pct > 0 ? "up" as const : "down" as const;
  const trendVal = (pct: number | undefined) =>
    pct == null || pct === 0 ? undefined : `${pct > 0 ? "+" : ""}${pct}%`;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Admin Overview</h1>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-surface p-1">
            {RANGE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setDays(p.days)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  days === p.days ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[...Array(3)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {[...Array(2)].map((_, i) => <div key={i} className="h-64 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
        </div>
      ) : !hasData ? (
        <div className="space-y-6">
          {data?.web_push ? (
            <FadeIn delay={0.1}>
              <DesktopPushHealthCard
                snapshot={data.web_push}
                onSnapshotChange={(next) => setData((current) => (current ? { ...current, web_push: next } : current))}
              />
            </FadeIn>
          ) : null}

          <div className="flex flex-col items-center justify-center py-20">
            <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
              <LayoutDashboard className="h-10 w-10 text-text-muted" />
            </div>
            <h3 className="text-xl font-semibold mb-2">No platform data yet</h3>
            <p className="text-sm text-text-secondary max-w-md text-center mb-6">
              Once users register, hosts come online, and jobs are submitted, platform metrics will appear here.
            </p>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 w-full max-w-lg">
              <Card className="p-4 text-center border-dashed">
                <Users className="h-5 w-5 text-text-muted mx-auto mb-2" />
                <p className="text-xs text-text-muted">Users</p>
                <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
              </Card>
              <Card className="p-4 text-center border-dashed">
                <Server className="h-5 w-5 text-text-muted mx-auto mb-2" />
                <p className="text-xs text-text-muted">Hosts</p>
                <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
              </Card>
              <Card className="p-4 text-center border-dashed">
                <DollarSign className="h-5 w-5 text-text-muted mx-auto mb-2" />
                <p className="text-xs text-text-muted">Revenue</p>
                <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
              </Card>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Primary KPIs */}
          <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StaggerItem><StatCard label="Total Users" value={<CountUp value={k.total_users} />} icon={Users} glow="cyan" trend={trendDir(t?.users_pct)} trendValue={trendVal(t?.users_pct)} /></StaggerItem>
            <StaggerItem><StatCard label="Active Hosts" value={<CountUp value={k.active_hosts} />} icon={Server} glow="violet" trend={trendDir(t?.hosts_pct)} trendValue={trendVal(t?.hosts_pct)} /></StaggerItem>
            <StaggerItem><StatCard label="Running Jobs" value={<CountUp value={k.running_jobs} />} icon={Activity} glow="emerald" trend={trendDir(t?.jobs_pct)} trendValue={trendVal(t?.jobs_pct)} /></StaggerItem>
            <StaggerItem><StatCard label="Revenue MTD" value={<CountUp value={k.revenue_mtd} prefix="$" />} icon={DollarSign} glow="gold" trend={trendDir(t?.revenue_pct)} trendValue={trendVal(t?.revenue_pct)} /></StaggerItem>
          </StaggerList>
          <FadeIn delay={0.15}>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-6">
              <StatCard label="Total Jobs" value={<CountUp value={k.total_jobs} />} icon={Zap} />
              <StatCard label="Total Revenue" value={<CountUp value={k.revenue_total} prefix="$" />} icon={TrendingUp} />
              <StatCard label="GPU Hours" value={<CountUp value={k.total_gpu_hours} />} icon={Clock} />
              <StatCard label="GPU Utilization" value={<CountUp value={k.gpu_utilization} suffix="%" />} icon={Gauge} />
              <StatCard label="Failure Rate" value={<CountUp value={k.job_failure_rate} suffix="%" />} icon={AlertTriangle} />
              <StatCard label="ARPU" value={<CountUp value={k.arpu} prefix="$" />} icon={UserCheck} />
            </div>
          </FadeIn>

          {data?.web_push ? (
            <FadeIn delay={0.2}>
              <DesktopPushHealthCard
                snapshot={data.web_push}
                onSnapshotChange={(next) => setData((current) => (current ? { ...current, web_push: next } : current))}
              />
            </FadeIn>
          ) : null}

          {/* Charts */}
          <FadeIn delay={0.25}>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Revenue Trend */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Revenue ({days}d)</CardTitle></CardHeader>
                <CardContent>
                  {!data?.daily_revenue?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No revenue data yet</p>
                  ) : (
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                        <AreaChart data={data.daily_revenue}>
                          <defs>
                            <linearGradient id="revGrad" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => String(v).slice(5)} />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => `$${v}`} />
                          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [`$${Number(v).toFixed(2)}`, "Revenue"]} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Area type="monotone" dataKey="revenue" stroke="#10b981" fill="url(#revGrad)" strokeWidth={2} name="Revenue" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>

              {/* Job Activity */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Jobs ({days}d)</CardTitle></CardHeader>
                <CardContent>
                  {!data?.daily_jobs?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No job data yet</p>
                  ) : (
                    <div className="h-56">
                      <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                        <BarChart data={data.daily_jobs}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => String(v).slice(5)} />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="jobs" fill="#38bdf8" radius={[4, 4, 0, 0]} name="Jobs" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>

              {/* Signups Trend */}
              <HoverCard className="lg:col-span-2">
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">New Signups ({days}d)</CardTitle></CardHeader>
                <CardContent>
                  {!data?.daily_signups?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No signup data yet</p>
                  ) : (
                    <div className="h-48">
                      <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                        <BarChart data={data.daily_signups}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => String(v).slice(5)} />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" allowDecimals={false} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="signups" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Signups" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>
            </div>
          </FadeIn>
        </>
      )}
    </div>
  );
}
