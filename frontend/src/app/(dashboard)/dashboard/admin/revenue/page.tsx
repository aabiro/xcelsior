"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { FadeIn, ScrollReveal, StaggerList, StaggerItem, CountUp, HoverCard } from "@/components/ui/motion";
import { DollarSign, RefreshCw, TrendingUp, BarChart3, Download, GitCompareArrows } from "lucide-react";
import * as api from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell, Legend,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "@/lib/recharts";

const COLORS = ["#dc2626", "#f59e0b", "#38bdf8", "#10b981", "#8b5cf6", "#ec4899", "#6366f1", "#14b8a6"];
const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

const RANGE_PRESETS = [
  { label: "30d", days: 30 },
  { label: "60d", days: 60 },
  { label: "90d", days: 90 },
  { label: "YTD", days: 0 },
];

export default function AdminRevenuePage() {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.fetchAdminRevenue>> | null>(null);
  const [prevData, setPrevData] = useState<Awaited<ReturnType<typeof api.fetchAdminRevenue>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(90);
  const [showComparison, setShowComparison] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    const d = days === 0
      ? Math.ceil((Date.now() - new Date(new Date().getFullYear(), 0, 1).getTime()) / 86400000)
      : days;
    const promises: Promise<unknown>[] = [
      api.fetchAdminRevenue(d).then(setData),
    ];
    if (showComparison) {
      promises.push(
        api.fetchAdminRevenue(d * 2).then((full) => {
          // Extract only the "previous" half of the data
          const cutoff = full.daily.length - d;
          const prev = { ...full, daily: full.daily.slice(0, Math.max(0, cutoff)) };
          setPrevData(prev);
        }).catch(() => setPrevData(null)),
      );
    } else {
      setPrevData(null);
    }
    Promise.allSettled(promises)
      .catch(() => toast.error("Failed to load revenue data"))
      .finally(() => setLoading(false));
  }, [days, showComparison]);

  useEffect(() => { load(); }, [load]);

  useEventStream({
    eventTypes: ["payment_received", "job_completed"],
    onEvent: load,
  });

  const totalRevenue = data?.daily?.reduce((s, d) => s + d.revenue, 0) ?? 0;
  const totalJobs = data?.daily?.reduce((s, d) => s + d.jobs, 0) ?? 0;
  const totalGpuHours = data?.daily?.reduce((s, d) => s + d.gpu_hours, 0) ?? 0;
  const hasData = (data?.daily?.length ?? 0) > 0;

  const exportCsv = () => {
    if (!data?.daily?.length) { toast.error("No data to export"); return; }
    const keys = ["date", "revenue", "jobs", "gpu_hours"] as const;
    const csv = [keys.join(","), ...data.daily.map((r) => keys.map((k) => `"${r[k]}"`).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `xcelsior-revenue-${days || "ytd"}.csv`; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Revenue</h1>
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
          <Button variant="outline" size="sm" onClick={() => setShowComparison((v) => !v)} className={showComparison ? "border-accent-cyan text-accent-cyan" : ""}>
            <GitCompareArrows className="h-3.5 w-3.5" /> Compare
          </Button>
          <Button variant="outline" size="sm" onClick={exportCsv}>
            <Download className="h-3.5 w-3.5" /> CSV
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[...Array(3)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="h-72 rounded-xl bg-surface skeleton-pulse" />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {[...Array(2)].map((_, i) => <div key={i} className="h-64 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
        </div>
      ) : !hasData ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
            <DollarSign className="h-10 w-10 text-text-muted" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No revenue data yet</h3>
          <p className="text-sm text-text-secondary max-w-md text-center mb-6">
            Revenue will appear here once jobs are billed on the platform.
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 w-full max-w-lg">
            <Card className="p-4 text-center border-dashed">
              <DollarSign className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">Revenue</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
            <Card className="p-4 text-center border-dashed">
              <BarChart3 className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">Jobs</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
            <Card className="p-4 text-center border-dashed">
              <TrendingUp className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">GPU Hours</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
          </div>
        </div>
      ) : (
        <>
          {/* KPIs */}
          <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StaggerItem><StatCard label={`Revenue (${days || "YTD"})`} value={<CountUp value={totalRevenue} prefix="$" />} icon={DollarSign} glow="gold" /></StaggerItem>
            <StaggerItem><StatCard label="Jobs" value={<CountUp value={totalJobs} />} icon={BarChart3} glow="cyan" /></StaggerItem>
            <StaggerItem><StatCard label="GPU Hours" value={<CountUp value={Math.round(totalGpuHours * 10) / 10} />} icon={TrendingUp} glow="emerald" /></StaggerItem>
          </StaggerList>

          {/* Revenue Area Chart */}
          <FadeIn delay={0.15}>
            <HoverCard>
            <Card className="glow-card brand-top-accent">
              <CardHeader>
                <CardTitle className="text-sm">
                  Daily Revenue
                  {showComparison && <span className="ml-2 text-xs text-text-muted font-normal">vs previous period</span>}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                    <AreaChart data={(() => {
                      const current = data!.daily;
                      if (!showComparison || !prevData?.daily?.length) return current;
                      return current.map((d, i) => ({
                        ...d,
                        prev_revenue: prevData.daily[i]?.revenue ?? 0,
                      }));
                    })()}>
                      <defs>
                        <linearGradient id="revGrad2" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="prevGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#94a3b8" stopOpacity={0.15} />
                          <stop offset="95%" stopColor="#94a3b8" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => String(v).slice(5)} />
                      <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => `$${v}`} />
                      <Tooltip contentStyle={tooltipStyle} formatter={(v: any, name: any) => [`$${Number(v).toFixed(2)}`, name === "prev_revenue" ? "Previous" : "Revenue"]} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {showComparison && (
                        <Area type="monotone" dataKey="prev_revenue" stroke="#94a3b8" fill="url(#prevGrad)" strokeWidth={1} strokeDasharray="4 4" name="Previous" />
                      )}
                      <Area type="monotone" dataKey="revenue" stroke="#10b981" fill="url(#revGrad2)" strokeWidth={2} name="Revenue" />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
            </HoverCard>
          </FadeIn>

          <FadeIn delay={0.25}>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* By GPU Model (Pie) */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Revenue by GPU</CardTitle></CardHeader>
                <CardContent>
                  {!data?.by_gpu?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No data</p>
                  ) : (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                        <PieChart>
                          <Pie
                            data={data.by_gpu}
                            dataKey="revenue"
                            nameKey="gpu_model"
                            cx="50%" cy="50%"
                            outerRadius={90}
                            label={(props: any) => `${props.name || ""} ${((props.percent || 0) * 100).toFixed(0)}%`}
                            labelLine={false}
                            fontSize={11}
                          >
                            {data.by_gpu.map((_, i) => (
                              <Cell key={i} fill={COLORS[i % COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [`$${Number(v).toFixed(2)}`, "Revenue"]} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>

              {/* By Province (Bar) */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Revenue by Province</CardTitle></CardHeader>
                <CardContent>
                  {!data?.by_province?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No data</p>
                  ) : (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                        <BarChart data={data.by_province} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => `$${v}`} />
                          <YAxis type="category" dataKey="province" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" width={80} />
                          <Tooltip contentStyle={tooltipStyle} formatter={(v: any) => [`$${Number(v).toFixed(2)}`, "Revenue"]} />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar dataKey="revenue" fill="#38bdf8" radius={[0, 4, 4, 0]} name="Revenue" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>
            </div>
          </FadeIn>

          {/* Top Tables */}
          <ScrollReveal>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Top Customers */}
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Top Customers</CardTitle></CardHeader>
                <CardContent>
                  {!data?.top_customers?.length ? (
                    <p className="text-sm text-text-muted py-4">No data</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border text-text-secondary">
                            <th className="py-2 text-left font-medium">#</th>
                            <th className="py-2 text-left font-medium">Email</th>
                            <th className="py-2 text-right font-medium">Spend</th>
                            <th className="py-2 text-right font-medium">Jobs</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.top_customers.map((c, i) => (
                            <tr key={c.email} className="border-b border-border/50 hover:bg-surface-hover hover:border-l-2 hover:border-l-accent-cyan transition-colors">
                              <td className="py-2 text-text-muted">{i + 1}</td>
                              <td className="py-2 font-medium">{c.email}</td>
                              <td className="py-2 text-right font-mono">${c.total_spend.toFixed(2)}</td>
                              <td className="py-2 text-right font-mono">{c.jobs}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Top Providers */}
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Top Providers</CardTitle></CardHeader>
                <CardContent>
                  {!data?.top_providers?.length ? (
                    <p className="text-sm text-text-muted py-4">No data</p>
                  ) : (
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border text-text-secondary">
                            <th className="py-2 text-left font-medium">#</th>
                            <th className="py-2 text-left font-medium">Provider</th>
                            <th className="py-2 text-right font-medium">Earnings</th>
                            <th className="py-2 text-right font-medium">Jobs</th>
                          </tr>
                        </thead>
                        <tbody>
                          {data.top_providers.map((p, i) => (
                            <tr key={p.provider_id} className="border-b border-border/50 hover:bg-surface-hover hover:border-l-2 hover:border-l-accent-cyan transition-colors">
                              <td className="py-2 text-text-muted">{i + 1}</td>
                              <td className="py-2 font-medium">{p.provider_id}</td>
                              <td className="py-2 text-right font-mono">${p.earnings.toFixed(2)}</td>
                              <td className="py-2 text-right font-mono">{p.jobs}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>
          </ScrollReveal>
        </>
      )}
    </div>
  );
}
