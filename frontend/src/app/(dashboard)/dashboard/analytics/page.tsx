"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatCard } from "@/components/ui/stat-card";
import { FadeIn, ScrollReveal, StaggerList, StaggerItem, CountUp, HoverCard } from "@/components/ui/motion";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import * as directApi from "@/lib/api";
import type { WalletTransaction } from "@/lib/api";
import { toast } from "sonner";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Cell, Legend,
} from "recharts";
import {
  BarChart3, RefreshCw, Download, TrendingUp, Clock, Cpu, DollarSign, Zap, Gauge, Wallet,
} from "lucide-react";

const COLORS = ["#dc2626", "#f59e0b", "#38bdf8", "#10b981", "#8b5cf6", "#ec4899"];
const RANGE_PRESETS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "YTD", days: 0 },
];

const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

type AnalyticsRow = {
  period: string;
  job_count: number;
  total_cost_cad: number;
  total_gpu_hours: number;
  avg_gpu_utilization_pct: number;
  canadian_jobs: number;
  international_jobs: number;
};


type EnhancedAnalytics = {
  avg_cost_per_hour_trend?: Array<{ date: string; cost_per_hour: number }>;
  cumulative_spend?: Array<{ date: string; running_total: number }>;
  duration_histogram?: Array<{ bucket: string; count: number }>;
  job_status_breakdown?: { completed: number; failed: number; cancelled: number };
  top_hosts_used?: Array<{ host_id: string; job_count: number; total_cost: number }>;
  daily_gpu_hours?: Array<{ date: string; hours: number }>;
};

function asTrend(current: number, previous: number): { trend?: "up" | "down" | "flat"; trendValue?: string } {
  if (!previous) return {};
  const pct = ((current - previous) / previous) * 100;
  if (pct === 0) return { trend: "flat", trendValue: "0.0%" };
  return {
    trend: pct > 0 ? "up" : "down",
    trendValue: `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`,
  };
}

function toCurrency(v: number) {
  return `$${Number(v || 0).toFixed(2)}`;
}

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [gpuBreakdown, setGpuBreakdown] = useState<any[]>([]);
  const [provinceBreakdown, setProvinceBreakdown] = useState<any[]>([]);
  const [previousSummary, setPreviousSummary] = useState<any | null>(null);
  const [enhanced, setEnhanced] = useState<EnhancedAnalytics | null>(null);
  const [walletTransactions, setWalletTransactions] = useState<WalletTransaction[]>([]);
  const [spotPrices, setSpotPrices] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState(30);
  const api = useApi();
  const { t } = useLocale();

  const load = useCallback(async () => {
    setLoading(true);
    const days = range === 0
      ? Math.ceil((Date.now() - new Date(new Date().getFullYear(), 0, 1).getTime()) / 86400000)
      : range;
    try {
      const [usageRes, gpuRes, provinceRes, prevWindowRes, enhancedRes, meRes, spotRes] = await Promise.all([
        api.fetchAnalytics({ days: String(days), group_by: "day" }),
        api.fetchAnalytics({ days: String(days), group_by: "gpu_model" }),
        api.fetchAnalytics({ days: String(days), group_by: "province" }),
        api.fetchAnalytics({ days: String(days), group_by: "day", offset_days: String(days) }),
        api.fetchEnhancedAnalytics({ days: String(days) }).catch(() => null),
        directApi.getMe().catch(() => null),
        api.fetchSpotPrices().catch(() => null),
      ]);

      let walletHistory: { transactions: WalletTransaction[] } | null = null;
      const customerId = meRes?.user?.customer_id || meRes?.user?.user_id;
      if (customerId) {
        walletHistory = await api.fetchWalletHistory(customerId, 40).catch(() => null);
      }

      setData(usageRes);
      setGpuBreakdown(((gpuRes as any)?.analytics ?? []) as any[]);
      setProvinceBreakdown(((provinceRes as any)?.analytics ?? []) as any[]);
      setPreviousSummary((prevWindowRes as any)?.summary ?? null);
      setEnhanced((enhancedRes as any)?.ok ? (enhancedRes as EnhancedAnalytics) : null);
      setWalletTransactions(walletHistory?.transactions ?? []);
      setSpotPrices(((spotRes as any)?.spot_prices || (spotRes as any)?.prices || {}) as Record<string, number>);
    } catch {
      toast.error("Failed to load analytics");
    } finally {
      setLoading(false);
    }
  }, [api, range]);

  useEffect(() => { load(); }, [load]);

  const analytics: AnalyticsRow[] = data?.analytics || [];
  const summary = data?.summary || {};
  const hasData = analytics.length > 0 || (summary.total_jobs != null && summary.total_jobs > 0);

  const jobsOverTime = analytics.map((r) => ({ date: r.period, count: r.job_count }));
  const spendOverTime = analytics.map((r) => ({ date: r.period, amount: r.total_cost_cad }));
  const utilOverTime = analytics.map((r) => ({ date: r.period, util: Number(r.avg_gpu_utilization_pct ?? 0) }));
  const sovereigntyMixOverTime = analytics.map((r) => ({
    date: r.period,
    canadian: Number(r.canadian_jobs ?? 0),
    international: Number(r.international_jobs ?? 0),
  }));

  const topGpuSeries = gpuBreakdown
    .map((r: any) => ({
      name: String(r.period || "Unknown"),
      spend: Number(r.total_cost_cad ?? 0),
      jobs: Number(r.job_count ?? 0),
      hours: Number(r.total_gpu_hours ?? 0),
    }))
    .sort((a, b) => b.spend - a.spend)
    .slice(0, 6);

  const provinceSeries = provinceBreakdown
    .map((r: any) => ({
      name: String(r.period || "Unknown"),
      value: Number(r.total_cost_cad ?? 0),
      jobs: Number(r.job_count ?? 0),
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 6);

  const totalJobs = Number(summary.total_jobs ?? 0);
  const totalSpend = Number(summary.total_spend_cad ?? 0);
  const totalGpuHours = Number(summary.total_gpu_hours ?? 0);
  const avgUtil = Number(summary.avg_gpu_utilization_pct ?? 0);

  const avgCostPerJob = totalJobs > 0 ? totalSpend / totalJobs : 0;
  const avgJobDurationHours = totalJobs > 0 ? totalGpuHours / totalJobs : 0;
  const canadianJobs = sovereigntyMixOverTime.reduce((sum, item) => sum + item.canadian, 0);
  const canadianPct = totalJobs > 0 ? (canadianJobs / totalJobs) * 100 : 0;
  const topGpuModel = topGpuSeries[0]?.name || "—";

  const cumulativeSpendDerived = useMemo(() => {
    let running = 0;
    return spendOverTime.map((item) => {
      running += Number(item.amount || 0);
      return { date: item.date, running_total: Number(running.toFixed(2)) };
    });
  }, [spendOverTime]);

  const costPerHourDerived = useMemo(
    () => analytics.map((r) => ({
      date: r.period,
      cost_per_hour: Number(r.total_gpu_hours) > 0 ? Number(r.total_cost_cad) / Number(r.total_gpu_hours) : 0,
    })),
    [analytics],
  );

  const walletChart = walletTransactions
    .map((tx) => ({
      date: String(tx.created_at).slice(0, 10),
      deposits: tx.amount_cad > 0 ? tx.amount_cad : 0,
      charges: tx.amount_cad < 0 ? Math.abs(tx.amount_cad) : 0,
      tx_type: tx.type,
    }))
    .reverse();

  const walletKpis = walletTransactions.reduce(
    (acc, tx) => {
      if (tx.amount_cad > 0) acc.deposited += tx.amount_cad;
      if (tx.amount_cad < 0) acc.spent += Math.abs(tx.amount_cad);
      acc.balance += tx.amount_cad;
      return acc;
    },
    { balance: 0, deposited: 0, spent: 0 },
  );

  const spotSeries = Object.entries(spotPrices)
    .map(([model, price]) => ({ model, price: Number(price ?? 0) }))
    .sort((a, b) => b.price - a.price)
    .slice(0, 10);

  const durationHistogram = enhanced?.duration_histogram ?? [];
  const gpuHoursDaily = enhanced?.daily_gpu_hours ?? analytics.map((r) => ({ date: r.period, hours: r.total_gpu_hours }));
  const costPerHourSeries = enhanced?.avg_cost_per_hour_trend ?? costPerHourDerived;
  const cumulativeSpendSeries = enhanced?.cumulative_spend ?? cumulativeSpendDerived;
  const topHosts = enhanced?.top_hosts_used ?? [];
  const jobStatusBreakdown = enhanced?.job_status_breakdown ?? { completed: 0, failed: 0, cancelled: 0 };

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">{t("dash.analytics.title")}</h1>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-surface p-1">
            {RANGE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setRange(p.days)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  range === p.days ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" onClick={() => {
            if (!analytics.length) { toast.error("No data to export"); return; }
            const keys = Object.keys(analytics[0]);
            const csv = [keys.join(","), ...analytics.map((r: Record<string, unknown>) => keys.map((k) => `"${r[k] ?? ""}"`).join(","))].join("\n");
            const blob = new Blob([csv], { type: "text/csv" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a"); a.href = url; a.download = `analytics-${range}d.csv`; a.click();
            URL.revokeObjectURL(url);
          }}>
            <Download className="h-3.5 w-3.5" /> {t("dash.analytics.csv")}
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {[...Array(4)].map((_, i) => <div key={i} className="h-64 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
        </div>
      ) : !hasData ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
            <BarChart3 className="h-10 w-10 text-text-muted" />
          </div>
          <h3 className="text-xl font-semibold mb-2">{t("dash.analytics.empty")}</h3>
          <p className="text-sm text-text-secondary max-w-md text-center mb-6">
            {t("dash.analytics.empty_desc")}
          </p>
        </div>
      ) : (
        <>
          <section className="space-y-3">
            <div>
              <h2 className="text-lg font-semibold">Overview</h2>
              <p className="text-sm text-text-secondary">Core usage KPIs and period-over-period deltas.</p>
            </div>
            <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <StaggerItem>
                <StatCard label="Total Jobs" value={<CountUp value={totalJobs} />} icon={Zap} glow="cyan" {...asTrend(totalJobs, Number(previousSummary?.total_jobs ?? 0))} />
              </StaggerItem>
              <StaggerItem>
                <StatCard label="Total Spend" value={<CountUp value={totalSpend} prefix="$" />} icon={DollarSign} glow="gold" {...asTrend(totalSpend, Number(previousSummary?.total_spend_cad ?? 0))} />
              </StaggerItem>
              <StaggerItem>
                <StatCard label="GPU Hours" value={<CountUp value={totalGpuHours} />} icon={Clock} glow="emerald" {...asTrend(totalGpuHours, Number(previousSummary?.total_gpu_hours ?? 0))} />
              </StaggerItem>
              <StaggerItem>
                <StatCard label="Avg Utilization" value={<CountUp value={avgUtil} suffix="%" />} icon={Gauge} glow="violet" />
              </StaggerItem>
            </StaggerList>
            <FadeIn delay={0.12}>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <StatCard label="Avg Cost / Job" value={toCurrency(avgCostPerJob)} icon={DollarSign} />
                <StatCard label="Avg Job Duration" value={`${avgJobDurationHours.toFixed(2)} h`} icon={Clock} />
                <StatCard label="Canadian Compute" value={`${canadianPct.toFixed(1)}%`} icon={TrendingUp} />
                <StatCard label="Top GPU Model" value={topGpuModel} icon={Cpu} />
              </div>
            </FadeIn>
          </section>

          <FadeIn delay={0.2}>
            <section className="space-y-3">
              <div>
                <h2 className="text-lg font-semibold">Compute Trends</h2>
                <p className="text-sm text-text-secondary">Job volume, spend, and utilization over time.</p>
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {jobsOverTime.length > 0 && (
                  <HoverCard>
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_jobs")}</CardTitle></CardHeader>
                      <CardContent><div className="h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={jobsOverTime}><defs><linearGradient id="jobsGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#38bdf8" stopOpacity={0.35} /><stop offset="95%" stopColor="#38bdf8" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} /><Area type="monotone" dataKey="count" stroke="#38bdf8" fill="url(#jobsGrad)" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}

                {spendOverTime.length > 0 && (
                  <HoverCard>
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_spend")}</CardTitle></CardHeader>
                      <CardContent><div className="h-56"><ResponsiveContainer width="100%" height="100%"><BarChart data={spendOverTime}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [toCurrency(Number(value)), "Spend"]} /><Bar dataKey="amount" fill="#f59e0b" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}

                {utilOverTime.length > 0 && (
                  <HoverCard>
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">GPU Utilization Trend</CardTitle></CardHeader>
                      <CardContent><div className="h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={utilOverTime}><defs><linearGradient id="utilGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#10b981" stopOpacity={0.35} /><stop offset="95%" stopColor="#10b981" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" domain={[0, 100]} /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${Number(value).toFixed(1)}%`, "Avg Util"]} /><Area type="monotone" dataKey="util" stroke="#10b981" fill="url(#utilGrad)" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}

                {sovereigntyMixOverTime.length > 0 && (
                  <HoverCard>
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">Canadian vs International Jobs</CardTitle></CardHeader>
                      <CardContent><div className="h-56"><ResponsiveContainer width="100%" height="100%"><BarChart data={sovereigntyMixOverTime}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} /><Legend /><Bar stackId="jobs" dataKey="canadian" name="Canadian" fill="#38bdf8" radius={[3, 3, 0, 0]} /><Bar stackId="jobs" dataKey="international" name="International" fill="#f59e0b" radius={[3, 3, 0, 0]} /></BarChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}
              </div>
            </section>
          </FadeIn>

          <ScrollReveal>
            <section className="space-y-3">
              <div>
                <h2 className="text-lg font-semibold">Resource Breakdown</h2>
                <p className="text-sm text-text-secondary">GPU models, provinces, and job-level rollup.</p>
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {topGpuSeries.length > 0 && (
                  <HoverCard>
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">Top GPU Models by Spend</CardTitle></CardHeader>
                      <CardContent><div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={topGpuSeries} layout="vertical"><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis type="category" dataKey="name" tick={{ fill: "#94a3b8", fontSize: 11 }} width={120} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value, key) => key === "spend" ? [toCurrency(Number(value)), "Spend"] : [value, String(key)]} /><Bar dataKey="spend" fill="#8b5cf6" radius={[0, 4, 4, 0]} /></BarChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}

                {provinceSeries.length > 0 && (
                  <HoverCard>
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">Spend by Province</CardTitle></CardHeader>
                      <CardContent><div className="h-64"><ResponsiveContainer width="100%" height="100%"><PieChart><Pie data={provinceSeries} dataKey="value" nameKey="name" outerRadius={95} innerRadius={50} paddingAngle={2}>{provinceSeries.map((entry, index) => <Cell key={`province-${entry.name}`} fill={COLORS[index % COLORS.length]} />)}</Pie><Tooltip contentStyle={tooltipStyle} formatter={(value) => [toCurrency(Number(value)), "Spend"]} /><Legend /></PieChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}
              </div>
              {analytics.length > 0 && (
                <Card>
                  <CardHeader><CardTitle className="text-sm">{t("dash.analytics.breakdown")}</CardTitle></CardHeader>
                  <CardContent>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b border-border text-left">
                            <th className="py-2 pr-4 font-medium">{t("dash.analytics.col_category")}</th>
                            <th className="py-2 pr-4 font-medium text-right">{t("dash.analytics.col_jobs")}</th>
                            <th className="py-2 pr-4 font-medium text-right">{t("dash.analytics.col_hours")}</th>
                            <th className="py-2 font-medium text-right">{t("dash.analytics.col_spend")}</th>
                          </tr>
                        </thead>
                        <tbody>
                          {analytics.map((row, i) => (
                            <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                              <td className="py-2 pr-4 font-medium">{row.period || "—"}</td>
                              <td className="py-2 pr-4 text-right font-mono text-xs">{row.job_count ?? "—"}</td>
                              <td className="py-2 pr-4 text-right font-mono text-xs">{row.total_gpu_hours?.toFixed(1) ?? "—"}</td>
                              <td className="py-2 text-right font-mono text-xs">{row.total_cost_cad != null ? toCurrency(Number(row.total_cost_cad)) : "—"}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              )}
            </section>
          </ScrollReveal>

          <ScrollReveal>
            <section className="space-y-3">
              <div>
                <h2 className="text-lg font-semibold">Financial</h2>
                <p className="text-sm text-text-secondary">Wallet activity, market spot pricing, and cost efficiency trends.</p>
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">Wallet Activity</CardTitle></CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                        <StatCard label="Current Balance" value={toCurrency(walletKpis.balance)} icon={Wallet} />
                        <StatCard label="Total Deposited" value={toCurrency(walletKpis.deposited)} icon={TrendingUp} />
                        <StatCard label="Total Spent" value={toCurrency(walletKpis.spent)} icon={DollarSign} />
                      </div>
                      {walletChart.length > 0 ? (
                        <div className="h-56"><ResponsiveContainer width="100%" height="100%"><BarChart data={walletChart}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [toCurrency(Number(value)), "Amount"]} /><Legend /><Bar dataKey="deposits" fill="#10b981" stackId="wallet" /><Bar dataKey="charges" fill="#f59e0b" stackId="wallet" /></BarChart></ResponsiveContainer></div>
                      ) : (
                        <p className="text-sm text-text-muted py-6 text-center">No wallet transactions in this period.</p>
                      )}
                    </CardContent>
                  </Card>
                </HoverCard>

                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">Spot Price Overview</CardTitle></CardHeader>
                    <CardContent>
                      {spotSeries.length > 0 ? (
                        <div className="h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={spotSeries} layout="vertical"><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis type="category" dataKey="model" tick={{ fill: "#94a3b8", fontSize: 11 }} width={140} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [`$${Number(value).toFixed(4)}/hr`, "Spot Price"]} /><Bar dataKey="price" fill="#38bdf8" radius={[0, 4, 4, 0]} /></BarChart></ResponsiveContainer></div>
                      ) : (
                        <p className="text-sm text-text-muted py-6 text-center">No spot price data available.</p>
                      )}
                    </CardContent>
                  </Card>
                </HoverCard>
              </div>

              <div className="grid grid-cols-1 gap-6">
                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">Cost per GPU Hour Trend</CardTitle></CardHeader>
                    <CardContent><div className="h-60"><ResponsiveContainer width="100%" height="100%"><AreaChart data={costPerHourSeries}><defs><linearGradient id="costHourGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.35} /><stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [`$${Number(value).toFixed(2)}/GPUh`, "Cost"]} /><Area dataKey="cost_per_hour" stroke="#8b5cf6" fill="url(#costHourGrad)" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></CardContent>
                  </Card>
                </HoverCard>

                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">Cumulative Spend</CardTitle></CardHeader>
                    <CardContent><div className="h-60"><ResponsiveContainer width="100%" height="100%"><AreaChart data={cumulativeSpendSeries}><defs><linearGradient id="cumSpendGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#f59e0b" stopOpacity={0.35} /><stop offset="95%" stopColor="#f59e0b" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [toCurrency(Number(value)), "Running Total"]} /><Area dataKey="running_total" stroke="#f59e0b" fill="url(#cumSpendGrad)" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></CardContent>
                  </Card>
                </HoverCard>
              </div>
            </section>
          </ScrollReveal>

          <ScrollReveal>
            <section className="space-y-3">
              <div>
                <h2 className="text-lg font-semibold">Advanced Insights</h2>
                <p className="text-sm text-text-secondary">Duration distribution, daily GPU hours, and host usage concentration.</p>
              </div>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">Job Status Breakdown</CardTitle></CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                        <StatCard label="Completed" value={jobStatusBreakdown.completed} icon={TrendingUp} />
                        <StatCard label="Failed" value={jobStatusBreakdown.failed} icon={BarChart3} />
                        <StatCard label="Cancelled" value={jobStatusBreakdown.cancelled} icon={Clock} />
                      </div>
                    </CardContent>
                  </Card>
                </HoverCard>
                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">Job Duration Distribution</CardTitle></CardHeader>
                    <CardContent><div className="h-56"><ResponsiveContainer width="100%" height="100%"><BarChart data={durationHistogram}><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="bucket" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} /><Bar dataKey="count" fill="#10b981" radius={[4, 4, 0, 0]} /></BarChart></ResponsiveContainer></div></CardContent>
                  </Card>
                </HoverCard>
                <HoverCard>
                  <Card className="glow-card brand-top-accent">
                    <CardHeader><CardTitle className="text-sm">GPU Hours Over Time</CardTitle></CardHeader>
                    <CardContent><div className="h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={gpuHoursDaily}><defs><linearGradient id="gpuHoursGrad" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35} /><stop offset="95%" stopColor="#06b6d4" stopOpacity={0} /></linearGradient></defs><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} formatter={(value) => [`${Number(value).toFixed(2)} GPUh`, "Hours"]} /><Area dataKey="hours" stroke="#06b6d4" fill="url(#gpuHoursGrad)" strokeWidth={2} /></AreaChart></ResponsiveContainer></div></CardContent>
                  </Card>
                </HoverCard>
                {topHosts.length > 0 && (
                  <HoverCard className="lg:col-span-2">
                    <Card className="glow-card brand-top-accent">
                      <CardHeader><CardTitle className="text-sm">Top Hosts Used</CardTitle></CardHeader>
                      <CardContent><div className="h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={topHosts} layout="vertical"><CartesianGrid strokeDasharray="3 3" stroke="#1e293b" /><XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><YAxis type="category" dataKey="host_id" width={180} tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" /><Tooltip contentStyle={tooltipStyle} /><Legend /><Bar dataKey="job_count" fill="#8b5cf6" name="Jobs" /><Bar dataKey="total_cost" fill="#f59e0b" name="Cost" /></BarChart></ResponsiveContainer></div></CardContent>
                    </Card>
                  </HoverCard>
                )}
              </div>
            </section>
          </ScrollReveal>
        </>
      )}
    </div>
  );
}
