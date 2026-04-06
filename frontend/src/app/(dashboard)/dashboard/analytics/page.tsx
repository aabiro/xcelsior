"use client";

import { useEffect, useState, useCallback, useMemo, useRef } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { FadeIn, StaggerList, StaggerItem, CountUp, ScrollReveal } from "@/components/ui/motion";
import { cn } from "@/lib/utils";
import {
  BarChart3, RefreshCw, Download, TrendingUp, Clock, Cpu, Zap,
  DollarSign, Gauge, Server, Globe, Wallet, ArrowUpRight,
  Activity, Shield, ChartArea, LayoutGrid, Sparkles, AlertTriangle, Brain,
} from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import type { EnhancedAnalytics } from "@/lib/api";
import {
  SpendTrendChart, JobsTrendChart, UtilizationChart,
  CumulativeSpendChart, CostPerHourChart, GpuHoursChart,
  DurationHistogramChart, SovereigntyChart, TopGpuChart,
  ProvinceDonutChart, GpuPerformanceRadar, HourlyHeatmap,
  ProviderRevenueTrendChart, WalletActivityChart,
  TopEntitiesTable, GpuPerformanceTable, PeakDaysCards,
  InsightCards, Sparkline,
} from "./charts";
import type { Insight } from "./charts";
import { AnalyticsAiPanel } from "./analytics-ai-panel";

// ── Constants ──────────────────────────────────────────────────────────

const RANGE_PRESETS = [
  { label: "7d", days: 7 },
  { label: "14d", days: 14 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "YTD", days: 0 },
];

type Tab = "overview" | "compute" | "financial" | "provider";

const TABS: { key: Tab; label: string; icon: React.ElementType; description: string; providerOnly?: boolean }[] = [
  { key: "overview", label: "Overview", icon: LayoutGrid, description: "Key metrics at a glance" },
  { key: "compute", label: "Compute", icon: Cpu, description: "GPU usage & performance" },
  { key: "financial", label: "Financial", icon: DollarSign, description: "Spend, efficiency & wallet" },
  { key: "provider", label: "Provider", icon: Server, description: "Host earnings & uptime", providerOnly: true },
];

// ── Auto-Insights Engine ───────────────────────────────────────────────

function generateInsights(
  analytics: any[],
  summary: any,
  previousSummary: any,
  enhanced: EnhancedAnalytics | null,
): Insight[] {
  const insights: Insight[] = [];
  if (!analytics?.length) return insights;

  const totalJobs = Number(summary?.total_jobs ?? 0);
  const prevJobs = Number(previousSummary?.total_jobs ?? 0);
  const totalSpend = Number(summary?.total_spend_cad ?? 0);
  const prevSpend = Number(previousSummary?.total_spend_cad ?? 0);
  const avgUtil = Number(summary?.avg_gpu_utilization_pct ?? 0);
  const prevUtil = Number(previousSummary?.avg_gpu_utilization_pct ?? 0);
  const totalGpuHrs = Number(summary?.total_gpu_hours ?? 0);
  const prevGpuHrs = Number(previousSummary?.total_gpu_hours ?? 0);

  // Job trend
  if (prevJobs > 0 && totalJobs > 0) {
    const pct = ((totalJobs - prevJobs) / prevJobs) * 100;
    if (Math.abs(pct) >= 10) {
      insights.push({
        type: pct > 0 ? "positive" : "negative",
        title: pct > 0 ? "Job volume surging" : "Job volume declining",
        detail: `${Math.abs(pct).toFixed(0)}% ${pct > 0 ? "more" : "fewer"} jobs compared to the previous period.`,
        metric: `${totalJobs} jobs (was ${prevJobs})`,
      });
    }
  }

  // Spend trend
  if (prevSpend > 0 && totalSpend > 0) {
    const pct = ((totalSpend - prevSpend) / prevSpend) * 100;
    if (Math.abs(pct) >= 15) {
      insights.push({
        type: pct > 0 ? "negative" : "positive",
        title: pct > 0 ? "Spend increasing" : "Spend optimized",
        detail: `Your spend is ${Math.abs(pct).toFixed(0)}% ${pct > 0 ? "higher" : "lower"} than the previous period.`,
        metric: `$${totalSpend.toFixed(2)} (was $${prevSpend.toFixed(2)})`,
      });
    }
  }

  // Utilization alert
  if (avgUtil > 0 && avgUtil < 40) {
    insights.push({
      type: "negative",
      title: "Low GPU utilization",
      detail: "Average GPU utilization is below 40%. Consider using smaller instances or optimizing your workloads.",
      metric: `${avgUtil.toFixed(1)}% avg`,
    });
  } else if (avgUtil >= 85) {
    insights.push({
      type: "positive",
      title: "Excellent GPU utilization",
      detail: "You're making great use of your GPU resources with very high utilization.",
      metric: `${avgUtil.toFixed(1)}% avg`,
    });
  }

  // Utilization trend (compare periods)
  if (prevUtil > 0 && avgUtil > 0) {
    const pct = ((avgUtil - prevUtil) / prevUtil) * 100;
    if (pct >= 15) {
      insights.push({
        type: "positive",
        title: "Utilization improving",
        detail: `GPU utilization rose ${pct.toFixed(0)}% — your workloads are using hardware more effectively.`,
        metric: `${avgUtil.toFixed(1)}% (was ${prevUtil.toFixed(1)}%)`,
      });
    } else if (pct <= -15) {
      insights.push({
        type: "negative",
        title: "Utilization declining",
        detail: `GPU utilization dropped ${Math.abs(pct).toFixed(0)}% — you may be over-provisioning resources.`,
        metric: `${avgUtil.toFixed(1)}% (was ${prevUtil.toFixed(1)}%)`,
      });
    }
  }

  // GPU hours trend
  if (prevGpuHrs > 0 && totalGpuHrs > 0) {
    const pct = ((totalGpuHrs - prevGpuHrs) / prevGpuHrs) * 100;
    if (pct >= 25) {
      insights.push({
        type: "info",
        title: "GPU hours ramping up",
        detail: `You consumed ${pct.toFixed(0)}% more GPU hours this period — workloads are scaling up.`,
        metric: `${totalGpuHrs.toFixed(1)}h (was ${prevGpuHrs.toFixed(1)}h)`,
      });
    }
  }

  // Data sovereignty
  const caPct = enhanced?.sovereignty?.canadian_pct ?? 0;
  if (caPct > 0 && caPct >= 90) {
    insights.push({
      type: "positive",
      title: "Strong Canadian sovereignty",
      detail: `${caPct.toFixed(0)}% of your compute runs on Canadian infrastructure.`,
      metric: `${caPct.toFixed(0)}% Canadian`,
    });
  } else if (caPct > 0 && caPct < 50) {
    insights.push({
      type: "info",
      title: "Mostly international compute",
      detail: `Only ${caPct.toFixed(0)}% of jobs run on Canadian GPUs. Consider Canadian hosts for data sovereignty.`,
      metric: `${caPct.toFixed(0)}% Canadian`,
    });
  }

  // Peak usage pattern
  const peakDays = enhanced?.peak_days ?? [];
  if (peakDays.length > 0) {
    const peak = peakDays[0];
    insights.push({
      type: "info",
      title: "Peak usage detected",
      detail: `Your busiest day had ${peak.jobs} jobs consuming ${peak.gpu_hours.toFixed(1)}h of GPU time.`,
      metric: `$${peak.spend.toFixed(2)} on peak day`,
    });
  }

  // Cost efficiency
  if (totalJobs > 0 && totalSpend > 0) {
    const costPerJob = totalSpend / totalJobs;
    if (costPerJob < 1.0) {
      insights.push({
        type: "positive",
        title: "Efficient job costs",
        detail: `Your average cost per job is under $1, indicating efficient resource usage.`,
        metric: `$${costPerJob.toFixed(2)}/job`,
      });
    } else if (costPerJob > 10.0) {
      insights.push({
        type: "info",
        title: "High cost per job",
        detail: `Average job cost is $${costPerJob.toFixed(2)} — consider shorter runs or lower-tier GPUs for test jobs.`,
        metric: `$${costPerJob.toFixed(2)}/job`,
      });
    }
  }

  // GPU model efficiency (best value)
  const gpuPerf = enhanced?.gpu_performance ?? [];
  if (gpuPerf.length >= 2) {
    const withEfficiency = gpuPerf
      .filter(g => g.avg_cost_per_hour > 0 && g.avg_util > 0)
      .map(g => ({ ...g, efficiency: g.avg_util / g.avg_cost_per_hour }))
      .sort((a, b) => b.efficiency - a.efficiency);
    if (withEfficiency.length >= 2) {
      const best = withEfficiency[0];
      insights.push({
        type: "positive",
        title: `${best.gpu_model} is your best value`,
        detail: `Highest efficiency score across your GPU models — ${best.avg_util.toFixed(0)}% utilization at $${best.avg_cost_per_hour.toFixed(2)}/hr.`,
        metric: `${best.jobs} jobs, ${best.gpu_hours.toFixed(1)}h`,
      });
    }
  }

  // Cost per hour trend direction
  const cphTrend = enhanced?.cost_per_hour_trend ?? [];
  if (cphTrend.length >= 7) {
    const rates = cphTrend.map(d => d.cost_per_hour).filter(v => v > 0);
    if (rates.length >= 7) {
      const half = Math.floor(rates.length / 2);
      const firstHalf = rates.slice(0, half).reduce((s, v) => s + v, 0) / half;
      const secondHalf = rates.slice(half).reduce((s, v) => s + v, 0) / (rates.length - half);
      const changePct = ((secondHalf - firstHalf) / firstHalf) * 100;
      if (changePct <= -10) {
        insights.push({
          type: "positive",
          title: "Cost per hour dropping",
          detail: `Your cost per GPU hour decreased ${Math.abs(changePct).toFixed(0)}% over the period — good cost discipline.`,
          metric: `$${secondHalf.toFixed(3)}/hr (was $${firstHalf.toFixed(3)}/hr)`,
        });
      } else if (changePct >= 15) {
        insights.push({
          type: "negative",
          title: "Cost per hour rising",
          detail: `GPU hour costs increased ${changePct.toFixed(0)}% — you may be using pricier models or shorter, less efficient jobs.`,
          metric: `$${secondHalf.toFixed(3)}/hr (was $${firstHalf.toFixed(3)}/hr)`,
        });
      }
    }
  }

  // Heatmap — busiest time slot
  const heatmap = enhanced?.hourly_heatmap ?? [];
  if (heatmap.length > 0) {
    const peak = heatmap.reduce((a, b) => a.count > b.count ? a : b);
    const dowNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
    if (peak.count >= 3) {
      insights.push({
        type: "info",
        title: `Most active: ${dowNames[peak.dow]}s at ${peak.hour}:00`,
        detail: `Your peak activity slot — ${peak.count} jobs typically run at this time.`,
        metric: `${dowNames[peak.dow]} ${peak.hour}:00`,
      });
    }
  }

  return insights.slice(0, 6);
}

// ── Main Page ──────────────────────────────────────────────────────────

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [enhanced, setEnhanced] = useState<EnhancedAnalytics | null>(null);
  const [gpuBreakdown, setGpuBreakdown] = useState<any[]>([]);
  const [provinceBreakdown, setProvinceBreakdown] = useState<any[]>([]);
  const [previousSummary, setPreviousSummary] = useState<any | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [range, setRange] = useState(30);
  const [tab, setTab] = useState<Tab>("overview");
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [aiOpen, setAiOpen] = useState(false);
  const sseDebounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const api = useApi();
  const { user } = useAuth();
  const { t } = useLocale();

  const isProvider = !!user?.provider_id;
  const isAdmin = !!user?.is_admin;

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    const days = range === 0
      ? Math.ceil((Date.now() - new Date(new Date().getFullYear(), 0, 1).getTime()) / 86400000)
      : range;
    try {
      const results = await Promise.allSettled([
        api.fetchAnalytics({ days: String(days), group_by: "day" }),
        api.fetchAnalytics({ days: String(days), group_by: "gpu_model" }),
        api.fetchAnalytics({ days: String(days), group_by: "province" }),
        api.fetchAnalytics({ days: String(days), group_by: "day", offset_days: String(days) }),
        api.fetchEnhancedAnalytics(days),
      ]);
      const [usageRes, gpuRes, provinceRes, prevWindowRes, enhancedRes] = results;
      if (usageRes.status === "fulfilled") setData(usageRes.value);
      if (enhancedRes.status === "fulfilled") setEnhanced(enhancedRes.value as EnhancedAnalytics);
      setGpuBreakdown(gpuRes.status === "fulfilled" ? (((gpuRes.value as any)?.analytics ?? []) as any[]) : []);
      setProvinceBreakdown(provinceRes.status === "fulfilled" ? (((provinceRes.value as any)?.analytics ?? []) as any[]) : []);
      setPreviousSummary(prevWindowRes.status === "fulfilled" ? ((prevWindowRes.value as any)?.summary ?? null) : null);
      setLastUpdated(new Date());
      // Show warning if any failed but not all
      const failures = results.filter(r => r.status === "rejected");
      if (failures.length > 0 && failures.length < results.length) {
        toast.warning("Some analytics data couldn't be loaded");
      } else if (failures.length === results.length) {
        setLoadError("Failed to load analytics");
        toast.error("Failed to load analytics");
      }
    } catch (err: any) {
      const msg = err?.message || "Failed to load analytics";
      setLoadError(msg);
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  }, [api, range]);

  useEffect(() => { load(); }, [load]);

  // Debounced SSE handler — avoid flooding on rapid events
  const debouncedLoad = useCallback(() => {
    if (sseDebounceRef.current) clearTimeout(sseDebounceRef.current);
    sseDebounceRef.current = setTimeout(() => { load(); }, 3000);
  }, [load]);

  useEventStream({
    eventTypes: ["job_status", "job_submitted", "usage_recorded"],
    onEvent: debouncedLoad,
  });

  // Keyboard navigation for tabs
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const idx = availableTabs.findIndex(t => t.key === tab);
        if (e.key === "ArrowLeft" && idx > 0) setTab(availableTabs[idx - 1].key);
        if (e.key === "ArrowRight" && idx < availableTabs.length - 1) setTab(availableTabs[idx + 1].key);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  // ── Derived data ──────────────────────────────────────────────────

  const analytics: any[] = data?.analytics || [];
  const summary = data?.summary || {};
  const hasData = analytics.length > 0 || (summary.total_jobs != null && summary.total_jobs > 0);

  const jobsOverTime = useMemo(() => analytics.map((r: any) => ({ date: r.period, count: r.job_count })), [analytics]);
  const spendOverTime = useMemo(() => analytics.map((r: any) => ({ date: r.period, spend: r.total_cost_cad })), [analytics]);
  const utilOverTime = useMemo(() => analytics.map((r: any) => ({ date: r.period, util: Number(r.avg_gpu_utilization_pct ?? 0) })), [analytics]);
  const sovereigntyOverTime = useMemo(() => analytics.map((r: any) => ({
    date: r.period,
    canadian: Number(r.canadian_jobs ?? 0),
    international: Number(r.international_jobs ?? 0),
  })), [analytics]);

  const topGpuSeries = useMemo(() => gpuBreakdown
    .map((r: any) => ({ name: String(r.period || "Unknown"), spend: Number(r.total_cost_cad ?? 0), jobs: Number(r.job_count ?? 0), hours: Number(r.total_gpu_hours ?? 0) }))
    .sort((a, b) => b.spend - a.spend).slice(0, 8), [gpuBreakdown]);

  const provinceSeries = useMemo(() => provinceBreakdown
    .map((r: any) => ({ name: String(r.period || "Unknown"), value: Number(r.total_cost_cad ?? 0) }))
    .sort((a, b) => b.value - a.value).slice(0, 8), [provinceBreakdown]);

  // Sparkline data extracted from time series
  const jobSparkline = useMemo(() => jobsOverTime.map(d => d.count), [jobsOverTime]);
  const spendSparkline = useMemo(() => spendOverTime.map(d => d.spend), [spendOverTime]);
  const utilSparkline = useMemo(() => utilOverTime.map(d => d.util), [utilOverTime]);
  const gpuHoursSparkline = useMemo(
    () => (enhanced?.daily_gpu_hours ?? []).map((d: any) => d.hours),
    [enhanced?.daily_gpu_hours],
  );

  // ── Auto-Insights ─────────────────────────────────────────────────

  const insights = useMemo(
    () => generateInsights(analytics, summary, previousSummary, enhanced),
    [analytics, summary, previousSummary, enhanced],
  );

  // ── Deltas ────────────────────────────────────────────────────────

  const compare = (current: number, previous: number | null | undefined) => {
    if (previous === null || previous === undefined || previous === 0) return null;
    return Math.round(((current - previous) / previous) * 1000) / 10;
  };

  const jobsDelta = compare(Number(summary.total_jobs ?? 0), Number(previousSummary?.total_jobs ?? 0));
  const spendDelta = compare(Number(summary.total_spend_cad ?? 0), Number(previousSummary?.total_spend_cad ?? 0));
  const gpuHoursDelta = compare(Number(summary.total_gpu_hours ?? 0), Number(previousSummary?.total_gpu_hours ?? 0));
  const utilDelta = compare(Number(summary.avg_gpu_utilization_pct ?? 0), Number(previousSummary?.avg_gpu_utilization_pct ?? 0));

  const trendDir = (pct: number | null) => pct == null || pct === 0 ? undefined : pct > 0 ? "up" as const : "down" as const;
  const trendVal = (pct: number | null) => pct == null || pct === 0 ? undefined : `${pct > 0 ? "+" : ""}${pct}%`;

  // Derived secondary KPIs
  const totalJobs = Number(summary.total_jobs ?? 0);
  const totalSpend = Number(summary.total_spend_cad ?? 0);
  const totalGpuHours = Number(summary.total_gpu_hours ?? 0);
  const avgCostPerJob = totalJobs > 0 ? totalSpend / totalJobs : 0;
  const avgJobDuration = totalJobs > 0 ? totalGpuHours / totalJobs : 0;
  const canadianPct = enhanced?.sovereignty?.canadian_pct ?? 0;
  const topGpu = topGpuSeries[0]?.name ?? "—";

  // CSV export of the analytics data
  const exportCsv = useCallback(() => {
    if (!analytics.length) { toast.error("No data to export"); return; }
    const keys = Object.keys(analytics[0]);
    const escapeCsvCell = (val: unknown) => {
      const s = String(val ?? "");
      // Prevent CSV formula injection: prefix formula characters with a single quote
      const safe = /^[=+\-@\t\r]/.test(s) ? `'${s}` : s;
      return `"${safe.replace(/"/g, '""')}"`;
    };
    const csv = [keys.join(","), ...analytics.map((r: Record<string, unknown>) => keys.map((k) => escapeCsvCell(r[k])).join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = `xcelsior-analytics-${range}d.csv`; a.click();
    URL.revokeObjectURL(url);
  }, [analytics, range]);

  // Available tabs based on user role
  const availableTabs = useMemo(() =>
    TABS.filter((t) => !t.providerOnly || isProvider || isAdmin),
    [isProvider, isAdmin],
  );

  // Format last updated time
  const lastUpdatedLabel = lastUpdated
    ? lastUpdated.toLocaleTimeString("en-CA", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
    : null;

  return (
    <div className="space-y-6">
      {/* ── Header ─────────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{t("dash.analytics.title")}</h1>
          <div className="flex items-center gap-3 mt-0.5">
            <p className="text-sm text-text-secondary">
              {isProvider ? "Your hosting & renting analytics" : "Your compute usage insights"}
            </p>
            {lastUpdatedLabel && !loading && (
              <span className="inline-flex items-center gap-1.5 text-[10px] text-text-muted font-mono">
                <span className="live-dot" />
                Updated {lastUpdatedLabel}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-surface p-1">
            {RANGE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setRange(p.days)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-all duration-200 ${
                  range === p.days
                    ? "bg-card text-text-primary shadow-sm ring-1 ring-accent-cyan/30"
                    : "text-text-muted hover:text-text-primary hover:bg-surface-hover"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" onClick={exportCsv}>
            <Download className="h-3.5 w-3.5" /> {t("dash.analytics.csv")}
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} /> {t("common.refresh")}
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setAiOpen(true)}
            className={cn(
              "relative gap-1.5 transition-all duration-300",
              aiOpen
                ? "ring-1 ring-accent-cyan/40 bg-accent-cyan/5 text-accent-cyan"
                : "hover:border-accent-cyan/30 hover:text-accent-cyan",
            )}
          >
            <Brain className="h-3.5 w-3.5" />
            Ask AI
            <span className="flex h-1.5 w-1.5 rounded-full bg-emerald-400 animate-pulse" />
          </Button>
        </div>
      </div>

      {/* ── Tab Navigation ─────────────────────────────────── */}
      <div className="relative flex items-center gap-1 border-b border-border overflow-x-auto pb-0">
        {availableTabs.map((t) => {
          const Icon = t.icon;
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              className={`
                group relative flex items-center gap-2 px-4 py-2.5 text-sm font-medium
                border-b-2 transition-all duration-200 whitespace-nowrap
                ${active
                  ? "border-accent-cyan text-text-primary"
                  : "border-transparent text-text-muted hover:text-text-secondary hover:border-border"
                }
              `}
            >
              <Icon className={`h-4 w-4 transition-colors ${active ? "text-accent-cyan" : "text-text-muted group-hover:text-text-secondary"}`} />
              {t.label}
              {active && (
                <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-accent-cyan tab-active-indicator" />
              )}
            </button>
          );
        })}
      </div>

      {/* ── Loading State (shimmer skeletons) ─────────────── */}
      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-28 rounded-xl analytics-skeleton" />
            ))}
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-20 rounded-xl analytics-skeleton" style={{ animationDelay: `${i * 0.15}s` }} />
            ))}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-64 rounded-xl analytics-skeleton" style={{ animationDelay: `${i * 0.1}s` }} />
            ))}
          </div>
        </div>

      /* ── Error State ───────────────────────────────────── */
      ) : loadError ? (
        <FadeIn>
          <div className="flex flex-col items-center justify-center py-16">
            <div className="relative flex h-20 w-20 items-center justify-center rounded-2xl bg-accent-red/5 border border-accent-red/20 mb-6">
              <AlertTriangle className="h-10 w-10 text-accent-red" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Failed to load analytics</h3>
            <p className="text-sm text-text-secondary max-w-md text-center mb-6">{loadError}</p>
            <Button variant="outline" onClick={load}>
              <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Try Again
            </Button>
          </div>
        </FadeIn>

      /* ── Empty State ───────────────────────────────────── */
      ) : !hasData ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="relative flex h-24 w-24 items-center justify-center rounded-2xl bg-surface mb-6">
            <BarChart3 className="h-12 w-12 text-text-muted" />
            <div className="absolute -top-1 -right-1 h-4 w-4 rounded-full bg-accent-cyan/30 animate-pulse" />
          </div>
          <h3 className="text-xl font-semibold mb-2">{t("dash.analytics.empty")}</h3>
          <p className="text-sm text-text-secondary max-w-md text-center mb-8">
            {t("dash.analytics.empty_desc")}
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 w-full max-w-lg">
            <Card className="p-5 text-center border-dashed hover:border-accent-cyan/30 transition-colors">
              <TrendingUp className="h-6 w-6 text-accent-cyan mx-auto mb-3" />
              <p className="text-xs text-text-muted">Spend trends</p>
              <p className="text-lg font-bold font-mono text-text-muted/50 mt-1">—</p>
            </Card>
            <Card className="p-5 text-center border-dashed hover:border-emerald/30 transition-colors">
              <Cpu className="h-6 w-6 text-emerald mx-auto mb-3" />
              <p className="text-xs text-text-muted">GPU hours</p>
              <p className="text-lg font-bold font-mono text-text-muted/50 mt-1">—</p>
            </Card>
            <Card className="p-5 text-center border-dashed hover:border-accent-gold/30 transition-colors">
              <Clock className="h-6 w-6 text-accent-gold mx-auto mb-3" />
              <p className="text-xs text-text-muted">Job insights</p>
              <p className="text-lg font-bold font-mono text-text-muted/50 mt-1">—</p>
            </Card>
          </div>
          <p className="text-xs text-text-muted mt-8">
            Analytics populate automatically when you launch instances or when jobs run on your hosts
          </p>
          <div className="mt-6 rounded-xl border border-border/30 bg-surface/60 px-6 py-4 max-w-lg w-full">
            <p className="text-xs font-semibold text-text-secondary mb-3 uppercase tracking-wide">How to unlock analytics</p>
            <div className="space-y-2.5">
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-cyan/15 text-accent-cyan text-[10px] font-bold">1</div>
                <p className="text-xs text-text-secondary">
                  <span className="font-medium text-text-primary">Rent GPU compute</span> — launch an instance from the{" "}
                  <Link href="/dashboard/instances" className="text-accent-cyan hover:underline">Instances</Link> page. Spend trends and job insights appear after your first job.
                </p>
              </div>
              <div className="flex items-start gap-3">
                <div className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-accent-violet/15 text-accent-violet text-[10px] font-bold">2</div>
                <p className="text-xs text-text-secondary">
                  <span className="font-medium text-text-primary">Become a GPU provider</span> — connect Stripe on the{" "}
                  <Link href="/dashboard/earnings" className="text-accent-cyan hover:underline">Earnings</Link> page to unlock the <span className="font-medium">Provider</span> analytics tab with host earnings, utilization &amp; uptime.
                </p>
              </div>
            </div>
          </div>
        </div>

      /* ── Dashboard Content ─────────────────────────────── */
      ) : (
        <>
          {/* ── OVERVIEW TAB ──────────────────────────────── */}
          {tab === "overview" && (
            <div className="space-y-6">
              {/* Primary KPIs with glow + sparklines */}
              <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <StaggerItem>
                  <StatCard
                    label="Total Jobs"
                    value={
                      <div className="flex items-center gap-3">
                        <CountUp value={totalJobs} />
                        <Sparkline data={jobSparkline} color="#00d4ff" />
                      </div>
                    }
                    icon={Zap}
                    glow="cyan"
                    trend={trendDir(jobsDelta)}
                    trendValue={trendVal(jobsDelta)}
                  />
                </StaggerItem>
                <StaggerItem>
                  <StatCard
                    label="Total Spend"
                    value={
                      <div className="flex items-center gap-3">
                        <CountUp value={totalSpend} prefix="$" />
                        <Sparkline data={spendSparkline} color="#f59e0b" />
                      </div>
                    }
                    icon={DollarSign}
                    glow="gold"
                    trend={trendDir(spendDelta)}
                    trendValue={trendVal(spendDelta)}
                  />
                </StaggerItem>
                <StaggerItem>
                  <StatCard
                    label="GPU Hours"
                    value={
                      <div className="flex items-center gap-3">
                        <CountUp value={totalGpuHours} />
                        <Sparkline data={gpuHoursSparkline} color="#10b981" />
                      </div>
                    }
                    icon={Clock}
                    glow="emerald"
                    trend={trendDir(gpuHoursDelta)}
                    trendValue={trendVal(gpuHoursDelta)}
                  />
                </StaggerItem>
                <StaggerItem>
                  <StatCard
                    label="Avg Utilization"
                    value={
                      <div className="flex items-center gap-3">
                        <CountUp value={Number(summary.avg_gpu_utilization_pct ?? 0)} suffix="%" />
                        <Sparkline data={utilSparkline} color="#8b5cf6" />
                      </div>
                    }
                    icon={Gauge}
                    glow="violet"
                    trend={trendDir(utilDelta)}
                    trendValue={trendVal(utilDelta)}
                  />
                </StaggerItem>
              </StaggerList>

              {/* Auto-Insights */}
              {insights.length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Sparkles className="h-4 w-4 text-accent-cyan" />
                    <h3 className="text-sm font-semibold">Auto-Insights</h3>
                    <span className="text-[10px] text-text-muted font-mono">{insights.length} detected</span>
                  </div>
                  <InsightCards insights={insights} />
                </div>
              )}

              {/* Secondary KPIs */}
              <FadeIn delay={0.15}>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <StatCard label="Avg Cost/Job" value={<CountUp value={avgCostPerJob} prefix="$" />} icon={ArrowUpRight} />
                  <StatCard label="Avg Job Duration" value={<CountUp value={avgJobDuration} suffix="h" />} icon={Activity} />
                  <StatCard label="Canadian Compute" value={<CountUp value={canadianPct} suffix="%" />} icon={Shield} />
                  <StatCard label="Top GPU" value={<span className="text-sm font-mono">{topGpu}</span>} icon={ChartArea} />
                </div>
              </FadeIn>

              {/* Hero charts */}
              <FadeIn delay={0.25}>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <JobsTrendChart data={jobsOverTime} />
                  <SpendTrendChart data={spendOverTime} />
                </div>
              </FadeIn>

              {/* Activity heatmap */}
              <HourlyHeatmap data={enhanced?.hourly_heatmap ?? []} />

              {/* Peak days */}
              <PeakDaysCards data={enhanced?.peak_days ?? []} />
            </div>
          )}

          {/* ── COMPUTE TAB ──────────────────────────────── */}
          {tab === "compute" && (
            <div className="space-y-6">
              <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <StaggerItem>
                  <StatCard label="Total Jobs" value={<CountUp value={totalJobs} />} icon={Zap} glow="cyan" />
                </StaggerItem>
                <StaggerItem>
                  <StatCard label="GPU Hours" value={<CountUp value={totalGpuHours} />} icon={Clock} glow="emerald" />
                </StaggerItem>
                <StaggerItem>
                  <StatCard label="Avg Utilization" value={<CountUp value={Number(summary.avg_gpu_utilization_pct ?? 0)} suffix="%" />} icon={Gauge} glow="violet" />
                </StaggerItem>
                <StaggerItem>
                  <StatCard label="Canadian %" value={<CountUp value={canadianPct} suffix="%" />} icon={Globe} glow="gold" />
                </StaggerItem>
              </StaggerList>

              <FadeIn delay={0.15}>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <UtilizationChart data={utilOverTime} />
                  <GpuHoursChart data={enhanced?.daily_gpu_hours ?? []} />
                </div>
              </FadeIn>

              <FadeIn delay={0.2}>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <DurationHistogramChart data={enhanced?.duration_histogram ?? []} />
                  <SovereigntyChart data={sovereigntyOverTime} />
                </div>
              </FadeIn>

              <FadeIn delay={0.25}>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <GpuPerformanceRadar data={enhanced?.gpu_performance ?? []} />
                  <TopGpuChart data={topGpuSeries} />
                </div>
              </FadeIn>

              <GpuPerformanceTable data={enhanced?.gpu_performance ?? []} />
              <TopEntitiesTable data={enhanced?.top_entities ?? []} entityLabel={isAdmin ? "Customer" : "Host"} />
            </div>
          )}

          {/* ── FINANCIAL TAB ────────────────────────────── */}
          {tab === "financial" && (
            <div className="space-y-6">
              <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                <StaggerItem>
                  <StatCard label="Total Spend" value={<CountUp value={totalSpend} prefix="$" />} icon={DollarSign} glow="gold" trend={trendDir(spendDelta)} trendValue={trendVal(spendDelta)} />
                </StaggerItem>
                <StaggerItem>
                  <StatCard label="Avg Cost/Job" value={<CountUp value={avgCostPerJob} prefix="$" />} icon={ArrowUpRight} glow="cyan" />
                </StaggerItem>
                <StaggerItem>
                  <StatCard
                    label="CA Spend"
                    value={<CountUp value={enhanced?.sovereignty?.canadian_spend ?? 0} prefix="$" />}
                    icon={Shield}
                    glow="emerald"
                  />
                </StaggerItem>
                <StaggerItem>
                  <StatCard
                    label="Int'l Spend"
                    value={<CountUp value={enhanced?.sovereignty?.international_spend ?? 0} prefix="$" />}
                    icon={Globe}
                    glow="violet"
                  />
                </StaggerItem>
              </StaggerList>

              <FadeIn delay={0.15}>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <SpendTrendChart data={spendOverTime} />
                  <CostPerHourChart data={enhanced?.cost_per_hour_trend ?? []} />
                </div>
              </FadeIn>

              <CumulativeSpendChart data={enhanced?.cumulative_spend ?? []} />

              <FadeIn delay={0.25}>
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <WalletActivityChart data={enhanced?.wallet_activity ?? []} />
                  <ProvinceDonutChart data={provinceSeries} />
                </div>
              </FadeIn>
            </div>
          )}

          {/* ── PROVIDER TAB ─────────────────────────────── */}
          {tab === "provider" && (isProvider || isAdmin) && (
            <div className="space-y-6">
              {enhanced?.provider_summary ? (
                <>
                  <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <StaggerItem>
                      <StatCard
                        label="Jobs Served"
                        value={<CountUp value={enhanced.provider_summary.total_jobs_served} />}
                        icon={Zap}
                        glow="cyan"
                      />
                    </StaggerItem>
                    <StaggerItem>
                      <StatCard
                        label="Total Revenue"
                        value={<CountUp value={enhanced.provider_summary.total_revenue} prefix="$" />}
                        icon={DollarSign}
                        glow="gold"
                      />
                    </StaggerItem>
                    <StaggerItem>
                      <StatCard
                        label="GPU Hours Served"
                        value={<CountUp value={enhanced.provider_summary.total_gpu_hours} />}
                        icon={Clock}
                        glow="emerald"
                      />
                    </StaggerItem>
                    <StaggerItem>
                      <StatCard
                        label="Avg Utilization"
                        value={<CountUp value={enhanced.provider_summary.avg_util} suffix="%" />}
                        icon={Gauge}
                        glow="violet"
                      />
                    </StaggerItem>
                  </StaggerList>

                  <FadeIn delay={0.15}>
                    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                      <ProviderRevenueTrendChart data={enhanced.provider_daily ?? []} />
                      <UtilizationChart data={(enhanced.provider_daily ?? []).map((d) => ({ date: d.date, util: d.avg_util }))} />
                    </div>
                  </FadeIn>
                </>
              ) : (
                <FadeIn>
                  <div className="flex flex-col items-center justify-center py-16">
                    <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
                      <Server className="h-10 w-10 text-text-muted" />
                    </div>
                    <h3 className="text-xl font-semibold mb-2">No provider activity yet</h3>
                    <p className="text-sm text-text-secondary text-center max-w-md">
                      {isAdmin
                        ? "Provider analytics appear once hosts serve jobs through the platform."
                        : "Your provider account is connected — analytics will populate here once your first GPU job runs on your host."
                      }
                    </p>
                    {!isAdmin && (
                      <div className="mt-6 rounded-xl border border-border/30 bg-surface/60 px-5 py-4 max-w-sm w-full text-left">
                        <p className="text-xs font-semibold text-text-secondary uppercase tracking-wide mb-3">Next steps</p>
                        <div className="space-y-2">
                          <div className="flex items-start gap-2.5">
                            <div className="mt-0.5 flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-emerald/15 text-emerald text-[9px] font-bold">1</div>
                            <p className="text-xs text-text-secondary"><span className="font-medium text-text-primary">Install the worker agent</span> on your GPU host via the <Link href="/dashboard/hosts" className="text-accent-cyan hover:underline">Hosts</Link> page.</p>
                          </div>
                          <div className="flex items-start gap-2.5">
                            <div className="mt-0.5 flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-accent-cyan/15 text-accent-cyan text-[9px] font-bold">2</div>
                            <p className="text-xs text-text-secondary"><span className="font-medium text-text-primary">Set your pricing</span> and make your GPU available to renters.</p>
                          </div>
                          <div className="flex items-start gap-2.5">
                            <div className="mt-0.5 flex h-4.5 w-4.5 shrink-0 items-center justify-center rounded-full bg-accent-violet/15 text-accent-violet text-[9px] font-bold">3</div>
                            <p className="text-xs text-text-secondary">Earnings, uptime, and utilization charts appear automatically as jobs run.</p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </FadeIn>
              )}
            </div>
          )}
        </>
      )}
      {/* ── Analytics AI Panel ─────────────────────── */}
      <AnalyticsAiPanel
        open={aiOpen}
        onClose={() => setAiOpen(false)}
        tab={tab}
        summary={summary}
        enhanced={enhanced}
        previousSummary={previousSummary}
        range={range === 0 ? Math.ceil((Date.now() - new Date(new Date().getFullYear(), 0, 1).getTime()) / 86400000) : range}
      />
    </div>
  );
}
