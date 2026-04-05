"use client";

import { useMemo, useState } from "react";
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line, ComposedChart,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ReferenceLine,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { HoverCard, ScrollReveal, FadeIn } from "@/components/ui/motion";
import { TrendingUp, TrendingDown, Flame, Zap, Clock, BarChart3 } from "lucide-react";

// ── Theme ──────────────────────────────────────────────────────────────

const CHART_COLORS = {
  cyan: "#00d4ff",
  emerald: "#10b981",
  gold: "#f59e0b",
  violet: "#8b5cf6",
  red: "#dc2626",
  pink: "#ec4899",
  orange: "#f97316",
  blue: "#3b82f6",
};

const PALETTE = Object.values(CHART_COLORS);

const tooltipStyle: React.CSSProperties = {
  backgroundColor: "rgba(13, 19, 32, 0.95)",
  backdropFilter: "blur(12px)",
  border: "1px solid rgba(0, 212, 255, 0.15)",
  borderRadius: "12px",
  boxShadow: "0 8px 32px rgba(0, 0, 0, 0.6), 0 0 1px rgba(0, 212, 255, 0.2)",
  padding: "10px 14px",
  fontSize: "12px",
};

const axisStyle = { fill: "#5a6a7e", fontSize: 10, fontFamily: "'JetBrains Mono', monospace" };
const gridStroke = "rgba(26, 34, 53, 0.6)";
const axisStroke = "rgba(26, 34, 53, 0.4)";

function formatCurrency(value: number) {
  if (value >= 1000) return `$${(value / 1000).toFixed(1)}k`;
  return `$${value.toLocaleString("en-CA", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

function formatShortDate(v: string) {
  const s = String(v);
  if (s.length >= 10) {
    const d = new Date(s);
    return d.toLocaleDateString("en-CA", { month: "short", day: "numeric" });
  }
  return s.slice(5);
}

function formatNumber(v: number) {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}k`;
  return v.toLocaleString();
}

// ── Gradient Definitions ───────────────────────────────────────────────

function GradientDef({ id, color, opacity = 0.35 }: { id: string; color: string; opacity?: number }) {
  return (
    <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor={color} stopOpacity={opacity} />
      <stop offset="50%" stopColor={color} stopOpacity={opacity * 0.3} />
      <stop offset="100%" stopColor={color} stopOpacity={0} />
    </linearGradient>
  );
}

function GlowFilter({ id, color }: { id: string; color: string }) {
  return (
    <filter id={id} x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
      <feFlood floodColor={color} floodOpacity="0.3" result="color" />
      <feComposite in="color" in2="blur" operator="in" result="glow" />
      <feMerge>
        <feMergeNode in="glow" />
        <feMergeNode in="SourceGraphic" />
      </feMerge>
    </filter>
  );
}

// ── Chart Empty State ──────────────────────────────────────────────────

function ChartEmptyState({ message = "No data available for this metric" }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-center">
      <div className="relative h-12 w-12 rounded-xl bg-gradient-to-br from-surface-hover to-surface flex items-center justify-center mb-3 ring-1 ring-border/20">
        <BarChart3 className="h-5 w-5 text-text-muted/60" />
        <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-accent-cyan/5 to-accent-violet/5 animate-pulse" />
      </div>
      <p className="text-xs text-text-muted/80 font-medium">{message}</p>
      <p className="text-[10px] text-text-muted/50 mt-1">Data will appear when activity is recorded</p>
    </div>
  );
}

// ── Custom Tooltip ────────────────────────────────────────────────────

function CustomTooltip({ active, payload, label, formatter, unit }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div
      style={{
        background: "rgba(13, 19, 32, 0.95)",
        backdropFilter: "blur(16px)",
        border: "1px solid rgba(0, 212, 255, 0.12)",
        borderRadius: "12px",
        padding: "12px 16px",
        boxShadow: "0 12px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255,255,255,0.03)",
        minWidth: "140px",
      }}
    >
      <p className="text-[10px] text-text-muted mb-2 font-mono uppercase tracking-wider">
        {label ? formatShortDate(String(label)) : ""}
      </p>
      {payload.map((p: any, i: number) => (
        <div key={i} className="flex items-center justify-between gap-4 py-0.5">
          <div className="flex items-center gap-1.5">
            <div className="h-2 w-2 rounded-full" style={{ background: p.color || CHART_COLORS.cyan }} />
            <span className="text-xs text-text-secondary">{p.name || p.dataKey}</span>
          </div>
          <span className="text-xs font-mono font-semibold text-text-primary">
            {formatter ? formatter(p.value) : unit ? `${p.value?.toFixed?.(1) ?? p.value}${unit}` : formatNumber(Number(p.value ?? 0))}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Sparkline (inline mini-chart for stat cards) ─────────────────────

export function Sparkline({ data, color = CHART_COLORS.cyan, height = 32, width = 100 }: { data: number[]; color?: string; height?: number; width?: number }) {
  if (!data?.length || data.length < 2) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * width},${height - ((v - min) / range) * (height - 4) - 2}`).join(" ");
  const areaPoints = `${points} ${width},${height} 0,${height}`;
  const gradientId = `spark-${color.replace("#", "")}`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.25} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <polygon points={areaPoints} fill={`url(#${gradientId})`} />
      <polyline points={points} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
      {/* Terminus dot */}
      {(() => {
        const lastPair = points.split(" ").pop();
        const cy = lastPair ? parseFloat(lastPair.split(",")[1] ?? "") : NaN;
        return !isNaN(cy) ? (
          <circle cx={width} cy={cy} r="2.5" fill={color}>
            <animate attributeName="r" values="2.5;3.5;2.5" dur="2s" repeatCount="indefinite" />
          </circle>
        ) : null;
      })()}
    </svg>
  );
}

// ── Metric Ring (circular progress) ──────────────────────────────────

export function MetricRing({ value, max = 100, color = CHART_COLORS.cyan, size = 56, strokeWidth = 5, label }: {
  value: number; max?: number; color?: string; size?: number; strokeWidth?: number; label?: string;
}) {
  const r = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * r;
  const pct = Math.min(value / max, 1);
  const offset = circumference * (1 - pct);

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgba(26, 34, 53, 0.8)" strokeWidth={strokeWidth} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={color} strokeWidth={strokeWidth} strokeLinecap="round"
          strokeDasharray={circumference} strokeDashoffset={offset}
          className="metric-ring"
          style={{ filter: `drop-shadow(0 0 4px ${color}40)` }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-[10px] font-mono font-bold">{label ?? `${Math.round(pct * 100)}%`}</span>
      </div>
    </div>
  );
}

// ── Chart Card Wrapper ─────────────────────────────────────────────────

function ChartCard({
  title,
  subtitle,
  badge,
  children,
  className = "",
  fullWidth = false,
  noPadding = false,
}: {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  fullWidth?: boolean;
  noPadding?: boolean;
}) {
  return (
    <ScrollReveal className={fullWidth ? "lg:col-span-2" : ""}>
      <HoverCard>
        <Card className={`glow-card overflow-hidden ${className}`}>
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-sm font-medium">{title}</CardTitle>
                {subtitle && <p className="text-[11px] text-text-muted mt-0.5">{subtitle}</p>}
              </div>
              {badge}
            </div>
          </CardHeader>
          <CardContent className={noPadding ? "px-0 pb-4" : ""}>{children}</CardContent>
        </Card>
      </HoverCard>
    </ScrollReveal>
  );
}

// ── Trend Badge ────────────────────────────────────────────────────────

function TrendBadge({ values, suffix = "" }: { values: number[]; suffix?: string }) {
  if (!values || values.length < 2) return null;
  const first = values[0];
  const last = values[values.length - 1];
  if (first === 0 && last === 0) return null;
  const pct = first !== 0 ? ((last - first) / Math.abs(first)) * 100 : last > 0 ? 100 : 0;
  const up = pct >= 0;

  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium ${
      up ? "bg-emerald/10 text-emerald" : "bg-accent-red/10 text-accent-red"
    }`}>
      {up ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
      {up ? "+" : ""}{pct.toFixed(1)}%{suffix}
    </span>
  );
}

// ── 1. Spend Trend (AreaChart with gradient + reference line) ────────

export function SpendTrendChart({ data }: { data: { date: string; spend: number }[] }) {
  if (!data?.length) return <ChartCard title="Spend Over Time" subtitle="Daily compute spend"><ChartEmptyState /></ChartCard>;
  const avg = data.reduce((s, d) => s + d.spend, 0) / data.length;
  return (
    <ChartCard
      title="Spend Over Time"
      subtitle="Daily compute spend"
      badge={<TrendBadge values={data.map(d => d.spend)} />}
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="spendGrad" color={CHART_COLORS.gold} />
              <GlowFilter id="spendGlow" color={CHART_COLORS.gold} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} tickFormatter={(v) => `$${v}`} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip formatter={formatCurrency} />} />
            <ReferenceLine y={avg} stroke={CHART_COLORS.gold} strokeDasharray="4 4" strokeOpacity={0.4} label={{ value: `avg $${avg.toFixed(0)}`, position: "right", fill: "#5a6a7e", fontSize: 9 }} />
            <Area type="monotone" dataKey="spend" stroke={CHART_COLORS.gold} fill="url(#spendGrad)" strokeWidth={2.5} dot={false} animationDuration={1200} animationEasing="ease-out" filter="url(#spendGlow)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 2. Jobs Over Time ─────────────────────────────────────────────────

export function JobsTrendChart({ data }: { data: { date: string; count: number }[] }) {
  if (!data?.length) return <ChartCard title="Jobs Over Time" subtitle="Daily job submissions"><ChartEmptyState /></ChartCard>;
  const avg = data.reduce((s, d) => s + d.count, 0) / data.length;
  return (
    <ChartCard
      title="Jobs Over Time"
      subtitle="Daily job submissions"
      badge={<TrendBadge values={data.map(d => d.count)} />}
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="jobsGrad" color={CHART_COLORS.cyan} />
              <GlowFilter id="jobsGlow" color={CHART_COLORS.cyan} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} allowDecimals={false} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={avg} stroke={CHART_COLORS.cyan} strokeDasharray="4 4" strokeOpacity={0.4} />
            <Area type="monotone" dataKey="count" stroke={CHART_COLORS.cyan} fill="url(#jobsGrad)" strokeWidth={2.5} dot={false} name="Jobs" animationDuration={1200} animationEasing="ease-out" filter="url(#jobsGlow)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 3. GPU Utilization Trend ──────────────────────────────────────────

export function UtilizationChart({ data }: { data: { date: string; util: number }[] }) {
  if (!data?.length) return <ChartCard title="GPU Utilization" subtitle="Average GPU utilization over time"><ChartEmptyState /></ChartCard>;
  const avg = data.reduce((s, d) => s + d.util, 0) / data.length;
  return (
    <ChartCard
      title="GPU Utilization"
      subtitle="Average GPU utilization over time"
      badge={
        <MetricRing value={avg} color={avg >= 70 ? CHART_COLORS.emerald : avg >= 40 ? CHART_COLORS.gold : CHART_COLORS.red} size={40} strokeWidth={4} />
      }
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="utilGrad" color={CHART_COLORS.emerald} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} domain={[0, 100]} tickFormatter={(v) => `${v}%`} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip unit="%" />} />
            <ReferenceLine y={80} stroke={CHART_COLORS.emerald} strokeDasharray="4 4" strokeOpacity={0.3} label={{ value: "target 80%", position: "right", fill: "#5a6a7e", fontSize: 9 }} />
            <Area type="monotone" dataKey="util" stroke={CHART_COLORS.emerald} fill="url(#utilGrad)" strokeWidth={2.5} dot={false} animationDuration={1200} animationEasing="ease-out" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 4. Cumulative Spend ───────────────────────────────────────────────

export function CumulativeSpendChart({ data }: { data: { date: string; total: number }[] }) {
  if (!data?.length) return <ChartCard title="Cumulative Spend" subtitle="Running total over the period" fullWidth><ChartEmptyState /></ChartCard>;
  const total = data[data.length - 1]?.total ?? 0;
  return (
    <ChartCard
      title="Cumulative Spend"
      subtitle="Running total over the period"
      badge={<span className="text-xs font-mono font-bold text-accent-gold">{formatCurrency(total)}</span>}
      fullWidth
    >
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="cumulativeGrad" color={CHART_COLORS.violet} opacity={0.4} />
              <GlowFilter id="cumulGlow" color={CHART_COLORS.violet} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} tickFormatter={(v) => `$${formatNumber(v)}`} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip formatter={formatCurrency} />} />
            <Area type="monotone" dataKey="total" stroke={CHART_COLORS.violet} fill="url(#cumulativeGrad)" strokeWidth={2.5} dot={false} animationDuration={1400} animationEasing="ease-out" filter="url(#cumulGlow)" />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 5. Cost Per GPU Hour Trend ────────────────────────────────────────

export function CostPerHourChart({ data }: { data: { date: string; cost_per_hour: number }[] }) {
  if (!data?.length) return <ChartCard title="Cost per GPU Hour" subtitle="Efficiency trend — lower is better"><ChartEmptyState /></ChartCard>;
  const avg = data.reduce((s, d) => s + d.cost_per_hour, 0) / data.length;
  return (
    <ChartCard
      title="Cost per GPU Hour"
      subtitle="Efficiency trend — lower is better"
      badge={<TrendBadge values={data.map(d => d.cost_per_hour)} suffix="" />}
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="cphGrad" color={CHART_COLORS.orange} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} tickFormatter={(v) => `$${v}`} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip formatter={formatCurrency} />} />
            <ReferenceLine y={avg} stroke={CHART_COLORS.orange} strokeDasharray="4 4" strokeOpacity={0.3} label={{ value: `avg $${avg.toFixed(2)}`, position: "right", fill: "#5a6a7e", fontSize: 9 }} />
            <Area type="monotone" dataKey="cost_per_hour" stroke="none" fill="url(#cphGrad)" animationDuration={1200} />
            <Line type="monotone" dataKey="cost_per_hour" stroke={CHART_COLORS.orange} strokeWidth={2.5} dot={false} name="$/GPU hr" animationDuration={1200} animationEasing="ease-out" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 6. GPU Hours Over Time ────────────────────────────────────────────

export function GpuHoursChart({ data }: { data: { date: string; hours: number }[] }) {
  if (!data?.length) return <ChartCard title="GPU Hours Consumed" subtitle="Daily compute hours"><ChartEmptyState /></ChartCard>;
  const total = data.reduce((s, d) => s + d.hours, 0);
  return (
    <ChartCard
      title="GPU Hours Consumed"
      subtitle="Daily compute hours"
      badge={<span className="text-xs font-mono font-bold text-accent-cyan">{total.toFixed(1)}h total</span>}
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <BarChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <linearGradient id="gpuHoursBar" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS.cyan} stopOpacity={0.9} />
                <stop offset="100%" stopColor={CHART_COLORS.cyan} stopOpacity={0.4} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} tickFormatter={(v) => `${v}h`} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip unit="h" />} />
            <Bar dataKey="hours" fill="url(#gpuHoursBar)" radius={[4, 4, 0, 0]} animationDuration={1000} animationEasing="ease-out" name="GPU Hours" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 7. Duration Histogram ─────────────────────────────────────────────

export function DurationHistogramChart({ data }: { data: { bucket: string; count: number; total_cost: number }[] }) {
  if (!data?.length) return <ChartCard title="Job Duration Distribution" subtitle="How long your jobs typically run"><ChartEmptyState /></ChartCard>;
  const maxCount = Math.max(...data.map(d => d.count));
  return (
    <ChartCard title="Job Duration Distribution" subtitle="How long your jobs typically run">
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <BarChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <linearGradient id="durationBar" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={CHART_COLORS.violet} stopOpacity={0.9} />
                <stop offset="100%" stopColor={CHART_COLORS.violet} stopOpacity={0.4} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="bucket" tick={axisStyle} stroke={axisStroke} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} allowDecimals={false} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Bar dataKey="count" fill="url(#durationBar)" radius={[4, 4, 0, 0]} animationDuration={1000} animationEasing="ease-out" name="Jobs" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 8. Sovereignty ────────────────────────────────────────────────────

export function SovereigntyChart({ data }: { data: { date: string; canadian: number; international: number }[] }) {
  if (!data?.length) return null;
  const totalCA = data.reduce((s, d) => s + d.canadian, 0);
  const totalInt = data.reduce((s, d) => s + d.international, 0);
  const caPct = totalCA + totalInt > 0 ? Math.round((totalCA / (totalCA + totalInt)) * 100) : 0;
  return (
    <ChartCard
      title="Data Sovereignty"
      subtitle="Canadian vs International compute"
      badge={<MetricRing value={caPct} color={CHART_COLORS.cyan} size={40} strokeWidth={4} label={`${caPct}%`} />}
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <AreaChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="caGrad" color={CHART_COLORS.cyan} opacity={0.3} />
              <GradientDef id="intlGrad" color={CHART_COLORS.gold} opacity={0.2} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} allowDecimals={false} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip />} />
            <Area type="monotone" dataKey="canadian" stackId="sov" stroke={CHART_COLORS.cyan} fill="url(#caGrad)" strokeWidth={2} name="Canadian" animationDuration={1200} />
            <Area type="monotone" dataKey="international" stackId="sov" stroke={CHART_COLORS.gold} fill="url(#intlGrad)" strokeWidth={2} name="International" animationDuration={1200} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 9. Top GPU Models ─────────────────────────────────────────────────

export function TopGpuChart({ data }: { data: { name: string; spend: number; jobs: number; hours: number }[] }) {
  if (!data?.length) return null;
  const maxSpend = Math.max(...data.map(d => d.spend));
  return (
    <ChartCard title="Top GPU Models" subtitle="By total spend">
      <div className="space-y-2">
        {data.map((d, i) => {
          const pct = maxSpend > 0 ? (d.spend / maxSpend) * 100 : 0;
          const color = PALETTE[i % PALETTE.length];
          return (
            <div key={d.name} className="table-row-animate">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium truncate max-w-[140px]">{d.name}</span>
                <span className="text-xs font-mono text-text-secondary">{formatCurrency(d.spend)}</span>
              </div>
              <div className="relative h-2 rounded-full bg-surface-hover overflow-hidden">
                <div
                  className="absolute inset-y-0 left-0 rounded-full transition-all duration-700"
                  style={{ width: `${pct}%`, background: `linear-gradient(90deg, ${color}99, ${color})`, boxShadow: `0 0 8px ${color}40` }}
                />
              </div>
              <div className="flex gap-4 mt-0.5 text-[10px] text-text-muted">
                <span>{d.jobs} jobs</span>
                <span>{d.hours.toFixed(1)}h GPU</span>
              </div>
            </div>
          );
        })}
      </div>
    </ChartCard>
  );
}

// ── 10. Spend by Province (donut + center stat) ──────────────────────

export function ProvinceDonutChart({ data }: { data: { name: string; value: number }[] }) {
  if (!data?.length) return null;
  const total = data.reduce((s, d) => s + d.value, 0);
  return (
    <ChartCard title="Spend by Province" subtitle="Geographic distribution">
      <div className="h-64 relative">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="name"
              outerRadius={90}
              innerRadius={55}
              paddingAngle={3}
              strokeWidth={0}
              animationDuration={1200}
              animationEasing="ease-out"
            >
              {data.map((entry, index) => (
                <Cell
                  key={`province-${entry.name}`}
                  fill={PALETTE[index % PALETTE.length]}
                  style={{ filter: `drop-shadow(0 0 4px ${PALETTE[index % PALETTE.length]}30)` }}
                />
              ))}
            </Pie>
            <Tooltip content={<CustomTooltip formatter={formatCurrency} />} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
          </PieChart>
        </ResponsiveContainer>
        {/* Center stat */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 text-center pointer-events-none" style={{ marginTop: "-12px" }}>
          <p className="text-lg font-bold font-mono">{formatCurrency(total)}</p>
          <p className="text-[9px] text-text-muted uppercase tracking-wider">Total</p>
        </div>
      </div>
    </ChartCard>
  );
}

// ── 11. GPU Performance Radar ─────────────────────────────────────────

export function GpuPerformanceRadar({ data }: { data: { gpu_model: string; avg_util: number; avg_cost_per_hour: number; avg_duration_min: number }[] }) {
  if (!data?.length || data.length < 3) return <ChartCard title="GPU Performance Radar" subtitle="Compare GPU models across metrics"><ChartEmptyState message="Need at least 3 GPU models for radar view" /></ChartCard>;

  const maxCPH = Math.max(...data.map((d) => d.avg_cost_per_hour), 1);
  const maxDur = Math.max(...data.map((d) => d.avg_duration_min), 1);
  const radarData = data.slice(0, 6).map((d) => ({
    model: d.gpu_model?.replace("NVIDIA ", "").replace("GeForce ", "") || "Unknown",
    utilization: d.avg_util,
    cost_eff: Math.round((1 - d.avg_cost_per_hour / maxCPH) * 100),
    duration: Math.round((d.avg_duration_min / maxDur) * 100),
  }));

  return (
    <ChartCard title="GPU Performance Radar" subtitle="Compare GPU models across metrics">
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <RadarChart data={radarData} outerRadius={90}>
            <PolarGrid stroke="rgba(26, 34, 53, 0.8)" strokeDasharray="3 3" />
            <PolarAngleAxis dataKey="model" tick={{ fill: "#5a6a7e", fontSize: 9, fontFamily: "'JetBrains Mono', monospace" }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
            <Radar name="Utilization" dataKey="utilization" stroke={CHART_COLORS.emerald} fill={CHART_COLORS.emerald} fillOpacity={0.15} strokeWidth={2} animationDuration={1200} />
            <Radar name="Cost Efficiency" dataKey="cost_eff" stroke={CHART_COLORS.gold} fill={CHART_COLORS.gold} fillOpacity={0.15} strokeWidth={2} animationDuration={1200} />
            <Radar name="Job Length" dataKey="duration" stroke={CHART_COLORS.violet} fill={CHART_COLORS.violet} fillOpacity={0.15} strokeWidth={2} animationDuration={1200} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Tooltip contentStyle={tooltipStyle} />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 12. Hourly Heatmap (enhanced with tooltip + animation) ───────────

const DOW_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export function HourlyHeatmap({ data }: { data: { dow: number; hour: number; count: number }[] }) {
  if (!data?.length) return <ChartCard title="Activity Heatmap" subtitle="When you run the most jobs" fullWidth><ChartEmptyState /></ChartCard>;

  const maxCount = Math.max(...data.map((d) => d.count), 1);
  const grid: Record<string, number> = {};
  let totalJobs = 0;
  data.forEach((d) => { grid[`${d.dow}-${d.hour}`] = d.count; totalJobs += d.count; });

  // Find peak hour
  const peak = data.reduce((a, b) => (a.count > b.count ? a : b));
  const peakLabel = `${DOW_LABELS[peak.dow]} ${peak.hour}:00`;

  return (
    <ChartCard
      title="Activity Heatmap"
      subtitle="When you run the most jobs"
      badge={
        <span className="inline-flex items-center gap-1 text-[10px] text-accent-cyan">
          <Flame className="h-3 w-3" /> Peak: {peakLabel}
        </span>
      }
      fullWidth
    >
      <div className="overflow-x-auto">
        <div className="min-w-[640px]">
          <div className="flex gap-[2px]">
            <div className="w-10 shrink-0" />
            {Array.from({ length: 24 }, (_, h) => (
              <div key={h} className="flex-1 text-center text-[9px] text-text-muted font-mono">
                {h === 0 ? "12a" : h < 12 ? `${h}a` : h === 12 ? "12p" : `${h - 12}p`}
              </div>
            ))}
          </div>
          {DOW_LABELS.map((label, dow) => (
            <div key={dow} className="flex gap-[2px] mt-[2px]">
              <div className="w-10 shrink-0 text-[10px] text-text-muted font-mono flex items-center">
                {label}
              </div>
              {Array.from({ length: 24 }, (_, hour) => {
                const count = grid[`${dow}-${hour}`] || 0;
                const intensity = count / maxCount;
                return (
                  <div
                    key={hour}
                    className="heatmap-cell flex-1 rounded-[3px] aspect-square relative group cursor-default"
                    style={{
                      backgroundColor: count === 0
                        ? "rgba(26, 34, 53, 0.3)"
                        : `rgba(0, 212, 255, ${0.1 + intensity * 0.8})`,
                      boxShadow: intensity > 0.7 ? `0 0 6px rgba(0, 212, 255, ${intensity * 0.3})` : "none",
                    }}
                  >
                    {/* Tooltip */}
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 rounded-md bg-[#0d1320] border border-border/50 text-[10px] font-mono text-text-primary shadow-xl opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-20">
                      {count} job{count !== 1 ? "s" : ""} · {label} {hour}:00
                    </div>
                  </div>
                );
              })}
            </div>
          ))}
          <div className="flex items-center justify-between mt-3">
            <span className="text-[10px] text-text-muted font-mono">{totalJobs} total jobs</span>
            <div className="flex items-center gap-1.5">
              <span className="text-[10px] text-text-muted">Less</span>
              {[0.1, 0.25, 0.45, 0.65, 0.85].map((o) => (
                <div key={o} className="h-3 w-3 rounded-[2px]" style={{ backgroundColor: `rgba(0, 212, 255, ${o})` }} />
              ))}
              <span className="text-[10px] text-text-muted">More</span>
            </div>
          </div>
        </div>
      </div>
    </ChartCard>
  );
}

// ── 13. Provider Revenue Trend ────────────────────────────────────────

export function ProviderRevenueTrendChart({ data }: { data: { date: string; total_revenue: number; jobs_served: number }[] }) {
  if (!data?.length) return null;
  const totalRev = data.reduce((s, d) => s + d.total_revenue, 0);
  return (
    <ChartCard
      title="Earnings Over Time"
      subtitle="Daily revenue from hosted jobs"
      badge={<span className="text-xs font-mono font-bold text-emerald">{formatCurrency(totalRev)} earned</span>}
    >
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <ComposedChart data={data} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <defs>
              <GradientDef id="provRevGrad" color={CHART_COLORS.emerald} opacity={0.4} />
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis yAxisId="rev" tick={axisStyle} stroke={axisStroke} tickFormatter={(v) => `$${v}`} tickLine={false} axisLine={false} />
            <YAxis yAxisId="jobs" orientation="right" tick={axisStyle} stroke={axisStroke} allowDecimals={false} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip formatter={(v: number) => formatCurrency(v)} />} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Area yAxisId="rev" type="monotone" dataKey="total_revenue" stroke={CHART_COLORS.emerald} fill="url(#provRevGrad)" strokeWidth={2.5} dot={false} name="Revenue" animationDuration={1200} />
            <Line yAxisId="jobs" type="monotone" dataKey="jobs_served" stroke={CHART_COLORS.cyan} strokeWidth={1.5} dot={false} strokeDasharray="4 4" name="Jobs Served" animationDuration={1200} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 14. Wallet Activity Chart ─────────────────────────────────────────

export function WalletActivityChart({ data }: { data: { date: string; tx_type: string; total_amount: number; tx_count: number }[] }) {
  if (!data?.length) return null;

  const byDate: Record<string, { date: string; deposit: number; charge: number; refund: number }> = {};
  data.forEach((d) => {
    if (!byDate[d.date]) byDate[d.date] = { date: d.date, deposit: 0, charge: 0, refund: 0 };
    const entry = byDate[d.date];
    const t = d.tx_type?.toLowerCase() || "";
    if (t.includes("deposit") || t.includes("topup") || t.includes("credit")) {
      entry.deposit += d.total_amount;
    } else if (t.includes("charge") || t.includes("debit") || t.includes("usage")) {
      entry.charge += Math.abs(d.total_amount);
    } else if (t.includes("refund")) {
      entry.refund += d.total_amount;
    } else {
      entry.charge += Math.abs(d.total_amount);
    }
  });
  const pivoted = Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date));

  return (
    <ChartCard title="Wallet Activity" subtitle="Deposits, charges, and refunds">
      <div className="h-56">
        <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
          <BarChart data={pivoted} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} vertical={false} />
            <XAxis dataKey="date" tick={axisStyle} stroke={axisStroke} tickFormatter={formatShortDate} tickLine={false} />
            <YAxis tick={axisStyle} stroke={axisStroke} tickFormatter={(v) => `$${v}`} tickLine={false} axisLine={false} />
            <Tooltip content={<CustomTooltip formatter={formatCurrency} />} />
            <Legend wrapperStyle={{ fontSize: 10 }} />
            <Bar dataKey="deposit" stackId="wallet" fill={CHART_COLORS.emerald} name="Deposits" radius={[0, 0, 0, 0]} animationDuration={1000} />
            <Bar dataKey="charge" stackId="wallet" fill={CHART_COLORS.red} name="Charges" radius={[0, 0, 0, 0]} animationDuration={1000} />
            <Bar dataKey="refund" stackId="wallet" fill={CHART_COLORS.blue} name="Refunds" radius={[3, 3, 0, 0]} animationDuration={1000} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </ChartCard>
  );
}

// ── 15. Top Entities Table (enhanced with bars) ──────────────────────

export function TopEntitiesTable({ data, entityLabel }: { data: { entity: string; job_count: number; total_cost: number; gpu_hours: number }[]; entityLabel: string }) {
  if (!data?.length) return null;
  const maxJobs = Math.max(...data.map(d => d.job_count));
  return (
    <ScrollReveal>
      <Card className="glow-card overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">Top {entityLabel}s</CardTitle>
          <p className="text-[11px] text-text-muted mt-0.5">Most active by job count</p>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="py-2 pr-4 font-medium text-text-secondary text-xs">#</th>
                  <th className="py-2 pr-4 font-medium text-text-secondary text-xs">{entityLabel}</th>
                  <th className="py-2 pr-4 font-medium text-text-secondary text-xs">Activity</th>
                  <th className="py-2 pr-4 font-medium text-right text-text-secondary text-xs">Jobs</th>
                  <th className="py-2 pr-4 font-medium text-right text-text-secondary text-xs">GPU Hours</th>
                  <th className="py-2 font-medium text-right text-text-secondary text-xs">Spend</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, i) => {
                  const pct = maxJobs > 0 ? (row.job_count / maxJobs) * 100 : 0;
                  return (
                    <tr key={i} className="border-b border-border/20 hover:bg-surface/50 transition-colors table-row-animate">
                      <td className="py-2.5 pr-4 text-xs text-text-muted">{i + 1}</td>
                      <td className="py-2.5 pr-4 font-medium font-mono text-xs max-w-[200px] truncate">{row.entity || "—"}</td>
                      <td className="py-2.5 pr-4 w-32">
                        <div className="h-1.5 rounded-full bg-surface-hover overflow-hidden">
                          <div className="h-full rounded-full bg-accent-cyan/70 transition-all duration-700" style={{ width: `${pct}%` }} />
                        </div>
                      </td>
                      <td className="py-2.5 pr-4 text-right font-mono text-xs">{row.job_count}</td>
                      <td className="py-2.5 pr-4 text-right font-mono text-xs">{row.gpu_hours.toFixed(1)}h</td>
                      <td className="py-2.5 text-right font-mono text-xs">{formatCurrency(row.total_cost)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </ScrollReveal>
  );
}

// ── 16. GPU Performance Table (enhanced with badges) ─────────────────

export function GpuPerformanceTable({
  data,
}: {
  data: { gpu_model: string; jobs: number; avg_util: number; avg_duration_min: number; total_cost: number; gpu_hours: number; avg_cost_per_hour: number }[];
}) {
  if (!data?.length) return null;
  return (
    <ScrollReveal>
      <Card className="glow-card overflow-hidden">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium">GPU Model Performance</CardTitle>
          <p className="text-[11px] text-text-muted mt-0.5">Detailed breakdown by GPU model</p>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="py-2 pr-3 font-medium text-text-secondary text-xs">GPU Model</th>
                  <th className="py-2 pr-3 font-medium text-right text-text-secondary text-xs">Jobs</th>
                  <th className="py-2 pr-3 font-medium text-right text-text-secondary text-xs">GPU Hours</th>
                  <th className="py-2 pr-3 font-medium text-right text-text-secondary text-xs">Avg Util</th>
                  <th className="py-2 pr-3 font-medium text-right text-text-secondary text-xs">Avg $/hr</th>
                  <th className="py-2 font-medium text-right text-text-secondary text-xs">Total Spend</th>
                </tr>
              </thead>
              <tbody>
                {data.map((row, i) => {
                  const utilColor = row.avg_util >= 80 ? CHART_COLORS.emerald : row.avg_util >= 50 ? CHART_COLORS.gold : CHART_COLORS.red;
                  return (
                    <tr key={i} className="border-b border-border/20 hover:bg-surface/50 transition-colors table-row-animate">
                      <td className="py-2.5 pr-3 font-medium text-xs">{row.gpu_model || "Unknown"}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-xs">{row.jobs}</td>
                      <td className="py-2.5 pr-3 text-right font-mono text-xs">{row.gpu_hours.toFixed(1)}h</td>
                      <td className="py-2.5 pr-3 text-right">
                        <span className="inline-flex items-center gap-1.5">
                          <MetricRing value={row.avg_util} color={utilColor} size={22} strokeWidth={2.5} label="" />
                          <span className="font-mono text-xs" style={{ color: utilColor }}>{row.avg_util.toFixed(1)}%</span>
                        </span>
                      </td>
                      <td className="py-2.5 pr-3 text-right font-mono text-xs">{formatCurrency(row.avg_cost_per_hour)}</td>
                      <td className="py-2.5 text-right font-mono text-xs font-semibold">{formatCurrency(row.total_cost)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </ScrollReveal>
  );
}

// ── 17. Peak Days Cards (enhanced) ────────────────────────────────────

export function PeakDaysCards({ data }: { data: { date: string; jobs: number; gpu_hours: number; spend: number; avg_util: number }[] }) {
  if (!data?.length) return null;
  const best = data[0];
  return (
    <ScrollReveal>
      <Card className="glow-card overflow-hidden">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-sm font-medium">Peak Usage Days</CardTitle>
              <p className="text-[11px] text-text-muted mt-0.5">Your busiest days in this period</p>
            </div>
            <span className="inline-flex items-center gap-1 text-[10px] text-accent-cyan font-mono">
              <Flame className="h-3 w-3 text-orange" /> {best.jobs} peak jobs
            </span>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
            {data.slice(0, 5).map((day, i) => {
              const isBest = i === 0;
              return (
                <div
                  key={i}
                  className={`rounded-lg p-3 border transition-colors table-row-animate ${
                    isBest
                      ? "bg-accent-cyan/5 border-accent-cyan/20 ring-1 ring-accent-cyan/10"
                      : "bg-surface-hover border-border/30 hover:border-border/60"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <p className="text-[10px] text-text-muted font-mono">{formatShortDate(day.date)}</p>
                    {isBest && <Flame className="h-3 w-3 text-orange" />}
                  </div>
                  <div className="flex items-baseline gap-1 mt-1.5">
                    <p className="text-xl font-bold font-mono">{day.jobs}</p>
                    <span className="text-[10px] text-text-muted">jobs</span>
                  </div>
                  <div className="flex justify-between mt-2 text-[10px] text-text-secondary">
                    <span className="flex items-center gap-0.5"><Clock className="h-2.5 w-2.5" />{day.gpu_hours.toFixed(1)}h</span>
                    <span className="font-mono">{formatCurrency(day.spend)}</span>
                  </div>
                  {day.avg_util > 0 && (
                    <div className="mt-1.5 h-1 rounded-full bg-surface overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${Math.min(day.avg_util, 100)}%`,
                          background: day.avg_util >= 80 ? CHART_COLORS.emerald : day.avg_util >= 50 ? CHART_COLORS.gold : CHART_COLORS.red,
                        }}
                      />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>
    </ScrollReveal>
  );
}

// ── 18. Insight Card (auto-detected) ──────────────────────────────────

export interface Insight {
  type: "positive" | "negative" | "info";
  title: string;
  detail: string;
  metric?: string;
}

export function InsightCards({ insights }: { insights: Insight[] }) {
  if (!insights?.length) return null;
  const iconMap = {
    positive: <TrendingUp className="h-4 w-4 text-emerald" />,
    negative: <TrendingDown className="h-4 w-4 text-accent-red" />,
    info: <Zap className="h-4 w-4 text-accent-cyan" />,
  };
  const borderMap = {
    positive: "border-emerald/20 bg-emerald/[0.03]",
    negative: "border-accent-red/20 bg-accent-red/[0.03]",
    info: "border-accent-cyan/20 bg-accent-cyan/[0.03]",
  };

  return (
    <FadeIn delay={0.3}>
      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {insights.slice(0, 6).map((ins, i) => (
          <div
            key={i}
            className={`insight-card rounded-xl border p-4 transition-colors hover:bg-surface/50 ${borderMap[ins.type]}`}
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5 shrink-0">{iconMap[ins.type]}</div>
              <div className="min-w-0">
                <p className="text-xs font-semibold">{ins.title}</p>
                <p className="text-[11px] text-text-secondary mt-0.5 leading-relaxed">{ins.detail}</p>
                {ins.metric && (
                  <p className="text-sm font-mono font-bold mt-1.5">{ins.metric}</p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </FadeIn>
  );
}
