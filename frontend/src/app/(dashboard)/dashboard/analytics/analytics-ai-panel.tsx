"use client";

import {
  useState, useCallback, useRef, useEffect, useMemo,
} from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Sparkles, X, Send, Loader2, MessageSquare, Copy, Check,
  TrendingUp, TrendingDown, BarChart3, DollarSign, Cpu,
  ChevronDown, Brain, Zap, Clock, Database, ArrowUpRight,
  RotateCcw, Maximize2, Minimize2,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { formatMarkdown } from "@/lib/format-markdown";
import type { EnhancedAnalytics } from "@/lib/api";

// ── Types ──────────────────────────────────────────────────────────────

interface AiMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  tokenCount?: number;
}

interface Suggestion {
  label: string;
  prompt: string;
  icon: React.ElementType;
  color: string;
}

// ── Suggestion chips per tab ───────────────────────────────────────────

const OVERVIEW_SUGGESTIONS: Suggestion[] = [
  { label: "Full dashboard summary", prompt: "Give me a comprehensive summary of my entire analytics dashboard — all KPIs, trends, and key takeaways. What story does my data tell?", icon: BarChart3, color: "cyan" },
  { label: "What trends stand out?", prompt: "Analyse all of my trends across spending, jobs, utilisation, and GPU hours. What are the most notable patterns? Are things improving, stable, or declining?", icon: TrendingUp, color: "emerald" },
  { label: "Spending efficiency audit", prompt: "Do a deep-dive efficiency analysis: compare my cost per job, cost per GPU hour, and spending trajectory. Am I getting good value? Where can I optimise?", icon: DollarSign, color: "gold" },
  { label: "Activity patterns & peaks", prompt: "Analyse my activity heatmap, peak days, and usage timing. When am I most active? Are there patterns I should know about? Any scheduling optimisations I could make?", icon: Cpu, color: "violet" },
];

const COMPUTE_SUGGESTIONS: Suggestion[] = [
  { label: "Best GPU for my workloads?", prompt: "Compare all my GPU models across utilisation, cost per hour, job duration, and total hours. Which gives me the best value? Which should I use more or less?", icon: Cpu, color: "cyan" },
  { label: "Utilisation deep-dive", prompt: "Walk me through my GPU utilisation trend day by day. Is it healthy? What's the trajectory? What specific actions could improve it?", icon: TrendingUp, color: "emerald" },
  { label: "Job duration analysis", prompt: "Analyse my duration histogram in detail. What's the distribution shape? Are most jobs short or long? Are there outliers? How does duration relate to cost?", icon: Clock, color: "violet" },
  { label: "GPU hours breakdown", prompt: "Break down my daily GPU hours — total, average, peak, and trend. How does GPU hour consumption correlate with my spending?", icon: Zap, color: "gold" },
];

const FINANCIAL_SUGGESTIONS: Suggestion[] = [
  { label: "Complete spending breakdown", prompt: "Give me a full spending breakdown: cumulative spend trajectory, daily averages, cost per hour trends, and spending by GPU model. Where is my money going?", icon: DollarSign, color: "gold" },
  { label: "Cost per hour analysis", prompt: "Deep-dive into my cost-per-GPU-hour trend. What's driving changes? How does it vary by GPU model? What's the optimal rate for my workloads?", icon: TrendingUp, color: "emerald" },
  { label: "Wallet health check", prompt: "Analyse my wallet activity — deposits, charges, and net flow. Am I topping up enough? Any unusual transactions? What's my burn rate?", icon: Database, color: "cyan" },
  { label: "Cost optimisation plan", prompt: "Based on all my financial data, create a specific action plan to reduce costs while maintaining performance. Give me 3-5 concrete recommendations with expected savings.", icon: ArrowUpRight, color: "violet" },
];

const PROVIDER_SUGGESTIONS: Suggestion[] = [
  { label: "Revenue deep-dive", prompt: "Analyse my provider revenue trend day by day. What's the trajectory? How does it compare to my total jobs served? Where are the growth opportunities?", icon: DollarSign, color: "gold" },
  { label: "Utilisation optimisation", prompt: "My host utilisation directly impacts revenue. Analyse my utilisation patterns and give specific recommendations to maximise GPU uptime and earnings.", icon: TrendingUp, color: "emerald" },
  { label: "Provider performance score", prompt: "Rate my overall provider performance across all metrics: revenue, utilisation, jobs served, and reliability. Where do I rank and how can I improve?", icon: Zap, color: "cyan" },
];

const TAB_SUGGESTIONS: Record<string, Suggestion[]> = {
  overview: OVERVIEW_SUGGESTIONS,
  compute: COMPUTE_SUGGESTIONS,
  financial: FINANCIAL_SUGGESTIONS,
  provider: PROVIDER_SUGGESTIONS,
};

// ── Serialise analytics for context ────────────────────────────────────

function pctChange(current: number, previous: number): string | null {
  if (!previous || previous === 0) return null;
  const pct = ((current - previous) / previous) * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(1)}%`;
}

function serializeAnalytics(
  summary: any,
  enhanced: EnhancedAnalytics | null,
  previousSummary: any,
  range: number,
): string {
  const sections: string[] = [];

  // ── Period & role context
  sections.push(`=== ANALYTICS PERIOD: Last ${range} days ===`);

  // ── Core KPIs with deltas
  if (summary) {
    const totalJobs = Number(summary.total_jobs ?? 0);
    const totalSpend = Number(summary.total_spend_cad ?? 0);
    const totalHours = Number(summary.total_gpu_hours ?? 0);
    const avgUtil = Number(summary.avg_gpu_utilization_pct ?? 0);
    const costPerJob = totalJobs > 0 ? totalSpend / totalJobs : 0;
    const costPerHour = totalHours > 0 ? totalSpend / totalHours : 0;
    const avgDuration = totalJobs > 0 ? (totalHours * 60) / totalJobs : 0;

    const prevJobs = Number(previousSummary?.total_jobs ?? 0);
    const prevSpend = Number(previousSummary?.total_spend_cad ?? 0);
    const prevHours = Number(previousSummary?.total_gpu_hours ?? 0);
    const prevUtil = Number(previousSummary?.avg_gpu_utilization_pct ?? 0);

    sections.push(`\n=== CORE KPIs ===`);
    sections.push(`Total Jobs: ${totalJobs}${prevJobs ? ` (prev: ${prevJobs}, change: ${pctChange(totalJobs, prevJobs)})` : ""}`);
    sections.push(`Total Spend: $${totalSpend.toFixed(2)} CAD${prevSpend ? ` (prev: $${prevSpend.toFixed(2)}, change: ${pctChange(totalSpend, prevSpend)})` : ""}`);
    sections.push(`Total GPU Hours: ${totalHours.toFixed(2)}h${prevHours ? ` (prev: ${prevHours.toFixed(2)}h, change: ${pctChange(totalHours, prevHours)})` : ""}`);
    sections.push(`Avg GPU Utilisation: ${avgUtil.toFixed(1)}%${prevUtil ? ` (prev: ${prevUtil.toFixed(1)}%, change: ${pctChange(avgUtil, prevUtil)})` : ""}`);
    sections.push(`\n=== DERIVED METRICS ===`);
    sections.push(`Cost Per Job: $${costPerJob.toFixed(3)}`);
    sections.push(`Cost Per GPU Hour: $${costPerHour.toFixed(3)}/hr`);
    sections.push(`Avg Job Duration: ${avgDuration.toFixed(1)} minutes`);

    if (totalJobs > 0 && range > 0) {
      sections.push(`Daily Avg Jobs: ${(totalJobs / range).toFixed(1)}/day`);
      sections.push(`Daily Avg Spend: $${(totalSpend / range).toFixed(2)}/day`);
    }
  }

  if (enhanced) {
    // ── Cost per hour trend with statistics
    if (enhanced.cost_per_hour_trend?.length) {
      const cph = enhanced.cost_per_hour_trend;
      const rates = cph.map(d => d.cost_per_hour).filter(v => v > 0);
      if (rates.length > 0) {
        const avg = rates.reduce((s, v) => s + v, 0) / rates.length;
        const min = Math.min(...rates);
        const max = Math.max(...rates);
        const sorted = [...rates].sort((a, b) => a - b);
        const median = sorted[Math.floor(sorted.length / 2)];
        // Trend direction: compare first half vs second half
        const half = Math.floor(rates.length / 2);
        const firstHalf = rates.slice(0, half).reduce((s, v) => s + v, 0) / (half || 1);
        const secondHalf = rates.slice(half).reduce((s, v) => s + v, 0) / (rates.length - half || 1);
        const trendDir = secondHalf > firstHalf * 1.05 ? "INCREASING" : secondHalf < firstHalf * 0.95 ? "DECREASING" : "STABLE";

        sections.push(`\n=== COST PER HOUR TREND (${cph.length} data points) ===`);
        sections.push(`Average: $${avg.toFixed(3)}/hr | Median: $${median.toFixed(3)}/hr | Range: $${min.toFixed(3)}-$${max.toFixed(3)}/hr`);
        sections.push(`Trend direction: ${trendDir} (first-half avg: $${firstHalf.toFixed(3)}, second-half avg: $${secondHalf.toFixed(3)})`);
        sections.push(`Recent 5 days: ${cph.slice(-5).map(d => `${d.date}: $${d.cost_per_hour.toFixed(3)}/hr (${d.gpu_hours.toFixed(1)}h, $${d.spend.toFixed(2)})`).join(" | ")}`);
      }
    }

    // ── Cumulative spend trajectory
    if (enhanced.cumulative_spend?.length) {
      const cs = enhanced.cumulative_spend;
      const finalTotal = cs[cs.length - 1]?.total ?? 0;
      const midpoint = cs[Math.floor(cs.length / 2)]?.total ?? 0;
      const quarter = cs[Math.floor(cs.length / 4)]?.total ?? 0;
      sections.push(`\n=== CUMULATIVE SPEND ===`);
      sections.push(`Final total: $${finalTotal.toFixed(2)} over ${cs.length} days`);
      sections.push(`25% mark: $${quarter.toFixed(2)} | 50% mark: $${midpoint.toFixed(2)} | 100%: $${finalTotal.toFixed(2)}`);
      if (cs.length >= 7) {
        const last7Spend = (cs[cs.length - 1]?.total ?? 0) - (cs[cs.length - 8]?.total ?? 0);
        sections.push(`Last 7 days spend: $${last7Spend.toFixed(2)}`);
      }
    }

    // ── Duration histogram with analysis
    if (enhanced.duration_histogram?.length) {
      const dh = enhanced.duration_histogram;
      const totalDurJobs = dh.reduce((s, d) => s + d.count, 0);
      const peakBucket = dh.reduce((a, b) => a.count > b.count ? a : b);
      sections.push(`\n=== DURATION HISTOGRAM (${totalDurJobs} total jobs) ===`);
      sections.push(`Peak bucket: "${peakBucket.bucket}" with ${peakBucket.count} jobs ($${peakBucket.total_cost.toFixed(2)} total cost)`);
      sections.push(`Full breakdown: ${dh.map(d => `${d.bucket}: ${d.count} jobs ($${d.total_cost.toFixed(2)})`).join(" | ")}`);
    }

    // ── Daily GPU hours with trend
    if (enhanced.daily_gpu_hours?.length) {
      const gh = enhanced.daily_gpu_hours;
      const total = gh.reduce((s, d) => s + d.hours, 0);
      const daily = total / gh.length;
      const peakDay = gh.reduce((a, b) => a.hours > b.hours ? a : b);
      const minDay = gh.reduce((a, b) => a.hours < b.hours ? a : b);
      sections.push(`\n=== GPU HOURS (${gh.length} days) ===`);
      sections.push(`Total: ${total.toFixed(1)}h | Daily average: ${daily.toFixed(2)}h/day`);
      sections.push(`Peak: ${peakDay.date} (${peakDay.hours.toFixed(2)}h) | Min: ${minDay.date} (${minDay.hours.toFixed(2)}h)`);
    }

    // ── Heatmap with top patterns
    if (enhanced.hourly_heatmap?.length) {
      const hm = enhanced.hourly_heatmap;
      const dowNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
      const sorted = [...hm].sort((a, b) => b.count - a.count);
      const top5 = sorted.slice(0, 5);
      const totalHmJobs = hm.reduce((s, d) => s + d.count, 0);

      // Aggregate by day of week
      const byDow: Record<number, number> = {};
      hm.forEach(d => { byDow[d.dow] = (byDow[d.dow] || 0) + d.count; });
      const busiestDow = Object.entries(byDow).sort(([, a], [, b]) => b - a)[0];

      // Aggregate by hour
      const byHour: Record<number, number> = {};
      hm.forEach(d => { byHour[d.hour] = (byHour[d.hour] || 0) + d.count; });
      const busiestHour = Object.entries(byHour).sort(([, a], [, b]) => b - a)[0];

      sections.push(`\n=== ACTIVITY HEATMAP (${totalHmJobs} total jobs) ===`);
      sections.push(`Busiest day: ${dowNames[Number(busiestDow?.[0] ?? 0)]} (${busiestDow?.[1] ?? 0} jobs)`);
      sections.push(`Busiest hour: ${busiestHour?.[0] ?? 0}:00 (${busiestHour?.[1] ?? 0} jobs)`);
      sections.push(`Top 5 slots: ${top5.map(d => `${dowNames[d.dow]} ${d.hour}:00 (${d.count} jobs)`).join(" | ")}`);
    }

    // ── Data sovereignty
    if (enhanced.sovereignty) {
      const sov = enhanced.sovereignty;
      sections.push(`\n=== DATA SOVEREIGNTY ===`);
      sections.push(`Canadian: ${sov.canadian_jobs} jobs (${sov.canadian_pct.toFixed(1)}%) — $${sov.canadian_spend.toFixed(2)} spend`);
      sections.push(`International: ${sov.total_jobs - sov.canadian_jobs} jobs (${(100 - sov.canadian_pct).toFixed(1)}%) — $${sov.international_spend.toFixed(2)} spend`);
    }

    // ── GPU performance comparison
    if (enhanced.gpu_performance?.length) {
      const gpus = enhanced.gpu_performance.slice(0, 8);
      sections.push(`\n=== GPU MODEL PERFORMANCE (${gpus.length} models) ===`);
      gpus.forEach((g, i) => {
        const efficiencyScore = g.avg_util > 0 ? (g.avg_util / g.avg_cost_per_hour).toFixed(1) : "N/A";
        sections.push(`${i + 1}. ${g.gpu_model}: ${g.jobs} jobs, ${g.avg_util.toFixed(1)}% util, $${g.avg_cost_per_hour.toFixed(3)}/hr, ${g.avg_duration_min.toFixed(0)}min avg, ${g.gpu_hours.toFixed(1)}h total, $${g.total_cost.toFixed(2)} cost, efficiency=${efficiencyScore}`);
      });
      // Best value GPU
      const bestValue = gpus.filter(g => g.avg_cost_per_hour > 0).sort((a, b) => (b.avg_util / b.avg_cost_per_hour) - (a.avg_util / a.avg_cost_per_hour))[0];
      if (bestValue) {
        sections.push(`Best value GPU: ${bestValue.gpu_model} (highest util/cost ratio)`);
      }
    }

    // ── Top entities
    if (enhanced.top_entities?.length) {
      sections.push(`\n=== TOP ENTITIES ===`);
      enhanced.top_entities.slice(0, 6).forEach((e, i) => {
        sections.push(`${i + 1}. ${e.entity}: ${e.job_count} jobs, $${e.total_cost.toFixed(2)} spend, ${e.gpu_hours.toFixed(1)}h`);
      });
    }

    // ── Wallet activity
    if (enhanced.wallet_activity?.length) {
      const wa = enhanced.wallet_activity;
      const deposits = wa.filter(w => w.tx_type === "deposit" || w.tx_type === "credit");
      const charges = wa.filter(w => w.tx_type === "charge" || w.tx_type === "debit");
      const totalIn = deposits.reduce((s, d) => s + Math.abs(d.total_amount), 0);
      const totalOut = charges.reduce((s, d) => s + Math.abs(d.total_amount), 0);
      sections.push(`\n=== WALLET ACTIVITY (${wa.length} transactions) ===`);
      sections.push(`Total deposits/credits: $${totalIn.toFixed(2)} | Total charges/debits: $${totalOut.toFixed(2)}`);
      sections.push(`Net flow: ${totalIn - totalOut >= 0 ? "+" : ""}$${(totalIn - totalOut).toFixed(2)}`);
      sections.push(`Recent activity: ${wa.slice(0, 5).map(w => `${w.date}: ${w.tx_type} $${w.total_amount.toFixed(2)} (×${w.tx_count})`).join(" | ")}`);
    }

    // ── Peak days
    if (enhanced.peak_days?.length) {
      const pd = enhanced.peak_days;
      sections.push(`\n=== PEAK DAYS (Top ${pd.length}) ===`);
      pd.forEach((d, i) => {
        sections.push(`${i + 1}. ${d.date}: ${d.jobs} jobs, ${d.gpu_hours.toFixed(1)}h, $${d.spend.toFixed(2)}, ${d.avg_util.toFixed(1)}% util`);
      });
    }

    // ── Provider data
    if (enhanced.provider_summary) {
      const ps = enhanced.provider_summary;
      sections.push(`\n=== PROVIDER SUMMARY ===`);
      sections.push(`Jobs served: ${ps.total_jobs_served} | Revenue: $${ps.total_revenue.toFixed(2)} | GPU hours: ${ps.total_gpu_hours.toFixed(1)}h | Avg util: ${ps.avg_util.toFixed(1)}%`);
      const revenuePerJob = ps.total_jobs_served > 0 ? ps.total_revenue / ps.total_jobs_served : 0;
      const revenuePerHour = ps.total_gpu_hours > 0 ? ps.total_revenue / ps.total_gpu_hours : 0;
      sections.push(`Revenue per job: $${revenuePerJob.toFixed(3)} | Revenue per GPU hour: $${revenuePerHour.toFixed(3)}/hr`);
    }

    if (enhanced.provider_daily?.length) {
      const pd = enhanced.provider_daily;
      const totalRev = pd.reduce((s, d) => s + d.total_revenue, 0);
      const avgDailyRev = totalRev / pd.length;
      sections.push(`\n=== PROVIDER DAILY (${pd.length} days) ===`);
      sections.push(`Total revenue: $${totalRev.toFixed(2)} | Daily average: $${avgDailyRev.toFixed(2)}/day`);
      sections.push(`Recent 5 days: ${pd.slice(-5).map(d => `${d.date}: $${d.total_revenue.toFixed(2)} (${d.jobs_served} jobs, ${d.avg_util.toFixed(1)}% util)`).join(" | ")}`);
    }
  }

  return sections.join("\n");
}

// ── Helpers ────────────────────────────────────────────────────────────

function formatTime(ts: number): string {
  return new Date(ts).toLocaleTimeString("en-CA", { hour: "2-digit", minute: "2-digit" });
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(() => {
    navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, [text]);
  return (
    <button
      onClick={copy}
      className="opacity-0 group-hover:opacity-100 transition-opacity h-5 w-5 rounded flex items-center justify-center hover:bg-surface-hover"
      title="Copy message"
    >
      {copied ? <Check className="h-3 w-3 text-emerald" /> : <Copy className="h-3 w-3 text-text-muted" />}
    </button>
  );
}

function ThinkingIndicator() {
  return (
    <div className="flex items-center gap-2 px-1">
      <div className="flex gap-1">
        {[0, 1, 2].map(i => (
          <motion.div
            key={i}
            className="h-1.5 w-1.5 rounded-full bg-accent-cyan"
            animate={{ opacity: [0.3, 1, 0.3], scale: [0.8, 1.1, 0.8] }}
            transition={{ duration: 1.2, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
      <span className="text-xs text-text-muted">Analysing your data…</span>
    </div>
  );
}

function DataContextBadge({ dataPoints }: { dataPoints: number }) {
  return (
    <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-emerald/5 border border-emerald/15 text-[10px] text-emerald font-mono">
      <Database className="h-2.5 w-2.5" />
      {dataPoints} data points loaded
    </div>
  );
}

const SUGGESTION_COLORS: Record<string, { bg: string; border: string; text: string; icon: string }> = {
  cyan: { bg: "bg-accent-cyan/5", border: "border-accent-cyan/15 hover:border-accent-cyan/35", text: "text-accent-cyan", icon: "text-accent-cyan/60 group-hover:text-accent-cyan" },
  emerald: { bg: "bg-emerald/5", border: "border-emerald/15 hover:border-emerald/35", text: "text-emerald", icon: "text-emerald/60 group-hover:text-emerald" },
  gold: { bg: "bg-accent-gold/5", border: "border-accent-gold/15 hover:border-accent-gold/35", text: "text-accent-gold", icon: "text-accent-gold/60 group-hover:text-accent-gold" },
  violet: { bg: "bg-accent-violet/5", border: "border-accent-violet/15 hover:border-accent-violet/35", text: "text-accent-violet", icon: "text-accent-violet/60 group-hover:text-accent-violet" },
};

// ── Panel Component ────────────────────────────────────────────────────

export function AnalyticsAiPanel({
  open,
  onClose,
  tab,
  summary,
  enhanced,
  previousSummary,
  range,
}: {
  open: boolean;
  onClose: () => void;
  tab: string;
  summary: any;
  enhanced: EnhancedAnalytics | null;
  previousSummary: any;
  range: number;
}) {
  const [messages, setMessages] = useState<AiMessage[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Auto-scroll to the bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Focus input when panel opens
  useEffect(() => {
    if (open && inputRef.current) {
      setTimeout(() => inputRef.current?.focus(), 250);
    }
  }, [open]);

  // Escape key to close
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape" && !isStreaming) onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose, isStreaming]);

  // Current suggestions based on active tab
  const suggestions = useMemo(() => TAB_SUGGESTIONS[tab] ?? OVERVIEW_SUGGESTIONS, [tab]);

  // Serialise data for AI context
  const analyticsContext = useMemo(
    () => serializeAnalytics(summary, enhanced, previousSummary, range),
    [summary, enhanced, previousSummary, range],
  );

  // Count data points for badge
  const dataPointCount = useMemo(() => {
    let count = 0;
    if (summary?.total_jobs) count += 4; // core KPIs
    if (previousSummary?.total_jobs) count += 4;
    if (enhanced) {
      count += (enhanced.cost_per_hour_trend?.length ?? 0);
      count += (enhanced.cumulative_spend?.length ?? 0);
      count += (enhanced.duration_histogram?.length ?? 0);
      count += (enhanced.daily_gpu_hours?.length ?? 0);
      count += (enhanced.hourly_heatmap?.length ?? 0);
      count += (enhanced.gpu_performance?.length ?? 0);
      count += (enhanced.top_entities?.length ?? 0);
      count += (enhanced.wallet_activity?.length ?? 0);
      count += (enhanced.peak_days?.length ?? 0);
      count += (enhanced.provider_daily?.length ?? 0);
      if (enhanced.sovereignty) count += 5;
      if (enhanced.provider_summary) count += 4;
    }
    return count;
  }, [summary, enhanced, previousSummary]);

  // Auto-resize textarea
  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 120)}px`;
  }, []);

  const sendMessage = useCallback(async (messageText: string) => {
    if (!messageText.trim() || isStreaming) return;

    setError(null);

    const userMsg: AiMessage = {
      id: crypto.randomUUID(),
      role: "user",
      content: messageText.trim(),
      timestamp: Date.now(),
    };

    const assistantId = crypto.randomUUID();
    const assistantMsg: AiMessage = {
      id: assistantId,
      role: "assistant",
      content: "",
      timestamp: Date.now(),
    };

    setMessages(prev => [...prev, userMsg, assistantMsg]);
    setInput("");
    // Reset textarea height
    if (inputRef.current) inputRef.current.style.height = "auto";
    setIsStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const res = await fetch("/api/ai/analytics", {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: messageText.trim(),
          conversation_id: conversationId,
          chart_context: tab,
          analytics_summary: analyticsContext,
        }),
        signal: controller.signal,
      });

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Error ${res.status}`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response stream");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          try {
            const data = JSON.parse(line.slice(6));

            if (data.type === "meta" && data.conversation_id) {
              setConversationId(data.conversation_id);
            } else if (data.type === "token" && data.content) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + data.content }
                    : m,
                ),
              );
            } else if (data.type === "error") {
              setError(data.message || "An error occurred");
            }
          } catch {
            // Skip malformed lines
          }
        }
      }
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "Failed to get response");
      }
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, [isStreaming, conversationId, tab, analyticsContext]);

  const handleSubmit = useCallback((e?: React.FormEvent) => {
    e?.preventDefault();
    sendMessage(input);
  }, [sendMessage, input]);

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  }, [handleSubmit]);

  const handleNewChat = useCallback(() => {
    if (isStreaming && abortRef.current) {
      abortRef.current.abort();
    }
    setMessages([]);
    setConversationId(null);
    setError(null);
    setIsStreaming(false);
  }, [isStreaming]);

  if (!open) return null;

  const panelWidth = expanded ? "w-[680px]" : "w-[420px]";

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-40 bg-black/20 backdrop-blur-[2px]"
        onClick={onClose}
      />

      <AnimatePresence mode="wait">
        <motion.div
          key={expanded ? "expanded" : "collapsed"}
          initial={{ x: "100%", opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          exit={{ x: "100%", opacity: 0 }}
          transition={{ type: "spring", damping: 30, stiffness: 300 }}
          className={cn(
            "fixed top-0 right-0 h-full max-w-[calc(100vw-2rem)] z-50",
            panelWidth,
            "flex flex-col",
            "bg-[#0a0e1a]/98 backdrop-blur-2xl",
            "border-l border-accent-cyan/8",
            "shadow-[-8px_0_40px_rgba(0,0,0,0.5)]",
          )}
        >
          {/* ── Gradient top accent ─────────────────── */}
          <div className="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-accent-cyan/40 to-transparent" />

          {/* ── Header ─────────────────────────────────── */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border/30">
            <div className="flex items-center gap-3">
              <div className="relative">
                <div className="h-9 w-9 rounded-xl bg-gradient-to-br from-accent-cyan/15 to-accent-violet/15 flex items-center justify-center ring-1 ring-accent-cyan/20">
                  <Brain className="h-4.5 w-4.5 text-accent-cyan" />
                </div>
                <motion.div
                  className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-emerald border-2 border-[#0a0e1a]"
                  animate={{ scale: [1, 1.2, 1] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <h3 className="text-sm font-semibold bg-gradient-to-r from-text-primary to-text-secondary bg-clip-text">Analytics AI</h3>
                  <span className="text-[9px] font-mono px-1.5 py-0.5 rounded-full bg-accent-cyan/8 text-accent-cyan border border-accent-cyan/15">
                    LIVE
                  </span>
                </div>
                <p className="text-[10px] text-text-muted mt-0.5">Full dashboard awareness</p>
              </div>
            </div>
            <div className="flex items-center gap-1">
              <DataContextBadge dataPoints={dataPointCount} />
              {messages.length > 0 && (
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-7 px-2 text-[10px] text-text-muted hover:text-text-primary gap-1"
                  onClick={handleNewChat}
                >
                  <RotateCcw className="h-3 w-3" /> New
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="h-7 w-7 p-0"
                onClick={() => setExpanded(!expanded)}
                title={expanded ? "Collapse" : "Expand"}
              >
                {expanded ? <Minimize2 className="h-3.5 w-3.5" /> : <Maximize2 className="h-3.5 w-3.5" />}
              </Button>
              <Button variant="ghost" size="sm" className="h-7 w-7 p-0" onClick={onClose}>
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>

          {/* ── Messages Area ──────────────────────────── */}
          <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3 scroll-smooth">
            {messages.length === 0 ? (
              /* ── Empty State with suggestion chips ─── */
              <div className="flex flex-col items-center justify-center h-full px-2">
                <motion.div
                  initial={{ scale: 0.8, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: "spring", damping: 20 }}
                  className="relative h-20 w-20 rounded-2xl bg-gradient-to-br from-accent-cyan/8 to-accent-violet/8 flex items-center justify-center ring-1 ring-accent-cyan/10 mb-5"
                >
                  <Sparkles className="h-8 w-8 text-accent-cyan/80" />
                  <motion.div
                    className="absolute inset-0 rounded-2xl ring-1 ring-accent-cyan/20"
                    animate={{ scale: [1, 1.15, 1], opacity: [0.5, 0, 0.5] }}
                    transition={{ duration: 3, repeat: Infinity }}
                  />
                </motion.div>
                <h4 className="text-base font-semibold mb-1">Analytics Intelligence</h4>
                <p className="text-xs text-text-muted text-center max-w-[300px] mb-2 leading-relaxed">
                  I have full visibility into every chart, metric, and trend on your dashboard.
                </p>
                <p className="text-[10px] text-text-muted/70 text-center max-w-[280px] mb-6">
                  Ask me to explain patterns, compare periods, find anomalies, or suggest optimisations.
                </p>

                <div className="w-full space-y-2">
                  <p className="text-[10px] text-text-muted uppercase tracking-widest font-mono px-1 mb-1">
                    {tab === "overview" ? "Start exploring" : `${tab} questions`}
                  </p>
                  {suggestions.map((s, i) => {
                    const Icon = s.icon;
                    const colors = SUGGESTION_COLORS[s.color] ?? SUGGESTION_COLORS.cyan;
                    return (
                      <motion.button
                        key={i}
                        initial={{ opacity: 0, x: 12 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.1 + i * 0.06 }}
                        onClick={() => sendMessage(s.prompt)}
                        className={cn(
                          "w-full flex items-center gap-3 px-3 py-2.5 rounded-xl",
                          colors.bg, "border", colors.border,
                          "hover:bg-surface/60",
                          "transition-all duration-200 text-left group",
                        )}
                      >
                        <div className={cn(
                          "h-7 w-7 shrink-0 rounded-lg flex items-center justify-center",
                          "bg-surface/80 border border-border/30 group-hover:border-border/50 transition-colors",
                        )}>
                          <Icon className={cn("h-3.5 w-3.5 transition-colors", colors.icon)} />
                        </div>
                        <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors leading-snug">
                          {s.label}
                        </span>
                        <ArrowUpRight className="h-3 w-3 shrink-0 ml-auto text-text-muted/0 group-hover:text-text-muted transition-all group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
                      </motion.button>
                    );
                  })}
                </div>
              </div>
            ) : (
              /* ── Message bubbles ─────────────────────── */
              <>
                {messages.map((msg, idx) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}
                    className={cn(
                      "flex gap-2.5 group",
                      msg.role === "user" ? "flex-row-reverse" : "flex-row",
                    )}
                  >
                    {/* Avatar */}
                    {msg.role === "assistant" && (
                      <div className="h-6 w-6 shrink-0 rounded-lg bg-gradient-to-br from-accent-cyan/20 to-accent-violet/20 flex items-center justify-center ring-1 ring-accent-cyan/15 mt-0.5">
                        <Sparkles className="h-3 w-3 text-accent-cyan" />
                      </div>
                    )}

                    {/* Bubble */}
                    <div className="max-w-[88%] flex flex-col gap-1">
                      <div
                        className={cn(
                          "rounded-2xl px-3.5 py-2.5 text-[13px] leading-relaxed",
                          msg.role === "user"
                            ? "bg-accent-cyan/10 border border-accent-cyan/20 text-text-primary rounded-br-md"
                            : "bg-surface/40 border border-border/20 text-text-primary rounded-bl-md",
                        )}
                      >
                        {msg.role === "assistant" && msg.content ? (
                          <div
                            className="prose prose-invert prose-sm max-w-none prose-p:my-1.5 prose-li:my-0.5 prose-headings:my-2 prose-strong:text-accent-cyan prose-code:text-accent-cyan/90 prose-code:bg-surface prose-code:px-1 prose-code:rounded"
                            dangerouslySetInnerHTML={{ __html: formatMarkdown(msg.content) }}
                          />
                        ) : msg.role === "assistant" && !msg.content && isStreaming ? (
                          <ThinkingIndicator />
                        ) : (
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        )}
                      </div>

                      {/* Timestamp + copy */}
                      <div className={cn(
                        "flex items-center gap-1.5 px-1",
                        msg.role === "user" ? "justify-end" : "justify-start",
                      )}>
                        <span className="text-[9px] text-text-muted/50 font-mono">
                          {formatTime(msg.timestamp)}
                        </span>
                        {msg.role === "assistant" && msg.content && (
                          <CopyButton text={msg.content} />
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}

                {/* Follow-up suggestions after AI response */}
                {messages.length >= 2 && !isStreaming && (
                  <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                    className="flex flex-wrap gap-1.5 pt-1"
                  >
                    {["Tell me more", "What should I do?", "Compare with last period"].map((q, i) => (
                      <button
                        key={i}
                        onClick={() => sendMessage(q)}
                        className="px-2.5 py-1 rounded-lg text-[11px] bg-surface/50 border border-border/30 text-text-muted hover:text-text-primary hover:border-accent-cyan/20 transition-all"
                      >
                        {q}
                      </button>
                    ))}
                  </motion.div>
                )}
              </>
            )}

            {/* Error display */}
            {error && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                className="flex items-center gap-2 px-3 py-2.5 rounded-xl bg-accent-red/5 border border-accent-red/20 text-xs text-accent-red"
              >
                <span className="shrink-0">⚠</span>
                <span>{error}</span>
                <button onClick={() => setError(null)} className="ml-auto text-accent-red/60 hover:text-accent-red">
                  <X className="h-3 w-3" />
                </button>
              </motion.div>
            )}
          </div>

          {/* ── Input Area ─────────────────────────────── */}
          <div className="border-t border-border/30 px-4 py-3 bg-[#0a0e1a]/50">
            <form onSubmit={handleSubmit} className="relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder="Ask about your analytics…"
                rows={1}
                className={cn(
                  "w-full resize-none rounded-xl bg-surface/40 border border-border/30",
                  "px-4 py-3 pr-12 text-sm text-text-primary placeholder:text-text-muted/60",
                  "focus:outline-none focus:border-accent-cyan/30 focus:ring-1 focus:ring-accent-cyan/10",
                  "transition-all duration-200",
                )}
                style={{ minHeight: "44px", maxHeight: "120px" }}
                disabled={isStreaming}
              />
              <button
                type="submit"
                disabled={!input.trim() || isStreaming}
                className={cn(
                  "absolute right-2.5 bottom-2.5 h-8 w-8 rounded-lg flex items-center justify-center",
                  "transition-all duration-200",
                  input.trim() && !isStreaming
                    ? "bg-gradient-to-r from-accent-cyan to-accent-cyan/80 text-[#0a0e1a] shadow-lg shadow-accent-cyan/20 hover:shadow-accent-cyan/40"
                    : "bg-surface-hover text-text-muted cursor-not-allowed",
                )}
              >
                {isStreaming ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <Send className="h-3.5 w-3.5" />
                )}
              </button>
            </form>
            <div className="flex items-center justify-between mt-2 px-0.5">
              <p className="text-[9px] text-text-muted/50">
                Press <kbd className="px-1 py-0.5 rounded bg-surface text-[8px] border border-border/30 font-mono">Enter</kbd> to send · <kbd className="px-1 py-0.5 rounded bg-surface text-[8px] border border-border/30 font-mono">Esc</kbd> to close
              </p>
              <p className="text-[9px] text-text-muted/40">
                Powered by AI · May not be 100% accurate
              </p>
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </>
  );
}
