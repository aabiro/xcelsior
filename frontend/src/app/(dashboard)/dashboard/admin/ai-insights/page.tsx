"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { FadeIn, HoverCard, StaggerList, StaggerItem, CountUp } from "@/components/ui/motion";
import { StatCard } from "@/components/ui/stat-card";
import {
  MessageSquare, RefreshCw, Search, DollarSign,
  Zap, Hash, Bot, ChevronDown, ChevronUp, User, Clock,
} from "lucide-react";
import * as api from "@/lib/api";
import { toast } from "sonner";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, PieChart, Pie, Cell,
} from "recharts";
import { cn } from "@/lib/utils";

/* ───── Constants ──────────────────────────────────────────────────── */

const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

const SOURCE_META: Record<string, { label: string; color: string; dot: string; chart: string; badge: string }> = {
  support: {
    label: "Support AI",
    color: "text-ice-blue",
    dot: "bg-ice-blue",
    chart: "#7dd3fc",
    badge: "bg-ice-blue/15 text-ice-blue border border-ice-blue/30",
  },
  xcel: {
    label: "Xcel AI",
    color: "text-accent-violet",
    dot: "bg-accent-violet",
    chart: "#a78bfa",
    badge: "bg-accent-violet/15 text-accent-violet border border-accent-violet/30",
  },
  analytics: {
    label: "Analytics AI",
    color: "text-accent-cyan",
    dot: "bg-accent-cyan",
    chart: "#22d3ee",
    badge: "bg-accent-cyan/15 text-accent-cyan border border-accent-cyan/30",
  },
  wizard: {
    label: "Wizard AI",
    color: "text-accent-gold",
    dot: "bg-accent-gold",
    chart: "#fbbf24",
    badge: "bg-accent-gold/15 text-accent-gold border border-accent-gold/30",
  },
};

const SOURCES = ["all", "support", "xcel", "analytics", "wizard"] as const;

const RANGE_PRESETS = [
  { label: "3d", days: 3 },
  { label: "7d", days: 7 },
  { label: "14d", days: 14 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
];

/* ───── Helpers ────────────────────────────────────────────────────── */

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(n: number): string {
  return `$${n.toFixed(2)}`;
}

function relativeTime(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return `${Math.floor(diff / 86400)}d ago`;
}

/* ───── Page ───────────────────────────────────────────────────────── */

export default function AdminAiInsightsPage() {
  const [stats, setStats] = useState<api.AdminAiStats | null>(null);
  const [convosData, setConvosData] = useState<{
    conversations: api.AdminAiConversation[];
    total: number;
    page: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(30);
  const [source, setSource] = useState<string>("all");
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");
  const [page, setPage] = useState(1);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  const loadStats = useCallback(() => {
    api.fetchAdminAiStats(days).then(setStats).catch(() => toast.error("Failed to load AI stats"));
  }, [days]);

  const loadConvos = useCallback(() => {
    api.fetchAdminAiConversations(source, days, search, page)
      .then((d) => setConvosData({ conversations: d.conversations, total: d.total, page: d.page }))
      .catch(() => toast.error("Failed to load conversations"));
  }, [source, days, search, page]);

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([
      api.fetchAdminAiStats(days),
      api.fetchAdminAiConversations(source, days, search, page),
    ]).then(([s, c]) => {
      if (s.status === "fulfilled") setStats(s.value);
      if (c.status === "fulfilled")
        setConvosData({ conversations: c.value.conversations, total: c.value.total, page: c.value.page });
      setLoading(false);
    });
  }, [days, source, search, page]);

  useEffect(() => { load(); }, [load]);

  const toggleExpand = (id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const totalPages = convosData ? Math.max(1, Math.ceil(convosData.total / 30)) : 1;

  /* ─── Skeleton ──────────────────────────────────────────────────── */
  if (loading && !stats) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-text-primary">AI Insights</h2>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
        <div className="h-80 rounded-xl bg-surface skeleton-pulse" />
        <div className="space-y-3">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-20 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      </div>
    );
  }

  /* ─── Pie data ──────────────────────────────────────────────────── */
  const pieData = (stats?.by_source ?? [])
    .filter((s) => s.conversations > 0)
    .map((s) => ({
      name: SOURCE_META[s.source]?.label ?? s.source,
      value: s.conversations,
      color: SOURCE_META[s.source]?.chart ?? "#64748b",
    }));

  /* ─── Render ────────────────────────────────────────────────────── */
  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-xl font-semibold text-text-primary flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-accent-cyan" />
          AI Insights
        </h2>
        <div className="flex items-center gap-2">
          {/* Range pills */}
          <div className="flex items-center gap-1 rounded-lg bg-surface p-1">
            {RANGE_PRESETS.map((r) => (
              <button
                key={r.days}
                onClick={() => { setDays(r.days); setPage(1); }}
                className={cn(
                  "rounded-md px-2.5 py-1 text-xs font-medium transition-all",
                  days === r.days
                    ? "bg-card text-text-primary shadow-sm"
                    : "text-text-muted hover:text-text-primary",
                )}
              >
                {r.label}
              </button>
            ))}
          </div>
          <Button size="sm" variant="outline" onClick={load} className="gap-1.5">
            <RefreshCw className={cn("h-3.5 w-3.5", loading && "animate-spin")} />
            Refresh
          </Button>
        </div>
      </div>

      {/* ─── Stat cards ──────────────────────────────────────────── */}
      <StaggerList className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StaggerItem>
          <StatCard
            label="Total Conversations"
            value={<CountUp value={stats?.total_conversations ?? 0} />}
            icon={MessageSquare}
            glow="cyan"
          />
        </StaggerItem>
        <StaggerItem>
          <StatCard
            label="Total Messages"
            value={<CountUp value={stats?.total_messages ?? 0} />}
            icon={Hash}
            glow="violet"
          />
        </StaggerItem>
        <StaggerItem>
          <StatCard
            label="Total Tokens"
            value={formatTokens((stats?.total_input_tokens ?? 0) + (stats?.total_output_tokens ?? 0))}
            icon={Zap}
            glow="gold"
          />
        </StaggerItem>
        <StaggerItem>
          <StatCard
            label="Estimated Cost"
            value={formatCost(stats?.estimated_cost ?? 0)}
            icon={DollarSign}
            glow="emerald"
          />
        </StaggerItem>
      </StaggerList>

      {/* ─── Charts Row ──────────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Stacked area chart */}
        <FadeIn className="lg:col-span-2">
          <HoverCard>
            <Card className="glow-card brand-top-accent">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">
                  Conversations Over Time
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={stats?.daily ?? []}>
                    <defs>
                      {Object.entries(SOURCE_META).map(([key, m]) => (
                        <linearGradient key={key} id={`grad-${key}`} x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={m.chart} stopOpacity={0.3} />
                          <stop offset="95%" stopColor={m.chart} stopOpacity={0} />
                        </linearGradient>
                      ))}
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis
                      dataKey="date"
                      tick={{ fill: "#94a3b8", fontSize: 11 }}
                      tickFormatter={(v: string) => v.slice(5)}
                    />
                    <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} allowDecimals={false} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Legend
                      wrapperStyle={{ fontSize: 12, color: "#94a3b8" }}
                      formatter={(v: string) => SOURCE_META[v]?.label ?? v}
                    />
                    {Object.entries(SOURCE_META).map(([key, m]) => (
                      <Area
                        key={key}
                        type="monotone"
                        dataKey={key}
                        name={key}
                        stackId="1"
                        stroke={m.chart}
                        fill={`url(#grad-${key})`}
                        strokeWidth={2}
                      />
                    ))}
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </HoverCard>
        </FadeIn>

        {/* Pie chart */}
        <FadeIn>
          <HoverCard>
            <Card className="glow-card brand-top-accent">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">
                  By Source
                </CardTitle>
              </CardHeader>
              <CardContent className="flex items-center justify-center">
                {pieData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={280}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={3}
                        dataKey="value"
                        label={({ name, percent }) =>
                          `${name} ${((percent ?? 0) * 100).toFixed(0)}%`
                        }
                        labelLine={false}
                      >
                        {pieData.map((entry, i) => (
                          <Cell key={i} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={tooltipStyle} />
                    </PieChart>
                  </ResponsiveContainer>
                ) : (
                  <p className="text-text-muted text-sm py-12">No data in period</p>
                )}
              </CardContent>
            </Card>
          </HoverCard>
        </FadeIn>
      </div>

      {/* ─── Token breakdown ─────────────────────────────────────── */}
      <FadeIn>
        <HoverCard>
          <Card className="glow-card brand-top-accent">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-text-secondary">
                Token Usage &amp; Cost Breakdown
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                {(stats?.by_source ?? []).map((s) => {
                  const meta = SOURCE_META[s.source];
                  if (!meta) return null;
                  const totalTokens = s.input_tokens + s.output_tokens;
                  const cost = s.input_tokens * 0.000003 + s.output_tokens * 0.000015;
                  return (
                    <div key={s.source} className="rounded-lg bg-surface p-3 space-y-1.5">
                      <div className="flex items-center gap-1.5">
                        <span className={cn("h-2 w-2 rounded-full", meta.dot)} />
                        <span className={cn("text-xs font-medium", meta.color)}>{meta.label}</span>
                      </div>
                      <p className="text-lg font-bold text-text-primary">{formatTokens(totalTokens)}</p>
                      <p className="text-xs text-text-muted">
                        {formatTokens(s.input_tokens)} in / {formatTokens(s.output_tokens)} out
                      </p>
                      <p className="text-xs text-emerald">{formatCost(cost)}</p>
                    </div>
                  );
                })}
              </div>
            </CardContent>
          </Card>
        </HoverCard>
      </FadeIn>

      {/* ─── Top Users ───────────────────────────────────────────── */}
      {(stats?.top_users?.length ?? 0) > 0 && (
        <FadeIn>
          <HoverCard>
            <Card className="glow-card brand-top-accent">
              <CardHeader className="pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">
                  Most Active Users
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {stats!.top_users.map((u, i) => (
                    <div
                      key={u.user_id}
                      className="flex items-center justify-between rounded-lg bg-surface px-3 py-2"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-bold text-text-muted w-5">#{i + 1}</span>
                        <User className="h-3.5 w-3.5 text-text-muted" />
                        <span className="text-sm text-text-primary font-medium truncate max-w-[200px]">
                          {u.user_id}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-text-muted">
                        <span>{u.conversations} convos</span>
                        <span>{formatTokens(u.total_tokens)} tokens</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </HoverCard>
        </FadeIn>
      )}

      {/* ─── Filter / Search Bar ─────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-3">
        {/* Source pills */}
        <div className="flex items-center gap-1 rounded-lg bg-surface p-1">
          {SOURCES.map((s) => {
            const meta = SOURCE_META[s];
            return (
              <button
                key={s}
                onClick={() => { setSource(s); setPage(1); }}
                className={cn(
                  "flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-medium transition-all",
                  source === s
                    ? "bg-card text-text-primary shadow-sm"
                    : "text-text-muted hover:text-text-primary",
                )}
              >
                {meta && <span className={cn("h-1.5 w-1.5 rounded-full", meta.dot)} />}
                {s === "all" ? "All" : meta?.label ?? s}
              </button>
            );
          })}
        </div>
        {/* Search */}
        <form
          className="flex items-center gap-1.5"
          onSubmit={(e) => {
            e.preventDefault();
            setSearch(searchInput);
            setPage(1);
          }}
        >
          <Input
            placeholder="Search messages..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="h-8 w-48 text-xs"
          />
          <Button type="submit" size="sm" variant="outline" className="h-8 px-2">
            <Search className="h-3.5 w-3.5" />
          </Button>
        </form>
        {search && (
          <button
            onClick={() => { setSearch(""); setSearchInput(""); setPage(1); }}
            className="text-xs text-text-muted hover:text-accent-red transition-colors"
          >
            Clear search
          </button>
        )}
        <span className="ml-auto text-xs text-text-muted">
          {convosData?.total ?? 0} conversation{convosData?.total === 1 ? "" : "s"}
        </span>
      </div>

      {/* ─── Conversations List ──────────────────────────────────── */}
      <div className="space-y-3">
        {(convosData?.conversations ?? []).length === 0 && !loading ? (
          <FadeIn>
            <Card className="flex flex-col items-center justify-center py-16 text-center">
              <Bot className="h-12 w-12 text-text-muted/40 mb-3" />
              <p className="text-text-muted font-medium">No conversations found</p>
              <p className="text-text-muted/60 text-sm mt-1">
                Try adjusting filters or date range
              </p>
            </Card>
          </FadeIn>
        ) : (
          (convosData?.conversations ?? []).map((conv) => {
            const meta = SOURCE_META[conv.source] ?? SOURCE_META.xcel;
            const isOpen = expanded.has(conv.conversation_id);
            return (
              <FadeIn key={conv.conversation_id}>
                <Card className="overflow-hidden transition-shadow hover:shadow-md">
                  {/* Conversation header */}
                  <button
                    onClick={() => toggleExpand(conv.conversation_id)}
                    className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-surface/50 transition-colors"
                  >
                    <div className="flex items-center gap-3 min-w-0 flex-1">
                      <span className={cn("inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-semibold shrink-0", meta.badge)}>
                        <span className={cn("h-1.5 w-1.5 rounded-full", meta.dot)} />
                        {meta.label}
                      </span>
                      <span className="text-sm font-medium text-text-primary truncate">
                        {conv.title || "Untitled"}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0 ml-3">
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <User className="h-3 w-3" />
                        <span className="max-w-[120px] truncate">{conv.user}</span>
                      </div>
                      <div className="flex items-center gap-1 text-xs text-text-muted">
                        <Clock className="h-3 w-3" />
                        {relativeTime(conv.updated_at)}
                      </div>
                      <Badge variant="default">{conv.message_count} msg{conv.message_count !== 1 ? "s" : ""}</Badge>
                      {(conv.total_input_tokens + conv.total_output_tokens) > 0 && (
                        <span className="text-[10px] text-text-muted">
                          {formatTokens(conv.total_input_tokens + conv.total_output_tokens)} tok
                        </span>
                      )}
                      {isOpen ? (
                        <ChevronUp className="h-4 w-4 text-text-muted" />
                      ) : (
                        <ChevronDown className="h-4 w-4 text-text-muted" />
                      )}
                    </div>
                  </button>

                  {/* Expanded messages */}
                  {isOpen && (
                    <div className="border-t border-border bg-surface/30 px-4 py-3 space-y-2 max-h-[500px] overflow-y-auto">
                      {conv.messages.length === 0 ? (
                        <p className="text-xs text-text-muted italic">No messages</p>
                      ) : (
                        conv.messages.map((msg, i) => (
                          <div
                            key={i}
                            className={cn(
                              "rounded-lg px-3 py-2 text-sm",
                              msg.role === "user"
                                ? "bg-navy-lighter/50 border-l-2 border-accent-cyan/40"
                                : msg.role === "assistant"
                                ? "bg-surface border-l-2 border-accent-violet/40"
                                : "bg-surface/50 border-l-2 border-accent-gold/40",
                            )}
                          >
                            <div className="flex items-center gap-2 mb-1">
                              <span className={cn(
                                "text-[10px] font-semibold uppercase tracking-wider",
                                msg.role === "user" ? "text-accent-cyan" :
                                msg.role === "assistant" ? "text-accent-violet" : "text-accent-gold",
                              )}>
                                {msg.role}
                              </span>
                              {msg.tool_name && (
                                <span className="inline-flex items-center gap-1 rounded bg-navy-lighter px-1.5 py-0.5 text-[10px] text-text-muted">
                                  <Zap className="h-2.5 w-2.5" />
                                  {msg.tool_name}
                                </span>
                              )}
                              {(msg.tokens_in > 0 || msg.tokens_out > 0) && (
                                <span className="text-[10px] text-text-muted ml-auto">
                                  {msg.tokens_in > 0 && `${formatTokens(msg.tokens_in)} in`}
                                  {msg.tokens_in > 0 && msg.tokens_out > 0 && " / "}
                                  {msg.tokens_out > 0 && `${formatTokens(msg.tokens_out)} out`}
                                </span>
                              )}
                            </div>
                            <p className="text-text-secondary whitespace-pre-wrap break-words text-xs leading-relaxed">
                              {msg.content ? (
                                msg.content.length > 2000
                                  ? msg.content.slice(0, 2000) + "…"
                                  : msg.content
                              ) : (
                                <span className="italic text-text-muted">
                                  {msg.tool_name ? "(tool call)" : "(empty)"}
                                </span>
                              )}
                            </p>
                          </div>
                        ))
                      )}
                    </div>
                  )}
                </Card>
              </FadeIn>
            );
          })
        )}
      </div>

      {/* ─── Pagination ──────────────────────────────────────────── */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 pt-2">
          <Button
            size="sm"
            variant="outline"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            Previous
          </Button>
          <span className="text-xs text-text-muted">
            Page {page} of {totalPages}
          </span>
          <Button
            size="sm"
            variant="outline"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </Button>
        </div>
      )}
    </div>
  );
}
