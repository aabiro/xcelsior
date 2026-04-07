"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { ReputationBadge } from "@/components/ui/reputation-badge";
import {
  Star, RefreshCw, Users, Trophy, TrendingUp, BarChart3, ShieldCheck, ShieldOff,
} from "lucide-react";
import { useApi } from "@/lib/use-api";
import * as apiLib from "@/lib/api";
import type { ReputationEntry } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { toast } from "sonner";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import { cn } from "@/lib/utils";

// ── Tier config (self-contained — matches backend TIER_THRESHOLDS) ──────────

const TIERS = [
  {
    key: "new_user",
    label: "New User",
    icon: "🔘",
    threshold: 0,
    nextThreshold: 100,
    color: "#64748b",
    glow: "shadow-slate-500/30",
    searchBoost: "0.8×",
    pricingPremium: "0%",
    commission: "15%",
    perk: "Baseline access",
    unlock: "Create an account.",
  },
  {
    key: "bronze",
    label: "Bronze",
    icon: "🥉",
    threshold: 100,
    nextThreshold: 250,
    color: "#cd7f32",
    glow: "shadow-amber-700/40",
    searchBoost: "1.0×",
    pricingPremium: "0%",
    commission: "15%",
    perk: "Standard marketplace visibility",
    unlock: "Earn 100 reputation points.",
  },
  {
    key: "silver",
    label: "Silver",
    icon: "🥈",
    threshold: 250,
    nextThreshold: 450,
    color: "#c0c0c0",
    glow: "shadow-gray-400/40",
    searchBoost: "1.1×",
    pricingPremium: "5%",
    commission: "15%",
    perk: "Priority payout status",
    unlock: "Reach 250 points — verify phone or gov ID.",
  },
  {
    key: "gold",
    label: "Gold",
    icon: "🥇",
    threshold: 450,
    nextThreshold: 650,
    color: "#f59e0b",
    glow: "shadow-yellow-500/40",
    searchBoost: "1.25×",
    pricingPremium: "20%",
    commission: "12%",
    perk: "Verified badge + 20% pricing premium",
    unlock: "Reach 450 points — hardware audit recommended.",
  },
  {
    key: "platinum",
    label: "Platinum",
    icon: "💎",
    threshold: 650,
    nextThreshold: 850,
    color: "#38bdf8",
    glow: "shadow-cyan-400/40",
    searchBoost: "1.5×",
    pricingPremium: "40%",
    commission: "10%",
    perk: "Featured listing placement",
    unlock: "Reach 650 points — low failure rate required.",
  },
  {
    key: "diamond",
    label: "Diamond",
    icon: "👑",
    threshold: 850,
    nextThreshold: null,
    color: "#a78bfa",
    glow: "shadow-purple-400/40",
    searchBoost: "2.0×",
    pricingPremium: "50%",
    commission: "8%",
    perk: "Max visibility + reduced commission",
    unlock: "Reach 850 points — sustained through ongoing activity.",
  },
] as const;

type TierKey = (typeof TIERS)[number]["key"];

const TIER_MAP = Object.fromEntries(TIERS.map((t) => [t.key, t])) as Record<TierKey, (typeof TIERS)[number]>;

// ── Verification badge config ─────────────────────────────────────────────

const VERIFICATION_TYPES = [
  { key: "email", label: "Email", icon: "✉️" },
  { key: "phone", label: "Phone", icon: "📱" },
  { key: "gov_id", label: "Gov ID", icon: "🪪" },
  { key: "hardware_audit", label: "Hardware Audit", icon: "🔬" },
  { key: "incorporation", label: "Incorporation", icon: "🏢" },
  { key: "data_center", label: "Data Center", icon: "🏗️" },
] as const;

// ── Tier tooltip ──────────────────────────────────────────────────────────

function TierTooltip({ tier, visible, position }: { tier: (typeof TIERS)[number]; visible: boolean; position?: "left" | "center" | "right" }) {
  const alignClass =
    position === "left" ? "left-0 -translate-x-0" :
    position === "right" ? "right-0 left-auto translate-x-0" :
    "left-1/2 -translate-x-1/2";
  return (
    <div
      className={cn(
        "pointer-events-none absolute bottom-[calc(100%+10px)] z-50",
        alignClass,
        "w-64 rounded-xl border border-border p-3 shadow-xl text-xs",
        "transition-all duration-150",
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-1",
      )}
      style={{ backgroundColor: "#0f172a" }}
    >
      <div className="flex items-center gap-2 mb-2">
        <span className="text-lg">{tier.icon}</span>
        <span className="font-semibold text-sm" style={{ color: tier.color }}>{tier.label}</span>
        <span className="ml-auto text-text-muted font-mono">{tier.threshold}+ pts</span>
      </div>
      <p className="text-text-secondary mb-2">{tier.perk}</p>
      <div className="grid grid-cols-3 gap-1 mb-2">
        <div className="rounded bg-surface px-1.5 py-1 text-center">
          <div className="text-text-muted text-[10px]">Search</div>
          <div className="font-mono font-medium">{tier.searchBoost}</div>
        </div>
        <div className="rounded bg-surface px-1.5 py-1 text-center">
          <div className="text-text-muted text-[10px]">Premium</div>
          <div className="font-mono font-medium">+{tier.pricingPremium}</div>
        </div>
        <div className="rounded bg-surface px-1.5 py-1 text-center">
          <div className="text-text-muted text-[10px]">Commission</div>
          <div className="font-mono font-medium">{tier.commission}</div>
        </div>
      </div>
      <p className="text-text-muted text-[10px]">{tier.unlock}</p>
      {/* Downward arrow pointer */}
      <div className="absolute -bottom-[6px] left-1/2 -translate-x-1/2 w-3 h-3 rotate-45 border-r border-b border-border" style={{ backgroundColor: "#0f172a" }} />
    </div>
  );
}

// ── TierTrack ─────────────────────────────────────────────────────────────

function TierTrack({ currentTier }: { currentTier: TierKey }) {
  const currentIdx = TIERS.findIndex((t) => t.key === currentTier);
  const [hovered, setHovered] = useState<number | null>(null);

  return (
    <div className="relative flex items-center justify-between px-4 py-2 overflow-visible">
      {/* Connecting line */}
      <div className="absolute inset-y-1/2 left-6 right-6 h-0.5 bg-border -translate-y-1/2" />
      {/* Filled portion */}
      <div
        className="absolute inset-y-1/2 left-6 h-0.5 -translate-y-1/2 transition-all duration-500"
        style={{
          width: currentIdx === 0
            ? 0
            : `calc((100% - 3rem) * ${currentIdx / (TIERS.length - 1)})`,
          backgroundColor: TIERS[currentIdx]?.color ?? "#64748b",
        }}
      />
      {TIERS.map((tier, idx) => {
        const isReached = idx <= currentIdx;
        const isCurrent = idx === currentIdx;
        return (
          <div
            key={tier.key}
            className="relative z-10 flex flex-col items-center gap-1"
            onMouseEnter={() => setHovered(idx)}
            onMouseLeave={() => setHovered(null)}
          >
            <TierTooltip
              tier={tier}
              visible={hovered === idx}
              position={idx === 0 ? "left" : idx === TIERS.length - 1 ? "right" : "center"}
            />
            <button
              type="button"
              className={cn(
                "h-8 w-8 rounded-full border-2 flex items-center justify-center text-base transition-all duration-200",
                isReached ? "opacity-100" : "opacity-30",
                isCurrent && `shadow-lg ${tier.glow}`,
              )}
              style={{
                borderColor: isReached ? tier.color : "#334155",
                backgroundColor: isReached ? `${tier.color}22` : "transparent",
                boxShadow: isCurrent ? `0 0 12px 2px ${tier.color}55` : undefined,
              }}
            >
              {tier.icon}
            </button>
            <span
              className="text-[10px] font-medium"
              style={{ color: isReached ? tier.color : "#475569" }}
            >
              {tier.label}
            </span>
          </div>
        );
      })}
    </div>
  );
}

// ── ScoreBar ──────────────────────────────────────────────────────────────

function ScoreBar({
  label,
  value,
  max,
  color,
  suffix = "",
}: {
  label: string;
  value: number;
  max: number;
  color: string;
  suffix?: string;
}) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-text-secondary">{label}</span>
        <span className="font-mono text-text-primary">
          {typeof value === "number" ? value.toFixed(1) : value}{suffix}
          <span className="text-text-muted"> / {max}</span>
        </span>
      </div>
      <div className="h-2 rounded-full bg-border overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────

export default function ReputationPage() {
  const [leaderboard, setLeaderboard] = useState<ReputationEntry[]>([]);
  const [myRep, setMyRep] = useState<any>(null);
  const [history, setHistory] = useState<{ date: string; score: number }[]>([]);
  const [loading, setLoading] = useState(true);
  const api = useApi();
  const { user } = useAuth();
  const userId = user?.customer_id || user?.user_id || "";

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([
      api.fetchLeaderboard(),
      fetch("/api/reputation/me", { credentials: "include" }).then((r) => r.ok ? r.json() : Promise.reject()),
      userId ? apiLib.fetchReputationHistory(userId) : Promise.reject("no user"),
    ]).then(([lb, me, hist]) => {
      if (lb.status === "fulfilled") setLeaderboard(lb.value.leaderboard || []);
      if (me.status === "fulfilled") setMyRep(me.value);
      if (hist.status === "fulfilled") {
        const events: Array<{ created_at?: number; points_delta?: number; delta?: number }> =
          hist.value.events || [];
        let running = 0;
        setHistory(
          events.map((e) => {
            running += e.points_delta ?? e.delta ?? 0;
            const ts = (e.created_at ?? 0) * 1000;
            return {
              date: new Date(ts).toLocaleDateString("en-CA"),
              score: Math.max(0, running),
            };
          }),
        );
      }
      setLoading(false);
    });
  }, [api, userId]);

  useEffect(() => {
    load();
  }, [load]);

  const tierKey = (myRep?.tier || "new_user") as TierKey;
  const tierCfg = TIER_MAP[tierKey] ?? TIER_MAP.new_user;
  const finalScore: number = myRep?.final_score ?? myRep?.score ?? 0;
  const tierIdx = TIERS.findIndex((t) => t.key === tierKey);
  const nextTier = tierIdx < TIERS.length - 1 ? TIERS[tierIdx + 1] : null;
  const progressPct = nextTier
    ? Math.min(
        100,
        ((finalScore - tierCfg.threshold) / (nextTier.threshold - tierCfg.threshold)) * 100,
      )
    : 100;

  const jobsCompleted: number = myRep?.jobs_completed ?? 0;
  const jobsFailedHost: number = myRep?.jobs_failed_host ?? 0;
  const jobSuccessRate: number =
    jobsCompleted + jobsFailedHost > 0
      ? jobsCompleted / (jobsCompleted + jobsFailedHost)
      : 1.0;
  const uptimePct: number = myRep?.uptime_pct ?? 0;
  const reliabilityScore: number = myRep?.reliability_score ?? 1.0;
  const verificationPoints: number = myRep?.verification_points ?? 0;
  const activityPoints: number = myRep?.activity_points ?? 0;
  const penaltyPoints: number = myRep?.penalty_points ?? 0;

  const earnedVerifications: string[] = (() => {
    const raw = myRep?.verifications;
    if (Array.isArray(raw)) return raw as string[];
    if (typeof raw === "string") {
      try {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) return parsed as string[];
      } catch { /* ignore */ }
    }
    return [] as string[];
  })();

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Reputation</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5 mr-1.5" /> Refresh
        </Button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : (
        <>
          {/* Stats row */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <StatCard label="Jobs Completed" value={jobsCompleted} icon={Trophy} />
            <StatCard
              label="Job Success Rate"
              value={`${(jobSuccessRate * 100).toFixed(1)}%`}
              icon={TrendingUp}
            />
            <StatCard
              label="Uptime"
              value={`${(uptimePct * 100).toFixed(1)}%`}
              icon={BarChart3}
            />
            <StatCard
              label="Reliability"
              value={`${(reliabilityScore * 100).toFixed(1)}%`}
              icon={Star}
            />
          </div>

          {/* Hero card */}
          <Card className="overflow-hidden">
            <CardContent className="p-6 space-y-5">
              {/* Badge + score + perks */}
              <div className="flex flex-wrap items-start gap-5">
                <div
                  className={cn(
                    "flex h-20 w-20 items-center justify-center rounded-2xl border-2 text-5xl shadow-lg",
                    tierCfg.glow,
                  )}
                  style={{
                    borderColor: tierCfg.color,
                    backgroundColor: `${tierCfg.color}18`,
                    boxShadow: `0 0 24px 4px ${tierCfg.color}44`,
                  }}
                >
                  {tierCfg.icon}
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-2xl font-bold font-mono" style={{ color: tierCfg.color }}>
                      {finalScore.toFixed(1)}
                    </span>
                    <ReputationBadge tier={tierKey} size="md" />
                  </div>
                  <p className="text-sm text-text-secondary mb-3">{tierCfg.perk}</p>
                  {/* Progress to next tier */}
                  {nextTier ? (
                    <div>
                      <div className="flex items-center justify-between text-xs mb-1.5">
                        <span className="text-text-muted">
                          Progress to{" "}
                          <span className="font-medium" style={{ color: nextTier.color }}>
                            {nextTier.label}
                          </span>
                        </span>
                        <span className="font-mono">
                          {finalScore.toFixed(1)} / {nextTier.threshold}
                        </span>
                      </div>
                      <div className="h-2.5 rounded-full bg-border overflow-hidden">
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${progressPct}%`, backgroundColor: nextTier.color }}
                        />
                      </div>
                    </div>
                  ) : (
                    <p className="text-xs text-text-muted">Maximum tier achieved 🎉</p>
                  )}
                </div>
                {/* Current perks grid */}
                <div className="grid grid-cols-3 gap-2 text-center text-xs w-full sm:w-auto">
                  <div className="rounded-lg bg-surface p-2 border border-border">
                    <div className="text-text-muted text-[10px] mb-0.5">Search Boost</div>
                    <div className="font-mono font-semibold">{tierCfg.searchBoost}</div>
                  </div>
                  <div className="rounded-lg bg-surface p-2 border border-border">
                    <div className="text-text-muted text-[10px] mb-0.5">Price Premium</div>
                    <div className="font-mono font-semibold">+{tierCfg.pricingPremium}</div>
                  </div>
                  <div className="rounded-lg bg-surface p-2 border border-border">
                    <div className="text-text-muted text-[10px] mb-0.5">Commission</div>
                    <div className="font-mono font-semibold">{tierCfg.commission}</div>
                  </div>
                </div>
              </div>

              {/* Tier road map */}
              <div>
                <p className="text-xs text-text-muted mb-2 font-medium uppercase tracking-wide">
                  Trust Tier Roadmap
                </p>
                <TierTrack currentTier={tierKey} />
              </div>
            </CardContent>
          </Card>

          {/* Score Breakdown + Verification badges */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Score breakdown bars */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-4 w-4" /> Score Breakdown
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <ScoreBar
                  label="Verification Points"
                  value={verificationPoints}
                  max={250}
                  color="#38bdf8"
                />
                <ScoreBar
                  label="Activity Points"
                  value={activityPoints}
                  max={500}
                  color="#f59e0b"
                />
                <ScoreBar
                  label="Reliability Multiplier"
                  value={reliabilityScore * 100}
                  max={100}
                  color="#34d399"
                  suffix="%"
                />
                {penaltyPoints < 0 && (
                  <ScoreBar
                    label="Penalties"
                    value={Math.abs(penaltyPoints)}
                    max={500}
                    color="#f87171"
                  />
                )}
              </CardContent>
            </Card>

            {/* Verification badges */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <ShieldCheck className="h-4 w-4" /> Verification Badges
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                  {VERIFICATION_TYPES.map((v) => {
                    const earned = earnedVerifications.includes(v.key);
                    return (
                      <div
                        key={v.key}
                        className={cn(
                          "flex items-center gap-2 rounded-lg border p-2.5 text-xs transition-colors",
                          earned
                            ? "border-emerald-600 bg-emerald-900/20 text-emerald-300"
                            : "border-border bg-surface text-text-muted opacity-50",
                        )}
                      >
                        <span className="text-base">{v.icon}</span>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium truncate">{v.label}</div>
                          {earned ? (
                            <div className="text-[10px] text-emerald-400 flex items-center gap-0.5">
                              <ShieldCheck className="h-2.5 w-2.5" /> Verified
                            </div>
                          ) : (
                            <div className="text-[10px] flex items-center gap-0.5">
                              <ShieldOff className="h-2.5 w-2.5" /> Not verified
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Score History */}
          {history.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-4 w-4" /> Score History
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[220px]">
                  <ResponsiveContainer width="100%" height="100%" minHeight={180} minWidth={0} debounce={1}>
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="date"
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickFormatter={(d) =>
                          new Date(d).toLocaleDateString("en-CA", { month: "short", day: "numeric" })
                        }
                      />
                      <YAxis
                        domain={["auto", "auto"]}
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        width={36}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "#1e293b",
                          border: "1px solid #334155",
                          borderRadius: 8,
                        }}
                        labelFormatter={(d) => new Date(d).toLocaleDateString("en-CA")}
                        formatter={(value) => [Number(value).toFixed(1), "Score"]}
                      />
                      <Line
                        type="monotone"
                        dataKey="score"
                        stroke={tierCfg.color}
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Leaderboard */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-4 w-4" /> Leaderboard
              </CardTitle>
            </CardHeader>
            <CardContent>
              {leaderboard.length === 0 ? (
                <p className="text-sm text-text-muted">No reputation data available.</p>
              ) : (
                <div className="space-y-2">
                  {leaderboard.map((entry, i) => {
                    const isMe =
                      userId && (entry.entity_id === userId || entry.user_id === userId);
                    const entryTier = (entry.tier || "new_user") as TierKey;
                    const entryTierCfg = TIER_MAP[entryTier] ?? TIER_MAP.new_user;
                    return (
                      <div
                        key={entry.entity_id || i}
                        className={cn(
                          "flex items-center justify-between rounded-lg border p-3 transition-colors",
                          isMe ? "border-accent-gold/40 bg-accent-gold/5" : "border-border",
                        )}
                      >
                        <div className="flex items-center gap-3">
                          <span
                            className={cn(
                              "flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold",
                              i === 0
                                ? "bg-accent-gold/20 text-accent-gold"
                                : i === 1
                                ? "bg-gray-400/20 text-gray-300"
                                : i === 2
                                ? "bg-amber-700/20 text-amber-600"
                                : "bg-surface text-text-muted",
                            )}
                          >
                            {i + 1}
                          </span>
                          <span className="text-base">{entryTierCfg.icon}</span>
                          <span className="text-sm font-medium">
                            {entry.user_id || entry.entity_id}
                          </span>
                          {entry.tier && <ReputationBadge tier={entry.tier} size="sm" />}
                          {isMe && (
                            <span className="rounded-full bg-accent-gold/20 px-2 py-0.5 text-[10px] font-semibold text-accent-gold">
                              you
                            </span>
                          )}
                        </div>
                        <span className="font-mono text-sm text-text-secondary">
                          {entry.score?.toFixed(1)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}

