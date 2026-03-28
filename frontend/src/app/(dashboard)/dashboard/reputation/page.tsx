"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { ReputationBadge } from "@/components/ui/reputation-badge";
import { Star, RefreshCw, Users, Trophy, TrendingUp, BarChart3 } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import * as apiLib from "@/lib/api";
import type { ReputationEntry } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { toast } from "sonner";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar,
} from "recharts";

const TIER_ORDER = ["bronze", "silver", "gold", "platinum", "diamond"];
const TIER_THRESHOLDS = [0, 50, 70, 85, 95];
const TIER_COLORS: Record<string, string> = {
  bronze: "#cd7f32",
  silver: "#c0c0c0",
  gold: "#f59e0b",
  platinum: "#38bdf8",
  diamond: "#a78bfa",
};

export default function ReputationPage() {
  const [leaderboard, setLeaderboard] = useState<ReputationEntry[]>([]);
  const [myRep, setMyRep] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [breakdown, setBreakdown] = useState<{ category: string; score: number }[]>([]);
  const [history, setHistory] = useState<{ date: string; score: number }[]>([]);
  const api = useApi();
  const { t } = useLocale();
  const { user } = useAuth();
  const userId = user?.customer_id || user?.user_id || "";

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([
      api.fetchLeaderboard(),
      fetch("/api/reputation/me", { credentials: "include" }).then((r) => r.ok ? r.json() : Promise.reject()),
      userId ? apiLib.fetchReputationBreakdown(userId) : Promise.reject("no user"),
      userId ? apiLib.fetchReputationHistory(userId) : Promise.reject("no user"),
    ]).then(([lb, me, bd, hist]) => {
      if (lb.status === "fulfilled") setLeaderboard(lb.value.leaderboard || []);
      if (me.status === "fulfilled") setMyRep(me.value);
      if (bd.status === "fulfilled") {
        const b = bd.value.breakdown || {};
        setBreakdown(Object.entries(b).map(([key, val]) => ({
          category: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
          score: Number(val),
        })));
      }
      if (hist.status === "fulfilled") {
        const events = hist.value.events || [];
        let running = 0;
        setHistory(events.map((e: { timestamp: string; delta: number }) => {
          running += e.delta;
          return { date: e.timestamp.slice(0, 10), score: running };
        }));
      }
      setLoading(false);
    });
  }, [api, userId]);

  useEffect(() => { load(); }, [load]);

  // Tier progression
  const currentTier = myRep?.tier || "bronze";
  const currentScore = myRep?.score || 0;
  const tierIdx = TIER_ORDER.indexOf(currentTier);
  const nextTier = tierIdx < TIER_ORDER.length - 1 ? TIER_ORDER[tierIdx + 1] : null;
  const nextThreshold = nextTier ? TIER_THRESHOLDS[tierIdx + 1] : 100;
  const currentThreshold = TIER_THRESHOLDS[tierIdx] || 0;
  const progressPct = nextTier
    ? Math.min(100, ((currentScore - currentThreshold) / (nextThreshold - currentThreshold)) * 100)
    : 100;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.rep.title")}</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          {[...Array(3)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
        </div>
      ) : (
        <>
          {/* Stats */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StatCard label={t("dash.rep.my_score")} value={myRep?.score?.toFixed(1) || "—"} icon={Star} />
            <StatCard label={t("dash.rep.my_rank")} value={myRep?.rank || "—"} icon={Trophy} />
            <StatCard label={t("dash.rep.total_providers")} value={leaderboard.length} icon={Users} />
          </div>

          {/* Tier Progression */}
          {myRep?.tier && (
            <Card className="p-6">
              <div className="flex items-center gap-4 mb-4">
                <ReputationBadge tier={myRep.tier} size="lg" />
                <div className="flex-1">
                  <h3 className="text-lg font-semibold capitalize">{t("dash.rep.tier", { tier: myRep.tier.replace("_", " ") })}</h3>
                  <p className="text-sm text-text-secondary">
                    Score: {myRep.score?.toFixed(1)} · Uptime: {myRep.uptime || "—"}% · Jobs: {myRep.jobs_completed || 0}
                  </p>
                </div>
              </div>
              {nextTier && (
                <div>
                  <div className="flex items-center justify-between text-xs mb-1.5">
                    <span className="text-text-muted">{t("dash.rep.progress")} <span className="capitalize font-medium" style={{ color: TIER_COLORS[nextTier] }}>{nextTier}</span></span>
                    <span className="font-mono">{currentScore.toFixed(1)} / {nextThreshold}</span>
                  </div>
                  <div className="h-2 rounded-full bg-border overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{ width: `${progressPct}%`, backgroundColor: TIER_COLORS[nextTier] || "#38bdf8" }}
                    />
                  </div>
                </div>
              )}
              {/* Tier badges row */}
              <div className="flex items-center justify-between mt-4 px-2">
                {TIER_ORDER.map((t, i) => (
                  <div key={t} className={`flex flex-col items-center gap-1 ${tierIdx >= i ? "opacity-100" : "opacity-30"}`}>
                    <div
                      className="h-3 w-3 rounded-full"
                      style={{ backgroundColor: TIER_COLORS[t] }}
                    />
                    <span className="text-[10px] capitalize">{t}</span>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Breakdown + History */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Radar breakdown */}
            {breakdown.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><BarChart3 className="h-4 w-4" /> {t("dash.rep.breakdown")}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <RadarChart data={breakdown}>
                      <PolarGrid stroke="#334155" />
                      <PolarAngleAxis dataKey="category" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                      <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                      <Radar dataKey="score" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.2} strokeWidth={2} />
                    </RadarChart>
                  </ResponsiveContainer>
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-2">
                    {breakdown.map((b) => (
                      <div key={b.category} className="flex items-center justify-between rounded bg-surface px-2 py-1">
                        <span className="text-xs capitalize text-text-secondary">{b.category.replace(/_/g, " ")}</span>
                        <span className="text-xs font-mono font-medium">{b.score.toFixed(1)}</span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* History line chart */}
            {history.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><TrendingUp className="h-4 w-4" /> {t("dash.rep.history")}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                      <XAxis
                        dataKey="date"
                        tick={{ fill: "#94a3b8", fontSize: 10 }}
                        tickFormatter={(d) => new Date(d).toLocaleDateString("en-CA", { month: "short", day: "numeric" })}
                      />
                      <YAxis domain={[0, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 8 }}
                        labelFormatter={(d) => new Date(d).toLocaleDateString("en-CA")}
                        formatter={(value) => [Number(value).toFixed(1), "Score"]}
                      />
                      <Line type="monotone" dataKey="score" stroke="#f59e0b" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Leaderboard */}
          <Card>
            <CardHeader><CardTitle>{t("dash.rep.leaderboard")}</CardTitle></CardHeader>
            <CardContent>
              {leaderboard.length === 0 ? (
                <p className="text-sm text-text-muted">No reputation data available</p>
              ) : (
                <div className="space-y-2">
                  {leaderboard.map((entry, i) => (
                    <div key={entry.entity_id || i} className="flex items-center justify-between rounded-lg border border-border p-3">
                      <div className="flex items-center gap-3">
                        <span className={`flex h-7 w-7 items-center justify-center rounded-full text-xs font-bold ${
                          i === 0 ? "bg-accent-gold/20 text-accent-gold" :
                          i === 1 ? "bg-gray-400/20 text-gray-300" :
                          i === 2 ? "bg-amber-700/20 text-amber-600" :
                          "bg-surface text-text-muted"
                        }`}>
                          {i + 1}
                        </span>
                        <span className="text-sm font-medium">{entry.user_id || entry.entity_id}</span>
                        {entry.tier && <ReputationBadge tier={entry.tier} size="sm" />}
                      </div>
                      <span className="font-mono text-sm text-text-secondary">{entry.score?.toFixed(1)}</span>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
