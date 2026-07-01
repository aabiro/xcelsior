"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { FadeIn } from "@/components/ui/motion";
import { Scale, RefreshCw, TrendingUp, Users, Rocket, Zap, DollarSign } from "lucide-react";
import * as api from "@/lib/api";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "@/lib/recharts";

const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };
const RANGES = [30, 60, 90] as const;

type Data = Awaited<ReturnType<typeof api.fetchAdminUnitEconomics>>;

const money = (n?: number) => `$${(n ?? 0).toFixed(2)}`;

export default function AdminUnitEconomicsPage() {
  const [data, setData] = useState<Data | null>(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState<number>(30);

  const load = useCallback((refresh?: boolean) => {
    if (refresh) setLoading(true);
    api.fetchAdminUnitEconomics(days)
      .then(setData)
      .catch(() => toast.error("Failed to load unit economics"))
      .finally(() => setLoading(false));
  }, [days]);

  useEffect(() => { load(); }, [load]);

  const mk = data?.marketplace ?? {};
  const sl = data?.serverless ?? {};
  const slBySize = data?.serverless_by_model_size ?? [];
  const fn = data?.funnel ?? {};
  const lq = data?.liquidity ?? {};

  // Are we making money? Margin = platform fee kept on marketplace + serverless revenue.
  const makingMoney = (mk.platform_margin_cad ?? 0) + (sl.revenue_cad ?? 0);

  const funnelStages = [
    { label: "Signups", value: fn.signups ?? 0, icon: Users, color: "bg-text-muted" },
    { label: "Deployed a model", value: fn.deployed_model ?? 0, icon: Rocket, color: "bg-accent-violet" },
    { label: "Ran inference", value: fn.ran_inference ?? 0, icon: Zap, color: "bg-accent-cyan" },
    { label: "Paid", value: fn.paid ?? 0, icon: DollarSign, color: "bg-emerald" },
  ];
  const funnelBase = Math.max(1, fn.signups ?? 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Scale className="h-5 w-5 text-accent-cyan" />
          <h1 className="text-xl font-bold">Unit Economics</h1>
        </div>
        <div className="flex items-center gap-2">
          {RANGES.map((d) => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                days === d ? "bg-accent-cyan/15 text-accent-cyan" : "text-text-muted hover:bg-surface-hover"
              }`}
            >
              {d}d
            </button>
          ))}
          <Button size="sm" variant="outline" onClick={() => load(true)}>
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} /> Refresh
          </Button>
        </div>
      </div>

      {/* Are we making money? */}
      <FadeIn>
        <Card className={makingMoney >= 0 ? "border-emerald/30" : "border-accent-red/30"}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <TrendingUp className="h-4 w-4 text-emerald" /> Are we making money? (last {days}d)
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold">
              <span className={makingMoney >= 0 ? "text-emerald" : "text-accent-red"}>{money(makingMoney)}</span>
              <span className="ml-2 text-sm font-normal text-text-muted">platform margin + serverless revenue</span>
            </p>
          </CardContent>
        </Card>
      </FadeIn>

      {/* Marketplace margin */}
      <div>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-text-muted">Marketplace (dedicated)</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Gross charged" value={money(mk.gross_cad)} icon={DollarSign} glow="cyan" />
          <StatCard label="Paid to providers" value={money(mk.provider_payout_cad)} icon={Users} glow="violet" />
          <StatCard label="Platform margin" value={money(mk.platform_margin_cad)} icon={TrendingUp} glow="emerald" />
          <StatCard label="Margin %" value={`${(mk.margin_pct ?? 0).toFixed(1)}%`} icon={Scale} glow="gold" />
        </div>
      </div>

      {/* Serverless */}
      <div>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-text-muted">Serverless</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard label="Revenue" value={money(sl.revenue_cad)} icon={DollarSign} glow="cyan" />
          <StatCard label="Token revenue" value={money(sl.token_revenue_cad)} icon={TrendingUp} glow="violet" />
          <StatCard label="GPU-hours billed" value={`${(sl.gpu_hours ?? 0).toFixed(1)}h`} icon={Zap} glow="emerald" />
          <StatCard label="Billed cycles" value={String(sl.billed_cycles ?? 0)} icon={Rocket} glow="gold" />
        </div>
        <p className="mt-1.5 text-xs text-text-muted">
          Token revenue is the size-tiered token cost recorded per slice (charged once blended billing is enabled).
        </p>
      </div>

      {/* Margin-per-model-size (Phase 1 exit criterion) */}
      {slBySize.length > 0 && (
        <div>
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-text-muted">
            Serverless by model size (last {days}d)
          </h2>
          <Card>
            <CardContent className="overflow-x-auto p-0">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border/60 text-left text-xs uppercase tracking-wider text-text-muted">
                    <th className="px-4 py-2">Model size</th>
                    <th className="px-4 py-2">GPU-seconds revenue</th>
                    <th className="px-4 py-2">Token revenue</th>
                    <th className="px-4 py-2">GPU-hours</th>
                    <th className="px-4 py-2">Billed cycles</th>
                  </tr>
                </thead>
                <tbody>
                  {slBySize.map((row) => (
                    <tr key={row.band} className="border-b border-border/30 last:border-0">
                      <td className="px-4 py-2 font-medium">{row.band}</td>
                      <td className="px-4 py-2">{money(row.gpu_revenue_cad)}</td>
                      <td className="px-4 py-2">{money(row.token_revenue_cad)}</td>
                      <td className="px-4 py-2">{(row.gpu_seconds / 3600).toFixed(1)}h</td>
                      <td className="px-4 py-2">{row.billed_cycles}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </CardContent>
          </Card>
          <p className="mt-1.5 text-xs text-text-muted">
            GPU-seconds vs. token revenue per size band — the same comparison the blended meter uses,
            so you can see which model sizes are worth pricing by tokens before flipping
            <code className="mx-1">XCELSIOR_SERVERLESS_BLENDED_BILLING</code> on.
          </p>
        </div>
      )}

      {/* Supply liquidity */}
      <div>
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wider text-text-muted">Supply liquidity (last {days}d)</h2>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatCard
            label="Got a GPU"
            value={`${(lq.liquidity_pct ?? 0).toFixed(1)}%`}
            icon={Zap}
            glow={(lq.liquidity_pct ?? 0) >= 80 ? "emerald" : "gold"}
          />
          <StatCard label="Jobs fulfilled" value={String(lq.fulfilled ?? 0)} icon={Rocket} glow="cyan" />
          <StatCard label="Jobs requested" value={String(lq.requested ?? 0)} icon={Users} glow="violet" />
        </div>
        <p className="mt-1.5 text-xs text-text-muted">
          Share of submitted jobs that got a host assigned — target ≥ 70–80% during business hours.
        </p>
      </div>

      {/* Activation funnel */}
      <FadeIn>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Users className="h-4 w-4 text-accent-violet" /> Activation funnel (all-time)
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {funnelStages.map((s, i) => {
              const pct = Math.round((s.value / funnelBase) * 100);
              return (
                <div key={s.label} className="flex items-center gap-3">
                  <div className="flex w-36 shrink-0 items-center gap-2 text-sm">
                    <s.icon className="h-3.5 w-3.5 text-text-muted" />
                    {s.label}
                  </div>
                  <div className="relative h-7 flex-1 overflow-hidden rounded-lg bg-surface">
                    <div className={`h-full ${s.color} opacity-70 transition-all`} style={{ width: `${pct}%` }} />
                    <span className="absolute inset-y-0 left-2 flex items-center text-xs font-medium text-text-primary">
                      {s.value.toLocaleString()} {i > 0 && <span className="ml-1 text-text-muted">({pct}%)</span>}
                    </span>
                  </div>
                </div>
              );
            })}
            <p className="pt-1 text-xs text-text-muted">
              Activation (ran inference / signups): <span className="font-medium text-accent-cyan">{(fn.activation_pct ?? 0).toFixed(1)}%</span>
              {"  ·  "}Paid: <span className="font-medium text-emerald">{(fn.paid_pct ?? 0).toFixed(1)}%</span>
              {"  ·  "}target ≥ 25% activation in week one.
            </p>
          </CardContent>
        </Card>
      </FadeIn>

      {/* Weekly trend */}
      <FadeIn>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Weekly revenue & GPU-hours (8 weeks)</CardTitle>
          </CardHeader>
          <CardContent>
            {data?.weekly && data.weekly.length > 0 ? (
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data.weekly}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis dataKey="week" tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
                  <Tooltip contentStyle={tooltipStyle} />
                  <Bar dataKey="revenue" name="Revenue (CAD)" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="gpu_hours" name="GPU-hours" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <p className="py-8 text-center text-sm text-text-muted">No usage in the last 8 weeks yet.</p>
            )}
          </CardContent>
        </Card>
      </FadeIn>
    </div>
  );
}
