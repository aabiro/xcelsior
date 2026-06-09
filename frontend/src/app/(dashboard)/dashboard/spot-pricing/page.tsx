"use client";

import { useEffect, useState, useCallback, useMemo } from "react";
import Link from "next/link";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import {
  TrendingUp, RefreshCw, Loader2, DollarSign, Cpu, BarChart3, Rocket,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "@/lib/recharts";
import * as api from "@/lib/api";
import type { SpotPricePoint } from "@/lib/api";
import { toast } from "sonner";
import {
  SpotBadge,
  SpotSavingsPill,
  SpotSupplyIndicator,
  SpotSurface,
} from "@/components/spot/spot-surface";

export default function SpotPricingPage() {
  const [spotPrices, setSpotPrices] = useState<SpotPricePoint[]>([]);
  const [history, setHistory] = useState<Record<string, SpotPricePoint[]>>({});
  const [loading, setLoading] = useState(true);
  const [selectedModel, setSelectedModel] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const res = await api.fetchSpotPricesV2();
      const prices = res.spot_prices || [];
      setSpotPrices(prices);
      if (prices.length > 0) {
        setSelectedModel((prev) => prev || prices[0].gpu_model);
      }
    } catch {
      toast.error("Failed to load spot prices");
    } finally {
      setLoading(false);
    }
  }, []);

  const loadHistory = useCallback(async (model: string) => {
    try {
      const res = await api.fetchSpotHistory(model, 48);
      setHistory((prev) => ({ ...prev, [model]: res.history || [] }));
    } catch {
      // history is optional
    }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (selectedModel) loadHistory(selectedModel);
  }, [selectedModel, loadHistory]);

  const uniqueModels = useMemo(
    () => [...new Set(spotPrices.map((p) => p.gpu_model))],
    [spotPrices],
  );

  const heroStats = useMemo(() => {
    const withSavings = spotPrices.filter((p) => (p.savings_pct ?? 0) > 0);
    const avgSavings = withSavings.length
      ? Math.round(withSavings.reduce((s, p) => s + (p.savings_pct ?? 0), 0) / withSavings.length)
      : 0;
    const lowest = spotPrices.length
      ? spotPrices.reduce((min, p) => (p.spot_cents < min.spot_cents ? p : min), spotPrices[0])
      : null;
    return { avgSavings, lowest, modelCount: uniqueModels.length };
  }, [spotPrices, uniqueModels.length]);

  const selectedHistory = history[selectedModel] || [];
  const selectedPoint = spotPrices.find((p) => p.gpu_model === selectedModel);

  return (
    <div className="space-y-6">
      <SpotSurface variant="hero" className="p-5 md:p-6">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <SpotBadge size="md">Spot Instances</SpotBadge>
            </div>
            <h1 className="text-2xl font-bold tracking-tight">Spot Pricing</h1>
            <p className="text-sm text-text-secondary mt-1 max-w-xl">
              Published interruptible GPU rates — no bidding. Save up to 60% vs on-demand with
              automatic requeue when capacity returns.
            </p>
          </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
          <Button size="sm" asChild className="shadow-[0_0_20px_rgba(16,185,129,0.15)]">
            <Link href="/dashboard/instances?launch=true&mode=spot">
              <Rocket className="h-3.5 w-3.5" /> Launch spot instance
            </Link>
          </Button>
        </div>
      </SpotSurface>

      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard
          label="Avg Savings"
          value={heroStats.avgSavings > 0 ? `${heroStats.avgSavings}%` : "—"}
          icon={TrendingUp}
        />
        <StatCard label="GPU Models" value={heroStats.modelCount} icon={Cpu} />
        <StatCard
          label="Lowest Spot Rate"
          value={
            heroStats.lowest
              ? `$${((heroStats.lowest.rate_cad ?? heroStats.lowest.spot_cents / 100)).toFixed(2)}/hr`
              : "—"
          }
          icon={DollarSign}
        />
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Current Spot Prices</CardTitle>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
            </div>
          ) : spotPrices.length === 0 ? (
            <p className="text-text-muted text-center py-8">No spot prices available</p>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {spotPrices.map((p) => {
                const spotCad = p.rate_cad ?? p.spot_cents / 100;
                const onDemand = p.on_demand_cad;
                const selected = selectedModel === p.gpu_model;
                return (
                  <button
                    key={p.gpu_model}
                    type="button"
                    onClick={() => setSelectedModel(p.gpu_model)}
                    className={`group rounded-xl border p-4 text-left transition-all duration-200 ${
                      selected
                        ? "border-emerald/45 bg-emerald/[0.06] shadow-[0_0_24px_rgba(16,185,129,0.08)] ring-1 ring-emerald/20"
                        : "border-border hover:border-emerald/25 hover:bg-emerald/[0.03]"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold tracking-tight">
                        {p.gpu_model === "unknown" ? "Any GPU" : p.gpu_model}
                      </span>
                      <SpotBadge>Spot</SpotBadge>
                    </div>
                    <div className="mt-2 flex items-baseline gap-2 flex-wrap">
                      <span className="text-2xl font-bold font-mono text-emerald tabular-nums">
                        ${spotCad.toFixed(2)}
                      </span>
                      <span className="text-sm text-text-muted">/hr CAD</span>
                      <SpotSavingsPill pct={p.savings_pct ?? 0} />
                    </div>
                    {onDemand != null && onDemand > spotCad && (
                      <p className="mt-1 text-xs text-text-muted line-through font-mono tabular-nums">
                        ${onDemand.toFixed(2)}/hr on-demand
                      </p>
                    )}
                    <SpotSupplyIndicator
                      supply={p.supply}
                      demand={p.demand}
                      className="mt-2"
                    />
                    <Link
                      href={`/dashboard/instances?launch=true&mode=spot&gpu=${encodeURIComponent(p.gpu_model)}`}
                      onClick={(e) => e.stopPropagation()}
                      className="mt-3 inline-flex items-center gap-1 text-[11px] font-medium text-accent-cyan opacity-80 group-hover:opacity-100 hover:underline"
                    >
                      Launch {p.gpu_model} spot <Rocket className="h-3 w-3" />
                    </Link>
                  </button>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>

      {selectedModel && selectedHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              {selectedModel === "unknown" ? "GPU" : selectedModel} — 48h Price History
              {selectedPoint?.on_demand_cad != null && (
                <span className="ml-auto text-xs font-normal text-text-muted font-mono">
                  On-demand ${selectedPoint.on_demand_cad.toFixed(2)}/hr
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                <AreaChart
                  data={selectedHistory.slice(-96).map((p) => ({
                    time: new Date(p.recorded_at * 1000).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    }),
                    price: p.rate_cad != null ? p.rate_cad * 100 : p.spot_cents,
                    priceCad: p.rate_cad ?? p.spot_cents / 100,
                  }))}
                  margin={{ top: 4, right: 8, left: 0, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="spotGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                  <XAxis
                    dataKey="time"
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    interval="preserveStartEnd"
                    stroke="#334155"
                  />
                  <YAxis
                    tick={{ fill: "#64748b", fontSize: 11 }}
                    stroke="#334155"
                    tickFormatter={(v: number) => `$${(v / 100).toFixed(2)}`}
                    width={56}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      border: "1px solid #1e293b",
                      borderRadius: "8px",
                      fontSize: 12,
                    }}
                    labelStyle={{ color: "#94a3b8" }}
                    formatter={((value: any, _name: any, item: any) => {
                      const cad = item?.payload?.priceCad ?? Number(value) / 100;
                      return [`$${cad.toFixed(2)} CAD/hr`, "Spot Price"];
                    }) as any}
                  />
                  <Area
                    type="monotone"
                    dataKey="price"
                    stroke="#38bdf8"
                    strokeWidth={2}
                    fill="url(#spotGrad)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}