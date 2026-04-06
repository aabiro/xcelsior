"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  TrendingUp, RefreshCw, Loader2, DollarSign, Cpu, BarChart3,
} from "lucide-react";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";
import * as api from "@/lib/api";
import type { SpotPricePoint } from "@/lib/api";
import { toast } from "sonner";

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
      if (prices.length > 0 && !selectedModel) {
        setSelectedModel(prices[0].gpu_model);
      }
    } catch {
      toast.error("Failed to load spot prices");
    } finally {
      setLoading(false);
    }
  }, [selectedModel]);

  const loadHistory = useCallback(async (model: string) => {
    try {
      const res = await api.fetchSpotHistory(model, 48);
      setHistory((prev) => ({ ...prev, [model]: res.history || [] }));
    } catch {
      // Silently fail for history
    }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    if (selectedModel) loadHistory(selectedModel);
  }, [selectedModel, loadHistory]);

  const uniqueModels = [...new Set(spotPrices.map((p) => p.gpu_model))];
  const avgPrice = spotPrices.length
    ? Math.round(spotPrices.reduce((s, p) => s + p.spot_cents, 0) / spotPrices.length)
    : 0;
  const minPrice = spotPrices.length
    ? Math.min(...spotPrices.map((p) => p.spot_cents))
    : 0;

  const selectedHistory = history[selectedModel] || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Spot Pricing</h1>
          <p className="text-sm text-text-muted">
            Real-time GPU spot prices. Save up to 60% vs on-demand.
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> Refresh
        </Button>
      </div>

      {/* Stats */}
      <div className="grid gap-4 sm:grid-cols-3">
        <StatCard label="GPU Models" value={uniqueModels.length} icon={Cpu} />
        <StatCard label="Avg Spot Price" value={`¢${avgPrice}/hr`} icon={DollarSign} />
        <StatCard label="Lowest Price" value={`¢${minPrice}/hr`} icon={TrendingUp} />
      </div>

      {/* Current Spot Prices */}
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
              {spotPrices.map((p) => (
                <button
                  key={p.gpu_model}
                  onClick={() => setSelectedModel(p.gpu_model)}
                  className={`rounded-lg border p-4 text-left transition-colors ${
                    selectedModel === p.gpu_model
                      ? "border-primary bg-primary/5"
                      : "border-border hover:bg-muted/50"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{p.gpu_model === "unknown" ? "Any GPU" : p.gpu_model}</span>
                    <Badge variant="info">SPOT</Badge>
                  </div>
                  <div className="mt-2 text-2xl font-bold">
                    ¢{p.spot_cents}<span className="text-sm font-normal text-text-muted">/hr</span>
                  </div>
                  <div className="mt-1 text-xs text-text-muted">
                    ${(p.spot_cents / 100).toFixed(2)} CAD/hr
                  </div>
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Price History Chart */}
      {selectedModel && selectedHistory.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              {selectedModel === "unknown" ? "GPU" : selectedModel} — 48h Price History
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-60">
            <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
              <AreaChart
                data={selectedHistory.slice(-96).map((p) => ({
                  time: new Date(p.recorded_at * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                  price: p.spot_cents,
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
                  tickFormatter={(v: number) => `¢${v}`}
                  width={48}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#0f172a",
                    border: "1px solid #1e293b",
                    borderRadius: "8px",
                    fontSize: 12,
                  }}
                  labelStyle={{ color: "#94a3b8" }}
                  // eslint-disable-next-line @typescript-eslint/no-explicit-any
                  formatter={((value: any) => [`¢${Number(value).toFixed(0)}/hr`, "Spot Price"]) as any}
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
