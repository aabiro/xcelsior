"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart3, RefreshCw, Calendar, Download, TrendingUp, Clock, Cpu } from "lucide-react";
import { useApi } from "@/lib/use-api";
import { useLocale } from "@/lib/locale";
import { toast } from "sonner";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
  PieChart, Pie, Cell, Legend,
} from "recharts";

const COLORS = ["#dc2626", "#f59e0b", "#38bdf8", "#10b981", "#8b5cf6", "#ec4899"];
const RANGE_PRESETS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
  { label: "YTD", days: 0 },
];

export default function AnalyticsPage() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [range, setRange] = useState(30);
  const api = useApi();
  const { t } = useLocale();

  const load = useCallback(() => {
    setLoading(true);
    const days = range === 0
      ? Math.ceil((Date.now() - new Date(new Date().getFullYear(), 0, 1).getTime()) / 86400000)
      : range;
    api.fetchAnalytics({ days: String(days) })
      .then(setData)
      .catch(() => toast.error("Failed to load analytics"))
      .finally(() => setLoading(false));
  }, [api, range]);

  useEffect(() => { load(); }, [load]);

  const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

  // Derive chart data from the API response shape: {analytics: [...], summary: {...}}
  const analytics: any[] = data?.analytics || [];
  const summary = data?.summary || {};
  const hasData = analytics.length > 0 || (summary.total_jobs != null && summary.total_jobs > 0);

  // Build chart-friendly arrays from the analytics time-series
  const jobsOverTime = analytics.map((r: any) => ({ date: r.period, count: r.job_count }));
  const spendOverTime = analytics.map((r: any) => ({ date: r.period, amount: r.total_cost_cad }));

  return (
    <div className="space-y-6">
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
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {[...Array(4)].map((_, i) => <div key={i} className="h-64 rounded-xl bg-surface skeleton-pulse" />)}
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
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 w-full max-w-lg">
            <Card className="p-4 text-center border-dashed">
              <TrendingUp className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">Spend trends</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
            <Card className="p-4 text-center border-dashed">
              <Cpu className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">GPU hours</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
            <Card className="p-4 text-center border-dashed">
              <Clock className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">Avg duration</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
          </div>
          <p className="text-xs text-text-muted mt-6">
            Launch an instance to start collecting analytics data
          </p>
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Card className="p-4">
              <p className="text-2xl font-bold font-mono">{summary.total_jobs ?? 0}</p>
              <p className="text-xs text-text-muted">{t("dash.analytics.total_jobs")}</p>
            </Card>
            <Card className="p-4">
              <p className="text-2xl font-bold font-mono">${Number(summary.total_spend_cad ?? 0).toFixed(2)}</p>
              <p className="text-xs text-text-muted">{t("dash.analytics.total_spend")}</p>
            </Card>
            <Card className="p-4">
              <p className="text-2xl font-bold font-mono">{Number(summary.total_gpu_hours ?? 0).toFixed(1)}</p>
              <p className="text-xs text-text-muted">{t("dash.analytics.gpu_hours")}</p>
            </Card>
            <Card className="p-4">
              <p className="text-2xl font-bold font-mono">{Number(summary.avg_gpu_utilization_pct ?? 0).toFixed(1)}%</p>
              <p className="text-xs text-text-muted">{t("dash.analytics.avg_util")}</p>
            </Card>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {/* Jobs Over Time */}
            {jobsOverTime.length > 0 && (
              <Card>
                <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_jobs")}</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={0} debounce={1}>
                      <LineChart data={jobsOverTime}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Line type="monotone" dataKey="count" stroke="#38bdf8" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Spend Over Time */}
            {spendOverTime.length > 0 && (
              <Card>
                <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_spend")}</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={0} debounce={1}>
                      <BarChart data={spendOverTime}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                        <Tooltip contentStyle={tooltipStyle} formatter={(value) => [`$${Number(value).toFixed(2)}`, "Spend"]} />
                        <Bar dataKey="amount" fill="#f59e0b" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Breakdown Table */}
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
                      {analytics.map((row: any, i: number) => (
                        <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                          <td className="py-2 pr-4 font-medium">{row.period || "—"}</td>
                          <td className="py-2 pr-4 text-right font-mono text-xs">{row.job_count ?? "—"}</td>
                          <td className="py-2 pr-4 text-right font-mono text-xs">{row.total_gpu_hours?.toFixed(1) ?? "—"}</td>
                          <td className="py-2 text-right font-mono text-xs">{row.total_cost_cad != null ? `$${Number(row.total_cost_cad).toFixed(2)}` : "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
