"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { BarChart3, RefreshCw, Calendar, Download } from "lucide-react";
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
            if (!data) return;
            const breakdown = data.breakdown || data.daily_breakdown || [];
            if (breakdown.length === 0) { toast.error("No data to export"); return; }
            const keys = Object.keys(breakdown[0]);
            const csv = [keys.join(","), ...breakdown.map((r: Record<string, unknown>) => keys.map((k) => `"${r[k] ?? ""}"`).join(","))].join("\n");
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
      ) : !data ? (
        <Card className="p-12 text-center">
          <BarChart3 className="mx-auto h-12 w-12 text-text-muted mb-4" />
          <h3 className="text-lg font-semibold mb-1">{t("dash.analytics.empty")}</h3>
          <p className="text-sm text-text-secondary">{t("dash.analytics.empty_desc")}</p>
        </Card>
      ) : (
        <>
          {/* Summary Stats */}
          {(data.summary_stats || data.totals) && (() => {
            const s = data.summary_stats || data.totals || {};
            return (
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {s.total_jobs != null && (
                  <Card className="p-4"><p className="text-2xl font-bold font-mono">{s.total_jobs}</p><p className="text-xs text-text-muted">{t("dash.analytics.total_jobs")}</p></Card>
                )}
                {s.total_spend != null && (
                  <Card className="p-4"><p className="text-2xl font-bold font-mono">${Number(s.total_spend).toFixed(2)}</p><p className="text-xs text-text-muted">{t("dash.analytics.total_spend")}</p></Card>
                )}
                {s.gpu_hours != null && (
                  <Card className="p-4"><p className="text-2xl font-bold font-mono">{Number(s.gpu_hours).toFixed(1)}</p><p className="text-xs text-text-muted">{t("dash.analytics.gpu_hours")}</p></Card>
                )}
                {s.avg_job_duration != null && (
                  <Card className="p-4"><p className="text-2xl font-bold font-mono">{Number(s.avg_job_duration).toFixed(0)}m</p><p className="text-xs text-text-muted">{t("dash.analytics.avg_duration")}</p></Card>
                )}
              </div>
            );
          })()}

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {data.jobs_over_time && (
              <Card>
                <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_jobs")}</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={data.jobs_over_time}>
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

            {data.spend_over_time && (
              <Card>
                <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_spend")}</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={data.spend_over_time}>
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

            {data.gpu_distribution && (
              <Card>
                <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_gpu")}</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={data.gpu_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                          {data.gpu_distribution.map((_: any, i: number) => (
                            <Cell key={i} fill={COLORS[i % COLORS.length]} />
                          ))}
                        </Pie>
                        <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
                        <Tooltip contentStyle={tooltipStyle} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}

            {data.region_distribution && (
              <Card>
                <CardHeader><CardTitle className="text-sm">{t("dash.analytics.chart_region")}</CardTitle></CardHeader>
                <CardContent>
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie data={data.region_distribution} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                          {data.region_distribution.map((_: any, i: number) => (
                            <Cell key={i} fill={COLORS[(i + 2) % COLORS.length]} />
                          ))}
                        </Pie>
                        <Legend wrapperStyle={{ fontSize: 11, color: "#94a3b8" }} />
                        <Tooltip contentStyle={tooltipStyle} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Breakdown Table */}
          {data.breakdown && data.breakdown.length > 0 && (
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
                      {data.breakdown.map((row: any, i: number) => (
                        <tr key={i} className="border-b border-border/50 hover:bg-surface/50">
                          <td className="py-2 pr-4 font-medium">{row.category || row.name || row.gpu_type || "Other"}</td>
                          <td className="py-2 pr-4 text-right font-mono text-xs">{row.jobs ?? "—"}</td>
                          <td className="py-2 pr-4 text-right font-mono text-xs">{row.gpu_hours?.toFixed(1) ?? "—"}</td>
                          <td className="py-2 text-right font-mono text-xs">{row.spend != null ? `$${Number(row.spend).toFixed(2)}` : "—"}</td>
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
