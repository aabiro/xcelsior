"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { FadeIn, StaggerList, StaggerItem, CountUp, HoverCard } from "@/components/ui/motion";
import { Server, RefreshCw, ShieldCheck, Award, ServerOff, Download } from "lucide-react";
import * as api from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import {
  PieChart, Pie, Cell, BarChart, Bar,
  XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

const COLORS = ["#dc2626", "#f59e0b", "#38bdf8", "#10b981", "#8b5cf6", "#ec4899", "#6366f1", "#14b8a6"];
const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

/* Host status colors matching the actual DB values from scheduler.py check_hosts() */
const STATUS_COLORS: Record<string, string> = {
  active: "#10b981",   // green — host is alive and responding
  dead: "#dc2626",     // red — host failed ping
  unknown: "#94a3b8",  // grey — no status set yet
};

/* Reputation tier colors matching the tier system in reputation.py */
const TIER_COLORS: Record<string, string> = {
  diamond: "#38bdf8",
  platinum: "#8b5cf6",
  gold: "#f59e0b",
  silver: "#94a3b8",
  bronze: "#cd7f32",
};

export default function AdminInfrastructurePage() {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.fetchAdminInfrastructure>> | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(() => {
    setLoading(true);
    api.fetchAdminInfrastructure()
      .then(setData)
      .catch(() => toast.error("Failed to load infrastructure data"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  useEventStream({
    eventTypes: ["host_registered", "host_removed"],
    onEvent: load,
  });

  const hasData = (data?.total_hosts ?? 0) > 0;

  const exportCsv = () => {
    if (!data?.by_gpu?.length) { toast.error("No data to export"); return; }
    const rows = [
      "section,key,value",
      ...data.by_state.map((s) => `status,${s.state},${s.count}`),
      ...data.by_gpu.map((g) => `gpu,${g.gpu_model},${g.count}`),
      ...data.by_province.map((p) => `province,${p.province},${p.count}`),
      ...data.verification.map((v) => `verification,${v.state},${v.count}`),
      ...data.reputation_tiers.map((t) => `reputation,${t.tier},${t.count}`),
    ];
    const csv = rows.join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url; a.download = "xcelsior-infrastructure.csv"; a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Infrastructure</h1>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={exportCsv}>
            <Download className="h-3.5 w-3.5" /> CSV
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[...Array(3)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {[...Array(4)].map((_, i) => <div key={i} className="h-64 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
        </div>
      ) : !hasData ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
            <ServerOff className="h-10 w-10 text-text-muted" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No hosts registered</h3>
          <p className="text-sm text-text-secondary max-w-md text-center mb-6">
            Infrastructure data will appear once GPU hosts are registered on the platform.
          </p>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 w-full max-w-lg">
            <Card className="p-4 text-center border-dashed">
              <Server className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">Hosts</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
            <Card className="p-4 text-center border-dashed">
              <ShieldCheck className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">Verified</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
            <Card className="p-4 text-center border-dashed">
              <Award className="h-5 w-5 text-text-muted mx-auto mb-2" />
              <p className="text-xs text-text-muted">GPU Models</p>
              <p className="text-lg font-bold font-mono text-text-muted/50">—</p>
            </Card>
          </div>
        </div>
      ) : (
        <>
          <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StaggerItem>
              <StatCard label="Total Hosts" value={<CountUp value={data!.total_hosts} />} icon={Server} glow="cyan" />
            </StaggerItem>
            <StaggerItem>
              <StatCard
                label="Verified"
                value={<CountUp value={data!.verification?.find((v) => v.state === "verified")?.count ?? 0} />}
                icon={ShieldCheck} glow="emerald"
              />
            </StaggerItem>
            <StaggerItem>
              <StatCard label="GPU Models" value={<CountUp value={data!.by_gpu?.length ?? 0} />} icon={Award} glow="violet" />
            </StaggerItem>
          </StaggerList>

          <FadeIn delay={0.15}>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Hosts by Status (Pie) */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Hosts by Status</CardTitle></CardHeader>
                <CardContent>
                  {!data?.by_state?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No hosts</p>
                  ) : (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={0} debounce={1}>
                        <PieChart>
                          <Pie
                            data={data.by_state}
                            dataKey="count"
                            nameKey="state"
                            cx="50%" cy="50%"
                            outerRadius={90}
                            label={(props: any) => `${props.name || ""} (${props.value || 0})`}
                            labelLine={false}
                            fontSize={11}
                          >
                            {data.by_state.map((d) => (
                              <Cell key={d.state} fill={STATUS_COLORS[d.state] || "#94a3b8"} />
                            ))}
                          </Pie>
                          <Tooltip contentStyle={tooltipStyle} />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>

              {/* Hosts by GPU Model (Bar) */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Hosts by GPU</CardTitle></CardHeader>
                <CardContent>
                  {!data?.by_gpu?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No data</p>
                  ) : (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={0} debounce={1}>
                        <BarChart data={data.by_gpu} layout="vertical">
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" allowDecimals={false} />
                          <YAxis type="category" dataKey="gpu_model" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" width={120} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>

              {/* By Province */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Hosts by Province</CardTitle></CardHeader>
                <CardContent>
                  {!data?.by_province?.length ? (
                    <p className="text-sm text-text-muted py-8 text-center">No data</p>
                  ) : (
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%" minHeight={1} minWidth={0} debounce={1}>
                        <BarChart data={data.by_province}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                          <XAxis dataKey="province" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                          <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" allowDecimals={false} />
                          <Tooltip contentStyle={tooltipStyle} />
                          <Bar dataKey="count" fill="#38bdf8" radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </CardContent>
              </Card>
              </HoverCard>

              {/* Verification + Reputation */}
              <HoverCard>
              <Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Verification & Reputation</CardTitle></CardHeader>
                <CardContent className="space-y-6">
                  {/* Verification States */}
                  <div>
                    <p className="text-xs text-text-muted mb-2 font-medium uppercase tracking-wide">Verification</p>
                    {!data?.verification?.length ? (
                      <p className="text-sm text-text-muted">No verification data</p>
                    ) : (
                      <div className="space-y-2">
                        {data.verification.map((v) => {
                          const total = data.verification.reduce((s, x) => s + x.count, 0);
                          const pct = total ? (v.count / total) * 100 : 0;
                          return (
                            <div key={v.state} className="flex items-center gap-3">
                              <span className="text-sm w-20 capitalize">{v.state}</span>
                              <div className="flex-1 h-2 rounded-full bg-surface-hover overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all"
                                  style={{ width: `${pct}%`, backgroundColor: STATUS_COLORS[v.state] || COLORS[0] }}
                                />
                              </div>
                              <span className="text-sm font-mono w-8 text-right">{v.count}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>

                  {/* Reputation Tiers */}
                  <div>
                    <p className="text-xs text-text-muted mb-2 font-medium uppercase tracking-wide">Reputation Tiers</p>
                    {!data?.reputation_tiers?.length ? (
                      <p className="text-sm text-text-muted">No reputation data</p>
                    ) : (
                      <div className="space-y-2">
                        {data.reputation_tiers.map((t) => {
                          const total = data.reputation_tiers.reduce((s, x) => s + x.count, 0);
                          const pct = total ? (t.count / total) * 100 : 0;
                          return (
                            <div key={t.tier} className="flex items-center gap-3">
                              <span className="text-sm w-20 capitalize">{t.tier}</span>
                              <div className="flex-1 h-2 rounded-full bg-surface-hover overflow-hidden">
                                <div
                                  className="h-full rounded-full transition-all"
                                  style={{ width: `${pct}%`, backgroundColor: TIER_COLORS[t.tier] || COLORS[0] }}
                                />
                              </div>
                              <span className="text-sm font-mono w-8 text-right">{t.count}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>
              </HoverCard>
            </div>
          </FadeIn>
        </>
      )}
    </div>
  );
}
