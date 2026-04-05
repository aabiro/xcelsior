"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { FadeIn, HoverCard, StaggerList, StaggerItem, CountUp } from "@/components/ui/motion";
import { StatCard } from "@/components/ui/stat-card";
import {
  Activity, RefreshCw, ShieldCheck, Bell, Shield,
  CheckCircle, XCircle, Zap, Cpu, MapPin, DollarSign,
} from "lucide-react";
import * as api from "@/lib/api";
import { useEventStream } from "@/hooks/useEventStream";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend,
} from "recharts";

const tooltipStyle = { backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px" };

const EVENT_COLORS: Record<string, string> = {
  job_submitted: "bg-accent-cyan/10 text-accent-cyan border-accent-cyan/30",
  job_completed: "bg-emerald/10 text-emerald border-emerald/30",
  job_failed: "bg-accent-red/10 text-accent-red border-accent-red/30",
  host_registered: "bg-accent-violet/10 text-accent-violet border-accent-violet/30",
  host_removed: "bg-accent-red/10 text-accent-red border-accent-red/30",
  user_registered: "bg-accent-gold/10 text-accent-gold border-accent-gold/30",
  payment_received: "bg-emerald/10 text-emerald border-emerald/30",
};

const RANGE_PRESETS = [
  { label: "3d", days: 3 },
  { label: "7d", days: 7 },
  { label: "14d", days: 14 },
  { label: "30d", days: 30 },
];

export default function AdminActivityPage() {
  const [data, setData] = useState<Awaited<ReturnType<typeof api.fetchAdminActivity>> | null>(null);
  const [loading, setLoading] = useState(true);
  const [days, setDays] = useState(7);

  // Audit
  const [auditResult, setAuditResult] = useState<{ valid: boolean; details: string } | null>(null);
  const [verifyingAudit, setVerifyingAudit] = useState(false);

  // Alerts — real ALERT_CONFIG from scheduler.py
  const [alertConfig, setAlertConfig] = useState<Record<string, unknown> | null>(null);

  // Verification queue
  const [verificationQueue, setVerificationQueue] = useState<{
    host_id: string; hostname?: string; state: string; overall_score: number;
    last_check_at: string | null; gpu_model: string; province: string; cost_per_hour: number;
  }[]>([]);

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([
      api.fetchAdminActivity(days),
      api.fetchAlertConfig(),
      api.fetchAdminVerificationQueue(),
    ]).then(([a, ac, vq]) => {
      if (a.status === "fulfilled") setData(a.value);
      if (ac.status === "fulfilled") setAlertConfig(ac.value);
      if (vq.status === "fulfilled") setVerificationQueue(Array.isArray(vq.value.queue) ? vq.value.queue : []);
      setLoading(false);
    });
  }, [days]);

  useEffect(() => { load(); }, [load]);

  useEventStream({
    onEvent: () => load(),
    eventTypes: ["job_submitted", "job_completed", "job_failed", "host_registered", "host_removed"],
  });

  const handleVerifyAudit = async () => {
    setVerifyingAudit(true);
    try {
      const res = await api.verifyAuditChain();
      const ci = res.chain_integrity;
      setAuditResult({
        valid: ci.valid,
        details: ci.valid
          ? "All audit records are cryptographically linked. No tampering detected."
          : `Chain break detected at: ${ci.break_point || "unknown"}`,
      });
      toast.success(ci.valid ? "Audit chain verified" : "Audit chain integrity issue detected");
    } catch { toast.error("Failed to verify audit chain"); }
    finally { setVerifyingAudit(false); }
  };

  const handleAlertToggle = async (key: string) => {
    if (!alertConfig) return;
    const updated = { ...alertConfig, [key]: !alertConfig[key] };
    try {
      await api.updateAlertConfig(updated);
      setAlertConfig(updated);
      toast.success("Alert configuration updated");
    } catch { toast.error("Failed to update alert config"); }
  };

  const handleVerificationAction = async (hostId: string, action: "approve" | "reject") => {
    try {
      await (action === "approve" ? api.approveHost(hostId) : api.rejectHost(hostId));
      setVerificationQueue((q) => q.filter((h) => h.host_id !== hostId));
      toast.success(`Host ${action}d`);
    } catch { toast.error(`Failed to ${action} host`); }
  };

  const Toggle = ({ enabled, onToggle, label }: { enabled: boolean; onToggle: () => void; label: string }) => (
    <button
      type="button"
      role="switch"
      aria-checked={enabled}
      aria-label={label}
      onClick={onToggle}
      className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors ${enabled ? "bg-emerald" : "bg-border"}`}
    >
      <span className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${enabled ? "translate-x-6" : "translate-x-1"}`} />
    </button>
  );

  const hasEvents = (data?.events?.length ?? 0) > 0 || (data?.daily_jobs?.length ?? 0) > 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <h1 className="text-2xl font-bold">Activity</h1>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg bg-surface p-1">
            {RANGE_PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setDays(p.days)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                  days === p.days ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> Refresh
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="space-y-6">
          <div className="h-64 rounded-xl bg-surface skeleton-pulse" />
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {[...Array(2)].map((_, i) => <div key={i} className="h-64 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            {[...Array(3)].map((_, i) => <div key={i} className="h-48 rounded-xl bg-surface skeleton-pulse" />)}
          </div>
        </div>
      ) : !hasEvents ? (
        <div className="flex flex-col items-center justify-center py-20">
          <div className="flex h-20 w-20 items-center justify-center rounded-2xl bg-surface mb-6">
            <Zap className="h-10 w-10 text-text-muted" />
          </div>
          <h3 className="text-xl font-semibold mb-2">No activity yet</h3>
          <p className="text-sm text-text-secondary max-w-md text-center">
            Events will appear here as jobs are submitted, hosts register, and users sign up.
          </p>
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <StaggerList className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            <StaggerItem>
              <StatCard label="Total Events" value={<CountUp value={data?.events?.length ?? 0} />} icon={Activity} glow="cyan" />
            </StaggerItem>
            <StaggerItem>
              <StatCard label="Jobs This Period" value={<CountUp value={data?.daily_jobs?.reduce((s, d) => s + (d.submitted ?? 0), 0) ?? 0} />} icon={Zap} glow="emerald" />
            </StaggerItem>
            <StaggerItem>
              <StatCard label="Failed Jobs" value={<CountUp value={data?.daily_jobs?.reduce((s, d) => s + (d.failed ?? 0), 0) ?? 0} />} icon={XCircle} glow="gold" trend={((data?.daily_jobs?.reduce((s, d) => s + (d.failed ?? 0), 0) ?? 0) > 0) ? "down" : "flat"} trendValue={`${data?.daily_jobs?.reduce((s, d) => s + (d.failed ?? 0), 0) ?? 0} failures`} />
            </StaggerItem>
          </StaggerList>

          {/* Job Activity Chart */}
          <FadeIn>
            <HoverCard><Card className="glow-card brand-top-accent">
              <CardHeader><CardTitle className="text-sm">Job Activity ({days} days)</CardTitle></CardHeader>
              <CardContent>
                {!data?.daily_jobs?.length ? (
                  <p className="text-sm text-text-muted py-8 text-center">No job activity</p>
                ) : (
                  <div className="h-56">
                    <ResponsiveContainer width="100%" height="100%" minHeight={200} minWidth={0} debounce={1}>
                      <BarChart data={data.daily_jobs}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                        <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" tickFormatter={(v: any) => String(v).slice(5)} />
                        <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" allowDecimals={false} />
                        <Tooltip contentStyle={tooltipStyle} />
                        <Legend wrapperStyle={{ fontSize: 12 }} />
                        <Bar dataKey="submitted" fill="#38bdf8" radius={[4, 4, 0, 0]} name="Submitted" />
                        <Bar dataKey="completed" fill="#10b981" radius={[4, 4, 0, 0]} name="Completed" />
                        <Bar dataKey="failed" fill="#dc2626" radius={[4, 4, 0, 0]} name="Failed" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </CardContent>
            </Card></HoverCard>
          </FadeIn>

          <FadeIn delay={0.15}>
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              {/* Events by Type */}
              <HoverCard><Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Events by Type</CardTitle></CardHeader>
                <CardContent>
                  {!data?.by_type?.length ? (
                    <p className="text-sm text-text-muted py-4">No events</p>
                  ) : (
                    <div className="space-y-2">
                      {data.by_type.map((e) => {
                        const total = data.by_type.reduce((s, x) => s + x.count, 0);
                        const pct = total ? (e.count / total) * 100 : 0;
                        return (
                          <div key={e.event_type} className="flex items-center gap-3">
                            <span className="text-xs w-32 truncate font-mono">{e.event_type}</span>
                            <div className="flex-1 h-2 rounded-full bg-surface-hover overflow-hidden">
                              <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: "#38bdf8" }} />
                            </div>
                            <span className="text-xs font-mono w-12 text-right">{e.count}</span>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </CardContent>
              </Card></HoverCard>

              {/* Recent Events Feed */}
              <HoverCard><Card className="glow-card brand-top-accent">
                <CardHeader><CardTitle className="text-sm">Recent Events</CardTitle></CardHeader>
                <CardContent>
                  {!data?.events?.length ? (
                    <p className="text-sm text-text-muted py-4">No recent events</p>
                  ) : (
                    <div className="space-y-2 max-h-[400px] overflow-y-auto pr-1">
                      {data.events.slice(-50).reverse().map((e) => (
                        <div key={e.event_id} className={`rounded-lg border p-3 text-xs ${EVENT_COLORS[e.event_type] || "bg-surface border-border"}`}>
                          <div className="flex items-center justify-between">
                            <span className="font-mono font-medium">{e.event_type}</span>
                            <span className="text-[10px] opacity-60">
                              {new Date(e.timestamp).toLocaleString()}
                            </span>
                          </div>
                          <div className="mt-1 opacity-80">
                            <span>{e.entity_type}/{e.entity_id}</span>
                            {e.actor && <span className="ml-2">by {e.actor}</span>}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card></HoverCard>
            </div>
          </FadeIn>
        </>
      )}

      {/* Audit, Alerts, Verification Queue — always visible */}
      <FadeIn delay={0.25}>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Audit Chain */}
          <HoverCard><Card className="glow-card brand-top-accent">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm"><ShieldCheck className="h-4 w-4" /> Audit Chain</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <Button size="sm" onClick={handleVerifyAudit} disabled={verifyingAudit}>
                {verifyingAudit ? <><RefreshCw className="h-3.5 w-3.5 animate-spin" /> Verifying…</> : <><ShieldCheck className="h-3.5 w-3.5" /> Verify</>}
              </Button>
              {auditResult && (
                <div className={`flex items-start gap-2 rounded-lg border p-3 text-xs ${
                  auditResult.valid ? "border-emerald/30 bg-emerald/5" : "border-accent-red/30 bg-accent-red/5"
                }`}>
                  {auditResult.valid ? <CheckCircle className="h-4 w-4 text-emerald shrink-0" /> : <XCircle className="h-4 w-4 text-accent-red shrink-0" />}
                  <p>{auditResult.details}</p>
                </div>
              )}
            </CardContent>
          </Card></HoverCard>

          {/* Alert Config — matches real ALERT_CONFIG from scheduler.py */}
          <HoverCard><Card className="glow-card brand-top-accent">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm"><Bell className="h-4 w-4" /> Alerts</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {!alertConfig ? (
                <p className="text-xs text-text-muted">Loading...</p>
              ) : (
                [
                  { key: "email_enabled", label: "Email Alerts" },
                  { key: "telegram_enabled", label: "Telegram Alerts" },
                ].map((item) => (
                  <div key={item.key} className="flex items-center justify-between py-1">
                    <span className="text-xs">{item.label}</span>
                    <Toggle
                      enabled={!!alertConfig[item.key]}
                      onToggle={() => handleAlertToggle(item.key)}
                      label={item.label}
                    />
                  </div>
                ))
              )}
              {alertConfig && (
                <div className="mt-3 space-y-1 pt-3 border-t border-border">
                  <p className="text-[10px] text-text-muted uppercase tracking-wide font-medium">SMTP</p>
                  <p className="text-xs text-text-secondary font-mono truncate">{String(alertConfig.smtp_host || "Not configured")}</p>
                  <p className="text-[10px] text-text-muted uppercase tracking-wide font-medium mt-2">Telegram</p>
                  <p className="text-xs text-text-secondary font-mono truncate">
                    {alertConfig.telegram_chat_id ? `Chat: ${alertConfig.telegram_chat_id === "***" ? "***" : alertConfig.telegram_chat_id}` : "Not configured"}
                  </p>
                </div>
              )}
            </CardContent>
          </Card></HoverCard>

          {/* Verification Queue */}
          <HoverCard><Card className="glow-card brand-top-accent">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <Shield className="h-4 w-4" /> Verification Queue
                {verificationQueue.length > 0 && (
                  <span className="ml-1 bg-accent-red text-white text-[10px] px-1.5 py-0.5 rounded-full">{verificationQueue.length}</span>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent>
              {verificationQueue.length === 0 ? (
                <div className="text-center py-4">
                  <CheckCircle className="mx-auto h-6 w-6 text-emerald mb-1" />
                  <p className="text-xs text-text-muted">All clear</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {verificationQueue.map((host) => (
                    <div key={host.host_id} className="rounded-lg border border-border p-3">
                      <p className="text-xs font-medium mb-1 font-mono truncate">{host.hostname || host.host_id}</p>
                      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-text-muted mb-2">
                        {host.gpu_model && <span className="flex items-center gap-0.5"><Cpu className="h-2.5 w-2.5" />{host.gpu_model}</span>}
                        {host.province && <span className="flex items-center gap-0.5"><MapPin className="h-2.5 w-2.5" />{host.province}</span>}
                        <span className="flex items-center gap-0.5"><DollarSign className="h-2.5 w-2.5" />${host.cost_per_hour?.toFixed(2) ?? "--"}/hr</span>
                        <span>Score: {host.overall_score?.toFixed(1) ?? "--"}</span>
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline" className="text-emerald border-emerald/30 hover:bg-emerald/10 text-xs h-7"
                          onClick={() => handleVerificationAction(host.host_id, "approve")}>
                          Approve
                        </Button>
                        <Button size="sm" variant="outline" className="text-accent-red border-accent-red/30 hover:bg-accent-red/10 text-xs h-7"
                          onClick={() => handleVerificationAction(host.host_id, "reject")}>
                          Reject
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card></HoverCard>
        </div>
      </FadeIn>
    </div>
  );
}
