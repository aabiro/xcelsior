"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge, StatusBadge } from "@/components/ui/badge";
import { Input, Label } from "@/components/ui/input";
import {
  Users, Server, Activity, Shield, RefreshCw, ShieldCheck, Bell,
  CheckCircle, XCircle, AlertTriangle, Loader2, Search, FileText,
} from "lucide-react";
import * as api from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import { useAuth } from "@/lib/auth";
import { useRouter } from "next/navigation";
import { useEventStream } from "@/hooks/useEventStream";

export default function AdminPage() {
  const { t } = useLocale();
  const { user } = useAuth();
  const router = useRouter();
  const isAdmin = !!user?.is_admin;

  useEffect(() => {
    if (user && !isAdmin) router.replace("/dashboard");
  }, [user, isAdmin, router]);

  if (!user || !isAdmin) return null;

  return <AdminContent />;
}

function AdminContent() {
  const { t } = useLocale();
  const [stats, setStats] = useState<any>(null);
  const [users, setUsers] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"users" | "audit" | "alerts" | "verification">("users");
  const [searchQuery, setSearchQuery] = useState("");

  // Audit
  const [auditResult, setAuditResult] = useState<{ valid: boolean; details: string } | null>(null);
  const [verifyingAudit, setVerifyingAudit] = useState(false);

  // Alerts
  const [alertConfig, setAlertConfig] = useState<any>(null);

  // Verification queue
  const [verificationQueue, setVerificationQueue] = useState<any[]>([]);

  const load = useCallback(() => {
    setLoading(true);
    Promise.allSettled([
      api.fetchAdminStats(),
      api.fetchAdminUsers(),
      api.fetchAlertConfig(),
      fetch("/api/admin/verification-queue", { credentials: "include" }).then((r) => r.ok ? r.json() : Promise.reject()),
    ]).then(([s, u, ac, vq]) => {
      if (s.status === "fulfilled") setStats(s.value);
      if (u.status === "fulfilled") setUsers(Array.isArray(u.value.users) ? u.value.users : []);
      if (ac.status === "fulfilled") setAlertConfig(ac.value);
      if (vq.status === "fulfilled") setVerificationQueue(Array.isArray(vq.value.queue) ? vq.value.queue : []);
      setLoading(false);
    });
  }, []);

  useEffect(() => { load(); }, [load]);

  // Live updates — refresh stats on job/host/user changes
  useEventStream({
    eventTypes: ["job_status", "job_submitted", "host_registered", "host_removed", "user_registered"],
    onEvent: () => { load(); },
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
      const res = await fetch(`/api/verify/${encodeURIComponent(hostId)}/${action}`, {
        method: "POST", credentials: "include",
      });
      if (!res.ok) throw new Error();
      setVerificationQueue((q) => q.filter((h) => h.host_id !== hostId));
      toast.success(`Host ${action}d`);
    } catch { toast.error(`Failed to ${action} host`); }
  };

  const filteredUsers = users.filter((u) =>
    !searchQuery || u.email?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const tabs = [
    { id: "users" as const, label: t("dash.admin.tab_users"), icon: Users },
    { id: "audit" as const, label: t("dash.admin.tab_audit"), icon: ShieldCheck },
    { id: "alerts" as const, label: t("dash.admin.tab_alerts"), icon: Bell },
    { id: "verification" as const, label: t("dash.admin.tab_verification"), icon: Shield, count: verificationQueue.length },
  ];

  const Toggle = ({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) => (
    <button
      onClick={onToggle}
      className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors ${enabled ? "bg-emerald" : "bg-border"}`}
    >
      <span className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${enabled ? "translate-x-6" : "translate-x-1"}`} />
    </button>
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.admin.title")}</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label={t("dash.admin.total_users")} value={stats?.total_users ?? "—"} icon={Users} />
        <StatCard label={t("dash.admin.active_hosts")} value={stats?.active_hosts ?? "—"} icon={Server} />
        <StatCard label={t("dash.admin.running_instances")} value={stats?.running_jobs ?? "—"} icon={Activity} />
        <StatCard label={t("dash.admin.revenue_mtd")} value={`$${stats?.revenue_mtd?.toFixed(2) || "0.00"}`} icon={Shield} />
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 rounded-lg bg-surface p-1">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              tab === t.id ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"
            }`}
          >
            <t.icon className="h-3.5 w-3.5" /> {t.label}
            {t.count ? <span className="ml-1 bg-accent-red text-white text-[10px] px-1.5 py-0.5 rounded-full">{t.count}</span> : null}
          </button>
        ))}
      </div>

      {/* ── Users Tab ── */}
      {tab === "users" && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Users</CardTitle>
              <div className="relative w-64">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-text-muted" />
                <Input
                  placeholder="Search by email..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9 h-8 text-sm"
                />
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-text-muted" /></div>
            ) : filteredUsers.length === 0 ? (
              <p className="text-sm text-text-muted">No users found</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-text-secondary">
                      <th className="py-3 pr-4 text-left font-medium">Email</th>
                      <th className="py-3 px-4 text-left font-medium">Role</th>
                      <th className="py-3 px-4 text-center font-medium">Access</th>
                      <th className="py-3 px-4 text-center font-medium">Status</th>
                      <th className="py-3 px-4 text-center font-medium">Created</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredUsers.slice(0, 50).map((u) => (
                      <tr key={u.id || u.email} className="border-b border-border/50 hover:bg-surface-hover">
                        <td className="py-3 pr-4 font-medium">{u.email}</td>
                        <td className="py-3 px-4">
                          <Badge variant="default">{u.role || "submitter"}</Badge>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <Badge variant={u.is_admin ? "active" : "default"}>
                            {u.is_admin ? "Platform Admin" : "Standard"}
                          </Badge>
                        </td>
                        <td className="py-3 px-4 text-center">
                          <StatusBadge status={u.is_active ? "active" : "offline"} />
                        </td>
                        <td className="py-3 px-4 text-center text-text-muted">
                          {u.created_at ? new Date(u.created_at).toLocaleDateString() : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Audit Chain Tab ── */}
      {tab === "audit" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><ShieldCheck className="h-4 w-4" /> {t("dash.admin.audit_title")}</CardTitle>
            <CardDescription>{t("dash.admin.audit_desc")}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={handleVerifyAudit} disabled={verifyingAudit}>
              {verifyingAudit ? <><Loader2 className="h-4 w-4 animate-spin" /> Verifying…</> : <><ShieldCheck className="h-4 w-4" /> {t("dash.admin.verify_btn")}</>}
            </Button>
            {auditResult && (
              <div className={`flex items-start gap-3 rounded-lg border p-4 ${
                auditResult.valid ? "border-emerald/30 bg-emerald/5" : "border-accent-red/30 bg-accent-red/5"
              }`}>
                {auditResult.valid ? (
                  <CheckCircle className="h-5 w-5 text-emerald mt-0.5" />
                ) : (
                  <XCircle className="h-5 w-5 text-accent-red mt-0.5" />
                )}
                <div>
                  <p className="text-sm font-medium">{auditResult.valid ? "Chain Verified" : "Integrity Issue Detected"}</p>
                  <p className="text-xs text-text-secondary mt-0.5">{auditResult.details}</p>
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Alerts Tab ── */}
      {tab === "alerts" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Bell className="h-4 w-4" /> {t("dash.admin.alerts_title")}</CardTitle>
            <CardDescription>{t("dash.admin.alerts_desc")}</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            {!alertConfig ? (
              <p className="text-sm text-text-muted">Loading alert configuration...</p>
            ) : (
              <>
                {[
                  { key: "sla_violations", label: "SLA Violations", desc: "Alert when a host breaches its SLA tier uptime target" },
                  { key: "payment_failures", label: "Payment Failures", desc: "Alert on failed Stripe payment intents" },
                  { key: "host_offline", label: "Host Offline", desc: "Alert when a host goes offline unexpectedly" },
                  { key: "security_events", label: "Security Events", desc: "Alert on suspicious login attempts or token abuse" },
                  { key: "low_balance", label: "Low Balance Warnings", desc: "Alert users when wallet balance drops below threshold" },
                ].map((item) => (
                  <div key={item.key} className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div>
                      <p className="text-sm font-medium">{item.label}</p>
                      <p className="text-xs text-text-secondary">{item.desc}</p>
                    </div>
                    <Toggle
                      enabled={alertConfig[item.key] ?? false}
                      onToggle={() => handleAlertToggle(item.key)}
                    />
                  </div>
                ))}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Verification Queue Tab ── */}
      {tab === "verification" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><Shield className="h-4 w-4" /> {t("dash.admin.host_queue")}</CardTitle>
            <CardDescription>{t("dash.admin.host_queue_desc")}</CardDescription>
          </CardHeader>
          <CardContent>
            {verificationQueue.length === 0 ? (
              <div className="text-center py-8">
                <CheckCircle className="mx-auto h-8 w-8 text-emerald mb-2" />
                <p className="text-sm text-text-muted">No pending verifications</p>
              </div>
            ) : (
              <div className="space-y-3">
                {verificationQueue.map((host) => (
                  <div key={host.host_id} className="rounded-lg border border-border p-4">
                    <div className="flex items-start justify-between">
                      <div>
                        <p className="text-sm font-medium">{host.hostname || host.host_id}</p>
                        <div className="flex items-center gap-3 text-xs text-text-muted mt-1">
                          {host.gpu_model && <span>GPU: {host.gpu_model}</span>}
                          {host.region && <span>Region: {host.region}</span>}
                          {host.owner_email && <span>Owner: {host.owner_email}</span>}
                          {host.submitted_at && <span>Submitted: {new Date(host.submitted_at).toLocaleDateString()}</span>}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          size="sm" variant="outline"
                          className="text-emerald border-emerald/30 hover:bg-emerald/10"
                          onClick={() => handleVerificationAction(host.host_id, "approve")}
                        >
                          <CheckCircle className="h-3.5 w-3.5" /> Approve
                        </Button>
                        <Button
                          size="sm" variant="outline"
                          className="text-accent-red border-accent-red/30 hover:bg-accent-red/10"
                          onClick={() => handleVerificationAction(host.host_id, "reject")}
                        >
                          <XCircle className="h-3.5 w-3.5" /> Reject
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
