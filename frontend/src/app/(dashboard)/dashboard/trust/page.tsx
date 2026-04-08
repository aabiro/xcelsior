"use client";

import { useEffect, useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { StatusBadge } from "@/components/ui/badge";
import { Input, Select } from "@/components/ui/input";
import {
  ShieldCheck, Search, RefreshCw, CheckCircle, XCircle, AlertTriangle, Lock, Globe, MapPin, Shield, type LucideIcon,
} from "lucide-react";
import {
  fetchVerifiedHosts, fetchTrustTiers, fetchTransparencyReport,
  approveHost, rejectHost,
} from "@/lib/api";
import type { VerifiedHost } from "@/lib/api";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";

const TIER_ICONS: Record<string, LucideIcon> = {
  community: Globe,
  residency: MapPin,
  sovereignty: Shield,
  regulated: Lock,
};

const TIER_COLORS: Record<string, string> = {
  community: "bg-ice-blue/10 border-ice-blue/20 text-ice-blue",
  residency: "bg-emerald/10 border-emerald/20 text-emerald",
  sovereignty: "bg-accent-gold/10 border-accent-gold/20 text-accent-gold",
  regulated: "bg-accent-red/10 border-accent-red/20 text-accent-red",
};

export default function TrustPage() {
  const { t } = useLocale();
  const [hosts, setHosts] = useState<VerifiedHost[]>([]);
  const [tiers, setTiers] = useState<Record<string, { min_score: number; requirements: string[] }>>({});
  const [report, setReport] = useState<{
    period_months: number;
    summary: {
      requests_received: number;
      complied: number;
      challenged: number;
      pending: number;
    };
    cloud_act_note: string;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");

  const load = () => {
    setLoading(true);
    Promise.allSettled([
      fetchVerifiedHosts(),
      fetchTrustTiers(),
      fetchTransparencyReport(),
    ]).then(([h, t, r]) => {
      if (h.status === "fulfilled") setHosts(h.value.hosts || []);
      if (t.status === "fulfilled") setTiers(t.value.tiers || {});
      if (r.status === "fulfilled") setReport(r.value);
      setLoading(false);
    });
  };

  useEffect(() => { load(); }, []);

  async function handleApprove(hostId: string) {
    try {
      await approveHost(hostId);
      toast.success(`Host ${hostId.slice(0, 8)} approved`);
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Approve failed");
    }
  }

  async function handleReject(hostId: string) {
    const reason = prompt("Reason for rejection:");
    if (reason === null) return;
    try {
      await rejectHost(hostId, reason);
      toast.success(`Host ${hostId.slice(0, 8)} rejected`);
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Reject failed");
    }
  }

  const filtered = hosts.filter((h) => {
    if (statusFilter !== "all" && h.status !== statusFilter) return false;
    if (search && !h.host_id.toLowerCase().includes(search.toLowerCase()) && !h.gpu_model?.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.trust.title")}</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {/* Trust Tiers */}
      {Object.keys(tiers).length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {Object.entries(tiers).map(([name, tier]) => (
            <Card key={name} className={`border ${TIER_COLORS[name] || ""}`}>
              {(() => {
                const TierIcon = TIER_ICONS[name] || ShieldCheck;
                return (
                  <div className="flex items-center gap-2 mb-3">
                    <span className="flex h-10 w-10 items-center justify-center rounded-xl border border-current/15 bg-black/10">
                      <TierIcon className="h-5 w-5" />
                    </span>
                    <h3 className="font-semibold capitalize">{name}</h3>
                  </div>
                );
              })()}
              <p className="text-xs text-text-muted mb-2">Min Score: {tier.min_score ?? 0}</p>
              <ul className="space-y-1">
                {(tier.requirements || []).map((req, i) => (
                  <li key={i} className="flex items-start gap-1.5 text-xs text-text-secondary">
                    <CheckCircle className="h-3 w-3 mt-0.5 text-emerald shrink-0" />
                    {req}
                  </li>
                ))}
              </ul>
            </Card>
          ))}
        </div>
      )}

      {/* Transparency Summary */}
      {report && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Lock className="h-4 w-4 text-accent-gold" />
            <h2 className="text-sm font-semibold text-text-secondary">
              Transparency Report ({report.period_months} months)
            </h2>
          </div>
          <div className="grid gap-4 sm:grid-cols-4">
            <div className="text-center">
              <p className="text-2xl font-bold font-mono">{report.summary.requests_received}</p>
              <p className="text-xs text-text-muted">Requests Received</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold font-mono text-emerald">{report.summary.complied}</p>
              <p className="text-xs text-text-muted">Complied</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold font-mono text-accent-gold">{report.summary.challenged}</p>
              <p className="text-xs text-text-muted">Challenged</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold font-mono text-ice-blue">{report.summary.pending}</p>
              <p className="text-xs text-text-muted">Pending</p>
            </div>
          </div>
          {report.cloud_act_note && (
            <div className="mt-4 rounded-lg bg-accent-red/5 border border-accent-red/20 p-3">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-accent-red mt-0.5 shrink-0" />
                <p className="text-xs text-text-secondary">{report.cloud_act_note}</p>
              </div>
            </div>
          )}
        </Card>
      )}

      {/* Verified Hosts Table */}
      <Card>
        <div className="flex items-center gap-2 mb-4">
          <ShieldCheck className="h-4 w-4 text-emerald" />
          <h2 className="text-sm font-semibold text-text-secondary">{t("dash.trust.verified_hosts")}</h2>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row mb-4">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-text-muted" />
            <Input className="pl-9" placeholder={t("dash.trust.search")} value={search} onChange={(e) => setSearch(e.target.value)} />
          </div>
          <Select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">{t("dash.trust.all_status")}</option>
            <option value="verified">{t("dash.trust.verified")}</option>
            <option value="unverified">{t("dash.trust.unverified")}</option>
            <option value="deverified">{t("dash.trust.deverified")}</option>
          </Select>
        </div>

        {loading ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-14 rounded-xl bg-surface skeleton-pulse" />
            ))}
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-8 text-center">
            <ShieldCheck className="mx-auto h-10 w-10 text-text-muted mb-3" />
            <p className="text-sm text-text-secondary">No hosts match your filters.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-text-secondary">
                  <th className="py-3 pr-4 text-left font-medium">Host</th>
                  <th className="py-3 px-4 text-left font-medium">GPU</th>
                  <th className="py-3 px-4 text-center font-medium">Status</th>
                  <th className="py-3 px-4 text-center font-medium">Score</th>
                  <th className="py-3 px-4 text-center font-medium">Last Check</th>
                  <th className="py-3 px-4 text-right font-medium">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((host) => (
                  <tr key={host.host_id} className="border-b border-border/50 hover:bg-surface-hover">
                    <td className="py-3 pr-4">
                      <span className="font-mono text-xs">{host.host_id.slice(0, 16)}</span>
                      {host.gpu_fingerprint && (
                        <p className="text-xs text-text-muted">FP: {host.gpu_fingerprint.slice(0, 12)}...</p>
                      )}
                    </td>
                    <td className="py-3 px-4 text-text-secondary">{host.gpu_model || "—"}</td>
                    <td className="py-3 px-4 text-center">
                      <StatusBadge status={host.status} />
                    </td>
                    <td className="py-3 px-4 text-center font-mono">
                      {host.overall_score.toFixed(1)}
                    </td>
                    <td className="py-3 px-4 text-center text-text-muted text-xs">
                      {host.last_check ? new Date(host.last_check).toLocaleDateString() : "—"}
                    </td>
                    <td className="py-3 px-4 text-right">
                      <div className="flex justify-end gap-1">
                        {host.status !== "verified" && (
                          <Button variant="ghost" size="sm" onClick={() => handleApprove(host.host_id)} className="text-emerald hover:text-emerald">
                            <CheckCircle className="h-3.5 w-3.5" />
                          </Button>
                        )}
                        {host.status !== "deverified" && (
                          <Button variant="ghost" size="sm" onClick={() => handleReject(host.host_id)} className="text-accent-red hover:text-accent-red">
                            <XCircle className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
