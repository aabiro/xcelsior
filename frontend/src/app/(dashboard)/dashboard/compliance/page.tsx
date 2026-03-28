"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  RefreshCw, CheckCircle, AlertTriangle, XCircle, MapPin,
  Shield, Target, Loader2, Info,
} from "lucide-react";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";

const PROVINCE_NAMES: Record<string, string> = {
  AB: "Alberta", BC: "British Columbia", MB: "Manitoba", NB: "New Brunswick",
  NL: "Newfoundland & Labrador", NS: "Nova Scotia", NT: "Northwest Territories",
  NU: "Nunavut", ON: "Ontario", PE: "Prince Edward Island", QC: "Quebec",
  SK: "Saskatchewan", YT: "Yukon",
};

interface ComplianceCheck {
  id: string;
  name: string;
  status: "pass" | "warn" | "fail";
  description: string;
  last_checked?: string;
}

interface Province {
  code: string;
  name: string;
  gst: number;
  pst: number;
  hst: number;
  total_rate: number;
}

interface SlaTier {
  tier: string;
  uptime_target: number;
  response_time_ms: number;
  penalty_rate: number;
}

interface SlaHostSummary {
  host_id: string;
  hostname: string;
  tier: string;
  uptime_pct: number;
  violations: number;
}

export default function CompliancePage() {
  const [checks, setChecks] = useState<ComplianceCheck[]>([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState<"checks" | "provinces" | "sla">("checks");
  const { t } = useLocale();

  // Province matrix
  const [provinces, setProvinces] = useState<Province[]>([]);
  const [quebecPia, setQuebecPia] = useState<{ required: boolean; reason: string } | null>(null);

  // SLA
  const [slaTiers, setSlaTiers] = useState<SlaTier[]>([]);
  const [slaHosts, setSlaHosts] = useState<SlaHostSummary[]>([]);

  const loadChecks = useCallback(() => {
    setLoading(true);
    fetch("/api/compliance/status", { credentials: "include" })
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then((d) => setChecks(Array.isArray(d.checks) ? d.checks : []))
      .catch(() => toast.error("Failed to load compliance data"))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    loadChecks();

    // Province data — transform rates Record to array
    api.fetchTaxRates().then((d) => {
      const rates = d.rates || {};
      const provinceList: Province[] = Object.entries(rates).map(([code, total]) => ({
        code,
        name: PROVINCE_NAMES[code] || code,
        gst: 0.05,
        pst: 0,
        hst: 0,
        total_rate: Number(total),
      }));
      setProvinces(provinceList);
    }).catch(() => {});
    api.checkQuebecPia({ data_origin_province: "QC", processing_province: "QC", data_contains_pi: true })
      .then((d) => setQuebecPia({ required: d.pia_required, reason: d.reason }))
      .catch(() => {});

    // SLA data — transform Record to array
    api.fetchSlaTargets().then((d) => {
      const tiers = d.tiers || {};
      const tierList: SlaTier[] = Object.entries(tiers).map(([name, t]) => ({
        tier: name,
        uptime_target: t.uptime_pct,
        response_time_ms: 0,
        penalty_rate: t.credit_pct_100 / 100,
      }));
      setSlaTiers(tierList);
    }).catch(() => {});
    api.fetchSlaHostsSummary().then((d) => {
      const hosts = (d.hosts || []).map((h) => ({
        host_id: h.host_id,
        hostname: h.host_id.slice(0, 12),
        tier: h.sla_tier,
        uptime_pct: h.uptime_30d_pct,
        violations: h.violation_count,
      }));
      setSlaHosts(hosts);
    }).catch(() => {});
  }, [loadChecks]);

  const pass = checks.filter((c) => c.status === "pass").length;
  const warn = checks.filter((c) => c.status === "warn").length;
  const fail = checks.filter((c) => c.status === "fail").length;

  const tabs = [
    { id: "checks" as const, label: t("dash.comp.tab_checks"), icon: Shield },
    { id: "provinces" as const, label: t("dash.comp.tab_tax"), icon: MapPin },
    { id: "sla" as const, label: t("dash.comp.tab_sla"), icon: Target },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.comp.title")}</h1>
        <Button variant="outline" size="sm" onClick={loadChecks}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <Card className="p-4 flex items-center gap-3">
          <CheckCircle className="h-8 w-8 text-emerald" />
          <div>
            <p className="text-2xl font-bold font-mono">{pass}</p>
            <p className="text-xs text-text-muted">{t("dash.comp.passing")}</p>
          </div>
        </Card>
        <Card className="p-4 flex items-center gap-3">
          <AlertTriangle className="h-8 w-8 text-accent-gold" />
          <div>
            <p className="text-2xl font-bold font-mono">{warn}</p>
            <p className="text-xs text-text-muted">{t("dash.comp.warnings")}</p>
          </div>
        </Card>
        <Card className="p-4 flex items-center gap-3">
          <XCircle className="h-8 w-8 text-accent-red" />
          <div>
            <p className="text-2xl font-bold font-mono">{fail}</p>
            <p className="text-xs text-text-muted">{t("dash.comp.failing")}</p>
          </div>
        </Card>
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
          </button>
        ))}
      </div>

      {/* ── Compliance Checks Tab ── */}
      {tab === "checks" && (
        <Card>
          <CardHeader><CardTitle>{t("dash.comp.checks_title")}</CardTitle></CardHeader>
          <CardContent>
            {loading ? (
              <div className="flex items-center justify-center py-8"><Loader2 className="h-6 w-6 animate-spin text-text-muted" /></div>
            ) : checks.length === 0 ? (
              <p className="text-sm text-text-muted">{t("dash.comp.no_data")}</p>
            ) : (
              <div className="space-y-3">
                {checks.map((check) => (
                  <div key={check.id || check.name} className="flex items-start justify-between rounded-lg border border-border p-4">
                    <div className="flex items-start gap-3">
                      {check.status === "pass" ? (
                        <CheckCircle className="h-5 w-5 text-emerald mt-0.5" />
                      ) : check.status === "warn" ? (
                        <AlertTriangle className="h-5 w-5 text-accent-gold mt-0.5" />
                      ) : (
                        <XCircle className="h-5 w-5 text-accent-red mt-0.5" />
                      )}
                      <div>
                        <p className="text-sm font-medium">{check.name}</p>
                        <p className="text-xs text-text-secondary mt-0.5">{check.description}</p>
                      </div>
                    </div>
                    <Badge variant={check.status === "pass" ? "completed" : check.status === "warn" ? "warning" : "failed"}>
                      {check.status}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* ── Province Tax Matrix Tab ── */}
      {tab === "provinces" && (
        <>
          {quebecPia && (
            <div className={`flex items-start gap-3 rounded-lg border p-4 ${
              quebecPia.required ? "border-accent-gold/30 bg-accent-gold/5" : "border-emerald/30 bg-emerald/5"
            }`}>
              {quebecPia.required ? (
                <AlertTriangle className="h-5 w-5 text-accent-gold mt-0.5" />
              ) : (
                <CheckCircle className="h-5 w-5 text-emerald mt-0.5" />
              )}
              <div>
                <p className="text-sm font-medium">{quebecPia.required ? t("dash.comp.pia_required") : t("dash.comp.pia_not_required")}</p>
                <p className="text-xs text-text-secondary mt-0.5">{quebecPia.reason}</p>
              </div>
            </div>
          )}

          <Card>
            <CardHeader>
              <CardTitle>{t("dash.comp.tax_title")}</CardTitle>
              <CardDescription>{t("dash.comp.tax_desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {provinces.length === 0 ? (
                <p className="text-sm text-text-muted">No tax rate data available</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left">
                        <th className="py-2 pr-4 font-medium">{t("dash.comp.col_province")}</th>
                        <th className="py-2 pr-4 font-medium text-right">{t("dash.comp.col_gst")}</th>
                        <th className="py-2 pr-4 font-medium text-right">{t("dash.comp.col_pst")}</th>
                        <th className="py-2 pr-4 font-medium text-right">{t("dash.comp.col_hst")}</th>
                        <th className="py-2 font-medium text-right">{t("dash.comp.col_total")}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {provinces.map((p) => (
                        <tr key={p.code} className="border-b border-border/50 hover:bg-surface/50">
                          <td className="py-2.5 pr-4">
                            <span className="font-medium">{p.name}</span>
                            <span className="text-text-muted ml-1.5 text-xs">({p.code})</span>
                          </td>
                          <td className="py-2.5 pr-4 text-right font-mono text-xs">{p.gst ? `${(p.gst * 100).toFixed(1)}%` : "—"}</td>
                          <td className="py-2.5 pr-4 text-right font-mono text-xs">{p.pst ? `${(p.pst * 100).toFixed(1)}%` : "—"}</td>
                          <td className="py-2.5 pr-4 text-right font-mono text-xs">{p.hst ? `${(p.hst * 100).toFixed(1)}%` : "—"}</td>
                          <td className="py-2.5 text-right font-mono text-xs font-bold">{(p.total_rate * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* ── SLA Targets Tab ── */}
      {tab === "sla" && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>{t("dash.comp.sla_title")}</CardTitle>
              <CardDescription>{t("dash.comp.sla_desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {slaTiers.length === 0 ? (
                <p className="text-sm text-text-muted">No SLA tier data available</p>
              ) : (
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  {slaTiers.map((tier) => (
                    <div key={tier.tier} className="rounded-lg border border-border p-4">
                      <p className="text-sm font-bold capitalize mb-2">{tier.tier}</p>
                      <div className="space-y-1.5 text-xs">
                        <div className="flex justify-between">
                          <span className="text-text-muted">Uptime target</span>
                          <span className="font-mono font-medium">{tier.uptime_target}%</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-text-muted">Response time</span>
                          <span className="font-mono font-medium">{tier.response_time_ms}ms</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-text-muted">Penalty rate</span>
                          <span className="font-mono font-medium text-accent-red">{(tier.penalty_rate * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t("dash.comp.host_sla")}</CardTitle>
              <CardDescription>{t("dash.comp.host_sla_desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {slaHosts.length === 0 ? (
                <p className="text-sm text-text-muted">No host SLA data available</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left">
                        <th className="py-2 pr-4 font-medium">Host</th>
                        <th className="py-2 pr-4 font-medium">Tier</th>
                        <th className="py-2 pr-4 font-medium text-right">Uptime</th>
                        <th className="py-2 font-medium text-right">Violations</th>
                      </tr>
                    </thead>
                    <tbody>
                      {slaHosts.map((h) => (
                        <tr key={h.host_id} className="border-b border-border/50 hover:bg-surface/50">
                          <td className="py-2.5 pr-4 font-medium">{h.hostname || h.host_id.slice(0, 8)}</td>
                          <td className="py-2.5 pr-4">
                            <Badge variant="default" className="capitalize text-xs">{h.tier}</Badge>
                          </td>
                          <td className="py-2.5 pr-4 text-right">
                            <span className={`font-mono text-xs font-medium ${h.uptime_pct >= 99.9 ? "text-emerald" : h.uptime_pct >= 99 ? "text-accent-gold" : "text-accent-red"}`}>
                              {h.uptime_pct.toFixed(2)}%
                            </span>
                          </td>
                          <td className="py-2.5 text-right">
                            <span className={`font-mono text-xs ${h.violations > 0 ? "text-accent-red font-medium" : "text-text-muted"}`}>
                              {h.violations}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}
    </div>
  );
}
