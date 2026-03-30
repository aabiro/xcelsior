"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DollarSign, TrendingUp, RefreshCw, ArrowUpRight, ExternalLink,
  Percent, AlertTriangle, CheckCircle, Loader2, LinkIcon,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { Payout } from "@/lib/api";
import { toast } from "sonner";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from "recharts";

interface ProviderInfo {
  provider_id: string;
  provider_type: string;
  status: string;
  corporation_name?: string;
  email: string;
  province?: string;
  created_at: string;
  onboarded_at?: string;
}

export default function EarningsPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const providerId = user?.provider_id || "";
  const customerId = user?.customer_id || user?.user_id || "";

  const [earnings, setEarnings] = useState<{
    total_jobs: number; total_earned_cad: number;
    total_platform_cad: number; total_tax_cad: number;
  } | null>(null);
  const [payouts, setPayouts] = useState<Payout[]>([]);
  const [provider, setProvider] = useState<ProviderInfo | null>(null);
  const [gst, setGst] = useState<{
    total_revenue_cad: number; threshold_cad: number;
    must_register: boolean;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [onboarding, setOnboarding] = useState(false);

  const load = useCallback(async () => {
    if (!providerId && !customerId) return;
    setLoading(true);
    try {
      // Try providerId first, fall back to customerId for provider lookup
      // (registerProvider uses customerId as provider_id when no providerId exists)
      const pid = providerId || customerId;
      const promises: Promise<any>[] = [];
      promises.push(api.fetchProviderEarnings(pid).catch(() => null));
      promises.push(api.fetchProvider(pid).catch(() => null));
      promises.push(api.fetchGstThreshold(customerId || providerId));
      const [earningsRes, providerRes, gstRes] = await Promise.allSettled(promises);
      if (earningsRes.status === "fulfilled" && earningsRes.value) {
        setEarnings(earningsRes.value.earnings);
        setPayouts(earningsRes.value.recent_payouts || []);
      }
      if (providerRes.status === "fulfilled" && providerRes.value?.provider) {
        setProvider(providerRes.value.provider as ProviderInfo);
      }
      if (gstRes.status === "fulfilled") {
        setGst({
          total_revenue_cad: gstRes.value.total_revenue_cad,
          threshold_cad: gstRes.value.threshold_cad,
          must_register: gstRes.value.must_register,
        });
      }
    } catch {
      toast.error("Failed to load earnings");
    } finally {
      setLoading(false);
    }
  }, [providerId, customerId]);

  useEffect(() => { load(); }, [load]);

  const handleStripeConnect = async () => {
    const pid = providerId || customerId;
    if (!pid || !user?.email) return;
    setOnboarding(true);
    try {
      const res = await api.registerProvider({
        provider_id: pid,
        email: user.email,
      });
      if (res.onboarding_url) {
        window.location.href = res.onboarding_url;
      } else {
        toast.success("Provider account created");
        load();
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to start onboarding");
    } finally {
      setOnboarding(false);
    }
  };

  // Build chart data from payouts
  const chartData = payouts
    .slice()
    .sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime())
    .reduce<{ date: string; amount: number }[]>((acc, p) => {
      const date = new Date(p.created_at).toLocaleDateString("en-CA", { month: "short", day: "numeric" });
      const existing = acc.find((d) => d.date === date);
      if (existing) existing.amount += p.provider_share_cad;
      else acc.push({ date, amount: p.provider_share_cad });
      return acc;
    }, [])
    .slice(-30);

  const netEarnings = (earnings?.total_earned_cad ?? 0) - (earnings?.total_platform_cad ?? 0) - (earnings?.total_tax_cad ?? 0);

  const statusColor: Record<string, string> = {
    active: "text-emerald",
    onboarding: "text-gold",
    pending: "text-text-muted",
    restricted: "text-accent-red",
    suspended: "text-accent-red",
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.earnings.title")}</h1>
        <Button variant="outline" size="sm" onClick={load}>
          <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
        </Button>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          {[...Array(4)].map((_, i) => <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />)}
        </div>
      ) : (
        <>
          {/* Stat Cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label={t("dash.earnings.total_earned")}
              value={`$${(earnings?.total_earned_cad ?? 0).toFixed(2)}`}
              icon={DollarSign}
            />
            <StatCard
              label={t("dash.earnings.net_earnings")}
              value={`$${netEarnings.toFixed(2)}`}
              icon={TrendingUp}
            />
            <StatCard
              label={t("dash.earnings.platform_fee")}
              value={`$${(earnings?.total_platform_cad ?? 0).toFixed(2)}`}
              icon={Percent}
            />
            <StatCard
              label={t("dash.earnings.instances_completed")}
              value={earnings?.total_jobs ?? 0}
              icon={ArrowUpRight}
            />
          </div>

          {/* Stripe Connect Status + GST Alert */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* Stripe Connect */}
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">{t("dash.earnings.stripe_title")}</CardTitle>
                <CardDescription>{t("dash.earnings.stripe_desc")}</CardDescription>
              </CardHeader>
              <CardContent>
                {provider?.status === "active" ? (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-emerald" />
                      <span className="text-sm font-medium text-emerald">{t("dash.earnings.stripe_connected")}</span>
                    </div>
                    <span className="text-xs text-text-muted">
                      Since {provider.onboarded_at ? new Date(provider.onboarded_at).toLocaleDateString() : "—"}
                    </span>
                  </div>
                ) : provider ? (
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className={`text-sm font-medium ${statusColor[provider.status] || "text-text-muted"}`}>
                        {provider.status.charAt(0).toUpperCase() + provider.status.slice(1)}
                      </span>
                    </div>
                    <Button variant="outline" size="sm" onClick={handleStripeConnect} disabled={onboarding}>
                      {onboarding ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <ExternalLink className="h-3.5 w-3.5" />}
                      Complete Setup
                    </Button>
                  </div>
                ) : (
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium">Become a Provider</p>
                      <p className="text-xs text-text-muted">Connect Stripe to earn from your GPU resources</p>
                    </div>
                    <Button variant="gold" size="sm" onClick={handleStripeConnect} disabled={onboarding}>
                      {onboarding ? (
                        <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Setting up…</>
                      ) : (
                        <><LinkIcon className="h-3.5 w-3.5" /> Connect Stripe</>
                      )}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* GST Threshold */}
            <Card className={gst?.must_register ? "border-gold/30 bg-gold/5" : ""}>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">{t("dash.earnings.gst_title")}</CardTitle>
                <CardDescription>{t("dash.earnings.gst_desc")}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-lg font-bold font-mono">
                      ${(gst?.total_revenue_cad ?? 0).toLocaleString("en-CA", { minimumFractionDigits: 2 })}
                    </p>
                    <p className="text-xs text-text-muted">
                      of ${(gst?.threshold_cad ?? 30000).toLocaleString()} threshold
                    </p>
                  </div>
                  {gst?.must_register ? (
                    <div className="flex items-center gap-1.5 text-gold">
                      <AlertTriangle className="h-4 w-4" />
                      <span className="text-sm font-medium">Registration required</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 text-text-muted">
                      <CheckCircle className="h-4 w-4" />
                      <span className="text-sm">Exempt</span>
                    </div>
                  )}
                </div>
                {/* Progress bar */}
                <div className="mt-3 h-2 rounded-full bg-background overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all ${gst?.must_register ? "bg-gold" : "bg-emerald"}`}
                    style={{ width: `${Math.min(((gst?.total_revenue_cad ?? 0) / (gst?.threshold_cad ?? 30000)) * 100, 100)}%` }}
                  />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Earnings Chart */}
          <Card>
            <CardHeader>
              <CardTitle>{t("dash.earnings.chart_title")}</CardTitle>
              <CardDescription>{t("dash.earnings.chart_desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {chartData.length === 0 ? (
                <p className="text-sm text-text-muted py-8 text-center">
                  {t("dash.earnings.no_earnings")}
                </p>
              ) : (
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%" minHeight={1}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                      <XAxis dataKey="date" tick={{ fill: "#94a3b8", fontSize: 11 }} stroke="#475569" />
                      <YAxis
                        tick={{ fill: "#94a3b8", fontSize: 11 }}
                        stroke="#475569"
                        tickFormatter={(v) => `$${v}`}
                      />
                      <Tooltip
                        contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: "8px", color: "#e2e8f0" }}
                        formatter={(value) => [`$${Number(value).toFixed(2)} CAD`, "Earned"]}
                      />
                      <Bar dataKey="amount" fill="#10b981" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Payout History */}
          <Card>
            <CardHeader>
              <CardTitle>{t("dash.earnings.payouts")}</CardTitle>
              <CardDescription>{t("dash.earnings.payouts_desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {payouts.length === 0 ? (
                <p className="text-sm text-text-muted">{t("dash.earnings.no_payouts")}</p>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-left text-text-muted">
                        <th className="pb-2 pr-4 font-medium">Job</th>
                        <th className="pb-2 pr-4 font-medium">Total</th>
                        <th className="pb-2 pr-4 font-medium">Your Share</th>
                        <th className="pb-2 pr-4 font-medium">Platform</th>
                        <th className="pb-2 pr-4 font-medium">Tax</th>
                        <th className="pb-2 font-medium">Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {payouts.map((p) => (
                        <tr key={`${p.job_id}-${p.created_at}`} className="border-b border-border/50">
                          <td className="py-2.5 pr-4 font-mono text-xs">{p.job_id.slice(0, 12)}…</td>
                          <td className="py-2.5 pr-4 font-mono">${p.total_cad.toFixed(2)}</td>
                          <td className="py-2.5 pr-4 font-mono text-emerald">${p.provider_share_cad.toFixed(2)}</td>
                          <td className="py-2.5 pr-4 font-mono text-text-muted">${p.platform_share_cad.toFixed(2)}</td>
                          <td className="py-2.5 pr-4 font-mono text-text-muted">${p.gst_hst_cad.toFixed(2)}</td>
                          <td className="py-2.5 text-text-muted">{new Date(p.created_at).toLocaleDateString()}</td>
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
