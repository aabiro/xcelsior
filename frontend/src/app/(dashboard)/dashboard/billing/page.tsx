"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { DepositModal } from "@/components/billing/deposit-modal";
import { CryptoDepositModal } from "@/components/billing/crypto-deposit-modal";
import {
  CreditCard, DollarSign, RefreshCw, Download, Plus, FileText,
  ArrowUpRight, ArrowDownRight, Leaf, Clock, Zap, Receipt, Loader2,
  Bitcoin, AlertTriangle,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { Wallet, WalletTransaction, Invoice, PricingReference } from "@/lib/api";
import { toast } from "sonner";

export default function BillingPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const customerId = user?.customer_id || user?.user_id || "";

  const [wallet, setWallet] = useState<Wallet | null>(null);
  const [transactions, setTransactions] = useState<WalletTransaction[]>([]);
  const [invoices, setInvoices] = useState<Invoice[]>([]);
  const [usage, setUsage] = useState<{
    job_count: number; total_gpu_hours: number; total_cost_cad: number;
    canadian_compute_cad: number; hosts_used: number;
  } | null>(null);
  const [reservedTiers, setReservedTiers] = useState<Record<string, {
    commitment: string; discount_pct: number; description: string;
    min_hours_per_day: number; sample_hourly_rates_cad: Record<string, number>;
  }> | null>(null);
  const [loading, setLoading] = useState(true);
  const [showDeposit, setShowDeposit] = useState(false);
  const [showCryptoDeposit, setShowCryptoDeposit] = useState(false);
  const [btcEnabled, setBtcEnabled] = useState(false);
  const [cafLoading, setCafLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [subscribing, setSubscribing] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!customerId) return;
    setLoading(true);
    try {
      const [walletRes, histRes, invoiceRes, usageRes, plansRes] = await Promise.allSettled([
        api.fetchWallet(customerId),
        api.fetchWalletHistory(customerId),
        api.fetchInvoices(customerId),
        api.fetchUsageSummary(customerId),
        api.fetchReservedPlans(),
      ]);
      if (walletRes.status === "fulfilled") setWallet(walletRes.value.wallet);
      if (histRes.status === "fulfilled") setTransactions(histRes.value.transactions || []);
      if (invoiceRes.status === "fulfilled") setInvoices(invoiceRes.value.invoices || []);
      if (usageRes.status === "fulfilled") {
        const u = usageRes.value;
        setUsage({ job_count: u.job_count, total_gpu_hours: u.total_gpu_hours, total_cost_cad: u.total_cost_cad, canadian_compute_cad: u.canadian_compute_cad, hosts_used: u.hosts_used });
      }
      if (plansRes.status === "fulfilled" && (plansRes.value as Record<string, unknown>).reserved_tiers) {
        setReservedTiers((plansRes.value as Record<string, unknown>).reserved_tiers as typeof reservedTiers);
      }
    } catch {
      toast.error("Failed to load billing data");
    } finally {
      setLoading(false);
    }
  }, [customerId]);

  useEffect(() => { load(); }, [load]);

  // Check if BTC deposits are enabled
  useEffect(() => {
    api.checkCryptoEnabled().then((r) => setBtcEnabled(r.enabled)).catch((e) => console.error("Failed to check crypto status", e));
  }, []);

  // CSV export from billing records
  const handleCsvExport = async () => {
    if (!customerId) return;
    setCsvLoading(true);
    try {
      const now = Math.floor(Date.now() / 1000);
      const thirtyDaysAgo = now - 30 * 86400;
      const blob = await api.downloadInvoice(customerId, "csv", thirtyDaysAgo, now);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `xcelsior-billing-${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
      toast.success("CSV downloaded");
    } catch {
      toast.error("CSV export failed");
    } finally {
      setCsvLoading(false);
    }
  };

  // CAF rebate export
  const handleCafExport = async () => {
    if (!customerId) return;
    setCafLoading(true);
    try {
      const now = Math.floor(Date.now() / 1000);
      const ninetyDaysAgo = now - 90 * 86400;
      const blob = await api.exportCaf(customerId, ninetyDaysAgo, now, "csv");
      if (blob instanceof Blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `xcelsior-caf-rebate-${new Date().toISOString().slice(0, 10)}.csv`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success("CAF report downloaded");
      }
    } catch {
      toast.error("CAF export failed");
    } finally {
      setCafLoading(false);
    }
  };

  // Invoice download
  const handleInvoiceDownload = async (inv: Invoice) => {
    if (!customerId) return;
    try {
      const blob = await api.downloadInvoice(
        customerId, "csv",
        Math.floor(new Date(inv.period_start).getTime() / 1000),
        Math.floor(new Date(inv.period_end).getTime() / 1000),
        inv.tax_rate,
      );
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `invoice-${inv.invoice_id}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      toast.error("Invoice download failed");
    }
  };

  // Reserved plan subscription
  const handleSubscribe = async (commitmentType: string, gpuModel: string) => {
    if (!customerId) return;
    setSubscribing(commitmentType);
    try {
      await api.createReservation({
        customer_id: customerId,
        gpu_model: gpuModel,
        commitment_type: commitmentType as "1_month" | "3_month" | "1_year",
      });
      toast.success(`Reserved ${commitmentType.replace("_", " ")} plan activated`);
      load();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Subscription failed");
    } finally {
      setSubscribing(null);
    }
  };

  const tierMeta: Record<string, { label: string; accent: string; icon: typeof Clock }> = {
    "1_month": { label: "1 Month", accent: "text-ice", icon: Clock },
    "3_month": { label: "3 Months", accent: "text-gold", icon: Zap },
    "1_year":  { label: "1 Year", accent: "text-emerald", icon: Leaf },
  };

  const isTestMode = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY?.startsWith("pk_test");

  return (
    <div className="space-y-6">
      {/* Test Mode Banner */}
      {isTestMode && (
        <div className="flex items-center gap-3 rounded-lg border border-gold/30 bg-gold/5 px-4 py-3">
          <AlertTriangle className="h-4 w-4 text-gold shrink-0" />
          <p className="text-sm text-gold">
            <span className="font-medium">Test Mode</span> — Payments are in sandbox mode. No real charges will be processed. We&apos;re working on enabling live payments shortly.
          </p>
        </div>
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">{t("dash.billing.title")}</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleCsvExport} disabled={csvLoading}>
            {csvLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
            {t("dash.billing.export_csv")}
          </Button>
          <Button variant="outline" size="sm" onClick={load}>
            <RefreshCw className="h-3.5 w-3.5" /> {t("common.refresh")}
          </Button>
        </div>
      </div>

      {loading ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-4">
          {[...Array(4)].map((_, i) => (
            <div key={i} className="h-28 rounded-xl bg-surface skeleton-pulse" />
          ))}
        </div>
      ) : (
        <>
          {/* Stat Cards */}
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard
              label={t("dash.billing.wallet_balance")}
              value={`$${(wallet?.balance_cad ?? 0).toFixed(2)}`}
              icon={DollarSign}
            />
            <StatCard
              label={t("dash.billing.total_spent")}
              value={`$${(usage?.total_cost_cad ?? 0).toFixed(2)}`}
              icon={CreditCard}
            />
            <StatCard
              label={t("dash.billing.gpu_hours")}
              value={(usage?.total_gpu_hours ?? 0).toFixed(1)}
              icon={Clock}
            />
            <StatCard
              label={t("dash.billing.jobs_run")}
              value={usage?.job_count ?? 0}
              icon={Zap}
            />
          </div>

          {/* Add Credits + CAF Banner */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card>
              <CardContent className="flex items-center justify-between p-5">
                <div>
                  <p className="font-medium">{t("dash.billing.wallet_credits")}</p>
                  <p className="text-2xl font-bold font-mono text-emerald">
                    ${(wallet?.balance_cad ?? 0).toFixed(2)} <span className="text-sm font-normal text-text-muted">CAD</span>
                  </p>
                </div>
                <Button variant="success" onClick={() => setShowDeposit(true)}>
                  <Plus className="h-4 w-4" /> {t("dash.billing.add_credits")}
                </Button>
              </CardContent>
            </Card>

            <Card className="border-gold/20 bg-gold/5">
              <CardContent className="flex items-center justify-between p-5">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Leaf className="h-4 w-4 text-gold" />
                    <p className="font-medium text-gold">{t("dash.billing.fund_title")}</p>
                  </div>
                  <p className="text-xs text-text-secondary">
                    {t("dash.billing.fund_desc")}
                  </p>
                </div>
                <Button variant="gold" size="sm" onClick={handleCafExport} disabled={cafLoading}>
                  {cafLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                  {t("dash.billing.export_caf")}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Bitcoin Deposit */}
          {btcEnabled && (
            <Card className="border-amber-500/20 bg-amber-500/5">
              <CardContent className="flex items-center justify-between p-5">
                <div>
                  <div className="flex items-center gap-2 mb-1">
                    <Bitcoin className="h-4 w-4 text-amber-500" />
                    <p className="font-medium text-amber-500">Bitcoin Deposits</p>
                  </div>
                  <p className="text-xs text-text-secondary">
                    Pay with BTC — zero processing fees, settled in CAD
                  </p>
                </div>
                <Button
                  size="sm"
                  className="bg-amber-500 hover:bg-amber-600 text-black"
                  onClick={() => setShowCryptoDeposit(true)}
                >
                  <Bitcoin className="h-3.5 w-3.5" /> Deposit BTC
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Reserved Plans */}
          {reservedTiers && (
            <div>
              <h2 className="text-lg font-semibold mb-3">{t("dash.billing.reserved_plans")}</h2>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                {Object.entries(reservedTiers).map(([key, tier]) => {
                  const meta = tierMeta[key] || { label: key, accent: "text-text-primary", icon: Clock };
                  const Icon = meta.icon;
                  const sample = Object.entries(tier.sample_hourly_rates_cad || {})[0];
                  return (
                    <Card key={key} className="relative overflow-hidden">
                      <CardHeader className="pb-2">
                        <div className="flex items-center gap-2">
                          <Icon className={`h-4 w-4 ${meta.accent}`} />
                          <CardTitle className="text-base">{meta.label}</CardTitle>
                        </div>
                        <CardDescription>{tier.description}</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        <div className="flex items-baseline gap-1">
                          <span className={`text-3xl font-bold font-mono ${meta.accent}`}>
                            {tier.discount_pct}%
                          </span>
                          <span className="text-sm text-text-muted">discount</span>
                        </div>
                        {sample && (
                          <p className="text-xs text-text-secondary">
                            e.g. {sample[0]}: <span className="font-mono">${sample[1].toFixed(2)}/hr</span>
                          </p>
                        )}
                        {tier.min_hours_per_day > 0 && (
                          <p className="text-xs text-text-muted">
                            Min {tier.min_hours_per_day}h/day commitment
                          </p>
                        )}
                        <Button
                          variant="outline"
                          size="sm"
                          className="w-full"
                          disabled={subscribing === key}
                          onClick={() => handleSubscribe(key, sample?.[0] || "RTX 4090")}
                        >
                          {subscribing === key ? (
                            <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Subscribing…</>
                          ) : (
                            t("dash.billing.subscribe")
                          )}
                        </Button>
                      </CardContent>
                    </Card>
                  );
                })}
              </div>
            </div>
          )}

          {/* Invoices */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle>{t("dash.billing.invoices")}</CardTitle>
                <CardDescription>{t("dash.billing.invoices_desc")}</CardDescription>
              </div>
              <Receipt className="h-4 w-4 text-text-muted" />
            </CardHeader>
            <CardContent>
              {invoices.length === 0 ? (
                <p className="text-sm text-text-muted">{t("dash.billing.no_invoices")}</p>
              ) : (
                <div className="space-y-2">
                  {invoices.map((inv) => (
                    <div
                      key={inv.invoice_id}
                      className="flex items-center justify-between rounded-lg border border-border p-3"
                    >
                      <div className="flex items-center gap-3">
                        <FileText className="h-4 w-4 text-text-muted" />
                        <div>
                          <p className="text-sm font-medium">
                            {new Date(inv.period_start).toLocaleDateString("en-CA", { year: "numeric", month: "short" })}
                            {" – "}
                            {new Date(inv.period_end).toLocaleDateString("en-CA", { year: "numeric", month: "short" })}
                          </p>
                          <p className="text-xs text-text-muted">
                            {inv.line_items} item{inv.line_items !== 1 ? "s" : ""} · Tax {(inv.tax_rate * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="text-right">
                          <p className="text-sm font-mono font-medium">${inv.total_cad.toFixed(2)}</p>
                          <Badge variant={inv.status === "paid" ? "active" : inv.status === "void" ? "dead" : "info"}>
                            {inv.status}
                          </Badge>
                        </div>
                        <Button variant="ghost" size="icon" onClick={() => handleInvoiceDownload(inv)}>
                          <Download className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Transaction History */}
          <Card>
            <CardHeader>
                <CardTitle>{t("dash.billing.transactions")}</CardTitle>
              <CardDescription>{t("dash.billing.transactions_desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              {transactions.length === 0 ? (
                <p className="text-sm text-text-muted">{t("dash.billing.no_transactions")}</p>
              ) : (
                <div className="space-y-2">
                  {transactions.map((tx) => {
                    const isCredit = tx.amount_cad > 0;
                    return (
                      <div
                        key={tx.tx_id}
                        className="flex items-center justify-between rounded-lg border border-border p-3"
                      >
                        <div className="flex items-center gap-3">
                          {isCredit ? (
                            <ArrowDownRight className="h-4 w-4 text-emerald" />
                          ) : (
                            <ArrowUpRight className="h-4 w-4 text-accent-red" />
                          )}
                          <div>
                            <p className="text-sm font-medium">
                              {tx.description || tx.type || "Transaction"}
                            </p>
                            <p className="text-xs text-text-muted">
                              {tx.created_at ? new Date(tx.created_at).toLocaleString() : "—"}
                              {tx.job_id && <span className="ml-2">· Job {tx.job_id.slice(0, 8)}</span>}
                            </p>
                          </div>
                        </div>
                        <span className={`font-mono text-sm font-medium ${isCredit ? "text-emerald" : "text-accent-red"}`}>
                          {isCredit ? "+" : ""}${tx.amount_cad.toFixed(2)}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </>
      )}

      {/* Deposit Modal */}
      {showDeposit && customerId && (
        <DepositModal
          customerId={customerId}
          onClose={() => setShowDeposit(false)}
          onSuccess={(newBalance) => {
            setWallet((w) => w ? { ...w, balance_cad: newBalance } : w);
            setShowDeposit(false);
            load();
          }}
        />
      )}

      {/* Crypto Deposit Modal */}
      {showCryptoDeposit && customerId && (
        <CryptoDepositModal
          customerId={customerId}
          onClose={() => setShowCryptoDeposit(false)}
          onSuccess={(newBalance) => {
            setWallet((w) => w ? { ...w, balance_cad: newBalance } : w);
            setShowCryptoDeposit(false);
            load();
          }}
        />
      )}
    </div>
  );
}
