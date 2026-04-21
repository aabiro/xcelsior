"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { StatCard } from "@/components/ui/stat-card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { DepositModal } from "@/components/billing/deposit-modal";
import { CryptoDepositModal } from "@/components/billing/crypto-deposit-modal";
import { LightningDepositModal } from "@/components/billing/lightning-deposit-modal";
import {
  CreditCard, DollarSign, RefreshCw, Download, Plus, FileText,
  ArrowUpRight, ArrowDownRight, HardDrive, Leaf, Clock, Zap, Receipt, Loader2,
  Bitcoin, Activity, Gift, Sparkles, CheckCircle2, RotateCcw,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { Wallet, WalletTransaction, Invoice } from "@/lib/api";
import { toast } from "sonner";
import { AnimatePresence, animate, motion, type AnimationPlaybackControls } from "framer-motion";

const FREE_CREDIT_AMOUNT = 10;
const FREE_CREDIT_TRANSFER_DURATION_S = 2.2;
const FREE_CREDIT_TRANSFER_DURATION_MS = FREE_CREDIT_TRANSFER_DURATION_S * 1000;
const FREE_CREDIT_SUCCESS_HOLD_MS = 1400;
const DEFAULT_BTC_STATUS = {
  enabled: false,
  available: false,
  reason: "",
};
const DEFAULT_LIGHTNING_STATUS = {
  enabled: false,
  available: false,
  reason: "",
  node_alias: "",
  num_active_channels: 0,
};

type FreeCreditAnimationState = "idle" | "transferring" | "complete";

interface CreditTransfer {
  amount: number;
  from: number;
  to: number;
}

function formatCad(amount: number) {
  return `$${amount.toFixed(2)}`;
}

export default function BillingPage() {
  const { user } = useAuth();
  const { t } = useLocale();
  const customerId = user?.customer_id || user?.user_id || "";
  const isPlatformAdmin = !!user?.is_admin || user?.role === "admin";

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
  const searchParams = useSearchParams();

  // Auto-open deposit modal from ?topup=true (Credits button link)
  useEffect(() => {
    if (searchParams.get("topup") === "true") setShowDeposit(true);
  }, [searchParams]);

  const [showCryptoDeposit, setShowCryptoDeposit] = useState(false);
  const [showLightningDeposit, setShowLightningDeposit] = useState(false);
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [btcStatus, setBtcStatus] = useState(DEFAULT_BTC_STATUS);
  const [lnStatus, setLnStatus] = useState(DEFAULT_LIGHTNING_STATUS);
  const [paymentRailsLoading, setPaymentRailsLoading] = useState(true);
  const [cafLoading, setCafLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [subscribing, setSubscribing] = useState<string | null>(null);
  const [resettingWallet, setResettingWallet] = useState(false);
  const [freeCreditsAvailable, setFreeCreditsAvailable] = useState(false);
  const [claimingCredits, setClaimingCredits] = useState(false);
  const [freeCreditAnimationState, setFreeCreditAnimationState] = useState<FreeCreditAnimationState>("idle");
  const [displayWalletBalance, setDisplayWalletBalance] = useState(0);
  const [creditTransfer, setCreditTransfer] = useState<CreditTransfer | null>(null);

  const walletValueRef = useRef<HTMLSpanElement | null>(null);
  const walletAnimationRef = useRef<AnimationPlaybackControls | null>(null);
  const freeCreditTimersRef = useRef<number[]>([]);
  const hasCryptoPaymentRails = true; // Always show crypto deposit options

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

  const clearFreeCreditTimers = useCallback(() => {
    for (const timerId of freeCreditTimersRef.current) {
      window.clearTimeout(timerId);
    }
    freeCreditTimersRef.current = [];
  }, []);

  const stopWalletAnimation = useCallback(() => {
    walletAnimationRef.current?.stop();
    walletAnimationRef.current = null;
  }, []);

  const animateWalletBalance = useCallback((from: number, to: number) => {
    stopWalletAnimation();
    setDisplayWalletBalance(from);
    walletAnimationRef.current = animate(from, to, {
      duration: FREE_CREDIT_TRANSFER_DURATION_S,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (value) => setDisplayWalletBalance(Number(value.toFixed(2))),
      onComplete: () => {
        setDisplayWalletBalance(to);
        walletAnimationRef.current = null;
      },
    });
  }, [stopWalletAnimation]);

  useEffect(() => { load(); }, [load]);

  // Check crypto payment rails together so the funding section does not pop in later.
  useEffect(() => {
    let active = true;
    const btcCtrl = new AbortController();
    const lnCtrl = new AbortController();
    const btcTimer = window.setTimeout(() => btcCtrl.abort(), 4000);
    const lnTimer = window.setTimeout(() => lnCtrl.abort(), 4000);

    setPaymentRailsLoading(true);

    Promise.allSettled([
      api.checkCryptoEnabled({ signal: btcCtrl.signal }),
      api.checkLightningEnabled({ signal: lnCtrl.signal }),
    ])
      .then(([btcRes, lnRes]) => {
        if (!active) return;

        if (btcRes.status === "fulfilled") {
          const nextBtcStatus = {
            enabled: btcRes.value.enabled,
            available: btcRes.value.available ?? btcRes.value.enabled,
            reason: btcRes.value.reason ?? "",
          };
          if (nextBtcStatus.reason && !nextBtcStatus.available) {
            console.warn("Bitcoin deposits unavailable", nextBtcStatus.reason);
          }
          setBtcStatus(nextBtcStatus);
        } else {
          if (!(btcRes.reason instanceof DOMException && btcRes.reason.name === "AbortError")) {
            console.warn("Failed to check crypto status", btcRes.reason);
          }
          setBtcStatus(DEFAULT_BTC_STATUS);
        }

        if (lnRes.status === "fulfilled") {
          setLnStatus({
            enabled: lnRes.value.enabled,
            available: lnRes.value.available ?? lnRes.value.enabled,
            reason: lnRes.value.reason ?? "",
            node_alias: lnRes.value.node_alias ?? "",
            num_active_channels: lnRes.value.num_active_channels ?? 0,
          });
        } else {
          setLnStatus(DEFAULT_LIGHTNING_STATUS);
        }
      })
      .finally(() => {
        window.clearTimeout(btcTimer);
        window.clearTimeout(lnTimer);
        if (active) setPaymentRailsLoading(false);
      });

    return () => {
      active = false;
      window.clearTimeout(btcTimer);
      window.clearTimeout(lnTimer);
      btcCtrl.abort();
      lnCtrl.abort();
    };
  }, []);

  // Check if free signup credits are available
  useEffect(() => {
    if (!customerId) return;
    api.checkFreeCreditsStatus(customerId)
      .then((r) => setFreeCreditsAvailable(!r.claimed))
      .catch(() => setFreeCreditsAvailable(false));
  }, [customerId]);

  useEffect(() => {
    if (freeCreditAnimationState !== "idle") return;
    setDisplayWalletBalance(wallet?.balance_cad ?? 0);
  }, [wallet?.balance_cad, freeCreditAnimationState]);

  useEffect(() => {
    if (freeCreditAnimationState !== "transferring" || !creditTransfer || !customerId) return;

    clearFreeCreditTimers();

    animateWalletBalance(creditTransfer.from, creditTransfer.to);

    freeCreditTimersRef.current.push(window.setTimeout(() => {
      setFreeCreditAnimationState("complete");
    }, FREE_CREDIT_TRANSFER_DURATION_MS));

    freeCreditTimersRef.current.push(window.setTimeout(() => {
      setFreeCreditsAvailable(false);
      setFreeCreditAnimationState("idle");
      setCreditTransfer(null);
      void api.fetchWalletHistory(customerId)
        .then((r) => setTransactions(r.transactions || []))
        .catch(() => {});
    }, FREE_CREDIT_TRANSFER_DURATION_MS + FREE_CREDIT_SUCCESS_HOLD_MS));
  }, [
    animateWalletBalance,
    clearFreeCreditTimers,
    creditTransfer,
    customerId,
    freeCreditAnimationState,
  ]);

  useEffect(() => () => {
    clearFreeCreditTimers();
    stopWalletAnimation();
  }, [clearFreeCreditTimers, stopWalletAnimation]);

  const handleClaimFreeCredits = async () => {
    if (!customerId || claimingCredits || freeCreditAnimationState !== "idle") return;
    setClaimingCredits(true);
    try {
      const result = await api.claimFreeCredits(customerId);
      if (result.already_claimed) {
        toast.info(t("dash.billing.credits_already_claimed"));
        setFreeCreditsAvailable(false);
      } else {
        const nextBalance = result.balance_cad;
        const previousBalance = wallet?.balance_cad ?? displayWalletBalance;
        clearFreeCreditTimers();
        setCreditTransfer({
          amount: result.amount_cad || FREE_CREDIT_AMOUNT,
          from: previousBalance,
          to: nextBalance,
        });
        setFreeCreditAnimationState("transferring");
        setWallet((w) => (
          w
            ? { ...w, balance_cad: nextBalance }
            : { customer_id: customerId, balance_cad: nextBalance, currency: "CAD" }
        ));
        toast.success(t("dash.billing.credits_claimed_toast"));
      }
    } catch {
      toast.error("Failed to claim free credits");
    } finally {
      setClaimingCredits(false);
    }
  };

  const handleResetWalletTestingState = async () => {
    if (!customerId || !isPlatformAdmin || resettingWallet) return;
    setResettingWallet(true);
    try {
      const result = await api.resetWalletTestingState(customerId);
      clearFreeCreditTimers();
      stopWalletAnimation();
      setClaimingCredits(false);
      setCreditTransfer(null);
      setFreeCreditAnimationState("idle");
      setWallet(result.wallet);
      setTransactions([]);
      setDisplayWalletBalance(result.wallet.balance_cad ?? 0);
      setFreeCreditsAvailable(result.promo_available);
      setShowResetConfirm(false);
      toast.success(t("dash.billing.admin_reset_success"));
    } catch {
      toast.error(t("dash.billing.admin_reset_failed"));
    } finally {
      setResettingWallet(false);
    }
  };

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

  const handleCafPrintable = async () => {
    if (!customerId) return;
    try {
      const now = Math.floor(Date.now() / 1000);
      const ninetyDaysAgo = now - 90 * 86400;
      const blob = await api.exportCaf(customerId, ninetyDaysAgo, now, "pdf");
      if (blob instanceof Blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `xcelsior-caf-${new Date().toISOString().slice(0, 10)}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
        toast.success("CAF claim form downloaded");
      }
    } catch {
      toast.error("Could not generate CAF claim form");
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

  const currentWalletBalance = displayWalletBalance;
  const freeCreditFlowActive = claimingCredits || freeCreditAnimationState !== "idle";

  return (
    <div className="space-y-6">
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
              value={(
                <span ref={walletValueRef} className="inline-block">
                  <motion.span
                    animate={freeCreditAnimationState === "transferring" ? { scale: [1, 1.08, 1] } : { scale: 1 }}
                    transition={{
                      duration: freeCreditAnimationState === "transferring" ? FREE_CREDIT_TRANSFER_DURATION_S : 0.2,
                      times: freeCreditAnimationState === "transferring" ? [0, 0.35, 1] : undefined,
                      ease: "easeOut",
                    }}
                    className="inline-block"
                  >
                    {formatCad(currentWalletBalance)}
                  </motion.span>
                </span>
              )}
              icon={DollarSign}
              glow="cyan"
            />
            <StatCard
              label={t("dash.billing.total_spent")}
              value={`$${(usage?.total_cost_cad ?? 0).toFixed(2)}`}
              icon={CreditCard}
              glow="violet"
            />
            <StatCard
              label={t("dash.billing.gpu_hours")}
              value={(usage?.total_gpu_hours ?? 0).toFixed(1)}
              icon={Clock}
              glow="emerald"
            />
            <StatCard
              label={t("dash.billing.jobs_run")}
              value={usage?.job_count ?? 0}
              icon={Zap}
              glow="gold"
            />
          </div>

          {/* Welcome $10 Free Credits Banner */}
          <AnimatePresence>
            {freeCreditsAvailable && (
              <motion.div
                initial={{ opacity: 0, y: -8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8, height: 0, marginTop: 0 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="relative overflow-hidden border-accent-cyan/30 bg-gradient-to-r from-accent-cyan/10 via-accent-violet/10 to-accent-cyan/10">
                  {/* Animated shimmer */}
                  <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent animate-shimmer" />
                  <CardContent className="relative flex flex-col gap-4 p-5 lg:flex-row lg:items-center lg:justify-between">
                    <div className="flex items-center gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-accent-cyan/20">
                        {freeCreditAnimationState === "complete" ? (
                          <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ type: "spring", stiffness: 300, damping: 15 }}
                          >
                            <CheckCircle2 className="h-6 w-6 text-emerald" />
                          </motion.div>
                        ) : (
                          <motion.div
                            animate={freeCreditAnimationState === "transferring" ? { rotate: [0, -6, 6, 0], scale: [1, 1.06, 1] } : { rotate: 0, scale: 1 }}
                            transition={{
                              duration: freeCreditAnimationState === "transferring" ? 1.1 : 0.2,
                              repeat: freeCreditAnimationState === "transferring" ? Infinity : 0,
                              ease: "easeInOut",
                            }}
                          >
                            <Gift className="h-6 w-6 text-accent-cyan" />
                          </motion.div>
                        )}
                      </div>
                      <div>
                        {freeCreditAnimationState === "complete" ? (
                          <motion.div
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                          >
                            <p className="font-semibold text-emerald">{t("dash.billing.credits_claimed_title")}</p>
                            <p className="text-sm text-text-secondary">{t("dash.billing.credits_claimed_desc")}</p>
                          </motion.div>
                        ) : freeCreditAnimationState === "transferring" ? (
                          <motion.div
                            initial={{ opacity: 0, y: 4 }}
                            animate={{ opacity: 1, y: 0 }}
                          >
                            <p className="font-semibold text-accent-cyan">{t("dash.billing.credits_transferring_title")}</p>
                            <p className="text-sm text-text-secondary">{t("dash.billing.credits_transferring_desc")}</p>
                          </motion.div>
                        ) : (
                          <>
                            <p className="font-semibold">{t("dash.billing.free_credits_title")}</p>
                            <p className="text-sm text-text-secondary">{t("dash.billing.free_credits_desc")}</p>
                          </>
                        )}
                      </div>
                    </div>
                    <div className="flex min-h-10 items-center justify-end">
                      {freeCreditAnimationState === "transferring" ? (
                        <motion.div
                          initial={{ opacity: 1, scale: 1 }}
                          animate={{ opacity: 0, scale: 0.8 }}
                          transition={{ duration: 0.3, ease: "easeOut" }}
                          className="rounded-full border border-emerald/30 bg-emerald/10 px-3 py-1 text-sm font-semibold font-mono text-emerald shadow-lg shadow-emerald/15"
                        >
                          +{formatCad(creditTransfer?.amount ?? FREE_CREDIT_AMOUNT)}
                        </motion.div>
                      ) : freeCreditAnimationState === "complete" ? (
                        <motion.div
                          initial={{ opacity: 0, scale: 0.8, y: 8 }}
                          animate={{ opacity: 1, scale: 1, y: 0 }}
                          className="inline-flex items-center gap-2 rounded-full border border-emerald/30 bg-emerald/10 px-3 py-1 text-sm font-semibold text-emerald"
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          {t("dash.billing.credits_added_badge")}
                        </motion.div>
                      ) : (
                        <Button
                          className="bg-accent-cyan text-navy font-semibold shadow-lg shadow-accent-cyan/20 hover:bg-accent-cyan/80"
                          onClick={handleClaimFreeCredits}
                          disabled={freeCreditFlowActive}
                        >
                          {claimingCredits ? (
                            <><Loader2 className="h-4 w-4 animate-spin" /> {t("dash.billing.claiming")}</>
                          ) : (
                            <><Sparkles className="h-4 w-4" /> {t("dash.billing.claim_credits")}</>
                          )}
                        </Button>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Add Credits + CAF Banner */}
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* Burn Rate / Depletion Estimate */}
            <Card className="border-ice/20">
              <CardContent className="p-5 space-y-3">
                <div className="flex items-center gap-2 mb-1">
                  <Activity className="h-4 w-4 text-ice" />
                  <p className="font-medium">{t("dash.billing.burn_rate")}</p>
                </div>
                {(() => {
                  const balance = currentWalletBalance;
                  const spent = usage?.total_cost_cad ?? 0;
                  const hours = usage?.total_gpu_hours ?? 0;
                  const burnPerHour = hours > 0 ? spent / hours : 0;
                  const daysRemaining = burnPerHour > 0 ? balance / (burnPerHour * 24) : Infinity;
                  const depletionDate = isFinite(daysRemaining)
                    ? new Date(Date.now() + daysRemaining * 86400_000)
                    : null;
                  return (
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-text-muted">Avg cost/GPU-hour</p>
                        <p className="text-lg font-bold font-mono text-ice">
                          ${burnPerHour.toFixed(2)}<span className="text-xs font-normal text-text-muted">/hr</span>
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-text-muted">Est. days remaining</p>
                        <p className={`text-lg font-bold font-mono ${daysRemaining < 7 ? "text-accent-red" : daysRemaining < 30 ? "text-gold" : "text-emerald"}`}>
                          {isFinite(daysRemaining) ? `${Math.floor(daysRemaining)}d` : "∞"}
                        </p>
                      </div>
                      {depletionDate && (
                        <div className="col-span-2">
                          <p className="text-xs text-text-muted">
                            Balance depletes ~{depletionDate.toLocaleDateString("en-CA", { month: "short", day: "numeric", year: "numeric" })}
                          </p>
                        </div>
                      )}
                    </div>
                  );
                })()}
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-5 space-y-4">
                <div className="flex items-center justify-between gap-4">
                  <div>
                    <p className="font-medium">{t("dash.billing.wallet_credits")}</p>
                    <p className="text-2xl font-bold font-mono text-emerald">
                      {formatCad(currentWalletBalance)} <span className="text-sm font-normal text-text-muted">CAD</span>
                    </p>
                  </div>
                  <Button variant="success" onClick={() => setShowDeposit(true)}>
                    <Plus className="h-4 w-4" /> {t("dash.billing.add_credits")}
                  </Button>
                </div>
                {isPlatformAdmin && (
                  <div className="flex flex-col gap-3 rounded-lg border border-accent-red/20 bg-accent-red/5 p-3 sm:flex-row sm:items-center sm:justify-between">
                    <div>
                      <p className="text-sm font-medium text-accent-red">{t("dash.billing.admin_reset_title")}</p>
                      <p className="text-xs text-text-secondary">{t("dash.billing.admin_reset_desc")}</p>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      className="border-accent-red/30 text-accent-red hover:bg-accent-red/10"
                      onClick={() => setShowResetConfirm(true)}
                      disabled={resettingWallet || freeCreditFlowActive}
                    >
                      {resettingWallet ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
                      {t("dash.billing.admin_reset_action")}
                    </Button>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
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
                <div className="flex items-center gap-2">
                  <Button variant="gold" size="sm" onClick={handleCafPrintable}>
                    <FileText className="h-3.5 w-3.5" />
                    Claim Form (PDF)
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleCafExport} disabled={cafLoading}>
                    {cafLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
                    CSV
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Storage Costs */}
            {(() => {
              const storageTxs = transactions.filter((tx) =>
                tx.description?.toLowerCase().includes("storage") || tx.description?.toLowerCase().includes("volume")
              );
              const storageCost = storageTxs.reduce((sum, tx) => sum + Math.abs(tx.amount_cad), 0);
              if (storageCost <= 0) return null;
              return (
                <Card className="border-ice-blue/20">
                  <CardContent className="p-5 space-y-3">
                    <div className="flex items-center gap-2 mb-1">
                      <HardDrive className="h-4 w-4 text-ice-blue" />
                      <p className="font-medium">Volume Storage</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-text-muted">Total storage cost</p>
                        <p className="text-lg font-bold font-mono text-ice-blue">{formatCad(storageCost)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-text-muted">Billing cycles</p>
                        <p className="text-lg font-bold font-mono">{storageTxs.length}</p>
                      </div>
                    </div>
                    <p className="text-[10px] text-text-muted/60">Persistent volumes billed per GB/month from your credits</p>
                  </CardContent>
                </Card>
              );
            })()}
          </div>

          {hasCryptoPaymentRails && (
            <Card>
              <CardHeader>
                <CardTitle>Crypto Deposits</CardTitle>
                <CardDescription>Fund your wallet with Bitcoin on-chain or Lightning Network. Zero processing fees, settled in CAD.</CardDescription>
              </CardHeader>
              <CardContent className="grid gap-4 lg:grid-cols-2">
                {paymentRailsLoading && (
                  <>
                    {[0, 1].map((index) => (
                      <div
                        key={index}
                        className="rounded-2xl border border-border bg-surface/40 p-5"
                      >
                        <div className="mb-4 h-4 w-32 animate-pulse rounded bg-border/60" />
                        <div className="mb-2 h-3 w-full animate-pulse rounded bg-border/40" />
                        <div className="h-3 w-2/3 animate-pulse rounded bg-border/30" />
                        <div className="mt-5 h-9 w-32 animate-pulse rounded-xl bg-border/40" />
                      </div>
                    ))}
                  </>
                )}

                {!paymentRailsLoading && btcStatus.enabled && (
                  <div className="rounded-2xl border border-amber-500/20 bg-amber-500/5 p-5">
                    <div className="flex items-center gap-2 mb-1">
                      <Bitcoin className="h-4 w-4 text-amber-500" />
                      <p className="font-medium text-amber-500">Bitcoin Deposits</p>
                      {!btcStatus.available && (
                        <Badge variant="info" className="border border-amber-500/30 bg-amber-500/10 text-amber-500">
                          Unavailable
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-text-secondary">
                      {btcStatus.available
                        ? "Pay with BTC - zero processing fees, settled in CAD"
                        : "Bitcoin deposits are temporarily unavailable."}
                    </p>
                    <Button
                      size="sm"
                      className="mt-5 bg-amber-500 hover:bg-amber-600 text-black"
                      onClick={() => setShowCryptoDeposit(true)}
                      disabled={!btcStatus.available}
                    >
                      <Bitcoin className="h-3.5 w-3.5" />
                      {btcStatus.available ? "Deposit BTC" : "Unavailable"}
                    </Button>
                  </div>
                )}

                {!paymentRailsLoading && (
                  <div className="rounded-2xl border border-violet-500/20 bg-violet-500/5 p-5">
                    <div className="flex items-center gap-2 mb-1">
                      <Zap className="h-4 w-4 text-violet-400" />
                      <p className="font-medium text-violet-400">Lightning Network</p>
                      {!lnStatus.enabled && (
                        <Badge variant="info" className="border border-violet-500/30 bg-violet-500/10 text-violet-400">
                          Coming Soon
                        </Badge>
                      )}
                      {lnStatus.enabled && !lnStatus.available && (
                        <Badge variant="info" className="border border-violet-500/30 bg-violet-500/10 text-violet-400">
                          Unavailable
                        </Badge>
                      )}
                      {lnStatus.available && (
                        <Badge variant="active" className="border border-emerald/30 bg-emerald/10 text-emerald text-[10px]">
                          Instant
                        </Badge>
                      )}
                    </div>
                    <p className="text-xs text-text-secondary">
                      {lnStatus.available
                        ? "Instant deposits via Lightning — zero fees, instant settlement"
                        : lnStatus.enabled
                          ? (lnStatus.reason || "Lightning deposits are temporarily unavailable.")
                          : "Lightning Network deposits are coming soon. Stay tuned!"}
                    </p>
                    <Button
                      size="sm"
                      className="mt-5 bg-gradient-to-r from-violet-500 to-violet-600 hover:from-violet-600 hover:to-violet-700 text-white"
                      onClick={() => setShowLightningDeposit(true)}
                      disabled={!lnStatus.available}
                      title={!lnStatus.available ? (lnStatus.enabled ? lnStatus.reason : "Coming soon") : undefined}
                    >
                      <Zap className="h-3.5 w-3.5" />
                      {lnStatus.available ? "Deposit via Lightning" : lnStatus.enabled ? "Unavailable" : "Coming Soon"}
                    </Button>
                  </div>
                )}
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

          {/* Usage Heatmap — hourly submission patterns */}
          <Card>
            <CardHeader>
              <CardTitle>Usage Heatmap</CardTitle>
              <CardDescription>Hourly compute usage patterns (last 30 days)</CardDescription>
            </CardHeader>
            <CardContent>
              {(() => {
                const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
                // Build 7×24 grid from transaction timestamps
                const grid: number[][] = Array.from({ length: 7 }, () => Array(24).fill(0));
                let maxCount = 1;
                for (const tx of transactions) {
                  if (!tx.created_at) continue;
                  const d = new Date(
                    typeof tx.created_at === "number"
                      ? tx.created_at * 1000
                      : tx.created_at
                  );
                  const day = d.getDay();
                  const hour = d.getHours();
                  grid[day][hour]++;
                  if (grid[day][hour] > maxCount) maxCount = grid[day][hour];
                }
                return (
                  <div className="overflow-x-auto">
                    <div className="min-w-[600px]">
                      {/* Hour labels */}
                      <div className="flex ml-10 mb-1">
                        {Array.from({ length: 24 }, (_, h) => (
                          <div key={h} className="flex-1 text-center text-[10px] text-text-muted">
                            {h % 6 === 0 ? `${h}h` : ""}
                          </div>
                        ))}
                      </div>
                      {/* Grid rows */}
                      {days.map((dayLabel, dayIdx) => (
                        <div key={dayLabel} className="flex items-center gap-1 mb-0.5">
                          <span className="w-9 text-xs text-text-muted text-right pr-1">{dayLabel}</span>
                          {grid[dayIdx].map((count, hour) => {
                            const intensity = count / maxCount;
                            const bg = count === 0
                              ? "bg-surface"
                              : intensity < 0.25
                              ? "bg-ice/20"
                              : intensity < 0.5
                              ? "bg-ice/40"
                              : intensity < 0.75
                              ? "bg-ice/60"
                              : "bg-ice";
                            return (
                              <div
                                key={hour}
                                className={`flex-1 h-4 rounded-sm ${bg} transition-colors`}
                                title={`${dayLabel} ${hour}:00 — ${count} transaction${count !== 1 ? "s" : ""}`}
                              />
                            );
                          })}
                        </div>
                      ))}
                      {/* Legend */}
                      <div className="flex items-center gap-2 mt-3 ml-10">
                        <span className="text-[10px] text-text-muted">Less</span>
                        <div className="w-3 h-3 rounded-sm bg-surface border border-border" />
                        <div className="w-3 h-3 rounded-sm bg-ice/20" />
                        <div className="w-3 h-3 rounded-sm bg-ice/40" />
                        <div className="w-3 h-3 rounded-sm bg-ice/60" />
                        <div className="w-3 h-3 rounded-sm bg-ice" />
                        <span className="text-[10px] text-text-muted">More</span>
                      </div>
                    </div>
                  </div>
                );
              })()}
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
      {showCryptoDeposit && customerId && btcStatus.available && (
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
      {showLightningDeposit && customerId && lnStatus.available && (
        <LightningDepositModal
          customerId={customerId}
          onClose={() => setShowLightningDeposit(false)}
          onSuccess={(newBalance) => {
            setWallet((w) => w ? { ...w, balance_cad: newBalance } : w);
            setShowLightningDeposit(false);
            load();
          }}
        />
      )}

      <ConfirmDialog
        open={showResetConfirm}
        title={t("dash.billing.admin_reset_confirm_title")}
        description={t("dash.billing.admin_reset_confirm_desc")}
        confirmLabel={t("dash.billing.admin_reset_confirm_action")}
        cancelLabel={t("common.cancel")}
        variant="danger"
        onConfirm={() => { void handleResetWalletTestingState(); }}
        onCancel={() => {
          if (!resettingWallet) setShowResetConfirm(false);
        }}
      />
    </div>
  );
}
