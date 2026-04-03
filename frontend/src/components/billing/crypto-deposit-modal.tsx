"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { BitcoinWalletConnectAction } from "@/components/billing/bitcoin-wallet-connect-action";
import {
  createCryptoDeposit,
  checkCryptoDeposit,
  refreshCryptoDeposit,
} from "@/lib/api";
import { toast } from "sonner";
import {
  X,
  Loader2,
  CheckCircle,
  Copy,
  RefreshCw,
  Clock,
  Bitcoin,
  ExternalLink,
  ArrowRight,
  Shield,
  Zap,
  Check,
  AlertTriangle,
} from "lucide-react";
import QRCode from "qrcode";
import { motion, AnimatePresence } from "framer-motion";

interface CryptoDepositModalProps {
  customerId: string;
  onClose: () => void;
  onSuccess: (newBalance: number) => void;
}

const PRESETS = [10, 25, 50, 100, 250, 500];
const POLL_INTERVAL = 8_000; // 8 seconds

function getFriendlyCryptoErrorMessage(message: string) {
  const normalized = message.toLowerCase();

  if (normalized.includes("not enabled")) {
    return "Bitcoin deposits are not enabled on this deployment.";
  }

  if (
    normalized.includes("no receiving keys")
    || normalized.includes("receiving keys")
    || normalized.includes("wallet is not ready")
    || normalized.includes("wallet file not specified")
  ) {
    return "Bitcoin wallet is not ready for fresh receiving addresses. Check the configured wallet or keypool.";
  }

  if (
    normalized.includes("offline or unavailable")
    || normalized.includes("connection refused")
    || normalized.includes("timed out")
    || normalized.includes("timeout")
    || normalized.includes("failed to establish a new connection")
    || normalized.includes("urlopen error")
  ) {
    return "Bitcoin checkout is temporarily unavailable because the node is offline.";
  }

  if (
    normalized.includes("pricing service")
    || normalized.includes("unable to fetch btc/cad rate")
    || normalized.includes("failed to fetch btc/cad rate")
  ) {
    return "Bitcoin pricing is temporarily unavailable. Try again in a moment.";
  }

  return message;
}

// ── Animated step wrapper ───────────────────────────────────────────
const stepVariants = {
  enter: { opacity: 0, x: 20, scale: 0.98 },
  center: { opacity: 1, x: 0, scale: 1 },
  exit: { opacity: 0, x: -20, scale: 0.98 },
};

// ── Pulsing ring animation for waiting state ────────────────────────
function PulseRing({ color = "rgb(245, 158, 11)" }: { color?: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute rounded-full border"
          style={{ borderColor: color }}
          initial={{ width: 200, height: 200, opacity: 0.4 }}
          animate={{ width: 280, height: 280, opacity: 0 }}
          transition={{
            duration: 2.5,
            repeat: Infinity,
            delay: i * 0.8,
            ease: "easeOut",
          }}
        />
      ))}
    </div>
  );
}

// ── Copy button with check animation ────────────────────────────────
function CopyBtn({
  text,
  label,
}: {
  text: string;
  label: string;
}) {
  const [copied, setCopied] = useState(false);
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    toast.success(`${label} copied`);
    setTimeout(() => setCopied(false), 2000);
  };
  return (
    <button
      onClick={handleCopy}
      className="shrink-0 rounded-lg p-2 hover:bg-navy-lighter transition-all active:scale-95"
      title={`Copy ${label.toLowerCase()}`}
    >
      <AnimatePresence mode="wait">
        {copied ? (
          <motion.div
            key="check"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
          >
            <Check className="h-4 w-4 text-emerald" />
          </motion.div>
        ) : (
          <motion.div
            key="copy"
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
          >
            <Copy className="h-4 w-4 text-text-muted" />
          </motion.div>
        )}
      </AnimatePresence>
    </button>
  );
}

// ── Confirmation step indicator ─────────────────────────────────────
function ConfirmationSteps({
  current,
  total,
  status,
}: {
  current: number;
  total: number;
  status: string;
}) {
  return (
    <div className="flex items-center gap-2">
      {Array.from({ length: total }).map((_, i) => {
        const done = i < current;
        const active = i === current && status === "confirming";
        return (
          <div key={i} className="flex items-center gap-2">
            <motion.div
              className={`relative flex h-9 w-9 items-center justify-center rounded-full border-2 transition-colors ${
                done
                  ? "border-amber-500 bg-amber-500"
                  : active
                    ? "border-amber-500 bg-amber-500/20"
                    : "border-border bg-navy-lighter"
              }`}
              animate={active ? { scale: [1, 1.1, 1] } : {}}
              transition={active ? { duration: 1.5, repeat: Infinity } : {}}
            >
              {done ? (
                <Check className="h-4 w-4 text-black" />
              ) : (
                <span
                  className={`text-xs font-bold ${active ? "text-amber-500" : "text-text-muted"}`}
                >
                  {i + 1}
                </span>
              )}
              {active && (
                <motion.div
                  className="absolute inset-0 rounded-full border-2 border-amber-500"
                  animate={{ scale: [1, 1.4], opacity: [0.6, 0] }}
                  transition={{ duration: 1.2, repeat: Infinity }}
                />
              )}
            </motion.div>
            {i < total - 1 && (
              <div
                className={`h-0.5 w-6 rounded-full transition-colors ${
                  i < current ? "bg-amber-500" : "bg-border"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}

export function CryptoDepositModal({
  customerId,
  onClose,
  onSuccess,
}: CryptoDepositModalProps) {
  const [amount, setAmount] = useState("");
  const [step, setStep] = useState<"amount" | "waiting" | "success">("amount");
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [deposit, setDeposit] = useState<{
    deposit_id: string;
    btc_address: string;
    amount_btc: number;
    amount_cad: number;
    btc_cad_rate: number;
    expires_at: number;
    qr_data: string;
  } | null>(null);
  const [confirmations, setConfirmations] = useState(0);
  const [status, setStatus] = useState("pending");
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [timeLeft, setTimeLeft] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const numericAmount = parseFloat(amount);
  const isValid =
    !isNaN(numericAmount) && numericAmount >= 1 && numericAmount <= 10000;

  // Generate QR code — white on transparent with Bitcoin-orange accents
  useEffect(() => {
    if (deposit?.qr_data) {
      QRCode.toDataURL(deposit.qr_data, {
        width: 240,
        margin: 3,
        color: { dark: "#f59e0b", light: "#00000000" },
        errorCorrectionLevel: "H",
      })
        .then(setQrDataUrl)
        .catch(() => setQrDataUrl(""));
    }
  }, [deposit?.qr_data]);

  // Countdown timer
  useEffect(() => {
    if (step !== "waiting" || !deposit) return;
    const update = () => {
      const left = Math.max(
        0,
        Math.floor(deposit.expires_at - Date.now() / 1000),
      );
      setTimeLeft(left);
    };
    update();
    timerRef.current = setInterval(update, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [step, deposit]);

  // Poll for confirmation
  const startPolling = useCallback(
    (depositId: string) => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const res = await checkCryptoDeposit(depositId);
          setConfirmations(res.confirmations ?? 0);
          setStatus(res.status);
          if (res.status === "credited" || res.status === "confirmed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setStep("success");
            toast.success(
              `$${(res.amount_cad ?? 0).toFixed(2)} CAD credited from Bitcoin deposit`,
            );
            setTimeout(
              () => onSuccess(res.balance_after_cad ?? res.amount_cad ?? 0),
              2500,
            );
          } else if (res.status === "expired") {
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch {
          // Silently retry on network errors
        }
      }, POLL_INTERVAL);
    },
    [onSuccess],
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleCreateDeposit = async () => {
    if (!isValid || submitting) return;
    setSubmitting(true);
    setErrorMsg("");
    try {
      const res = await createCryptoDeposit(customerId, numericAmount);
      setDeposit(res);
      setStep("waiting");
      startPolling(res.deposit_id);
    } catch (err) {
      const msg =
        err instanceof Error ? err.message : "Failed to create BTC deposit";
      const friendlyMsg = getFriendlyCryptoErrorMessage(msg);
      setErrorMsg(friendlyMsg);
      toast.error(friendlyMsg);
    } finally {
      setSubmitting(false);
    }
  };

  const handleRefresh = async () => {
    if (!deposit || refreshing) return;
    setRefreshing(true);
    try {
      const res = await refreshCryptoDeposit(deposit.deposit_id);
      setDeposit({
        ...deposit,
        amount_btc: res.amount_btc,
        btc_cad_rate: res.btc_cad_rate,
        expires_at: res.expires_at,
        qr_data: res.qr_data ?? deposit.qr_data,
      });
      setStatus("pending");
      startPolling(deposit.deposit_id);
      toast.success("Rate refreshed — new quote locked for 30 min");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to refresh rate";
      toast.error(getFriendlyCryptoErrorMessage(msg));
    } finally {
      setRefreshing(false);
    }
  };

  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  // Timer ring progress (0-1)
  const timerProgress =
    deposit && deposit.expires_at
      ? Math.max(0, timeLeft / 1800)
      : 0;

  const expired = status === "expired" || timeLeft === 0;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md" onClick={onClose}>
      <motion.div
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 24, scale: 0.96 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="relative w-full max-w-[460px] mx-4 overflow-hidden rounded-3xl border border-border bg-gradient-to-b from-surface to-navy shadow-[0_32px_80px_rgba(0,0,0,0.6)]"
      >
        {/* Decorative gradient top border */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-amber-500/0 via-amber-500 to-amber-500/0" />

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4">
          <div className="flex items-center gap-3">
            <div className="relative flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-amber-500/20 to-amber-600/10 ring-1 ring-amber-500/20">
              <Bitcoin className="h-6 w-6 text-amber-500" />
              <div className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-emerald ring-2 ring-surface" />
            </div>
            <div>
              <h2 className="text-lg font-semibold tracking-tight">
                Bitcoin Checkout
              </h2>
              <p className="text-xs text-text-muted">
                {step === "amount"
                  ? "Select deposit amount"
                  : step === "waiting"
                    ? "Awaiting payment"
                    : "Payment received"}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-xl text-text-muted hover:bg-navy-lighter hover:text-text-primary transition-all"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* Step progress dots */}
        <div className="flex items-center justify-center gap-2 pb-4">
          {["amount", "waiting", "success"].map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <div
                className={`h-1.5 rounded-full transition-all duration-500 ${
                  s === step
                    ? "w-8 bg-amber-500"
                    : ["amount", "waiting", "success"].indexOf(step) > i
                      ? "w-1.5 bg-amber-500/50"
                      : "w-1.5 bg-border"
                }`}
              />
            </div>
          ))}
        </div>

        <div className="px-6 pb-6">
          <AnimatePresence mode="wait">
            {/* ── Step 1: Amount Selection ─────────────────────────── */}
            {step === "amount" && (
              <motion.div
                key="amount"
                variants={stepVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={{ duration: 0.25 }}
              >
                {/* Quick amount presets */}
                <div className="mb-5">
                  <div className="flex items-center justify-between mb-2.5">
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                      Quick Select
                    </span>
                    <span className="text-xs text-text-muted flex items-center gap-1">
                      <Shield className="h-3 w-3" /> Zero fees
                    </span>
                  </div>
                  <div className="grid grid-cols-3 gap-2">
                    {PRESETS.map((p) => {
                      const selected = amount === String(p);
                      return (
                        <motion.button
                          key={p}
                          whileHover={{ scale: 1.02 }}
                          whileTap={{ scale: 0.97 }}
                          onClick={() => setAmount(String(p))}
                          className={`relative rounded-xl border px-3 py-3 text-sm font-semibold font-mono transition-all ${
                            selected
                              ? "border-amber-500 bg-amber-500/10 text-amber-500 shadow-[0_0_20px_rgba(245,158,11,0.15)]"
                              : "border-border bg-navy-light text-text-secondary hover:border-amber-500/30 hover:text-text-primary"
                          }`}
                        >
                          ${p}
                          {selected && (
                            <motion.div
                              layoutId="preset-indicator"
                              className="absolute inset-0 rounded-xl border-2 border-amber-500"
                              transition={{ type: "spring", stiffness: 500, damping: 30 }}
                            />
                          )}
                        </motion.button>
                      );
                    })}
                  </div>
                </div>

                {/* Custom amount */}
                <div className="mb-5">
                  <Label
                    htmlFor="btc-amount"
                    className="mb-1.5 block text-xs font-medium text-text-secondary uppercase tracking-wider"
                  >
                    Custom Amount
                  </Label>
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm font-semibold">
                      $
                    </span>
                    <Input
                      id="btc-amount"
                      type="number"
                      min="1"
                      max="10000"
                      step="0.01"
                      placeholder="0.00"
                      value={amount}
                      onChange={(e) => setAmount(e.target.value)}
                      className="pl-9 pr-16 font-mono text-base h-12 rounded-xl bg-navy-light border-border focus:border-amber-500 focus:ring-amber-500/20"
                    />
                    <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-text-muted font-medium">
                      CAD
                    </span>
                  </div>
                  {amount && !isValid && (
                    <p className="text-xs text-accent-red mt-1.5 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3" /> $1.00 – $10,000.00
                    </p>
                  )}
                </div>

                {errorMsg && (
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mb-4 rounded-xl border border-accent-red/20 bg-accent-red/5 p-3.5"
                  >
                    <p className="text-sm text-accent-red flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 shrink-0" />
                      {errorMsg}
                    </p>
                  </motion.div>
                )}

                {/* Trust badges */}
                <div className="mb-5 flex items-center gap-4 text-[11px] text-text-muted">
                  <span className="flex items-center gap-1">
                    <Zap className="h-3 w-3 text-amber-500" /> Instant quote
                  </span>
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3 text-emerald" /> FINTRAC compliant
                  </span>
                  <span className="flex items-center gap-1">
                    <Bitcoin className="h-3 w-3 text-amber-500" /> Direct node
                  </span>
                </div>

                {/* Continue button */}
                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    className="flex-1 h-12 rounded-xl"
                    onClick={onClose}
                  >
                    Cancel
                  </Button>
                  <Button
                    className="flex-1 h-12 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-black font-semibold shadow-lg shadow-amber-500/20"
                    onClick={handleCreateDeposit}
                    disabled={!isValid || submitting}
                  >
                    {submitting ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Creating…</span>
                      </>
                    ) : (
                      <>
                        <span>
                          Continue — ${isValid ? numericAmount.toFixed(2) : "0.00"}
                        </span>
                        <ArrowRight className="h-4 w-4" />
                      </>
                    )}
                  </Button>
                </div>
              </motion.div>
            )}

            {/* ── Step 2: Payment / QR Code ───────────────────────── */}
            {step === "waiting" && deposit && (
              <motion.div
                key="waiting"
                variants={stepVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={{ duration: 0.25 }}
              >
                {/* Timer bar */}
                <div className="mb-5 rounded-xl border border-border bg-navy-light p-3.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className={`flex h-7 w-7 items-center justify-center rounded-lg ${
                          expired
                            ? "bg-accent-red/10"
                            : timeLeft < 120
                              ? "bg-accent-red/10"
                              : "bg-amber-500/10"
                        }`}
                      >
                        <Clock
                          className={`h-3.5 w-3.5 ${
                            expired || timeLeft < 120
                              ? "text-accent-red"
                              : "text-amber-500"
                          }`}
                        />
                      </div>
                      <div>
                        <p className="text-xs text-text-muted">Quote expires in</p>
                        <p
                          className={`text-sm font-mono font-semibold ${
                            expired
                              ? "text-accent-red"
                              : timeLeft < 120
                                ? "text-accent-red"
                                : "text-text-primary"
                          }`}
                        >
                          {expired ? "Expired" : fmtTime(timeLeft)}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs text-text-muted">Rate</p>
                      <p className="text-sm font-mono text-text-primary">
                        ${deposit.btc_cad_rate.toLocaleString()}
                      </p>
                    </div>
                  </div>
                  {/* Progress bar */}
                  <div className="mt-2.5 h-1 rounded-full bg-navy-lighter overflow-hidden">
                    <motion.div
                      className={`h-full rounded-full ${
                        expired
                          ? "bg-accent-red"
                          : timerProgress < 0.1
                            ? "bg-accent-red"
                            : "bg-amber-500"
                      }`}
                      initial={{ width: "100%" }}
                      animate={{ width: `${timerProgress * 100}%` }}
                      transition={{ duration: 1 }}
                    />
                  </div>
                </div>

                {/* QR Code with decorative frame */}
                <div className="relative flex flex-col items-center mb-5">
                  <PulseRing
                    color={
                      confirmations > 0
                        ? "rgb(16, 185, 129)"
                        : "rgb(245, 158, 11)"
                    }
                  />
                  <div className="relative rounded-2xl bg-white p-4 shadow-xl shadow-amber-500/10">
                    {qrDataUrl ? (
                      <motion.img
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.4, ease: "easeOut" }}
                        src={qrDataUrl}
                        alt="Bitcoin payment QR code"
                        className="rounded-xl"
                        width={200}
                        height={200}
                      />
                    ) : (
                      <div className="h-[200px] w-[200px] rounded-xl bg-gray-100 animate-pulse" />
                    )}
                    {/* Bitcoin logo overlay on QR */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-md">
                        <Bitcoin className="h-6 w-6 text-amber-500" />
                      </div>
                    </div>
                  </div>
                  <p className="mt-3 text-[11px] text-text-muted">
                    Scan with any Bitcoin wallet
                  </p>
                </div>

                {/* Amount display — large and prominent */}
                <div className="mb-4 rounded-xl border border-amber-500/20 bg-amber-500/5 p-4 text-center">
                  <p className="text-xs text-text-muted mb-1">Send exactly</p>
                  <div className="flex items-center justify-center gap-2">
                    <span className="text-2xl font-bold font-mono text-amber-500">
                      {deposit.amount_btc.toFixed(8)}
                    </span>
                    <span className="text-sm font-semibold text-amber-500/60">
                      BTC
                    </span>
                    <CopyBtn
                      text={deposit.amount_btc.toFixed(8)}
                      label="Amount"
                    />
                  </div>
                  <p className="text-xs text-text-muted mt-1 font-mono">
                    ≈ ${deposit.amount_cad.toFixed(2)} CAD
                  </p>
                </div>

                {/* Address */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                      Destination Address
                    </span>
                  </div>
                  <div className="flex items-center gap-2 rounded-xl border border-border bg-navy-light p-3">
                    <code className="flex-1 text-[11px] font-mono break-all text-text-primary leading-relaxed">
                      {deposit.btc_address}
                    </code>
                    <CopyBtn text={deposit.btc_address} label="Address" />
                  </div>
                </div>

                <BitcoinWalletConnectAction
                  amountBtc={deposit.amount_btc}
                  recipient={deposit.btc_address}
                  disabled={expired}
                />

                {/* Open in wallet (BIP21 deep link) */}
                <a
                  href={deposit.qr_data}
                  className="mb-5 flex items-center justify-center gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm font-medium text-amber-500 hover:bg-amber-500/10 transition-all"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open in Wallet App
                </a>

                {/* Confirmation tracker */}
                <div className="mb-5 rounded-xl border border-border bg-navy-light p-4">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                      Confirmations
                    </span>
                    <span className="text-xs font-mono text-text-muted">
                      {confirmations} / 3
                    </span>
                  </div>
                  <div className="flex justify-center">
                    <ConfirmationSteps
                      current={confirmations}
                      total={3}
                      status={status}
                    />
                  </div>
                  <p className="mt-3 text-center text-xs text-text-muted">
                    {status === "pending" && confirmations === 0
                      ? "Listening for incoming transaction…"
                      : status === "confirming"
                        ? `Block confirmation ${confirmations} of 3 received`
                        : status === "expired"
                          ? "Quote expired — refresh to get a new rate"
                          : "Processing payment…"}
                  </p>
                  {status === "pending" && confirmations === 0 && (
                    <div className="mt-2 flex justify-center">
                      <motion.div
                        className="flex gap-1"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        <div className="h-1 w-1 rounded-full bg-amber-500" />
                        <div className="h-1 w-1 rounded-full bg-amber-500" />
                        <div className="h-1 w-1 rounded-full bg-amber-500" />
                      </motion.div>
                    </div>
                  )}
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <Button
                    variant="outline"
                    className="flex-1 h-11 rounded-xl"
                    onClick={onClose}
                  >
                    Cancel
                  </Button>
                  {expired && (
                    <Button
                      className="flex-1 h-11 rounded-xl bg-gradient-to-r from-amber-500 to-amber-600 hover:from-amber-600 hover:to-amber-700 text-black font-semibold"
                      onClick={handleRefresh}
                      disabled={refreshing}
                    >
                      {refreshing ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <RefreshCw className="h-4 w-4" />
                      )}
                      Refresh Quote
                    </Button>
                  )}
                </div>
              </motion.div>
            )}

            {/* ── Step 3: Success ──────────────────────────────────── */}
            {step === "success" && deposit && (
              <motion.div
                key="success"
                variants={stepVariants}
                initial="enter"
                animate="center"
                exit="exit"
                transition={{ duration: 0.3 }}
                className="flex flex-col items-center py-6"
              >
                {/* Animated checkmark */}
                <motion.div
                  initial={{ scale: 0, rotate: -180 }}
                  animate={{ scale: 1, rotate: 0 }}
                  transition={{
                    type: "spring",
                    stiffness: 200,
                    damping: 15,
                    delay: 0.1,
                  }}
                  className="relative mb-5"
                >
                  <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-br from-emerald to-emerald/80 shadow-lg shadow-emerald/30">
                    <CheckCircle className="h-10 w-10 text-white" />
                  </div>
                  {/* Success rings */}
                  <motion.div
                    className="absolute inset-0 rounded-full border-2 border-emerald"
                    initial={{ scale: 1, opacity: 0.6 }}
                    animate={{ scale: 1.6, opacity: 0 }}
                    transition={{ duration: 1, delay: 0.3 }}
                  />
                  <motion.div
                    className="absolute inset-0 rounded-full border-2 border-emerald"
                    initial={{ scale: 1, opacity: 0.4 }}
                    animate={{ scale: 2, opacity: 0 }}
                    transition={{ duration: 1.2, delay: 0.5 }}
                  />
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="text-center"
                >
                  <h3 className="text-xl font-bold mb-1">Payment Received</h3>
                  <p className="text-text-muted text-sm mb-4">
                    Your account has been credited
                  </p>

                  <div className="rounded-xl border border-emerald/20 bg-emerald/5 p-4 mb-2">
                    <p className="text-3xl font-bold font-mono text-emerald">
                      ${deposit.amount_cad.toFixed(2)}
                      <span className="text-base ml-1 font-normal text-emerald/60">
                        CAD
                      </span>
                    </p>
                    <p className="text-xs text-text-muted mt-1 font-mono">
                      {deposit.amount_btc.toFixed(8)} BTC • 3/3 confirmations
                    </p>
                  </div>
                </motion.div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </motion.div>
    </div>
  );
}
