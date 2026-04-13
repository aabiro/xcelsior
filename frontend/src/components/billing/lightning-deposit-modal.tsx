"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import {
  createLightningDeposit,
  checkLightningDeposit,
} from "@/lib/api";
import { toast } from "sonner";
import {
  X,
  Loader2,
  CheckCircle,
  Copy,
  Clock,
  ExternalLink,
  ArrowRight,
  Shield,
  Zap,
  Check,
  AlertTriangle,
  Radio,
} from "lucide-react";
import QRCode from "qrcode";
import { motion, AnimatePresence } from "framer-motion";

interface LightningDepositModalProps {
  customerId: string;
  onClose: () => void;
  onSuccess: (newBalance: number) => void;
}

const PRESETS = [5, 10, 25, 50, 100, 250];
const POLL_INTERVAL = 3_000; // 3s — Lightning is instant

// ── Animated step wrapper ───────────────────────────────────────────
const stepVariants = {
  enter: { opacity: 0, x: 20, scale: 0.98 },
  center: { opacity: 1, x: 0, scale: 1 },
  exit: { opacity: 0, x: -20, scale: 0.98 },
};

// ── Electric pulse rings for Lightning waiting state ────────────────
function ElectricPulse({ color = "rgb(124, 58, 237)" }: { color?: string }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
      {[0, 1, 2].map((i) => (
        <motion.div
          key={i}
          className="absolute rounded-full"
          style={{
            borderWidth: 2,
            borderStyle: "solid",
            borderColor: color,
            boxShadow: `0 0 20px ${color}40`,
          }}
          initial={{ width: 200, height: 200, opacity: 0.5 }}
          animate={{ width: 300, height: 300, opacity: 0 }}
          transition={{
            duration: 2,
            repeat: Infinity,
            delay: i * 0.6,
            ease: "easeOut",
          }}
        />
      ))}
    </div>
  );
}

// ── Lightning bolt SVG icon ─────────────────────────────────────────
function LightningBolt({ className = "h-6 w-6" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
    >
      <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" fill="currentColor" />
    </svg>
  );
}

// ── Copy button with check animation ────────────────────────────────
function CopyBtn({ text, label }: { text: string; label: string }) {
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
          <motion.div key="check" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
            <Check className="h-4 w-4 text-emerald" />
          </motion.div>
        ) : (
          <motion.div key="copy" initial={{ scale: 0 }} animate={{ scale: 1 }} exit={{ scale: 0 }}>
            <Copy className="h-4 w-4 text-text-muted" />
          </motion.div>
        )}
      </AnimatePresence>
    </button>
  );
}

export function LightningDepositModal({
  customerId,
  onClose,
  onSuccess,
}: LightningDepositModalProps) {
  const [amount, setAmount] = useState("");
  const [step, setStep] = useState<"amount" | "waiting" | "success">("amount");
  const [submitting, setSubmitting] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [deposit, setDeposit] = useState<{
    deposit_id: string;
    bolt11: string;
    payment_hash: string;
    amount_sats: number;
    amount_btc: number;
    amount_cad: number;
    btc_cad_rate: number;
    expires_at: number;
  } | null>(null);
  const [status, setStatus] = useState("pending");
  const [qrDataUrl, setQrDataUrl] = useState("");
  const [timeLeft, setTimeLeft] = useState(0);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 1 && numericAmount <= 1000;

  // Generate QR code — violet/purple for Lightning
  useEffect(() => {
    if (deposit?.bolt11) {
      QRCode.toDataURL(`lightning:${deposit.bolt11}`, {
        width: 240,
        margin: 3,
        color: { dark: "#7c3aed", light: "#00000000" },
        errorCorrectionLevel: "M",
      })
        .then(setQrDataUrl)
        .catch(() => setQrDataUrl(""));
    }
  }, [deposit?.bolt11]);

  // Countdown timer
  useEffect(() => {
    if (step !== "waiting" || !deposit) return;
    const update = () => {
      const left = Math.max(0, Math.floor(deposit.expires_at - Date.now() / 1000));
      setTimeLeft(left);
    };
    update();
    timerRef.current = setInterval(update, 1000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [step, deposit]);

  // Poll for payment
  const startPolling = useCallback(
    (depositId: string) => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const res = await checkLightningDeposit(depositId);
          setStatus(res.status);
          if (res.status === "paid" || res.status === "credited") {
            if (pollRef.current) clearInterval(pollRef.current);
            setStep("success");
            toast.success(`$${(res.amount_cad ?? 0).toFixed(2)} CAD credited instantly via Lightning`);
            setTimeout(() => onSuccess(res.amount_cad ?? 0), 2500);
          } else if (res.status === "expired") {
            if (pollRef.current) clearInterval(pollRef.current);
          }
        } catch {
          // Silently retry
        }
      }, POLL_INTERVAL);
    },
    [onSuccess],
  );

  // Cleanup
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
      const res = await createLightningDeposit(customerId, numericAmount);
      setDeposit(res);
      setStep("waiting");
      startPolling(res.deposit_id);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Failed to create Lightning invoice";
      setErrorMsg(msg);
      toast.error(msg);
    } finally {
      setSubmitting(false);
    }
  };

  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  const timerProgress = deposit?.expires_at ? Math.max(0, timeLeft / 600) : 0; // 10 min
  const expired = status === "expired" || timeLeft === 0;

  const formatSats = (sats: number) => {
    if (sats >= 1_000_000) return `${(sats / 1_000_000).toFixed(2)}M`;
    if (sats >= 1_000) return `${(sats / 1_000).toFixed(1)}k`;
    return sats.toLocaleString();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/70 px-4 py-4 backdrop-blur-md sm:items-center" onClick={onClose}>
      <motion.div
        onClick={(e: React.MouseEvent) => e.stopPropagation()}
        initial={{ opacity: 0, y: 24, scale: 0.96 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: 24, scale: 0.96 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="relative my-auto max-h-[calc(100vh-2rem)] w-full max-w-[460px] overflow-x-hidden overflow-y-auto rounded-3xl border border-border bg-gradient-to-b from-surface to-navy shadow-[0_32px_80px_rgba(0,0,0,0.6)]"
      >
        {/* Decorative gradient top border — violet/purple for Lightning */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-violet-500/0 via-violet-500 to-violet-500/0" />

        {/* Header */}
        <div className="flex items-center justify-between px-6 pt-6 pb-4">
          <div className="flex items-center gap-3">
            <div className="relative flex h-11 w-11 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500/20 to-violet-600/10 ring-1 ring-violet-500/20">
              <Zap className="h-6 w-6 text-violet-400" />
              <div className="absolute -top-0.5 -right-0.5 h-2.5 w-2.5 rounded-full bg-emerald ring-2 ring-surface" />
            </div>
            <div>
              <h2 className="text-lg font-semibold tracking-tight">
                Lightning Checkout
              </h2>
              <p className="text-xs text-text-muted">
                {step === "amount"
                  ? "Instant settlement, zero fees"
                  : step === "waiting"
                    ? "Scan to pay"
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
                    ? "w-8 bg-violet-500"
                    : ["amount", "waiting", "success"].indexOf(step) > i
                      ? "w-1.5 bg-violet-500/50"
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
                      <Zap className="h-3 w-3 text-violet-400" /> Instant
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
                              ? "border-violet-500 bg-violet-500/10 text-violet-400 shadow-[0_0_20px_rgba(124,58,237,0.15)]"
                              : "border-border bg-navy-light text-text-secondary hover:border-violet-500/30 hover:text-text-primary"
                          }`}
                        >
                          ${p}
                          {selected && (
                            <motion.div
                              layoutId="ln-preset-indicator"
                              className="absolute inset-0 rounded-xl border-2 border-violet-500"
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
                    htmlFor="ln-amount"
                    className="mb-1.5 block text-xs font-medium text-text-secondary uppercase tracking-wider"
                  >
                    Custom Amount
                  </Label>
                  <div className="relative">
                    <span className="absolute left-4 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm font-semibold">
                      $
                    </span>
                    <Input
                      id="ln-amount"
                      type="number"
                      min="1"
                      max="1000"
                      step="0.01"
                      placeholder="0.00"
                      value={amount}
                      onChange={(e) => setAmount(e.target.value)}
                      className="pl-9 pr-16 font-mono text-base h-12 rounded-xl bg-navy-light border-border focus:border-violet-500 focus:ring-violet-500/20"
                    />
                    <span className="absolute right-4 top-1/2 -translate-y-1/2 text-xs text-text-muted font-medium">
                      CAD
                    </span>
                  </div>
                  {amount && !isValid && (
                    <p className="text-xs text-accent-red mt-1.5 flex items-center gap-1">
                      <AlertTriangle className="h-3 w-3" /> $1.00 – $1,000.00
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
                    <Zap className="h-3 w-3 text-violet-400" /> Instant settlement
                  </span>
                  <span className="flex items-center gap-1">
                    <Shield className="h-3 w-3 text-emerald" /> Zero fees
                  </span>
                  <span className="flex items-center gap-1">
                    <Radio className="h-3 w-3 text-violet-400" /> Own node
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
                    className="flex-1 h-12 rounded-xl bg-gradient-to-r from-violet-500 to-violet-600 hover:from-violet-600 hover:to-violet-700 text-white font-semibold shadow-lg shadow-violet-500/20"
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

            {/* ── Step 2: Invoice / QR Code ──────────────────────── */}
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
                      <div className={`flex h-7 w-7 items-center justify-center rounded-lg ${expired ? "bg-accent-red/10" : timeLeft < 60 ? "bg-accent-red/10" : "bg-violet-500/10"}`}>
                        <Clock className={`h-3.5 w-3.5 ${expired || timeLeft < 60 ? "text-accent-red" : "text-violet-400"}`} />
                      </div>
                      <div>
                        <p className="text-xs text-text-muted">Invoice expires in</p>
                        <p className={`text-sm font-mono font-semibold ${expired ? "text-accent-red" : timeLeft < 60 ? "text-accent-red" : "text-text-primary"}`}>
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
                      className={`h-full rounded-full ${expired ? "bg-accent-red" : timerProgress < 0.15 ? "bg-accent-red" : "bg-violet-500"}`}
                      initial={{ width: "100%" }}
                      animate={{ width: `${timerProgress * 100}%` }}
                      transition={{ duration: 1 }}
                    />
                  </div>
                </div>

                {/* QR Code with electric pulse frame */}
                <div className="relative flex flex-col items-center mb-5">
                  <ElectricPulse color={status === "paid" ? "rgb(16, 185, 129)" : "rgb(124, 58, 237)"} />
                  <div className="relative rounded-2xl bg-white p-4 shadow-xl shadow-violet-500/10">
                    {qrDataUrl ? (
                      <motion.img
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.4, ease: "easeOut" }}
                        src={qrDataUrl}
                        alt="Lightning invoice QR code"
                        className="rounded-xl"
                        width={200}
                        height={200}
                      />
                    ) : (
                      <div className="h-[200px] w-[200px] rounded-xl bg-gray-100 animate-pulse" />
                    )}
                    {/* Lightning bolt overlay on QR */}
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                      <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white shadow-md">
                        <LightningBolt className="h-5 w-5 text-violet-500" />
                      </div>
                    </div>
                  </div>
                  <p className="mt-3 text-[11px] text-text-muted">
                    Scan with any Lightning wallet
                  </p>
                </div>

                {/* Amount display — sats prominent */}
                <div className="mb-4 rounded-xl border border-violet-500/20 bg-violet-500/5 p-4 text-center">
                  <p className="text-xs text-text-muted mb-1">Send exactly</p>
                  <div className="flex items-center justify-center gap-2">
                    <LightningBolt className="h-5 w-5 text-violet-400" />
                    <span className="text-2xl font-bold font-mono text-violet-400">
                      {deposit.amount_sats.toLocaleString()}
                    </span>
                    <span className="text-sm font-semibold text-violet-400/60">
                      sats
                    </span>
                    <CopyBtn text={String(deposit.amount_sats)} label="Amount" />
                  </div>
                  <p className="text-xs text-text-muted mt-1 font-mono">
                    {deposit.amount_btc.toFixed(8)} BTC ≈ ${deposit.amount_cad.toFixed(2)} CAD
                  </p>
                </div>

                {/* BOLT11 Invoice */}
                <div className="mb-4">
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">
                      Lightning Invoice
                    </span>
                  </div>
                  <div className="flex items-center gap-2 rounded-xl border border-border bg-navy-light p-3">
                    <code className="flex-1 text-[10px] font-mono break-all text-text-primary leading-relaxed line-clamp-3">
                      {deposit.bolt11}
                    </code>
                    <CopyBtn text={deposit.bolt11} label="Invoice" />
                  </div>
                </div>

                {/* Open in wallet (lightning: deep link) */}
                <a
                  href={`lightning:${deposit.bolt11}`}
                  className="mb-5 flex items-center justify-center gap-2 rounded-xl border border-violet-500/20 bg-violet-500/5 px-4 py-3 text-sm font-medium text-violet-400 hover:bg-violet-500/10 transition-all"
                >
                  <ExternalLink className="h-4 w-4" />
                  Open in Wallet App
                </a>

                {/* Listening indicator */}
                <div className="mb-5 rounded-xl border border-border bg-navy-light p-4">
                  <div className="flex items-center justify-center gap-3">
                    <div className="relative">
                      <motion.div
                        className="flex h-10 w-10 items-center justify-center rounded-full bg-violet-500/10"
                        animate={{ scale: [1, 1.1, 1] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        <Zap className="h-5 w-5 text-violet-400" />
                      </motion.div>
                      <motion.div
                        className="absolute inset-0 rounded-full border-2 border-violet-500"
                        animate={{ scale: [1, 1.5], opacity: [0.4, 0] }}
                        transition={{ duration: 1.2, repeat: Infinity }}
                      />
                    </div>
                    <div>
                      <p className="text-sm font-medium text-text-primary">
                        {expired ? "Invoice expired" : "Waiting for payment…"}
                      </p>
                      <p className="text-xs text-text-muted">
                        {expired
                          ? "Close and create a new invoice"
                          : "Payment confirms instantly via Lightning"}
                      </p>
                    </div>
                  </div>
                  {!expired && (
                    <div className="mt-3 flex justify-center">
                      <motion.div
                        className="flex gap-1.5"
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, repeat: Infinity }}
                      >
                        {[0, 1, 2, 3, 4].map((i) => (
                          <motion.div
                            key={i}
                            className="h-1 w-3 rounded-full bg-violet-500"
                            animate={{ scaleY: [1, 2.5, 1] }}
                            transition={{
                              duration: 0.8,
                              repeat: Infinity,
                              delay: i * 0.15,
                            }}
                          />
                        ))}
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
                    {expired ? "Close" : "Cancel"}
                  </Button>
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
                {/* Animated lightning bolt success */}
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
                  <h3 className="text-xl font-bold mb-1">Instant Payment Received</h3>
                  <p className="text-text-muted text-sm mb-4 flex items-center justify-center gap-1.5">
                    <Zap className="h-3.5 w-3.5 text-violet-400" />
                    Settled via Lightning Network
                  </p>

                  <div className="rounded-xl border border-emerald/20 bg-emerald/5 p-4 mb-2">
                    <p className="text-3xl font-bold font-mono text-emerald">
                      ${deposit.amount_cad.toFixed(2)}
                      <span className="text-base ml-1 font-normal text-emerald/60">CAD</span>
                    </p>
                    <p className="text-xs text-text-muted mt-1 font-mono">
                      {formatSats(deposit.amount_sats)} sats • instant confirmation
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
