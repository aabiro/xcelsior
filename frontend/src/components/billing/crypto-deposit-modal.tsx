"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
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
} from "lucide-react";
import QRCode from "qrcode";

interface CryptoDepositModalProps {
  customerId: string;
  onClose: () => void;
  onSuccess: (newBalance: number) => void;
}

const PRESETS = [10, 25, 50, 100, 250, 500];
const POLL_INTERVAL = 10_000; // 10 seconds

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

  // Generate QR code
  useEffect(() => {
    if (deposit?.qr_data) {
      QRCode.toDataURL(deposit.qr_data, {
        width: 200,
        margin: 2,
        color: { dark: "#000000", light: "#ffffff" },
      }).then(setQrDataUrl).catch(() => setQrDataUrl(""));
    }
  }, [deposit?.qr_data]);

  // Countdown timer
  useEffect(() => {
    if (step !== "waiting" || !deposit) return;
    const update = () => {
      const left = Math.max(0, Math.floor(deposit.expires_at - Date.now() / 1000));
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
            // Give user a moment to see success, then notify parent
            setTimeout(
              () => onSuccess(res.balance_after_cad ?? res.amount_cad ?? 0),
              1500,
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
      const msg = err instanceof Error ? err.message : "Failed to create BTC deposit";
      const isUnavailable = msg.includes("503") || msg.includes("unavailable") || msg.includes("not enabled") || msg.includes("Bad Gateway");
      setErrorMsg(isUnavailable
        ? "Bitcoin deposits are temporarily unavailable. The BTC node is offline — please use card deposit instead."
        : msg);
      toast.error(isUnavailable ? "Bitcoin node is offline" : msg);
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
      toast.success("Rate refreshed");
    } catch (err) {
      toast.error(
        err instanceof Error ? err.message : "Failed to refresh rate",
      );
    } finally {
      setRefreshing(false);
    }
  };

  const handleCopyAddress = () => {
    if (deposit?.btc_address) {
      navigator.clipboard.writeText(deposit.btc_address);
      toast.success("Address copied");
    }
  };

  const handleCopyAmount = () => {
    if (deposit?.amount_btc) {
      navigator.clipboard.writeText(deposit.amount_btc.toFixed(8));
      toast.success("Amount copied");
    }
  };

  const fmtTime = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return `${m}:${sec.toString().padStart(2, "0")}`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-500/10">
              <Bitcoin className="h-5 w-5 text-amber-500" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Bitcoin Deposit</h2>
              <p className="text-xs text-text-muted">
                {step === "amount"
                  ? "Choose deposit amount"
                  : step === "waiting"
                    ? "Send BTC to complete"
                    : "Deposit confirmed"}
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-text-muted hover:bg-background hover:text-text-primary transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {step === "success" ? (
          <div className="flex flex-col items-center py-8">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10 mb-4">
              <CheckCircle className="h-8 w-8 text-emerald" />
            </div>
            <p className="text-lg font-semibold">Deposit Confirmed</p>
            <p className="text-sm text-text-muted mt-1">
              ${deposit?.amount_cad.toFixed(2)} CAD has been added to your
              wallet
            </p>
            <p className="text-xs text-text-muted mt-1 font-mono">
              {deposit?.amount_btc.toFixed(8)} BTC received
            </p>
          </div>
        ) : step === "amount" ? (
          <>
            {/* Preset amounts */}
            <div className="mb-4">
              <Label className="mb-2 block text-text-secondary text-xs">
                Quick select
              </Label>
              <div className="grid grid-cols-3 gap-2">
                {PRESETS.map((p) => (
                  <button
                    key={p}
                    onClick={() => setAmount(String(p))}
                    className={`rounded-lg border px-3 py-2.5 text-sm font-mono transition-colors ${
                      amount === String(p)
                        ? "border-amber-500 bg-amber-500/10 text-amber-500"
                        : "border-border bg-background text-text-secondary hover:border-text-muted"
                    }`}
                  >
                    ${p}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom amount */}
            <div className="mb-6">
              <Label
                htmlFor="btc-amount"
                className="mb-1.5 block text-text-secondary text-xs"
              >
                Custom amount (CAD)
              </Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm">
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
                  className="pl-8 font-mono"
                />
              </div>
              {amount && !isValid && (
                <p className="text-xs text-accent-red mt-1">
                  Enter between $1.00 and $10,000.00
                </p>
              )}
            </div>

            {errorMsg && (
              <div className="mb-4 rounded-lg border border-accent-red/30 bg-accent-red/5 p-3">
                <p className="text-sm text-accent-red">{errorMsg}</p>
              </div>
            )}

            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={onClose}>
                Cancel
              </Button>
              <Button
                className="flex-1 bg-amber-500 hover:bg-amber-600 text-black"
                onClick={handleCreateDeposit}
                disabled={!isValid || submitting}
              >
                {submitting ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" /> Creating…
                  </>
                ) : (
                  <>
                    <Bitcoin className="h-4 w-4" /> Continue — $
                    {isValid ? numericAmount.toFixed(2) : "0.00"}
                  </>
                )}
              </Button>
            </div>
          </>
        ) : (
          /* Waiting for payment */
          <>
            {/* Timer + Rate */}
            <div className="mb-4 flex items-center justify-between rounded-lg border border-border bg-background p-3">
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-text-muted" />
                <span className="text-sm text-text-secondary">
                  Rate expires in
                </span>
              </div>
              <span
                className={`font-mono text-sm font-medium ${timeLeft < 120 ? "text-accent-red" : "text-text-primary"}`}
              >
                {timeLeft > 0 ? fmtTime(timeLeft) : "Expired"}
              </span>
            </div>

            {/* QR Code */}
            <div className="flex flex-col items-center mb-4">
              {qrDataUrl ? (
                <img
                  src={qrDataUrl}
                  alt="BTC QR code"
                  className="rounded-lg"
                  width={180}
                  height={180}
                />
              ) : (
                <div className="h-[180px] w-[180px] rounded-lg bg-background skeleton-pulse" />
              )}
            </div>

            {/* Address */}
            <div className="mb-3">
              <Label className="mb-1 block text-text-secondary text-xs">
                Send to address
              </Label>
              <div className="flex items-center gap-2 rounded-lg border border-border bg-background p-2.5">
                <code className="flex-1 text-xs font-mono break-all text-text-primary">
                  {deposit?.btc_address}
                </code>
                <button
                  onClick={handleCopyAddress}
                  className="shrink-0 rounded p-1 hover:bg-surface transition-colors"
                  title="Copy address"
                >
                  <Copy className="h-3.5 w-3.5 text-text-muted" />
                </button>
              </div>
            </div>

            {/* Amount */}
            <div className="mb-4">
              <Label className="mb-1 block text-text-secondary text-xs">
                Exact amount
              </Label>
              <div className="flex items-center gap-2 rounded-lg border border-border bg-background p-2.5">
                <code className="flex-1 text-sm font-mono text-amber-500 font-medium">
                  {deposit?.amount_btc.toFixed(8)} BTC
                </code>
                <button
                  onClick={handleCopyAmount}
                  className="shrink-0 rounded p-1 hover:bg-surface transition-colors"
                  title="Copy amount"
                >
                  <Copy className="h-3.5 w-3.5 text-text-muted" />
                </button>
              </div>
              <p className="text-xs text-text-muted mt-1 font-mono">
                ≈ ${deposit?.amount_cad.toFixed(2)} CAD @ $
                {deposit?.btc_cad_rate.toLocaleString()} BTC/CAD
              </p>
            </div>

            {/* Confirmation progress */}
            <div className="mb-4 rounded-lg border border-border bg-background p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs text-text-secondary">
                  Confirmations
                </span>
                <span className="text-xs font-mono">
                  {confirmations}/3
                </span>
              </div>
              <div className="h-2 rounded-full bg-surface overflow-hidden">
                <div
                  className="h-full rounded-full bg-amber-500 transition-all duration-500"
                  style={{ width: `${Math.min(100, (confirmations / 3) * 100)}%` }}
                />
              </div>
              <p className="text-xs text-text-muted mt-1.5">
                {status === "pending" && confirmations === 0
                  ? "Waiting for transaction…"
                  : status === "confirming"
                    ? `${confirmations} confirmation${confirmations !== 1 ? "s" : ""} received, waiting for ${3 - confirmations} more`
                    : status === "expired"
                      ? "Rate expired — refresh to continue"
                      : "Processing…"}
              </p>
            </div>

            {/* Actions */}
            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={onClose}>
                Cancel
              </Button>
              {(status === "expired" || timeLeft === 0) && (
                <Button
                  className="flex-1 bg-amber-500 hover:bg-amber-600 text-black"
                  onClick={handleRefresh}
                  disabled={refreshing}
                >
                  {refreshing ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <RefreshCw className="h-4 w-4" />
                  )}
                  Refresh Rate
                </Button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
