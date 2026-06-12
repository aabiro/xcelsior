"use client";

import { useState, useCallback, useEffect } from "react";
import { Elements, PaymentElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import {
  createPaymentIntent,
  fetchWallet,
  checkPayPalEnabled,
  createPayPalOrder,
  capturePayPalOrder,
} from "@/lib/api";
import { getStripeElementsOptions } from "@/lib/stripe-appearance";
import { getStripePromise } from "@/lib/stripe-client";
import { toast } from "sonner";
import { X, CreditCard, Loader2, CheckCircle, ShieldCheck, Sparkles } from "lucide-react";

interface DepositModalProps {
  customerId: string;
  onClose: () => void;
  onSuccess: (newBalance: number) => void;
}

const PRESETS = [10, 25, 50, 100, 250, 500];

async function pollWalletBalance(
  customerId: string,
  previousBalance: number,
  expectedIncrease: number,
  maxAttempts = 24,
): Promise<number> {
  for (let i = 0; i < maxAttempts; i++) {
    const { wallet } = await fetchWallet(customerId);
    if (wallet.balance_cad >= previousBalance + expectedIncrease - 0.01) {
      return wallet.balance_cad;
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error("Payment received but wallet not credited yet. Please refresh in a moment.");
}

function AmountStep({
  amount,
  setAmount,
  isValid,
  numericAmount,
  onCancel,
  onContinue,
  continueLabel,
  minAmount = 5,
  maxAmount = 10000,
}: {
  amount: string;
  setAmount: (v: string) => void;
  isValid: boolean;
  numericAmount: number;
  onCancel: () => void;
  onContinue: () => void;
  continueLabel?: string;
  minAmount?: number;
  maxAmount?: number;
}) {
  const [showAmountError, setShowAmountError] = useState(false);

  const handleContinue = () => {
    if (!isValid) {
      setShowAmountError(true);
      return;
    }
    setShowAmountError(false);
    onContinue();
  };

  return (
    <>
      <div className="mb-4">
        <Label className="mb-2 block text-text-secondary text-xs">Quick select</Label>
        <div className="grid grid-cols-3 gap-2">
          {PRESETS.map((p) => (
            <button
              key={p}
              onClick={() => setAmount(String(p))}
              className={`rounded-lg border px-3 py-2.5 text-sm font-mono transition-colors ${
                amount === String(p)
                  ? "border-emerald bg-emerald/10 text-emerald"
                  : "border-border bg-background text-text-secondary hover:border-text-muted"
              }`}
            >
              ${p}
            </button>
          ))}
        </div>
      </div>
      <div className="mb-6">
        <Label htmlFor="amount" className="mb-1.5 block text-text-secondary text-xs">
          Custom amount (CAD)
        </Label>
        <div className="relative">
          <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm">$</span>
          <Input
            id="amount"
            type="text"
            inputMode="decimal"
            placeholder="0.00"
            value={amount}
            onChange={(e) => {
              setAmount(e.target.value);
              setShowAmountError(false);
            }}
            className="pl-8 font-mono"
          />
        </div>
        {showAmountError && !isValid && (
          <p className="text-xs text-accent-red mt-1">
            Enter between ${minAmount.toFixed(2)} and ${maxAmount.toLocaleString("en-CA", { minimumFractionDigits: 2, maximumFractionDigits: 2 })} CAD
          </p>
        )}
      </div>
      <div className="flex gap-3">
        <Button variant="outline" className="flex-1" onClick={onCancel}>Cancel</Button>
        <Button variant="success" className="flex-1" onClick={handleContinue}>
          {continueLabel ?? `Continue — $${isValid ? numericAmount.toFixed(2) : "0.00"}`}
        </Button>
      </div>
    </>
  );
}

function ModalShell({
  onClose,
  subtitle,
  children,
}: {
  onClose: () => void;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div
        className="brand-top-accent w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald/20 to-accent-cyan/10 ring-1 ring-emerald/20">
              <CreditCard className="h-5 w-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Add Credits</h2>
              <p className="text-xs text-text-muted">{subtitle}</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-text-muted hover:bg-background hover:text-text-primary transition-colors"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function PayPalButton({
  customerId,
  amountCad,
  onSuccess,
  disabled,
}: {
  customerId: string;
  amountCad: number;
  onSuccess: (balance: number) => void;
  disabled?: boolean;
}) {
  const [loading, setLoading] = useState(false);

  const handlePayPal = async () => {
    setLoading(true);
    try {
      const { order_id } = await createPayPalOrder(customerId, amountCad);
      const approvalUrl = `https://www.${
        process.env.NEXT_PUBLIC_PAYPAL_MODE === "live" ? "" : "sandbox."
      }paypal.com/checkoutnow?token=${order_id}`;
      const popup = window.open(approvalUrl, "paypal_checkout", "width=500,height=700,left=200,top=100");
      if (!popup) {
        toast.error("Please allow popups for PayPal checkout");
        setLoading(false);
        return;
      }
      const poll = setInterval(async () => {
        if (!popup.closed) return;
        clearInterval(poll);
        try {
          const result = await capturePayPalOrder(customerId, order_id);
          toast.success(`$${result.amount_cad.toFixed(2)} CAD added via PayPal`);
          onSuccess(result.balance_cad);
        } catch {
          toast.info("PayPal payment was not completed");
        } finally {
          setLoading(false);
        }
      }, 500);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "PayPal checkout failed");
      setLoading(false);
    }
  };

  return (
    <button
      onClick={handlePayPal}
      disabled={disabled || loading}
      className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#ffc439] bg-[#ffc439] px-4 py-2.5 text-sm font-semibold text-[#003087] transition-colors hover:bg-[#f0b929] disabled:opacity-50"
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : null}
      {loading ? "Processing…" : "Pay with PayPal"}
    </button>
  );
}

function PayPalOnlyDepositForm({ customerId, onClose, onSuccess }: DepositModalProps) {
  const [amount, setAmount] = useState("");
  const [step, setStep] = useState<"amount" | "pay" | "success" | "unavailable">("amount");
  const [paypalAvailable, setPaypalAvailable] = useState<boolean | null>(null);
  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 5 && numericAmount <= 10000;

  useEffect(() => {
    checkPayPalEnabled()
      .then((r) => {
        setPaypalAvailable(r.enabled);
        if (!r.enabled) setStep("unavailable");
      })
      .catch(() => {
        setPaypalAvailable(false);
        setStep("unavailable");
      });
  }, []);

  if (step === "unavailable" || paypalAvailable === false) {
    return (
      <ModalShell onClose={onClose} subtitle="Payments unavailable">
        <p className="text-sm text-text-secondary mb-4">
          Card payments are not configured and PayPal is unavailable. Please contact support to add credits.
        </p>
        <Button variant="outline" className="w-full" onClick={onClose}>Close</Button>
      </ModalShell>
    );
  }

  if (paypalAvailable === null) {
    return (
      <ModalShell onClose={onClose} subtitle="Loading payment options">
        <div className="flex justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-text-muted" />
        </div>
      </ModalShell>
    );
  }

  return (
    <ModalShell onClose={onClose} subtitle={step === "success" ? "Payment complete" : "Pay with PayPal"}>
      {step === "success" ? (
        <div className="flex flex-col items-center py-8">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10 mb-4">
            <CheckCircle className="h-8 w-8 text-emerald" />
          </div>
          <p className="text-lg font-semibold">Payment Successful</p>
          <p className="text-sm text-text-muted mt-1">${numericAmount.toFixed(2)} CAD has been added to your wallet</p>
        </div>
      ) : step === "amount" ? (
        <AmountStep
          amount={amount}
          setAmount={setAmount}
          isValid={isValid}
          numericAmount={numericAmount}
          onCancel={onClose}
          onContinue={() => setStep("pay")}
        />
      ) : (
        <>
          <div className="mb-4 rounded-lg border border-border bg-background p-3 flex items-center justify-between">
            <span className="text-sm text-text-secondary">Deposit amount</span>
            <span className="text-lg font-bold font-mono">${numericAmount.toFixed(2)} CAD</span>
          </div>
          <PayPalButton
            customerId={customerId}
            amountCad={numericAmount}
            onSuccess={(balance) => {
              setStep("success");
              setTimeout(() => onSuccess(balance), 1200);
            }}
          />
          <p className="mt-2 text-[11px] leading-relaxed text-text-muted">
            PayPal here adds credits to your wallet only. GPU jobs always spend wallet balance.
          </p>
          <Button variant="outline" className="w-full mt-3" onClick={() => setStep("amount")}>Back</Button>
        </>
      )}
    </ModalShell>
  );
}

function DepositPaymentStep({
  customerId,
  numericAmount,
  onBack,
  onSuccess,
  paypalAvailable,
}: {
  customerId: string;
  numericAmount: number;
  onBack: () => void;
  onSuccess: (balance: number) => void;
  paypalAvailable: boolean;
}) {
  const stripe = useStripe();
  const elements = useElements();
  const [submitting, setSubmitting] = useState(false);
  const [paymentReady, setPaymentReady] = useState(false);

  const handlePayment = useCallback(async () => {
    if (!stripe || !elements || submitting) return;
    setSubmitting(true);
    try {
      const baseline = await fetchWallet(customerId);
      const previousBalance = baseline.wallet.balance_cad;
      const { error, paymentIntent } = await stripe.confirmPayment({
        elements,
        confirmParams: { return_url: window.location.href },
        redirect: "if_required",
      });
      if (error) {
        toast.error(error.message || "Payment failed");
        return;
      }
      if (paymentIntent?.status === "succeeded") {
        const newBalance = await pollWalletBalance(customerId, previousBalance, numericAmount);
        toast.success(`$${numericAmount.toFixed(2)} CAD added to wallet`);
        onSuccess(newBalance);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Payment failed");
    } finally {
      setSubmitting(false);
    }
  }, [stripe, elements, customerId, numericAmount, submitting, onSuccess]);

  return (
    <>
      <div className="mb-4 rounded-xl border border-border/80 bg-background/80 p-3.5 flex items-center justify-between">
        <span className="text-sm text-text-secondary">Deposit amount</span>
        <div className="flex items-center gap-2">
          <span className="text-lg font-bold font-mono">${numericAmount.toFixed(2)}</span>
          <span className="text-xs text-text-muted">CAD</span>
          <button onClick={onBack} className="text-xs text-accent-cyan hover:underline ml-1">Change</button>
        </div>
      </div>

      <div className="mb-4 rounded-xl border border-border bg-background/60 p-4">
        <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-text-muted">
          <Sparkles className="h-3.5 w-3.5 text-accent-cyan" />
          Secure payment
        </div>
        <PaymentElement
          onChange={(e) => setPaymentReady(e.complete)}
          options={{ layout: "tabs" }}
        />
      </div>

      <div className="mb-4 flex items-start gap-2 rounded-lg bg-accent-cyan/5 border border-accent-cyan/15 p-3">
        <ShieldCheck className="h-4 w-4 text-accent-cyan mt-0.5 shrink-0" />
        <p className="text-xs text-text-secondary">
          Embedded checkout — card details never touch our servers. Powered by Stripe.
        </p>
      </div>

      {paypalAvailable && (
        <div className="mb-6 rounded-xl border border-border/70 bg-background/70 p-4">
          <div className="mb-3 flex items-center gap-3">
            <div className="h-px flex-1 bg-border" />
            <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-text-muted">Or use PayPal</span>
            <div className="h-px flex-1 bg-border" />
          </div>
          <PayPalButton customerId={customerId} amountCad={numericAmount} onSuccess={onSuccess} disabled={submitting} />
        </div>
      )}

      <div className="flex gap-3">
        <Button variant="outline" className="flex-1" onClick={onBack} disabled={submitting}>Back</Button>
        <Button variant="success" className="flex-1" onClick={handlePayment} disabled={!paymentReady || submitting}>
          {submitting ? (
            <><Loader2 className="h-4 w-4 animate-spin" /> Processing…</>
          ) : (
            <>Pay ${numericAmount.toFixed(2)} CAD</>
          )}
        </Button>
      </div>
    </>
  );
}

function DepositForm({ customerId, onClose, onSuccess }: DepositModalProps) {
  const [amount, setAmount] = useState("");
  const [step, setStep] = useState<"amount" | "pay" | "success">("amount");
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [intentLoading, setIntentLoading] = useState(false);
  const [paypalAvailable, setPaypalAvailable] = useState(false);

  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 5 && numericAmount <= 10000;

  useEffect(() => {
    checkPayPalEnabled().then((r) => setPaypalAvailable(r.enabled)).catch(() => {});
  }, []);

  const handleProceedToPay = useCallback(async () => {
    if (!isValid || intentLoading) return;
    setIntentLoading(true);
    try {
      const { intent } = await createPaymentIntent(customerId, numericAmount);
      if (!intent.client_secret) {
        toast.error("Card payments are temporarily unavailable on the server.");
        return;
      }
      setClientSecret(intent.client_secret);
      setStep("pay");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not start checkout");
    } finally {
      setIntentLoading(false);
    }
  }, [customerId, numericAmount, isValid, intentLoading]);

  const stripePromise = getStripePromise();

  return (
    <ModalShell
      onClose={onClose}
      subtitle={
        step === "amount" ? "Choose deposit amount" : step === "pay" ? "Enter payment details" : "Payment complete"
      }
    >
      {step === "success" ? (
        <div className="flex flex-col items-center py-8">
          <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10 mb-4">
            <CheckCircle className="h-8 w-8 text-emerald" />
          </div>
          <p className="text-lg font-semibold">Payment Successful</p>
          <p className="text-sm text-text-muted mt-1">${numericAmount.toFixed(2)} CAD has been added to your wallet</p>
        </div>
      ) : step === "amount" ? (
        <AmountStep
          amount={amount}
          setAmount={setAmount}
          isValid={isValid}
          numericAmount={numericAmount}
          onCancel={onClose}
          onContinue={handleProceedToPay}
          continueLabel={intentLoading ? "Preparing checkout…" : undefined}
        />
      ) : clientSecret && stripePromise ? (
        <Elements stripe={stripePromise} options={getStripeElementsOptions(clientSecret)}>
          <DepositPaymentStep
            customerId={customerId}
            numericAmount={numericAmount}
            onBack={() => {
              setStep("amount");
              setClientSecret(null);
            }}
            onSuccess={(balance) => {
              setStep("success");
              setTimeout(() => onSuccess(balance), 1200);
            }}
            paypalAvailable={paypalAvailable}
          />
        </Elements>
      ) : (
        <div className="flex justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-text-muted" />
        </div>
      )}
    </ModalShell>
  );
}

export function DepositModal(props: DepositModalProps) {
  const stripe = getStripePromise();
  if (!stripe) {
    return <PayPalOnlyDepositForm {...props} />;
  }
  return <DepositForm {...props} />;
}