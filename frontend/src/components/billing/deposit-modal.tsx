"use client";

import { useState, useCallback, useEffect } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { Elements, CardElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { createPaymentIntent, depositWallet, checkPayPalEnabled, createPayPalOrder, capturePayPalOrder } from "@/lib/api";
import { toast } from "sonner";
import { X, CreditCard, Loader2, CheckCircle, ShieldCheck } from "lucide-react";

let stripePromise: ReturnType<typeof loadStripe> | null = null;
function getStripePromise() {
  if (!process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY) return null;
  if (!stripePromise) {
    stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY);
  }
  return stripePromise;
}

interface DepositModalProps {
  customerId: string;
  onClose: () => void;
  onSuccess: (newBalance: number) => void;
}

const PRESETS = [10, 25, 50, 100, 250, 500];

const CARD_ELEMENT_OPTIONS = {
  style: {
    base: {
      color: "#e2e8f0",
      fontFamily: "ui-monospace, monospace",
      fontSize: "14px",
      "::placeholder": { color: "#64748b" },
      iconColor: "#64748b",
    },
    invalid: { color: "#dc2626", iconColor: "#dc2626" },
  },
};

/* ── Amount selector (shared between Stripe and direct flows) ──────── */

function AmountStep({
  amount,
  setAmount,
  isValid,
  numericAmount,
  onCancel,
  onContinue,
}: {
  amount: string;
  setAmount: (v: string) => void;
  isValid: boolean;
  numericAmount: number;
  onCancel: () => void;
  onContinue: () => void;
}) {
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
            type="number"
            min="5"
            max="10000"
            step="0.01"
            placeholder="0.00"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            className="pl-8 font-mono"
          />
        </div>
        {amount && !isValid && (
          <p className="text-xs text-accent-red mt-1">Minimum top-up is $5.00 (max $10,000.00)</p>
        )}
      </div>
      <div className="flex gap-3">
        <Button variant="outline" className="flex-1" onClick={onCancel}>Cancel</Button>
        <Button variant="success" className="flex-1" onClick={onContinue} disabled={!isValid}>
          Continue — ${isValid ? numericAmount.toFixed(2) : "0.00"}
        </Button>
      </div>
    </>
  );
}

/* ── PayPal button (opens popup) ───────────────────────────────────── */

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
      // 1. Create order on backend
      const { order_id } = await createPayPalOrder(customerId, amountCad);

      // 2. Open PayPal approval popup
      const approvalUrl = `https://www.${
        process.env.NEXT_PUBLIC_PAYPAL_MODE === "live" ? "" : "sandbox."
      }paypal.com/checkoutnow?token=${order_id}`;

      const popup = window.open(approvalUrl, "paypal_checkout", "width=500,height=700,left=200,top=100");
      if (!popup) {
        toast.error("Please allow popups for PayPal checkout");
        setLoading(false);
        return;
      }

      // 3. Poll for popup close (user approved or cancelled)
      const poll = setInterval(async () => {
        if (!popup.closed) return;
        clearInterval(poll);

        // Try to capture — PayPal will fail if user cancelled
        try {
          const result = await capturePayPalOrder(customerId, order_id);
          toast.success(`$${result.amount_cad.toFixed(2)} CAD added via PayPal`);
          onSuccess(result.balance_cad);
        } catch {
          // User likely cancelled the PayPal flow
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
      {loading ? (
        <Loader2 className="h-4 w-4 animate-spin" />
      ) : (
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M7.076 21.337H2.47a.641.641 0 01-.633-.74L4.944 3.72a.774.774 0 01.763-.648h6.486c2.15 0 3.652.457 4.462 1.358.752.836.985 2.046.693 3.597l-.014.08v.717l.557.316c.46.244.826.532 1.1.87.364.45.599.997.697 1.626.102.645.066 1.41-.104 2.274-.196.997-.514 1.866-.944 2.578a5.267 5.267 0 01-1.456 1.606 5.605 5.605 0 01-1.92.89c-.714.193-1.53.29-2.43.29h-.577a1.735 1.735 0 00-1.715 1.467l-.043.223-.724 4.588-.033.163a.186.186 0 01-.046.106.144.144 0 01-.1.038z" fill="#253B80"/>
          <path d="M19.451 8.262c-.013.08-.027.163-.043.248-1.383 7.1-6.12 9.556-12.173 9.556H5.112a1.498 1.498 0 00-1.48 1.267l-1.2 7.605a.78.78 0 00.77.904h5.406a1.312 1.312 0 001.296-1.108l.053-.278.028-.146.719-4.56a1.735 1.735 0 011.715-1.467h.577c3.5 0 6.24-1.422 7.04-5.537.334-1.717.161-3.152-.722-4.16a3.452 3.452 0 00-.992-.724z" fill="#179BD7"/>
          <path d="M18.267 7.793a8.5 8.5 0 00-1.052-.233 13.34 13.34 0 00-2.12-.156h-6.43a1.27 1.27 0 00-1.254 1.073l-1.37 8.686-.039.252a1.498 1.498 0 011.48-1.267h2.123c6.053 0 10.79-2.458 12.173-9.556.041-.21.075-.414.104-.611a5.364 5.364 0 00-1.58-.654 10.578 10.578 0 00-2.035-.534z" fill="#222D65"/>
        </svg>
      )}
      {loading ? "Processing…" : "Pay with PayPal"}
    </button>
  );
}

/* ── Direct deposit (no Stripe) ────────────────────────────────────── */

function DirectDepositForm({ customerId, onClose, onSuccess }: DepositModalProps) {
  const [amount, setAmount] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [step, setStep] = useState<"amount" | "success">("amount");

  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 5 && numericAmount <= 10000;

  const handleDeposit = useCallback(async () => {
    if (!isValid || submitting) return;
    setSubmitting(true);
    try {
      const res = await depositWallet(customerId, numericAmount);
      toast.success(`$${numericAmount.toFixed(2)} CAD added to wallet`);
      setStep("success");
      setTimeout(() => onSuccess(res.balance_cad), 1200);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Deposit failed");
    } finally {
      setSubmitting(false);
    }
  }, [customerId, numericAmount, isValid, submitting, onSuccess]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald/10">
              <CreditCard className="h-5 w-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Add Credits</h2>
              <p className="text-xs text-text-muted">
                {step === "amount" ? "Choose deposit amount" : "Payment complete"}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-text-muted hover:bg-background hover:text-text-primary transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>

        {step === "success" ? (
          <div className="flex flex-col items-center py-8">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10 mb-4">
              <CheckCircle className="h-8 w-8 text-emerald" />
            </div>
            <p className="text-lg font-semibold">Payment Successful</p>
            <p className="text-sm text-text-muted mt-1">${numericAmount.toFixed(2)} CAD has been added to your wallet</p>
          </div>
        ) : (
          <AmountStep
            amount={amount}
            setAmount={setAmount}
            isValid={isValid}
            numericAmount={numericAmount}
            onCancel={onClose}
            onContinue={handleDeposit}
          />
        )}
      </div>
    </div>
  );
}

/* ── Card + PayPal deposit form ────────────────────────────────────── */

function DepositForm({ customerId, onClose, onSuccess }: DepositModalProps) {
  const stripe = useStripe();
  const elements = useElements();

  const [amount, setAmount] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [step, setStep] = useState<"amount" | "pay" | "success">("amount");
  const [cardComplete, setCardComplete] = useState(false);
  const [paypalAvailable, setPaypalAvailable] = useState(false);

  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 5 && numericAmount <= 10000;

  useEffect(() => {
    checkPayPalEnabled()
      .then((r) => setPaypalAvailable(r.enabled))
      .catch(() => {});
  }, []);

  const handleProceedToPay = () => {
    if (isValid) setStep("pay");
  };

  const handlePayment = useCallback(async () => {
    if (!isValid || submitting) return;

    if (!stripe || !elements) {
      setSubmitting(true);
      try {
        const res = await depositWallet(customerId, numericAmount);
        toast.success(`$${numericAmount.toFixed(2)} CAD added to wallet`);
        setStep("success");
        setTimeout(() => onSuccess(res.balance_cad), 1200);
      } catch (err) {
        toast.error(err instanceof Error ? err.message : "Deposit failed");
      } finally {
        setSubmitting(false);
      }
      return;
    }

    setSubmitting(true);
    try {
      const { intent } = await createPaymentIntent(customerId, numericAmount);
      const clientSecret = intent.client_secret;
      if (!clientSecret) {
        toast.error("Card payments are temporarily unavailable on the server.");
        return;
      }

      const cardElement = elements.getElement(CardElement);
      if (!cardElement) throw new Error("Card element not mounted");

      const { error, paymentIntent } = await stripe.confirmCardPayment(clientSecret, {
        payment_method: { card: cardElement },
      });

      if (error) {
        toast.error(error.message || "Payment failed");
        return;
      }

      if (paymentIntent?.status === "succeeded") {
        try { await depositWallet(customerId, numericAmount); } catch { /* webhook will handle */ }
        toast.success(`$${numericAmount.toFixed(2)} CAD added to wallet`);
        setStep("success");
        setTimeout(() => onSuccess(numericAmount), 1200);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Payment failed");
    } finally {
      setSubmitting(false);
    }
  }, [stripe, elements, customerId, numericAmount, isValid, submitting, onSuccess]);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald/10">
              <CreditCard className="h-5 w-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Add Credits</h2>
              <p className="text-xs text-text-muted">
                {step === "amount" ? "Choose deposit amount" : step === "pay" ? "Enter payment details" : "Payment complete"}
              </p>
            </div>
          </div>
          <button onClick={onClose} className="rounded-lg p-1.5 text-text-muted hover:bg-background hover:text-text-primary transition-colors">
            <X className="h-4 w-4" />
          </button>
        </div>

        {step === "success" ? (
          <div className="flex flex-col items-center py-8">
            <div className="flex h-16 w-16 items-center justify-center rounded-full bg-emerald/10 mb-4">
              <CheckCircle className="h-8 w-8 text-emerald" />
            </div>
            <p className="text-lg font-semibold">Payment Successful</p>
            <p className="text-sm text-text-muted mt-1">
              ${numericAmount.toFixed(2)} CAD has been added to your wallet
            </p>
          </div>
        ) : step === "amount" ? (
          <AmountStep
            amount={amount}
            setAmount={setAmount}
            isValid={isValid}
            numericAmount={numericAmount}
            onCancel={onClose}
            onContinue={handleProceedToPay}
          />
        ) : (
          <>
            {/* Amount summary */}
            <div className="mb-4 rounded-lg border border-border bg-background p-3 flex items-center justify-between">
              <span className="text-sm text-text-secondary">Deposit amount</span>
              <div className="flex items-center gap-2">
                <span className="text-lg font-bold font-mono">${numericAmount.toFixed(2)}</span>
                <span className="text-xs text-text-muted">CAD</span>
                <button onClick={() => setStep("amount")} className="text-xs text-ice hover:underline ml-1">
                  Change
                </button>
              </div>
            </div>
            <div className="mb-4">
              <Label className="mb-2 block text-text-secondary text-xs">Card details</Label>
              <div className="rounded-lg border border-border bg-background p-3.5">
                <CardElement
                  options={CARD_ELEMENT_OPTIONS}
                  onChange={(e) => setCardComplete(e.complete)}
                />
              </div>
            </div>

            <div className="mb-4 flex items-start gap-2 rounded-lg bg-ice/5 border border-ice/10 p-3">
              <ShieldCheck className="h-4 w-4 text-ice mt-0.5 shrink-0" />
              <p className="text-xs text-text-secondary">
                Payments are processed securely. Card details never touch our servers.
              </p>
            </div>

            {paypalAvailable && (
              <div className="mb-6 rounded-xl border border-border/70 bg-background/70 p-4">
                <div className="mb-3 flex items-center gap-3">
                  <div className="h-px flex-1 bg-border" />
                  <span className="text-[11px] font-medium uppercase tracking-[0.18em] text-text-muted">
                    Or use PayPal
                  </span>
                  <div className="h-px flex-1 bg-border" />
                </div>
                <PayPalButton
                  customerId={customerId}
                  amountCad={numericAmount}
                  onSuccess={(balance) => {
                    setStep("success");
                    setTimeout(() => onSuccess(balance), 1200);
                  }}
                  disabled={submitting}
                />
                <p className="mt-3 text-xs text-text-secondary">
                  Prefer PayPal? Complete the same top-up in a separate PayPal window.
                </p>
              </div>
            )}

            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={() => setStep("amount")} disabled={submitting}>
                Back
              </Button>
              <Button
                variant="success"
                className="flex-1"
                onClick={handlePayment}
                disabled={!cardComplete || submitting}
              >
                {submitting ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Processing…</>
                ) : (
                  <>Pay ${numericAmount.toFixed(2)} CAD</>
                )}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function DepositModal(props: DepositModalProps) {
  const stripe = getStripePromise();
  if (!stripe) {
    return <DirectDepositForm {...props} />;
  }
  return (
    <Elements stripe={stripe}>
      <DepositForm {...props} />
    </Elements>
  );
}
