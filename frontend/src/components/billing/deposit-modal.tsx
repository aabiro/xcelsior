"use client";

import { useState, useCallback } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { Elements, CardElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { createPaymentIntent, depositWallet } from "@/lib/api";
import { toast } from "sonner";
import { X, CreditCard, Loader2, CheckCircle, ShieldCheck } from "lucide-react";

const stripePromise = process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
  ? loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY)
  : null;

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

function DirectDepositForm({ customerId, onClose, onSuccess }: DepositModalProps) {
  const [amount, setAmount] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [step, setStep] = useState<"amount" | "success">("amount");

  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 1 && numericAmount <= 10000;

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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
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
          <>
            <div className="mb-4">
              <Label className="mb-2 block text-text-secondary text-xs">Quick select</Label>
              <div className="grid grid-cols-3 gap-2">
                {PRESETS.map((p) => (
                  <button
                    key={p}
                    onClick={() => setAmount(String(p))}
                    className={`rounded-lg border px-3 py-2.5 text-sm font-mono transition-colors ${
                      amount === String(p) ? "border-emerald bg-emerald/10 text-emerald" : "border-border bg-background text-text-secondary hover:border-text-muted"
                    }`}
                  >
                    ${p}
                  </button>
                ))}
              </div>
            </div>
            <div className="mb-6">
              <Label htmlFor="amount" className="mb-1.5 block text-text-secondary text-xs">Custom amount (CAD)</Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm">$</span>
                <Input id="amount" type="number" min="1" max="10000" step="0.01" placeholder="0.00" value={amount} onChange={(e) => setAmount(e.target.value)} className="pl-8 font-mono" />
              </div>
              {amount && !isValid && <p className="text-xs text-accent-red mt-1">Enter between $1.00 and $10,000.00</p>}
            </div>
            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={onClose}>Cancel</Button>
              <Button variant="success" className="flex-1" onClick={handleDeposit} disabled={!isValid || submitting}>
                {submitting ? <><Loader2 className="h-4 w-4 animate-spin" /> Processing…</> : <>Deposit ${isValid ? numericAmount.toFixed(2) : "0.00"}</>}
              </Button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function DepositForm({ customerId, onClose, onSuccess }: DepositModalProps) {
  const stripe = useStripe();
  const elements = useElements();

  const [amount, setAmount] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [step, setStep] = useState<"amount" | "card" | "success">("amount");
  const [cardComplete, setCardComplete] = useState(false);

  const numericAmount = parseFloat(amount);
  const isValid = !isNaN(numericAmount) && numericAmount >= 1 && numericAmount <= 10000;

  const handleProceedToCard = () => {
    if (isValid) setStep("card");
  };

  const handlePayment = useCallback(async () => {
    if (!isValid || submitting) return;

    // If Stripe Elements not available, fall back to direct wallet deposit
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
      // 1. Create PaymentIntent on backend
      const { intent } = await createPaymentIntent(customerId, numericAmount);
      const clientSecret = intent.client_secret;

      if (!clientSecret || clientSecret.startsWith("stub_")) {
        // Stub mode — directly credit wallet
        await depositWallet(customerId, numericAmount);
        toast.success(`$${numericAmount.toFixed(2)} CAD added to wallet`);
        setStep("success");
        setTimeout(() => onSuccess(numericAmount), 1200);
        return;
      }

      // 2. Confirm payment with Stripe.js
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
        // Wallet will be credited via webhook, but also credit directly for instant UX
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald/10">
              <CreditCard className="h-5 w-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Add Credits</h2>
              <p className="text-xs text-text-muted">
                {step === "amount" ? "Choose deposit amount" : step === "card" ? "Enter payment details" : "Payment complete"}
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
            <p className="text-lg font-semibold">Payment Successful</p>
            <p className="text-sm text-text-muted mt-1">
              ${numericAmount.toFixed(2)} CAD has been added to your wallet
            </p>
          </div>
        ) : step === "amount" ? (
          <>
            {/* Preset amounts */}
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

            {/* Custom amount */}
            <div className="mb-6">
              <Label htmlFor="amount" className="mb-1.5 block text-text-secondary text-xs">
                Custom amount (CAD)
              </Label>
              <div className="relative">
                <span className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted font-mono text-sm">$</span>
                <Input
                  id="amount"
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
                <p className="text-xs text-accent-red mt-1">Enter between $1.00 and $10,000.00</p>
              )}
            </div>

            <div className="flex gap-3">
              <Button variant="outline" className="flex-1" onClick={onClose}>Cancel</Button>
              <Button variant="success" className="flex-1" onClick={handleProceedToCard} disabled={!isValid}>
                Continue — ${isValid ? numericAmount.toFixed(2) : "0.00"}
              </Button>
            </div>
          </>
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

            {/* Stripe Card Element */}
            <div className="mb-4">
              <Label className="mb-2 block text-text-secondary text-xs">Card details</Label>
              <div className="rounded-lg border border-border bg-background p-3.5">
                <CardElement
                  options={CARD_ELEMENT_OPTIONS}
                  onChange={(e) => setCardComplete(e.complete)}
                />
              </div>
            </div>

            {/* Security note */}
            <div className="mb-6 flex items-start gap-2 rounded-lg bg-ice/5 border border-ice/10 p-3">
              <ShieldCheck className="h-4 w-4 text-ice mt-0.5 shrink-0" />
              <p className="text-xs text-text-secondary">
                Payment processed securely by Stripe. Card details never touch our servers.
              </p>
            </div>

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
  if (!stripePromise) {
    // Stripe not configured — direct wallet deposit only
    return <DirectDepositForm {...props} />;
  }
  return (
    <Elements stripe={stripePromise}>
      <DepositForm {...props} />
    </Elements>
  );
}
