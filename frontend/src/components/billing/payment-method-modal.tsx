"use client";

import { useCallback, useEffect, useState } from "react";
import { Elements, PaymentElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { Button } from "@/components/ui/button";
import { createSetupIntent } from "@/lib/api";
import { getStripeElementsOptions, STRIPE_PAYMENT_ELEMENT_OPTIONS } from "@/lib/stripe-appearance";
import { getStripePromise } from "@/lib/stripe-client";
import { toast } from "sonner";
import { X, CreditCard, Loader2, ShieldCheck, Sparkles } from "lucide-react";
import posthog from "posthog-js";

interface PaymentMethodModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

function ModalShell({ onClose, children }: { onClose: () => void; children: React.ReactNode }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="brand-top-accent w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald/20 to-accent-cyan/10 ring-1 ring-emerald/20">
              <CreditCard className="h-5 w-5 text-emerald" />
            </div>
            <div>
              <h2 className="text-lg font-semibold">Add a card</h2>
              <p className="text-xs text-text-muted">Save a card for automatic top-ups</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-text-muted transition-colors hover:bg-background hover:text-text-primary"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {children}
      </div>
    </div>
  );
}

function AddCardForm({ onClose, onSuccess, clientSecret }: PaymentMethodModalProps & { clientSecret: string }) {
  const stripe = useStripe();
  const elements = useElements();
  const [paymentReady, setPaymentReady] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const handleSave = useCallback(async () => {
    if (!stripe || !elements || submitting) return;
    setSubmitting(true);
    try {
      const { error, setupIntent } = await stripe.confirmSetup({
        elements,
        confirmParams: { return_url: window.location.href },
        redirect: "if_required",
      });
      if (error) {
        toast.error(error.message || "Could not save card");
        return;
      }
      if (setupIntent?.status === "succeeded") {
        posthog.capture("payment_method_added", { payment_type: "card" });
        toast.success("Card saved");
        onSuccess();
        onClose();
      }
    } catch (err) {
      posthog.captureException(err instanceof Error ? err : new Error(String(err)));
      toast.error(err instanceof Error ? err.message : "Could not save card");
    } finally {
      setSubmitting(false);
    }
  }, [stripe, elements, submitting, onSuccess, onClose]);

  return (
    <>
      <div className="mb-4 rounded-xl border border-border bg-background/60 p-4">
        <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-text-muted">
          <Sparkles className="h-3.5 w-3.5 text-accent-cyan" />
          Payment method
        </div>
        <PaymentElement
          onChange={(e) => setPaymentReady(e.complete)}
          options={STRIPE_PAYMENT_ELEMENT_OPTIONS}
        />
      </div>
      <div className="mb-4 flex items-start gap-2 rounded-lg border border-accent-cyan/15 bg-accent-cyan/5 p-3">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-accent-cyan" />
        <p className="text-xs text-text-secondary">
          Embedded checkout — your card is stored securely with Stripe for off-session wallet top-ups.
        </p>
      </div>
      <div className="flex gap-3">
        <Button variant="outline" className="flex-1" onClick={onClose} disabled={submitting}>
          Cancel
        </Button>
        <Button className="flex-1" onClick={handleSave} disabled={!paymentReady || submitting}>
          {submitting ? (
            <><Loader2 className="h-4 w-4 animate-spin" /> Saving…</>
          ) : (
            "Save card"
          )}
        </Button>
      </div>
    </>
  );
}

export function PaymentMethodModal({ onClose, onSuccess }: PaymentMethodModalProps) {
  const [clientSecret, setClientSecret] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const stripePromise = getStripePromise();

  useEffect(() => {
    createSetupIntent()
      .then((res) => {
        if (!res.client_secret) {
          toast.error("Saving cards is temporarily unavailable.");
          onClose();
          return;
        }
        setClientSecret(res.client_secret);
      })
      .catch((err) => {
        toast.error(err instanceof Error ? err.message : "Could not start card setup");
        onClose();
      })
      .finally(() => setLoading(false));
  }, [onClose]);

  if (!stripePromise) {
    return (
      <ModalShell onClose={onClose}>
        <p className="text-sm text-text-secondary">Card payments are not configured.</p>
        <Button variant="outline" className="mt-4 w-full" onClick={onClose}>Close</Button>
      </ModalShell>
    );
  }

  if (loading || !clientSecret) {
    return (
      <ModalShell onClose={onClose}>
        <div className="flex justify-center py-8">
          <Loader2 className="h-8 w-8 animate-spin text-text-muted" />
        </div>
      </ModalShell>
    );
  }

  return (
    <ModalShell onClose={onClose}>
      <Elements stripe={stripePromise} options={getStripeElementsOptions(clientSecret)}>
        <AddCardForm onClose={onClose} onSuccess={onSuccess} clientSecret={clientSecret} />
      </Elements>
    </ModalShell>
  );
}