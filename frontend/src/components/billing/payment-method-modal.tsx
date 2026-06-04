"use client";

import { useCallback, useState } from "react";
import { loadStripe } from "@stripe/stripe-js";
import { Elements, CardElement, useStripe, useElements } from "@stripe/react-stripe-js";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/input";
import { createSetupIntent } from "@/lib/api";
import { toast } from "sonner";
import { X, CreditCard, Loader2, ShieldCheck } from "lucide-react";

let stripePromise: ReturnType<typeof loadStripe> | null = null;
function getStripePromise() {
  if (!process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY) return null;
  if (!stripePromise) {
    stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY);
  }
  return stripePromise;
}

interface PaymentMethodModalProps {
  onClose: () => void;
  /** Called after a card is successfully saved, so the parent can refresh. */
  onSuccess: () => void;
}

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

function ModalShell({ onClose, children }: { onClose: () => void; children: React.ReactNode }) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald/10">
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

function AddCardForm({ onClose, onSuccess }: PaymentMethodModalProps) {
  const stripe = useStripe();
  const elements = useElements();
  const [cardComplete, setCardComplete] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  const handleSave = useCallback(async () => {
    if (!stripe || !elements || submitting) return;
    setSubmitting(true);
    try {
      const { client_secret } = await createSetupIntent();
      if (!client_secret) {
        toast.error("Saving cards is temporarily unavailable on the server.");
        return;
      }
      const cardElement = elements.getElement(CardElement);
      if (!cardElement) throw new Error("Card element not mounted");

      const { error, setupIntent } = await stripe.confirmCardSetup(client_secret, {
        payment_method: { card: cardElement },
      });
      if (error) {
        toast.error(error.message || "Could not save card");
        return;
      }
      if (setupIntent?.status === "succeeded") {
        toast.success("Card saved");
        onSuccess();
        onClose();
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not save card");
    } finally {
      setSubmitting(false);
    }
  }, [stripe, elements, submitting, onSuccess, onClose]);

  return (
    <>
      <div className="mb-4">
        <Label className="mb-2 block text-xs text-text-secondary">Card details</Label>
        <div className="rounded-lg border border-border bg-background p-3.5">
          <CardElement options={CARD_ELEMENT_OPTIONS} onChange={(e) => setCardComplete(e.complete)} />
        </div>
      </div>
      <div className="mb-4 flex items-start gap-2 rounded-lg border border-ice/10 bg-ice/5 p-3">
        <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-ice" />
        <p className="text-xs text-text-secondary">
          Your card is stored securely by Stripe and can be charged automatically only when you
          enable auto-reload. Card details never touch our servers.
        </p>
      </div>
      <div className="flex gap-3">
        <Button variant="outline" className="flex-1" onClick={onClose} disabled={submitting}>
          Cancel
        </Button>
        <Button
          variant="success"
          className="flex-1"
          onClick={handleSave}
          disabled={!cardComplete || submitting}
        >
          {submitting ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" /> Saving…
            </>
          ) : (
            "Save card"
          )}
        </Button>
      </div>
    </>
  );
}

export function PaymentMethodModal(props: PaymentMethodModalProps) {
  const stripe = getStripePromise();
  if (!stripe) {
    return (
      <ModalShell onClose={props.onClose}>
        <p className="text-sm text-text-secondary">
          Card payments are not configured on this deployment, so cards cannot be saved. Set
          <code className="mx-1 rounded bg-background px-1 py-0.5 text-xs">
            NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY
          </code>
          to enable this.
        </p>
        <div className="mt-6 flex justify-end">
          <Button variant="outline" onClick={props.onClose}>
            Close
          </Button>
        </div>
      </ModalShell>
    );
  }
  return (
    <Elements stripe={stripe}>
      <ModalShell onClose={props.onClose}>
        <AddCardForm {...props} />
      </ModalShell>
    </Elements>
  );
}
