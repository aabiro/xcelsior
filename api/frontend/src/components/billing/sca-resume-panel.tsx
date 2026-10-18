"use client";

/**
 * Finish a card payment the bank stopped for verification.
 *
 * P1's only browser detour. An off-session charge refused with
 * `authentication_required` is not failed and not complete — the issuer wants
 * the cardholder, and no retry from our side changes that. The terminal cannot
 * satisfy a bank's challenge, so the person comes here, does one thing, and
 * goes back.
 *
 * The plan is explicit that this is "a detour, not a destination", which is why
 * the success state says to return to the terminal rather than offering onward
 * navigation. Nothing here invites browsing.
 *
 * **The secret is fetched per intent, never listed.** `fetchPendingVerification`
 * says what is waiting and for how much; the `client_secret` arrives only from
 * `resumePendingVerification`, for one intent, behind `billing:write`.
 *
 * **Success comes from the processor, not from this page.** `handleNextAction`
 * resolving means the challenge was satisfied, not that the wallet moved — the
 * webhook credits the balance. The copy says so, because "done" on a screen that
 * has not yet been paid for is how people conclude they have funds they do not.
 */

import { useCallback, useEffect, useState } from "react";
import { Loader2, ShieldCheck, AlertTriangle, Terminal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { getStripePromise } from "@/lib/stripe-client";
import { resumePendingVerification, type PendingVerification } from "@/lib/api";

type Phase = "idle" | "loading" | "confirming" | "done" | "error";

interface Props {
  /** The Stripe PaymentIntent id taken from `?resume=`. */
  intentId: string;
  /** What the listing already knew, so the amount shows before any fetch. */
  known?: PendingVerification | null;
  /** Clear `?resume=` from the URL once the detour is over. */
  onFinished?: () => void;
}

export function ScaResumePanel({ intentId, known, onFinished }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const [amount, setAmount] = useState<number | null>(known?.amount_cad ?? null);
  const [error, setError] = useState<string>("");

  const confirm = useCallback(async () => {
    setError("");
    setPhase("loading");
    try {
      const stripePromise = getStripePromise();
      if (!stripePromise) {
        throw new Error("Card payments are not configured on this deployment.");
      }
      const stripe = await stripePromise;
      if (!stripe) {
        throw new Error("Could not load the payment library.");
      }

      const resumed = await resumePendingVerification(intentId);
      if (!resumed.resumable || !resumed.client_secret) {
        throw new Error(resumed.message || "This payment can no longer be confirmed.");
      }
      setAmount(resumed.amount_cad);
      setPhase("confirming");

      // `handleNextAction`, not `confirmPayment`: the intent already exists and
      // already has a payment method. There is nothing to collect — the only
      // thing outstanding is the issuer's challenge.
      const { error: stripeError } = await stripe.handleNextAction({
        clientSecret: resumed.client_secret,
      });
      if (stripeError) {
        throw new Error(stripeError.message || "The bank did not accept the verification.");
      }
      setPhase("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not complete the verification.");
      setPhase("error");
    }
  }, [intentId]);

  // Never auto-confirm. Opening a link is consent to look, not consent to pay,
  // and a challenge that fires on page load is one the person did not choose.
  useEffect(() => {
    setPhase("idle");
  }, [intentId]);

  if (phase === "done") {
    return (
      <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/5 p-4 sm:p-6">
        <div className="flex items-start gap-3">
          <ShieldCheck className="mt-0.5 h-5 w-5 shrink-0 text-emerald-500" />
          <div className="space-y-2">
            <p className="font-medium">Verification complete</p>
            <p className="text-sm text-muted-foreground">
              Your bank accepted it. The balance updates once the payment
              processor confirms, usually within a minute — you do not need to
              stay on this page.
            </p>
            <p className="flex items-center gap-2 text-sm font-medium">
              <Terminal className="h-4 w-4" aria-hidden />
              You can return to your terminal.
            </p>
            {onFinished ? (
              <Button variant="outline" size="sm" onClick={onFinished}>
                Dismiss
              </Button>
            ) : null}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4 sm:p-6">
      <div className="flex items-start gap-3">
        <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-500" aria-hidden />
        <div className="w-full space-y-3">
          <div className="space-y-1">
            <p className="font-medium">Your bank needs to verify this payment</p>
            <p className="text-sm text-muted-foreground">
              {amount != null ? (
                <>
                  A top-up of{" "}
                  <span className="font-medium text-foreground">
                    ${amount.toFixed(2)} CAD
                  </span>{" "}
                  was <span className="font-medium text-foreground">not charged</span>.
                </>
              ) : (
                <>This top-up was <span className="font-medium text-foreground">not charged</span>.</>
              )}{" "}
              Confirming here completes it. Nothing has been taken from your card
              yet.
            </p>
          </div>

          {error ? (
            <p role="alert" className="text-sm text-destructive">
              {error}
            </p>
          ) : null}

          <Button
            onClick={() => void confirm()}
            disabled={phase === "loading" || phase === "confirming"}
            className="w-full sm:w-auto"
          >
            {phase === "loading" || phase === "confirming" ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" aria-hidden />
                {phase === "confirming" ? "Waiting for your bank…" : "Preparing…"}
              </>
            ) : (
              "Verify with my bank"
            )}
          </Button>
        </div>
      </div>
    </div>
  );
}

export default ScaResumePanel;
