"use client";

/**
 * Stripe Connect embedded surface via AccountSession.
 * - mode "setup": onboarding + notification banner (incomplete accounts)
 * - mode "manage": notification banner + account management + payouts (active)
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { loadConnectAndInitialize } from "@stripe/connect-js/pure";
import posthog from "posthog-js";
import { ApiError, createProviderAccountSession } from "@/lib/api";
import { getStripePublishableKey } from "@/lib/stripe-client";
import { STRIPE_CONNECT_APPEARANCE } from "@/lib/stripe-appearance";
import { Button } from "@/components/ui/button";
import { ExternalLink, Loader2, RefreshCw, ShieldCheck } from "lucide-react";

export type ConnectEmbedMode = "setup" | "manage";

type Props = {
  providerId: string;
  /** When false, does not auto-mount. */
  active?: boolean;
  mode?: ConnectEmbedMode;
  /** Starts hosted onboarding again; the server replaces an account Stripe no longer recognizes. */
  onRestartOnboarding?: () => void;
  restartingOnboarding?: boolean;
};

type ConnectInstance = ReturnType<typeof loadConnectAndInitialize>;

export function StripeConnectEmbedded({
  providerId,
  active = true,
  mode = "setup",
  onRestartOnboarding,
  restartingOnboarding = false,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const instanceRef = useRef<ConnectInstance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [needsRestart, setNeedsRestart] = useState(false);

  const mount = useCallback(async () => {
    if (!providerId || !containerRef.current) return;
    const pk = getStripePublishableKey();
    if (!pk) {
      setError("Stripe publishable key not configured");
      return;
    }
    setLoading(true);
    setError(null);
    setReady(false);
    setNeedsRestart(false);
    const fetchSecret = async () => {
      const res = await createProviderAccountSession(providerId);
      if (!res?.client_secret) {
        throw new Error("Account session missing client_secret");
      }
      return res.client_secret;
    };
    try {
      // Fetch the first secret here, inside the try/catch. A 409 means Stripe
      // no longer recognizes the account: that is an expected state, so show
      // the restart action and do not hand connect-js a rejected promise.
      let firstSecret: string | null;
      try {
        firstSecret = await fetchSecret();
      } catch (err) {
        if (err instanceof ApiError && err.status === 409) {
          if (containerRef.current) containerRef.current.innerHTML = "";
          setError(err.message);
          setNeedsRestart(true);
          return;
        }
        throw err;
      }
      const instance = loadConnectAndInitialize({
        publishableKey: pk,
        // connect-js calls this again when the session expires, outside
        // mount's try/catch, so surface the failure in the inline error box
        // before rethrowing.
        fetchClientSecret: async () => {
          if (firstSecret) {
            const secret = firstSecret;
            firstSecret = null;
            return secret;
          }
          try {
            return await fetchSecret();
          } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to start Stripe session");
            setNeedsRestart(err instanceof ApiError && err.status === 409);
            setReady(false);
            throw err;
          }
        },
        // Shared brand palette so the payouts portal matches the rest of the
        // app's embedded Stripe surfaces (see lib/stripe-appearance.ts).
        appearance: STRIPE_CONNECT_APPEARANCE,
      });
      instanceRef.current = instance;
      containerRef.current.innerHTML = "";

      // Always mount requirements banner so restricted accounts surface actions.
      const banner = instance.create("notification-banner");
      containerRef.current.appendChild(banner as unknown as Node);

      if (mode === "setup") {
        const onboarding = instance.create("account-onboarding");
        containerRef.current.appendChild(onboarding as unknown as Node);
      } else {
        const management = instance.create("account-management");
        const payouts = instance.create("payouts");
        containerRef.current.appendChild(management as unknown as Node);
        containerRef.current.appendChild(payouts as unknown as Node);
      }
      setReady(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load Stripe Connect");
      setReady(false);
    } finally {
      setLoading(false);
    }
  }, [providerId, mode]);

  useEffect(() => {
    if (active && providerId) {
      void mount();
    }
    return () => {
      instanceRef.current = null;
    };
  }, [active, providerId, mount]);

  return (
    <div className="space-y-3 rounded-xl border border-border/80 bg-gradient-to-b from-surface/80 to-background/40 p-3 sm:p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2 text-sm font-medium text-text-primary">
            <ShieldCheck className="h-4 w-4 shrink-0 text-emerald" />
            {mode === "setup" ? "Stripe embedded setup" : "Stripe account & payouts"}
          </div>
          <p className="mt-0.5 text-xs text-text-muted leading-relaxed">
            {mode === "setup"
              ? "Complete identity and bank details without leaving Xcelsior. Requirements update automatically."
              : "Manage payout details and resolve open requirements without leaving the dashboard."}
          </p>
        </div>
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="shrink-0"
          onClick={() => void mount()}
          disabled={loading}
          aria-label="Reload Stripe Connect components"
        >
          {loading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <RefreshCw className="h-3.5 w-3.5" />
          )}
        </Button>
      </div>
      {error && (
        <div className="rounded-lg border border-accent-red/30 bg-accent-red/10 px-3 py-2 text-xs text-accent-red">
          {error}
        </div>
      )}
      {needsRestart && onRestartOnboarding && (
        <Button
          type="button"
          size="sm"
          variant="outline"
          className="w-full"
          onClick={() => {
            posthog.capture("stripe_onboarding_restarted", { source: "payouts_panel", mode });
            onRestartOnboarding();
          }}
          disabled={restartingOnboarding}
        >
          {restartingOnboarding ? (
            <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Redirecting to Stripe…</>
          ) : (
            <><ExternalLink className="h-3.5 w-3.5" /> Restart Stripe onboarding</>
          )}
        </Button>
      )}
      {ready && !error && (
        <p className="text-[11px] font-medium text-emerald">Connect components ready</p>
      )}
      <div
        ref={containerRef}
        className="min-h-[140px] rounded-lg border border-border/60 bg-background/70 p-2 sm:p-3"
        data-testid="stripe-connect-embed"
        data-mode={mode}
      />
    </div>
  );
}
