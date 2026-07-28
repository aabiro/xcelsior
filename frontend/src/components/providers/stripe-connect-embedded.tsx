"use client";

/**
 * Stripe Connect embedded surface via AccountSession.
 * - mode "setup": onboarding + notification banner (incomplete accounts)
 * - mode "manage": notification banner + account management + payouts (active)
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { loadConnectAndInitialize } from "@stripe/connect-js/pure";
import { createProviderAccountSession } from "@/lib/api";
import { getStripePublishableKey } from "@/lib/stripe-client";
import { Button } from "@/components/ui/button";
import { Loader2, RefreshCw, ShieldCheck } from "lucide-react";

export type ConnectEmbedMode = "setup" | "manage";

type Props = {
  providerId: string;
  /** When false, does not auto-mount. */
  active?: boolean;
  mode?: ConnectEmbedMode;
};

type ConnectInstance = ReturnType<typeof loadConnectAndInitialize>;

export function StripeConnectEmbedded({
  providerId,
  active = true,
  mode = "setup",
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const instanceRef = useRef<ConnectInstance | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);

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
    try {
      const instance = loadConnectAndInitialize({
        publishableKey: pk,
        fetchClientSecret: async () => {
          const res = await createProviderAccountSession(providerId);
          if (!res?.client_secret) {
            throw new Error("Account session missing client_secret");
          }
          return res.client_secret;
        },
        appearance: {
          overlays: "dialog",
          variables: {
            colorPrimary: "#34d399",
            colorBackground: "#0b1220",
            colorText: "#e2e8f0",
            colorDanger: "#f87171",
            borderRadius: "10px",
            fontFamily: "ui-sans-serif, system-ui, sans-serif",
          },
        },
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
