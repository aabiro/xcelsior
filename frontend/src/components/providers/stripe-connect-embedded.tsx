"use client";

/**
 * Minimal Stripe Connect embedded surface (onboarding + requirements banner).
 * Uses AccountSession — no redirect Account Link required for in-dashboard KYC.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { loadConnectAndInitialize } from "@stripe/connect-js";
import { createProviderAccountSession } from "@/lib/api";
import { getStripePublishableKey } from "@/lib/stripe-client";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

type Props = {
  providerId: string;
  /** When true, mounts Connect components into the page. */
  active?: boolean;
};

export function StripeConnectEmbedded({ providerId, active = true }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
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
    try {
      const instance = loadConnectAndInitialize({
        publishableKey: pk,
        fetchClientSecret: async () => {
          const res = await createProviderAccountSession(providerId);
          return res.client_secret;
        },
        appearance: {
          overlays: "dialog",
          variables: {
            colorPrimary: "#34d399",
            colorBackground: "#0b1220",
            colorText: "#e2e8f0",
            colorDanger: "#f87171",
            borderRadius: "8px",
          },
        },
      });
      // Clear previous children
      containerRef.current.innerHTML = "";
      const banner = instance.create("notification-banner");
      const onboarding = instance.create("account-onboarding");
      containerRef.current.appendChild(banner);
      containerRef.current.appendChild(onboarding);
      setReady(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load Stripe Connect");
    } finally {
      setLoading(false);
    }
  }, [providerId]);

  useEffect(() => {
    if (active && providerId) {
      void mount();
    }
  }, [active, providerId, mount]);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs text-text-muted">
          Embedded Stripe setup (KYC, requirements, bank). Stays inside Xcelsior.
        </p>
        <Button type="button" size="sm" variant="outline" onClick={() => void mount()} disabled={loading}>
          {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : "Reload"}
        </Button>
      </div>
      {error && <p className="text-xs text-accent-red">{error}</p>}
      {ready && !error && (
        <p className="text-[11px] text-emerald">Connect components loaded</p>
      )}
      <div ref={containerRef} className="min-h-[120px] rounded-lg border border-border bg-background/60 p-2" />
    </div>
  );
}
