"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  CheckCircle,
  ExternalLink,
  Loader2,
  LinkIcon,
  RefreshCw,
  Unlink,
  Zap,
  Percent,
  Layers,
  Info,
} from "lucide-react";
import { toast } from "sonner";
import * as api from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { PayPalLogo } from "@/components/ui/payment-logos";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

export interface PayPalProviderState {
  enabled: boolean;
  status: string;
  onboarded_at?: number;
}

interface PayPalConnectCardProps {
  providerId: string;
  paypal?: PayPalProviderState | null;
  platformPayPalEnabled?: boolean;
  onUpdated?: () => void;
}

const STATUS_STYLE: Record<string, string> = {
  active: "text-emerald",
  onboarding: "text-[#009cde]",
  not_started: "text-text-muted",
  restricted: "text-accent-red",
};

const BENEFIT_ICONS = [Zap, Percent, Layers] as const;
const BENEFIT_KEYS = [
  "dash.earnings.paypal_benefit_instant",
  "dash.earnings.paypal_benefit_fee",
  "dash.earnings.paypal_benefit_dual",
] as const;

export function PayPalConnectCard({
  providerId,
  paypal,
  platformPayPalEnabled = true,
  onUpdated,
}: PayPalConnectCardProps) {
  const { t } = useLocale();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [polling, setPolling] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [justConnected, setJustConnected] = useState(false);
  const [handledReturn, setHandledReturn] = useState(false);
  const [confirmDisconnect, setConfirmDisconnect] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);

  const status = paypal?.status || "not_started";
  const isActive = status === "active";

  const statusLabel =
    status === "onboarding"
      ? t("dash.earnings.paypal_status_onboarding")
      : status === "not_started"
        ? t("dash.earnings.paypal_status_setup")
        : status.charAt(0).toUpperCase() + status.slice(1);

  const pollUntilActive = useCallback(async () => {
    setPolling(true);
    let attempts = 0;
    const maxAttempts = 15;
    const tick = async () => {
      attempts += 1;
      try {
        const res = await api.refreshPayPalProvider(providerId);
        if (res.paypal?.status === "active") {
          setJustConnected(true);
          setPolling(false);
          onUpdated?.();
          toast.success(t("dash.earnings.paypal_connected_toast"), { duration: 10000 });
          return;
        }
      } catch {
        /* retry */
      }
      if (attempts >= maxAttempts) {
        setPolling(false);
        toast.info(t("dash.earnings.paypal_verify_timeout"), { duration: 10000 });
        return;
      }
      window.setTimeout(tick, 2000);
    };
    void tick();
  }, [providerId, onUpdated, t]);

  useEffect(() => {
    const paypalState = searchParams.get("paypal");
    if (!paypalState || handledReturn || !providerId) return;

    if (paypalState === "return") {
      toast.info(t("dash.earnings.paypal_verifying_toast"), { duration: 4000 });
      void pollUntilActive();
    } else if (paypalState === "refresh") {
      toast.info(t("dash.earnings.paypal_incomplete"), { duration: 8000 });
    }

    const next = new URLSearchParams(searchParams.toString());
    next.delete("paypal");
    next.delete("provider");
    const query = next.toString();
    router.replace(query ? `/dashboard/earnings?${query}` : "/dashboard/earnings", { scroll: false });
    setHandledReturn(true);
  }, [searchParams, handledReturn, providerId, router, pollUntilActive, t]);

  const handleConnect = async () => {
    if (!providerId) return;
    setError(null);
    setLoading(true);
    try {
      const res = await api.startPayPalProviderOnboard(providerId);
      if (res.onboarding_url) {
        window.location.href = res.onboarding_url;
        window.setTimeout(() => setLoading(false), 5000);
      } else {
        setError(t("dash.earnings.paypal_link_error"));
        setLoading(false);
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : t("dash.earnings.paypal_start_error");
      setError(msg);
      toast.error(msg, { duration: 8000 });
      setLoading(false);
    }
  };

  const handleDisconnect = async () => {
    setConfirmDisconnect(false);
    setDisconnecting(true);
    try {
      await api.disconnectPayPalProvider(providerId);
      setJustConnected(false);
      toast.success(t("dash.earnings.paypal_disconnected"));
      onUpdated?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.earnings.disconnect_failed"));
    } finally {
      setDisconnecting(false);
    }
  };

  const handleRefresh = async () => {
    setLoading(true);
    try {
      const res = await api.refreshPayPalProvider(providerId);
      if (res.paypal?.status === "active") {
        setJustConnected(true);
        toast.success(t("dash.earnings.paypal_connected_toast"));
      } else {
        toast.info(t("dash.earnings.paypal_still_pending"));
      }
      onUpdated?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.earnings.paypal_refresh_error"));
    } finally {
      setLoading(false);
    }
  };

  if (!platformPayPalEnabled) {
    return (
      <Card className="border-border/60 bg-surface/40 opacity-80">
        <CardHeader className="pb-2">
          <CardTitle className="text-base flex items-center gap-2.5">
            <PayPalLogo className="h-4" />
            <span className="text-text-secondary font-normal text-sm">{t("dash.earnings.paypal_subtitle")}</span>
          </CardTitle>
          <CardDescription>{t("dash.earnings.paypal_unavailable")}</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card
      className={
        isActive
          ? "border-[#009cde]/35 ring-1 ring-[#009cde]/15 bg-gradient-to-br from-surface via-surface to-[#003087]/[0.04]"
          : "border-border/80"
      }
    >
      <CardHeader className="pb-2">
        <CardTitle className="text-base flex items-center gap-2.5">
          <PayPalLogo className="h-4" />
          <span className="text-text-secondary font-normal text-sm">{t("dash.earnings.paypal_subtitle")}</span>
        </CardTitle>
        <CardDescription className="mt-1">{t("dash.earnings.paypal_desc")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {isActive ? (
          <div
            className={`rounded-lg border px-3 py-3 transition-colors ${
              justConnected ? "border-[#009cde]/40 bg-[#009cde]/8" : "border-[#009cde]/20 bg-[#009cde]/5"
            }`}
          >
            <div className="flex items-start gap-3">
              <div className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#009cde]/15">
                <CheckCircle className="h-4.5 w-4.5 text-[#009cde]" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-semibold text-text-primary">
                    {justConnected ? t("dash.earnings.paypal_just_connected") : t("dash.earnings.paypal_ready")}
                  </p>
                  <Badge className="border-emerald/25 bg-emerald/10 text-emerald text-[10px]">
                    {t("dash.earnings.paypal_connected")}
                  </Badge>
                </div>
                <p className="mt-0.5 text-xs text-text-secondary">{t("dash.earnings.paypal_ready_hint")}</p>
                {paypal?.onboarded_at ? (
                  <p className="mt-1 text-[11px] text-text-muted">
                    {t("dash.earnings.paypal_since")}{" "}
                    {new Date(paypal.onboarded_at * 1000).toLocaleDateString()}
                  </p>
                ) : null}
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-2 h-7 px-2 text-xs text-text-muted hover:text-accent-red hover:bg-accent-red/10"
                  onClick={() => setConfirmDisconnect(true)}
                  disabled={disconnecting}
                >
                  {disconnecting ? (
                    <><Loader2 className="h-3 w-3 animate-spin" /> {t("dash.earnings.disconnecting")}</>
                  ) : (
                    <><Unlink className="h-3 w-3" /> {t("dash.earnings.disconnect")}</>
                  )}
                </Button>
              </div>
            </div>
            <ConfirmDialog
              open={confirmDisconnect}
              title={t("dash.earnings.paypal_disconnect_title")}
              description={t("dash.earnings.paypal_disconnect_desc")}
              confirmLabel={t("dash.earnings.disconnect")}
              variant="danger"
              onConfirm={handleDisconnect}
              onCancel={() => setConfirmDisconnect(false)}
            />
          </div>
        ) : (
          <>
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <p className="text-sm font-medium text-text-primary">
                  {status === "onboarding"
                    ? t("dash.earnings.paypal_resume")
                    : t("dash.earnings.paypal_connect")}
                </p>
                <p className="mt-0.5 text-xs text-text-muted">{t("dash.earnings.paypal_fee_note")}</p>
              </div>
              <Badge
                className={`${STATUS_STYLE[status] || "text-text-muted"} border-current/20 bg-current/5 shrink-0`}
              >
                {statusLabel}
              </Badge>
            </div>

            <div className="flex items-start gap-2 rounded-lg border border-[#009cde]/15 bg-[#009cde]/5 px-3 py-2.5">
              <Info className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#009cde]" />
              <p className="text-xs leading-relaxed text-text-secondary">
                {t("dash.earnings.paypal_wallet_clarity")}
              </p>
            </div>

            <ul className="space-y-1.5 rounded-lg border border-border/50 bg-background/40 px-3 py-2.5">
              {BENEFIT_KEYS.map((key, i) => {
                const Icon = BENEFIT_ICONS[i];
                return (
                  <li key={key} className="flex items-start gap-2 text-xs text-text-secondary">
                    <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#009cde]" />
                    <span>{t(key)}</span>
                  </li>
                );
              })}
            </ul>

            <div className="flex flex-col gap-2 sm:flex-row">
              <Button
                size="sm"
                className="w-full border-[#ffc439]/60 bg-[#ffc439] text-[#003087] hover:bg-[#f0b929] font-semibold shadow-sm"
                onClick={handleConnect}
                disabled={loading || polling}
              >
                {loading ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    {t("dash.earnings.paypal_redirecting")}
                  </>
                ) : polling ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    {t("dash.earnings.paypal_verifying")}
                  </>
                ) : status === "onboarding" ? (
                  <>
                    <ExternalLink className="h-3.5 w-3.5" />
                    {t("dash.earnings.paypal_resume")}
                  </>
                ) : (
                  <>
                    <LinkIcon className="h-3.5 w-3.5" />
                    {t("dash.earnings.paypal_connect")}
                  </>
                )}
              </Button>
              {(status === "onboarding" || polling) && (
                <Button
                  variant="outline"
                  size="sm"
                  className="w-full border-[#009cde]/25 hover:bg-[#009cde]/5"
                  onClick={handleRefresh}
                  disabled={loading}
                >
                  <RefreshCw className="h-3.5 w-3.5" />
                  {t("dash.earnings.paypal_check_status")}
                </Button>
              )}
            </div>
            {error ? (
              <div className="rounded-md border border-accent-red/30 bg-accent-red/5 px-3 py-2">
                <p className="text-xs font-medium text-accent-red">{error}</p>
              </div>
            ) : null}
          </>
        )}
      </CardContent>
    </Card>
  );
}