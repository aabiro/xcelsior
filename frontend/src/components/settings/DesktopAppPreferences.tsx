"use client";

import { useEffect, useState } from "react";
import { BellRing, CheckCircle2, Download, Loader2, MonitorSmartphone } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  fetchPushSubscriptionStatus,
} from "@/lib/api";
import { usePwaInstallPrompt } from "@/hooks/usePwaInstallPrompt";
import {
  revokePushSubscriptionOnServer,
  syncPushSubscriptionWithServer,
} from "@/lib/pwa/web-push";

async function ensureServiceWorkerRegistration() {
  const existing = await navigator.serviceWorker.getRegistration("/");
  if (existing) return existing;

  return navigator.serviceWorker.register("/sw.js", {
    scope: "/",
    updateViaCache: "none",
  });
}

export function DesktopAppPreferences() {
  const [installing, setInstalling] = useState(false);
  const [pushConfigured, setPushConfigured] = useState(false);
  const [pushLoading, setPushLoading] = useState(false);
  const [pushPermission, setPushPermission] = useState<NotificationPermission>("default");
  const [pushSubscribed, setPushSubscribed] = useState(false);
  const [pushSupported, setPushSupported] = useState(false);
  const [vapidPublicKey, setVapidPublicKey] = useState("");
  const { canInstall, isDesktopDevice, isInstalled, promptToInstall } = usePwaInstallPrompt();

  useEffect(() => {
    if (!isDesktopDevice) return;

    let cancelled = false;

    async function loadPushState() {
      const supported =
        "Notification" in window &&
        "serviceWorker" in navigator &&
        "PushManager" in window;

      if (cancelled) return;

      setPushSupported(supported);
      setPushPermission(window.Notification?.permission ?? "default");

      if (!supported) return;

      try {
        const status = await fetchPushSubscriptionStatus();
        if (cancelled) return;

        setPushConfigured(status.configured);
        setVapidPublicKey(status.vapid_public_key);

        const registration = await navigator.serviceWorker.getRegistration("/");
        const subscription = await registration?.pushManager.getSubscription();
        if (cancelled) return;

        setPushSubscribed(Boolean(subscription));
      } catch (error) {
        if (!cancelled) {
          console.error("Failed to load web push status", error);
          setPushConfigured(false);
          setPushSubscribed(false);
        }
      }
    }

    loadPushState();

    return () => {
      cancelled = true;
    };
  }, [isDesktopDevice]);

  if (!isDesktopDevice) return null;

  const statusLabel = isInstalled
    ? "Installed on this desktop"
    : canInstall
      ? "Ready to install"
      : "Use your browser install menu";

  const statusTone = isInstalled
    ? "border-emerald/30 bg-emerald/10 text-emerald"
    : "border-border bg-navy-light text-text-secondary";

  const description = isInstalled
    ? "The installed app uses the same Xcelsior UI, supports offline fallback, and reopens where you left off."
    : canInstall
      ? "Install Xcelsior as a desktop app without changing the existing website layout or workflow."
      : "If your browser does not expose the install button here, use the install action in Chrome or Edge for this site.";

  const pushStatusLabel = !pushSupported
    ? "Unsupported in this browser"
    : pushSubscribed
      ? "Enabled"
      : !pushConfigured
        ? "Server not configured"
        : pushPermission === "denied"
          ? "Permission blocked"
          : "Off";

  const pushStatusTone = pushSubscribed
    ? "border-emerald/30 bg-emerald/10 text-emerald"
    : "border-border bg-navy-light text-text-secondary";

  const handleInstall = async () => {
    setInstalling(true);
    try {
      await promptToInstall();
    } finally {
      setInstalling(false);
    }
  };

  const handleEnablePush = async () => {
    if (!pushSupported) {
      toast.error("Desktop notifications are not supported in this browser.");
      return;
    }
    if (!pushConfigured || !vapidPublicKey) {
      toast.error("Desktop notifications are not configured for this deployment.");
      return;
    }

    setPushLoading(true);
    try {
      const permission =
        Notification.permission === "granted"
          ? "granted"
          : await Notification.requestPermission();

      setPushPermission(permission);
      if (permission !== "granted") {
        toast.error("Notification permission was not granted.");
        return;
      }

      const registration = await ensureServiceWorkerRegistration();
      const subscription = await syncPushSubscriptionWithServer({
        registration,
        vapidPublicKey,
      });
      if (!subscription) {
        throw new Error("Web push configuration is unavailable.");
      }

      setPushSubscribed(true);
      toast.success("Desktop notifications enabled.");
    } catch (error) {
      console.error("Failed to enable web push", error);
      toast.error("Failed to enable desktop notifications.");
    } finally {
      setPushLoading(false);
    }
  };

  const handleDisablePush = async () => {
    if (!pushSupported) {
      setPushSubscribed(false);
      return;
    }

    setPushLoading(true);
    try {
      const registration = await navigator.serviceWorker.getRegistration("/");
      const subscription = await registration?.pushManager.getSubscription();

      if (subscription?.endpoint) {
        await revokePushSubscriptionOnServer(subscription.endpoint);
        await subscription.unsubscribe();
      }

      setPushSubscribed(false);
      toast.success("Desktop notifications disabled.");
    } catch (error) {
      console.error("Failed to disable web push", error);
      toast.error("Failed to disable desktop notifications.");
    } finally {
      setPushLoading(false);
    }
  };

  return (
    <div className="mt-5 border-t border-border/60 pt-5">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <MonitorSmartphone className="h-4 w-4 text-accent-gold" />
            <p className="text-sm font-medium text-text-primary">Desktop app</p>
            <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${statusTone}`}>
              {statusLabel}
            </span>
          </div>
          <p className="max-w-2xl text-xs text-text-secondary">{description}</p>
        </div>

        {canInstall ? (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={handleInstall}
            disabled={installing}
            className="min-w-36"
          >
            <Download className="h-4 w-4" />
            {installing ? "Installing..." : "Install app"}
          </Button>
        ) : isInstalled ? (
          <div className="inline-flex h-8 items-center gap-2 rounded-lg border border-emerald/30 bg-emerald/10 px-3 text-xs font-medium text-emerald">
            <CheckCircle2 className="h-4 w-4" />
            Installed
          </div>
        ) : null}
      </div>

      <div className="mt-5 border-t border-border/60 pt-5">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <BellRing className="h-4 w-4 text-accent-gold" />
              <p className="text-sm font-medium text-text-primary">Desktop notifications</p>
              <span className={`rounded-full border px-2 py-0.5 text-[11px] font-medium ${pushStatusTone}`}>
                {pushStatusLabel}
              </span>
            </div>
            <p className="max-w-2xl text-xs text-text-secondary">
              Receive system notifications for instance activity and platform events through the installed app or your desktop browser.
            </p>
          </div>

          {pushSupported && pushConfigured ? (
            <Button
              type="button"
              variant={pushSubscribed ? "outline" : "secondary"}
              size="sm"
              onClick={pushSubscribed ? handleDisablePush : handleEnablePush}
              disabled={pushLoading}
              className="min-w-44"
            >
              {pushLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <BellRing className="h-4 w-4" />}
              {pushSubscribed ? "Disable notifications" : "Enable notifications"}
            </Button>
          ) : null}
        </div>
      </div>
    </div>
  );
}
