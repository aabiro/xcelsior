"use client";

import { useEffect } from "react";
import { syncPushSubscriptionWithServer } from "@/lib/pwa/web-push";

/**
 * Registers the Serwist-generated service worker.
 * Rendered once in the root layout — no UI output.
 */
export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if (!("serviceWorker" in navigator) || process.env.NODE_ENV !== "production") return;

    async function register() {
      const registration = await navigator.serviceWorker.register("/sw.js", {
        scope: "/",
        updateViaCache: "none",
      });

      if (!("Notification" in window) || !("PushManager" in window) || Notification.permission !== "granted") {
        return;
      }

      const subscription = await registration.pushManager.getSubscription();
      if (!subscription) return;

      await syncPushSubscriptionWithServer({
        registration,
        existingSubscription: subscription,
        createIfMissing: false,
      });
    }

    register().catch((err) => {
      console.warn("SW registration failed:", err);
    });
  }, []);

  return null;
}
