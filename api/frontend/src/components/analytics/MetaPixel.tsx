"use client";

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth";
import { CONSENT_CHANGED_EVENT, hasMarketingConsent } from "@/lib/cookie-consent";
import { clearMetaCookies, syncConsentToAccount, syncMetaCookies, trackPageView } from "@/lib/meta-events";

type Fbq = ((...args: unknown[]) => void) & { queue?: unknown[]; loaded?: boolean; version?: string };

declare global {
  interface Window {
    fbq?: Fbq;
    _fbq?: Fbq;
  }
}

function installFbq(pixelId: string) {
  if (typeof window.fbq === "function" && window.fbq.loaded) {
    window.fbq("init", pixelId);
    return;
  }
  const fbq: Fbq = function fbqStub(...args: unknown[]) {
    if (fbq.queue) fbq.queue.push(args);
  };
  fbq.queue = [];
  fbq.loaded = true;
  fbq.version = "2.0";
  window.fbq = fbq;
  window._fbq = fbq;
  const script = document.createElement("script");
  script.async = true;
  script.src = "https://connect.facebook.net/en_US/fbevents.js";
  document.head.appendChild(script);
  window.fbq("init", pixelId);
}

/** Loads the Pixel only after the visitor opts in to marketing cookies. */
export function MetaPixel({ pixelId }: { pixelId: string }) {
  const pathname = usePathname() || "/";
  const { user } = useAuth();
  const userId = user?.user_id;
  const [consented, setConsented] = useState(false);

  useEffect(() => {
    const update = () => {
      const granted = hasMarketingConsent();
      // fbevents stays on the page once loaded; tell it to stop or resume.
      if (typeof window.fbq === "function") window.fbq("consent", granted ? "grant" : "revoke");
      if (!granted) clearMetaCookies();
      setConsented(granted);
    };
    update();
    window.addEventListener(CONSENT_CHANGED_EVENT, update);
    return () => window.removeEventListener(CONSENT_CHANGED_EVENT, update);
  }, []);

  useEffect(() => {
    void syncConsentToAccount(userId);
    const sync = () => void syncConsentToAccount(userId);
    window.addEventListener(CONSENT_CHANGED_EVENT, sync);
    return () => window.removeEventListener(CONSENT_CHANGED_EVENT, sync);
  }, [userId]);

  useEffect(() => {
    if (!consented || !/^\d{8,20}$/.test(pixelId)) return;
    syncMetaCookies();
    installFbq(pixelId);
  }, [consented, pixelId]);

  useEffect(() => {
    if (!consented || !/^\d{8,20}$/.test(pixelId)) return;
    syncMetaCookies();
    trackPageView(pathname);
  }, [consented, pixelId, pathname]);

  return null;
}
