"use client";

import { useEffect } from "react";
import { usePathname } from "next/navigation";
import { syncMetaCookies, trackPageView } from "@/lib/meta-events";

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

export function MetaPixel({ pixelId }: { pixelId: string }) {
  const pathname = usePathname() || "/";

  useEffect(() => {
    if (!/^\d{8,20}$/.test(pixelId)) return;
    syncMetaCookies();
    installFbq(pixelId);
  }, [pixelId]);

  useEffect(() => {
    if (!/^\d{8,20}$/.test(pixelId)) return;
    syncMetaCookies();
    trackPageView(pathname);
  }, [pixelId, pathname]);

  return null;
}
