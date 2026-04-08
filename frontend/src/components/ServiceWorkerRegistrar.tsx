"use client";

import { useEffect } from "react";

/**
 * Registers the Serwist-generated service worker.
 * Rendered once in the root layout — no UI output.
 */
export function ServiceWorkerRegistrar() {
  useEffect(() => {
    if ("serviceWorker" in navigator && process.env.NODE_ENV === "production") {
      navigator.serviceWorker.register("/sw.js").catch((err) => {
        console.warn("SW registration failed:", err);
      });
    }
  }, []);

  return null;
}
