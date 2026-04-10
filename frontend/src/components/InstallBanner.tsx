"use client";

import { useEffect, useState } from "react";
import { usePwaInstallPrompt } from "@/hooks/usePwaInstallPrompt";

/**
 * Custom "Add to Home Screen" banner shown on mobile only (md:hidden).
 * Appears on 2nd visit, dismissible, non-annoying.
 */
export function InstallBanner() {
  const [show, setShow] = useState(false);
  const [visitCount, setVisitCount] = useState(0);
  const { canInstall, isDesktopDevice, isInstalled, promptToInstall } = usePwaInstallPrompt();

  useEffect(() => {
    if (isInstalled || isDesktopDevice) return;

    const visits = parseInt(
      localStorage.getItem("xcelsior-pwa-visits") || "0",
      10,
    );
    localStorage.setItem("xcelsior-pwa-visits", String(visits + 1));
    setVisitCount(visits);
  }, [isDesktopDevice, isInstalled]);

  useEffect(() => {
    if (isInstalled || isDesktopDevice) return;

    // Check if user previously dismissed
    const dismissed = localStorage.getItem("xcelsior-pwa-dismissed");
    if (dismissed) return;

    if (canInstall && visitCount >= 1) {
      setShow(true);
    } else if (!canInstall) {
      setShow(false);
    }
  }, [canInstall, isDesktopDevice, isInstalled, visitCount]);

  const handleInstall = async () => {
    const outcome = await promptToInstall();
    if (outcome === "accepted") {
      setShow(false);
    }
  };

  const handleDismiss = () => {
    setShow(false);
    localStorage.setItem("xcelsior-pwa-dismissed", "1");
  };

  if (!show) return null;

  return (
    <div className="fixed bottom-4 left-4 right-4 z-50 md:hidden animate-in slide-in-from-bottom-4 fade-in duration-300">
      <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-elevated p-4 shadow-lg backdrop-blur-sm">
        <img
          src="/xcelsior_icon_192x192.png"
          alt="Xcelsior"
          width={40}
          height={40}
          className="rounded-lg flex-shrink-0"
        />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-text-primary truncate">
            Install Xcelsior
          </p>
          <p className="text-xs text-text-secondary">
            Quick access from your home screen
          </p>
        </div>
        <button
          onClick={handleInstall}
          className="flex-shrink-0 rounded-lg bg-accent-cyan px-3 py-1.5 text-xs font-semibold text-navy transition-colors hover:bg-accent-cyan/80"
        >
          Install
        </button>
        <button
          onClick={handleDismiss}
          className="flex-shrink-0 p-1 text-text-muted hover:text-text-secondary transition-colors"
          aria-label="Dismiss install prompt"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M18 6 6 18" />
            <path d="m6 6 12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}
