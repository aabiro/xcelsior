"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePwaInstallPrompt } from "@/hooks/usePwaInstallPrompt";
import { useLocale } from "@/lib/locale";

const LS_DISMISSED = "xcelsior-pwa-dismissed";
const LS_VISITS = "xcelsior-pwa-visits";
const LS_LAUNCHED = "xcelsior-instance-launched";

type BannerVariant = "post-launch" | "frequent" | "default" | "desktop";

/**
 * Context-aware install banner.
 *
 * Mobile:
 *   - After first instance launch → "Nice — your instance is running" + tray alerts pitch
 *   - After 5+ visits (daily checker) → "You're here a lot" + offline pitch
 *   - Fallback after 2 visits → generic "Get the app"
 *
 * Desktop (non-PWA):
 *   - After first instance launch → suggest native Tauri app
 *   - After 5+ visits → suggest native Tauri app
 *
 * Never shows if already installed (PWA or desktop), or user dismissed.
 * Dismissal is soft: resets after 14 days so the prompt can resurface once.
 */
export function InstallBanner() {
  const [show, setShow] = useState(false);
  const [variant, setVariant] = useState<BannerVariant>("default");
  const { canInstall, isDesktopDevice, isInstalled, promptToInstall } =
    usePwaInstallPrompt();
  const { t } = useLocale();

  useEffect(() => {
    if (isInstalled) return;

    // Bump visit count
    const visits = parseInt(localStorage.getItem(LS_VISITS) || "0", 10) + 1;
    localStorage.setItem(LS_VISITS, String(visits));

    // Check soft-dismiss (expires after 14 days)
    const dismissedAt = localStorage.getItem(LS_DISMISSED);
    if (dismissedAt) {
      const elapsed = Date.now() - parseInt(dismissedAt, 10);
      if (elapsed < 14 * 24 * 60 * 60 * 1000) return;
      localStorage.removeItem(LS_DISMISSED);
    }

    const hasLaunched = localStorage.getItem(LS_LAUNCHED) === "1";

    if (isDesktopDevice) {
      // On desktop, suggest the native Tauri app
      if (hasLaunched || visits >= 5) {
        setVariant("desktop");
        setShow(true);
      }
      return;
    }

    // Mobile — need the browser install prompt to be available
    if (!canInstall) return;

    if (hasLaunched) {
      setVariant("post-launch");
      setShow(true);
    } else if (visits >= 5) {
      setVariant("frequent");
      setShow(true);
    } else if (visits >= 2) {
      setVariant("default");
      setShow(true);
    }
  }, [canInstall, isDesktopDevice, isInstalled]);

  const handleInstall = async () => {
    const outcome = await promptToInstall();
    if (outcome === "accepted") {
      setShow(false);
    }
  };

  const handleDismiss = () => {
    setShow(false);
    localStorage.setItem(LS_DISMISSED, String(Date.now()));
  };

  if (!show) return null;

  // Desktop variant → link to /download instead of PWA prompt
  if (variant === "desktop") {
    return (
      <div className="fixed bottom-24 right-4 z-40 hidden md:block animate-in slide-in-from-bottom-4 fade-in duration-300 max-w-sm">
        <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-elevated p-4 shadow-lg backdrop-blur-sm">
          <img
            src="/xcelsior_icon_192x192.png"
            alt="Xcelsior"
            width={36}
            height={36}
            className="rounded-lg flex-shrink-0"
          />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-text-primary">
              {t("pwa.install_desktop_title")}
            </p>
            <p className="text-xs text-text-secondary">
              {t("pwa.install_desktop_desc")}
            </p>
          </div>
          <Link
            href="/download"
            onClick={() => setShow(false)}
            className="flex-shrink-0 rounded-lg bg-accent-cyan px-3 py-1.5 text-xs font-semibold text-navy transition-colors hover:bg-accent-cyan/80"
          >
            {t("pwa.install_desktop_cta")}
          </Link>
          <button
            onClick={handleDismiss}
            className="flex-shrink-0 p-1 text-text-muted hover:text-text-secondary transition-colors"
            aria-label="Dismiss"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M18 6 6 18" /><path d="m6 6 12 12" />
            </svg>
          </button>
        </div>
      </div>
    );
  }

  // Mobile variants
  const titles: Record<Exclude<BannerVariant, "desktop">, string> = {
    "post-launch": t("pwa.install_after_launch_title"),
    frequent: t("pwa.install_frequent_title"),
    default: t("pwa.install_title"),
  };
  const descs: Record<Exclude<BannerVariant, "desktop">, string> = {
    "post-launch": t("pwa.install_after_launch_desc"),
    frequent: t("pwa.install_frequent_desc"),
    default: t("pwa.install_desc"),
  };

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
            {titles[variant as Exclude<BannerVariant, "desktop">]}
          </p>
          <p className="text-xs text-text-secondary">
            {descs[variant as Exclude<BannerVariant, "desktop">]}
          </p>
        </div>
        <button
          onClick={handleInstall}
          className="flex-shrink-0 rounded-lg bg-accent-cyan px-3 py-1.5 text-xs font-semibold text-navy transition-colors hover:bg-accent-cyan/80"
        >
          {t("pwa.install_cta")}
        </button>
        <button
          onClick={handleDismiss}
          className="flex-shrink-0 p-1 text-text-muted hover:text-text-secondary transition-colors"
          aria-label="Dismiss"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M18 6 6 18" /><path d="m6 6 12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}

/**
 * Call this after a successful instance launch to set the flag
 * that triggers the "post-launch" install banner variant.
 */
export function markInstanceLaunched() {
  if (typeof window !== "undefined") {
    localStorage.setItem(LS_LAUNCHED, "1");
  }
}
