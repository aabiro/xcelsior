"use client";

/* eslint-disable @next/next/no-img-element */

import { useEffect, useState } from "react";
import { usePathname } from "next/navigation";
import Link from "next/link";
import { usePwaInstallPrompt } from "@/hooks/usePwaInstallPrompt";
import { BRAND_PNG_ASSETS } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const LS_DISMISSED = "xcelsior-pwa-dismissed";
const LS_VISITS = "xcelsior-pwa-visits";
const LS_LAUNCHED = "xcelsior-instance-launched";

type BannerVariant = "post-launch" | "frequent" | "default" | "desktop";

/**
 * Context-aware install banner.
 *
 * Mobile:
 *   - After first instance launch → "Nice, your instance is running" + tray alerts pitch
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
const AUTO_DISMISS_MS = 12_000;
const EXIT_ANIMATION_MS = 300;

export function InstallBanner() {
  const [show, setShow] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const [variant, setVariant] = useState<BannerVariant>("default");
  const { canInstall, isDesktopDevice, isInstalled, promptToInstall } =
    usePwaInstallPrompt();
  const { t } = useLocale();
  const pathname = usePathname();
  const isDashboard = pathname?.startsWith("/dashboard");
  const allowPromo =
    pathname === "/" ||
    pathname?.startsWith("/dashboard") ||
    pathname?.startsWith("/pricing") ||
    pathname?.startsWith("/gpu-availability") ||
    pathname?.startsWith("/features") ||
    pathname?.startsWith("/download");

  useEffect(() => {
    if (!allowPromo || isInstalled) return;

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

    const frameId = requestAnimationFrame(() => {
      if (isDesktopDevice) {
        // On desktop, suggest the native Tauri app
        if (hasLaunched || visits >= 5) {
          setVariant("desktop");
          setShow(true);
        }
        return;
      }

      // Mobile, need the browser install prompt to be available
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
    });

    return () => cancelAnimationFrame(frameId);
  }, [allowPromo, canInstall, isDesktopDevice, isInstalled]);

  // Auto-dismiss: animate off after a while instead of lingering forever.
  // Session-only, the 14-day snooze is reserved for explicit dismissals.
  useEffect(() => {
    if (!show || leaving) return;
    const timer = window.setTimeout(() => setLeaving(true), AUTO_DISMISS_MS);
    return () => window.clearTimeout(timer);
  }, [show, leaving]);

  useEffect(() => {
    if (!leaving) return;
    const timer = window.setTimeout(() => { setShow(false); setLeaving(false); }, EXIT_ANIMATION_MS);
    return () => window.clearTimeout(timer);
  }, [leaving]);

  const handleInstall = async () => {
    const outcome = await promptToInstall();
    if (outcome === "accepted") {
      setShow(false);
    }
  };

  const handleDismiss = () => {
    setLeaving(true);
    localStorage.setItem(LS_DISMISSED, String(Date.now()));
  };

  if (!show) return null;

  const motionClasses = leaving ? "banner-exit" : "banner-enter";

  // Desktop variant → link to /download instead of PWA prompt
  if (variant === "desktop") {
    return (
      <div className={`fixed ${isDashboard ? "bottom-4" : "bottom-24"} right-4 z-40 hidden md:block ${motionClasses} max-w-sm`}>
        <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-overlay p-4 shadow-lg backdrop-blur-sm">
          <img
            src={BRAND_PNG_ASSETS.appGradientRounded512}
            alt="Xcelsior"
            width={36}
            height={36}
            className="h-9 w-9 flex-shrink-0 rounded-lg"
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
    <div className={`fixed bottom-4 left-4 right-4 z-50 md:hidden ${motionClasses}`}>
      <div className="flex items-center gap-3 rounded-xl border border-border bg-surface-overlay p-4 shadow-lg backdrop-blur-sm">
        <img
          src={BRAND_PNG_ASSETS.appGradientRounded512}
          alt="Xcelsior"
          width={40}
          height={40}
          className="h-10 w-10 flex-shrink-0 rounded-lg"
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
