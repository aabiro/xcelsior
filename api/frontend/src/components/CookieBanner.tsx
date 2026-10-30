"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useLocale } from "@/lib/locale";
import {
  OPEN_COOKIE_PREFERENCES_EVENT,
  readConsent,
  setMarketingConsent,
} from "@/lib/cookie-consent";

/**
 * Asks once for marketing cookies (Meta Pixel and Conversions API). Declining
 * is one click, the same as accepting. The footer's "Cookie preferences"
 * reopens it.
 */
export function CookieBanner() {
  const { t } = useLocale();
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      if (!readConsent()) setOpen(true);
    });
    const reopen = () => setOpen(true);
    window.addEventListener(OPEN_COOKIE_PREFERENCES_EVENT, reopen);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener(OPEN_COOKIE_PREFERENCES_EVENT, reopen);
    };
  }, []);

  if (!open) return null;

  const choose = (marketing: boolean) => {
    setMarketingConsent(marketing, "banner");
    setOpen(false);
  };

  return (
    <div
      role="dialog"
      aria-live="polite"
      aria-labelledby="cookie-banner-title"
      className="fixed bottom-4 left-4 right-4 z-[60] md:right-auto md:max-w-md"
    >
      <div className="rounded-xl border border-border bg-surface-overlay p-4 shadow-lg backdrop-blur-sm">
        <p id="cookie-banner-title" className="text-sm font-medium text-text-primary">
          {t("cookies.title")}
        </p>
        <p className="mt-1 text-xs leading-relaxed text-text-secondary">
          {t("cookies.body")}{" "}
          <Link href="/privacy" className="underline hover:text-text-primary">
            {t("cookies.privacy_link")}
          </Link>
        </p>
        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => choose(false)}
            className="flex-1 rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-text-primary transition-colors hover:bg-surface-hover"
          >
            {t("cookies.decline")}
          </button>
          <button
            type="button"
            onClick={() => choose(true)}
            className="flex-1 rounded-lg border border-border px-3 py-1.5 text-xs font-semibold text-text-primary transition-colors hover:bg-surface-hover"
          >
            {t("cookies.accept")}
          </button>
        </div>
      </div>
    </div>
  );
}
