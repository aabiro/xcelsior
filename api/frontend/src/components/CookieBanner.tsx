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
 * Asks once for optional cookies (analytics and ads). Declining is one click,
 * the same as accepting. The footer's "Cookie preferences" reopens it.
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
      role="region"
      aria-label={t("cookies.label")}
      className="fixed inset-x-0 bottom-0 z-[60] border-t border-border bg-surface-overlay/95 shadow-lg backdrop-blur-sm"
    >
      <div className="mx-auto flex max-w-6xl flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm text-text-secondary">
          {t("cookies.body")}{" "}
          <Link href="/privacy" className="underline hover:text-text-primary">
            {t("cookies.learn_more")}
          </Link>
        </p>
        <div className="flex shrink-0 gap-2">
          <button
            type="button"
            onClick={() => choose(false)}
            className="rounded-lg border border-border px-4 py-1.5 text-sm font-semibold text-text-primary transition-colors hover:bg-surface-hover"
          >
            {t("cookies.decline")}
          </button>
          <button
            type="button"
            onClick={() => choose(true)}
            className="rounded-lg border border-border px-4 py-1.5 text-sm font-semibold text-text-primary transition-colors hover:bg-surface-hover"
          >
            {t("cookies.accept")}
          </button>
        </div>
      </div>
    </div>
  );
}
