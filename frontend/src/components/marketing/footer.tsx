"use client";

import Link from "next/link";
import { useLocale } from "@/lib/locale";

export function Footer() {
  const { t } = useLocale();

  return (
    <footer className="border-t border-border bg-navy">
      <div className="mx-auto max-w-7xl px-6 py-12">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-4">
              <svg width="36" height="36" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="40" height="40" rx="10" fill="url(#footer-logo-bg)" />
                <path d="M12 12L20 22L28 12" stroke="url(#footer-logo-x)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
                <path d="M12 28L20 18L28 28" stroke="url(#footer-logo-x)" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" />
                <defs>
                  <linearGradient id="footer-logo-bg" x1="0" y1="0" x2="40" y2="40" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#0d1320" />
                    <stop offset="1" stopColor="#1a1f2e" />
                  </linearGradient>
                  <linearGradient id="footer-logo-x" x1="12" y1="12" x2="28" y2="28" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#00d4ff" />
                    <stop offset="0.5" stopColor="#7c3aed" />
                    <stop offset="1" stopColor="#dc2626" />
                  </linearGradient>
                </defs>
              </svg>
              <span className="text-xl font-bold">Xcelsior</span>
            </div>
            <p className="text-sm text-text-secondary leading-relaxed">
              {t("footer.tagline")}
              <br />
              <span className="text-accent-gold italic">{t("footer.motto")}</span>
            </p>
          </div>

          {/* Product */}
          <div>
            <h4 className="mb-3 text-sm font-semibold text-text-primary">{t("footer.product")}</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/features" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.features")}
                </Link>
              </li>
              <li>
                <Link href="/pricing" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.pricing")}
                </Link>
              </li>
              <li>
                <Link href="/sovereignty" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.sovereignty")}
                </Link>
              </li>
              <li>
                <a href="https://docs.xcelsior.ca" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.docs")}
                </a>
              </li>
            </ul>
          </div>

          {/* Company */}
          <div>
            <h4 className="mb-3 text-sm font-semibold text-text-primary">{t("footer.company")}</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/blog" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.blog")}
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.about")}
                </Link>
              </li>
              <li>
                <a href="mailto:hello@xcelsior.ca" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.contact")}
                </a>
              </li>
            </ul>
          </div>

          {/* Legal */}
          <div>
            <h4 className="mb-3 text-sm font-semibold text-text-primary">{t("footer.legal")}</h4>
            <ul className="space-y-2">
              <li>
                <Link href="/privacy" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.privacy")}
                </Link>
              </li>
              <li>
                <Link href="/terms" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.terms")}
                </Link>
              </li>
              <li>
                <Link href="/sovereignty" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.data_sovereignty")}
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 flex flex-col items-center justify-between gap-4 border-t border-border pt-8 md:flex-row">
          <p className="text-xs text-text-muted">
            {t("footer.copyright", { year: new Date().getFullYear() })}
          </p>
          <p className="text-xs text-text-muted">
            {t("footer.hydro")}
          </p>
        </div>
      </div>
    </footer>
  );
}
