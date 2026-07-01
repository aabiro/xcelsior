"use client";

import Link from "next/link";
import { SITE_ASSETS } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const footerColumns = [
  {
    head: "footer.product",
    items: [
      { href: "/features", key: "footer.features" },
      { href: "/pricing", key: "footer.pricing" },
      { href: "/download", key: "footer.download" },
      { href: "https://docs.xcelsior.ca", key: "footer.docs", external: true },
    ],
  },
  {
    head: "footer.company",
    items: [
      { href: "/blog", key: "footer.blog" },
      { href: "/about", key: "footer.about" },
      { href: "/support", key: "footer.support" },
      { href: "mailto:hello@xcelsior.ca", key: "footer.contact", external: true },
    ],
  },
  {
    head: "footer.legal",
    items: [
      { href: "/privacy", key: "footer.privacy" },
      { href: "/terms", key: "footer.terms" },
    ],
  },
];

export function Footer() {
  const { t } = useLocale();

  return (
    <footer className="site-footer">
      <div className="site-container">
        <div className="site-rails">
          <div className="site-footer-grid">
            <div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={SITE_ASSETS.iconGradient} style={{ width: 24, height: 24 }} alt="" aria-hidden />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={SITE_ASSETS.wordmarkLight} className="wm-light" style={{ height: 16 }} alt="Xcelsior" />
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={SITE_ASSETS.wordmarkDark} className="wm-dark" style={{ height: 16 }} alt="Xcelsior" />
              </div>
              <p style={{ maxWidth: 280, margin: "0 0 16px", color: "var(--text-3)", fontSize: 14, lineHeight: 1.6 }}>
                {t("footer.tagline")}
              </p>
              <div className="site-gradient-text site-mono" style={{ fontSize: 13, fontWeight: 600 }}>
                {t("footer.motto")}
              </div>
            </div>

            {footerColumns.map((column) => (
              <div key={column.head}>
                <div className="site-mono" style={{ marginBottom: 16, color: "var(--text-5)", fontSize: 11, fontWeight: 600, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                  {t(column.head)}
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 11 }}>
                  {column.items.map((item) => (
                    item.external ? (
                      <a key={item.href} href={item.href} style={{ fontSize: 14 }}>
                        {t(item.key)}
                      </a>
                    ) : (
                      <Link key={item.href} href={item.href} style={{ fontSize: 14 }}>
                        {t(item.key)}
                      </Link>
                    )
                  ))}
                </div>
              </div>
            ))}
          </div>

          <div className="site-footer-bottom">
            <span>{t("footer.copyright", { year: 2026 })}</span>
            <span>{t("footer.hydro")}</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
