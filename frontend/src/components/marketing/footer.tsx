"use client";

import Link from "next/link";
import Image from "next/image";
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
              <Image
                src="/xcelsior-logo-wordmark-iconbg.svg"
                alt="Xcelsior"
                width={160}
                height={44}
                className="h-11 w-auto block dark:block hidden"
                priority
              />
              <Image
                src="/xcelsior-logo-wordmark-iconbg-light.svg"
                alt="Xcelsior"
                width={160}
                height={44}
                className="h-11 w-auto block dark:hidden"
                priority
              />
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
