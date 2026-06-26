"use client";

import Link from "next/link";
import Image from "next/image";
import { BRAND_ASSETS } from "@/lib/brand-assets";
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
                src={BRAND_ASSETS.lockupLight}
                alt="Xcelsior"
                width={206}
                height={34}
                className="hidden h-auto w-40 dark:block"
              />
              <Image
                src={BRAND_ASSETS.lockupDark}
                alt="Xcelsior"
                width={206}
                height={34}
                className="block h-auto w-40 dark:hidden"
              />
            </div>
            <p className="text-sm text-text-secondary leading-relaxed">
              {t("footer.tagline")}
              <br />
              <span className="sr-only">{t("footer.motto")}</span>
              <Image
                src={BRAND_ASSETS.textEverSbLight}
                alt=""
                width={127}
                height={22}
                className="mt-2 hidden h-4 w-auto dark:block"
              />
              <Image
                src={BRAND_ASSETS.textEverSbDark}
                alt=""
                width={127}
                height={22}
                className="mt-2 block h-4 w-auto dark:hidden"
              />
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
                <Link href="/download" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.download")}
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
                <Link href="/support" className="text-sm text-text-secondary hover:text-text-primary">
                  {t("footer.support")}
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
            {t("footer.copyright", { year: 2026 })}
          </p>
          <a href="https://xcelsior.ca" className="inline-flex opacity-70 transition-opacity hover:opacity-100" aria-label="xcelsior.ca">
            <Image
              src={BRAND_ASSETS.textUrlMedLight}
              alt=""
              width={106}
              height={19}
              className="hidden h-3.5 w-auto dark:block"
            />
            <Image
              src={BRAND_ASSETS.textUrlMedDark}
              alt=""
              width={106}
              height={19}
              className="block h-3.5 w-auto dark:hidden"
            />
          </a>
          <p className="text-xs text-text-muted">
            {t("footer.hydro")}
          </p>
        </div>
      </div>
    </footer>
  );
}
