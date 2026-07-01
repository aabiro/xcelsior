"use client";

import Link from "next/link";
import { SITE_ASSETS, siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const products = [
  { key: "gpus", href: "/gpu-availability", icon: "gpu" },
  { key: "mcp", href: "/mcp", icon: "bot" },
  { key: "serverless", href: "/register", icon: "cloud" },
  { key: "instances", href: "/register", icon: "server" },
  { key: "hosting", href: "/register", icon: "coins" },
  { key: "xcelai", href: "/register", icon: "sparkle" },
  { key: "volumes", href: "/register", icon: "grid" },
] as const;

const foundations = [
  { icon: "shield", title: "features.sovereignty_title", desc: "features.sovereignty_desc", cat: "features.cat_sovereignty" },
  { icon: "route", title: "features.jurisdiction_title", desc: "features.jurisdiction_desc", cat: "features.cat_sovereignty" },
  { icon: "lock", title: "features.cloud_act_title", desc: "features.cloud_act_desc", cat: "features.cat_sovereignty" },
  { icon: "gpu", title: "features.marketplace_title", desc: "features.marketplace_desc", cat: "features.cat_compute" },
  { icon: "bolt", title: "features.spot_title", desc: "features.spot_desc", cat: "features.cat_compute" },
  { icon: "activity", title: "features.telemetry_title", desc: "features.telemetry_desc", cat: "features.cat_compute" },
  { icon: "badge", title: "features.trust_title", desc: "features.trust_desc", cat: "features.cat_trust" },
  { icon: "shield-check", title: "features.sla_title", desc: "features.sla_desc", cat: "features.cat_trust" },
  { icon: "check-circle", title: "features.compliance_title", desc: "features.compliance_desc", cat: "features.cat_trust" },
  { icon: "dollar", title: "features.billing_title", desc: "features.billing_desc", cat: "features.cat_billing" },
  { icon: "users", title: "features.payouts_title", desc: "features.payouts_desc", cat: "features.cat_billing" },
  { icon: "leaf", title: "features.green_title", desc: "features.green_desc", cat: "features.cat_billing" },
] as const;

function ThemeIcon({ name }: { name: string }) {
  return (
    <>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "dark")} className="site-theme-dark" alt="" aria-hidden />
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={siteIcon(name, "light")} className="site-theme-light" alt="" aria-hidden />
    </>
  );
}

function SectionMarker({ code, label }: { code: string; label: string }) {
  return (
    <div className="site-marker">
      <span className="site-marker-code">{code}</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
  );
}

export function FeaturesContent() {
  const { t } = useLocale();

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails">
            <div style={{ animation: "heroUp .7s ease both" }}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{t("features.platform_eyebrow")}</span>
              </div>
              <h1 className="site-hero-title">
                {t("features.title_1")} <span className="site-gradient-text">{t("features.title_accent")}</span>
              </h1>
              <p className="site-hero-copy">{t("features.platform_subtitle")}</p>
              <div className="site-hero-actions">
                <Link href="/gpu-availability" className="site-button site-button-primary">
                  {t("features.prod_gpus_cta")}
                </Link>
                <Link href="/mcp" className="site-button site-button-ghost">
                  {t("features.prod_mcp_cta")}
                </Link>
              </div>
            </div>

            <div className="site-telemetry-wrap" aria-hidden>
              <div className="site-telemetry-card">
                <div className="site-telemetry-head">
                  <div className="site-telemetry-model">
                    <span className="site-telemetry-mark">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={SITE_ASSETS.iconGradient} style={{ width: 20, height: 20 }} alt="" aria-hidden />
                    </span>
                    <div>
                      <div className="site-mono" style={{ color: "var(--text)", fontSize: 13, fontWeight: 600 }}>Platform</div>
                      <div className="site-mono" style={{ color: "var(--text-4)", fontSize: 11 }}>Sovereign backend</div>
                    </div>
                  </div>
                  <span className="site-live-badge">
                    <span className="site-live-dot" />
                    Live
                  </span>
                </div>
                {[
                  ["Products", "7", "86%"],
                  ["Compliance", "Built in", "100%"],
                  ["Billing", "CAD", "72%"],
                ].map(([label, value, width]) => (
                  <div key={label} className="site-meter">
                    <div className="site-meter-label">
                      <span style={{ color: "var(--text-4)" }}>{label}</span>
                      <span style={{ color: "var(--text-2)" }}>{value}</span>
                    </div>
                    <div className="site-meter-track">
                      <div className="site-meter-bar" style={{ width }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="01" label={t("features.platform_title")} />
          <h2 className="site-section-heading">{t("features.platform_title")}</h2>
          <p className="site-section-copy">{t("features.platform_subtitle")}</p>

          <div className="site-section-flush">
            {products.map((product, index) => (
              <article key={product.key} className="site-product-row">
                <div className="site-product-index">{String(index + 1).padStart(2, "0")}</div>
                <div className="site-product-main">
                  <div className="site-icon-box">
                    <ThemeIcon name={product.icon} />
                  </div>
                  <div className="site-product-badge site-gradient-text">{t(`features.prod_${product.key}_badge`)}</div>
                  <h3 className="site-product-title">{t(`features.prod_${product.key}_title`)}</h3>
                  <p className="site-card-copy">{t(`features.prod_${product.key}_desc`)}</p>
                  <Link href={product.href} className="site-product-cta" style={{ color: "var(--cyan)" }}>
                    {t(`features.prod_${product.key}_cta`)}
                  </Link>
                </div>
                <div className="site-product-points">
                  <p className="site-product-point">
                    <span style={{ color: "var(--green)" }}>+</span>
                    <span>{t(`features.prod_${product.key}_b1`)}</span>
                  </p>
                  <p className="site-product-point">
                    <span style={{ color: "var(--green)" }}>+</span>
                    <span>{t(`features.prod_${product.key}_b2`)}</span>
                  </p>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="02" label={t("features.everything_title")} />
          <h2 className="site-section-heading">{t("features.everything_title")}</h2>
          <div className="site-foundation-grid site-section-flush">
            {foundations.map((feature) => (
              <article key={feature.title} className="site-foundation-card">
                <div className="site-icon-box">
                  <ThemeIcon name={feature.icon} />
                </div>
                <div className="site-foundation-cat" style={{ color: "var(--cyan)", marginBottom: 12 }}>
                  {t(feature.cat)}
                </div>
                <h3 className="site-card-title">{t(feature.title)}</h3>
                <p className="site-card-copy">{t(feature.desc)}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </>
  );
}
