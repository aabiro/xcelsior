"use client";

import Link from "next/link";
import { AuthAwareLink } from "@/components/marketing/auth-aware-link";
import { SITE_ASSETS, siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const kpis = [
  ["home.stat_price_kicker", "home.stat_price"],
  ["home.stat_pipeda_kicker", "home.stat_pipeda"],
  ["home.stat_hydro_kicker", "home.stat_hydro"],
] as const;

const values = [
  ["01", "home.val_compliance_title", "home.val_compliance_desc"],
  ["02", "home.val_pricing_title", "home.val_pricing_desc"],
] as const;

const features = [
  ["gpu", "home.feat_marketplace_title", "home.feat_marketplace_desc"],
  ["shield-check", "home.feat_trust_title", "home.feat_trust_desc"],
  ["activity", "home.feat_telemetry_title", "home.feat_telemetry_desc"],
  ["route", "home.feat_jurisdiction_title", "home.feat_jurisdiction_desc"],
  ["dollar", "home.feat_spot_title", "home.feat_spot_desc"],
  ["leaf", "home.feat_green_title", "home.feat_green_desc"],
] as const;

const comparisonRows = [
  ["home.cmp_pipeda", "home.cmp_pipeda_x", "home.cmp_pipeda_aws", "home.cmp_pipeda_vast", "home.cmp_pipeda_rp"],
  ["home.cmp_price", "home.cmp_price_x", "home.cmp_price_aws", "home.cmp_price_vast", "home.cmp_price_rp"],
  ["home.cmp_cad", "home.cmp_cad_x", "home.cmp_cad_aws", "home.cmp_cad_vast", "home.cmp_cad_rp"],
  ["home.cmp_verification", "home.cmp_verification_x", "home.cmp_verification_aws", "home.cmp_verification_vast", "home.cmp_verification_rp"],
  ["home.cmp_green", "home.cmp_green_x", "home.cmp_green_aws", "home.cmp_green_vast", "home.cmp_green_rp"],
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
      <span className="site-marker-code">[ {code} ]</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
  );
}

export default function HomePage() {
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
                <span>{t("home.badge")}</span>
              </div>
              <h1 className="site-hero-title">
                {t("home.hero_line1")}
                <br />
                {t("home.hero_line2")} <span className="site-gradient-text">{t("home.hero_accent")}</span>
              </h1>
              <p className="site-hero-copy">{t("home.hero_desc")}</p>
              <div className="site-hero-actions">
                <AuthAwareLink intent="launch" className="site-button site-button-primary">
                  {t("home.cta_start")}
                </AuthAwareLink>
                <Link href="/pricing" className="site-button site-button-ghost">
                  {t("home.cta_pricing")}
                </Link>
              </div>
            </div>

            <div className="site-telemetry-wrap" aria-label="Live GPU market preview">
              <div className="site-telemetry-card">
                <div className="site-telemetry-head">
                  <div className="site-telemetry-model">
                    <span className="site-telemetry-mark">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={SITE_ASSETS.iconGradient} style={{ width: 20, height: 20 }} alt="" aria-hidden />
                    </span>
                    <div>
                      <div className="site-mono" style={{ color: "var(--text)", fontSize: 13, fontWeight: 600 }}>RTX 4090</div>
                      <div className="site-mono" style={{ color: "var(--text-4)", fontSize: 11 }}>24 GB VRAM</div>
                    </div>
                  </div>
                  <span className="site-live-badge">
                    <span className="site-live-dot" />
                    Live
                  </span>
                </div>

                {[
                  ["Utilization", "72%", "72%"],
                  ["Memory", "18.4 GB", "76%"],
                  ["Power", "310 W", "64%"],
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

                <div className="site-telemetry-price">
                  <div className="site-telemetry-price-copy">
                    <div className="site-telemetry-price-kicker">Spot · QC Hydro</div>
                    <div className="site-telemetry-price-value">
                      $0.30<span className="site-telemetry-price-unit"> CAD/hr</span>
                    </div>
                  </div>
                  <AuthAwareLink intent="launch" className="site-telemetry-action">
                    {t("gpus.deploy")} →
                  </AuthAwareLink>
                </div>
              </div>
            </div>
          </div>

          <div className="site-rails site-kpi-strip">
            {kpis.map(([kicker, label]) => (
              <div key={label} className="site-kpi">
                <div className="site-kpi-label">{t(kicker)}</div>
                <div className="site-kpi-value">{t(label)}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section">
          <SectionMarker code="01" label={t("home.why_title")} />
          <p className="site-section-copy">{t("home.why_desc")}</p>
          <div className="site-value-grid" style={{ marginTop: 54, gridTemplateColumns: "repeat(2, 1fr)" }}>
            {values.map(([number, title, desc]) => (
              <article key={title} className="site-value-card">
                <div className="site-number-badge">{number}</div>
                <h3 className="site-card-title">{t(title)}</h3>
                <p className="site-card-copy">{t(desc)}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="02" label={t("home.built_title")} />
          <div className="site-feature-grid site-section-flush">
            {features.map(([icon, title, desc]) => (
              <article key={title} className="site-feature-card">
                <div className="site-icon-box">
                  <ThemeIcon name={icon} />
                </div>
                <h3 className="site-card-title">{t(title)}</h3>
                <p className="site-card-copy">{t(desc)}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="03" label={t("home.compare_title")} />
          <div className="site-table-wrap" style={{ marginTop: 36 }}>
            <table className="site-table">
              <thead>
                <tr>
                  <th>{t("home.compare_feature")}</th>
                  <th className="site-table-head-x">{t("home.compare_xcelsior")}</th>
                  <th>{t("home.compare_aws")}</th>
                  <th>{t("home.compare_vast")}</th>
                  <th>{t("home.compare_runpod")}</th>
                </tr>
              </thead>
              <tbody>
                {comparisonRows.map(([feature, xcelsior, aws, vast, runpod]) => (
                  <tr key={feature}>
                    <td className="site-table-feature">{t(feature)}</td>
                    <td className="site-table-x">{t(xcelsior)}</td>
                    <td>{t(aws)}</td>
                    <td>{t(vast)}</td>
                    <td>{t(runpod)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="site-rails site-cta">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SITE_ASSETS.iconGradient} className="site-cta-mark" alt="" aria-hidden />
          <div className="site-eyebrow site-cta-eyebrow">{t("home.cta_eyebrow")}</div>
          <h2 className="site-cta-title">
            {t("home.cta_title_1")} <span className="site-gradient-text">{t("home.cta_title_2")}</span>
          </h2>
          <p className="site-section-copy" style={{ marginBottom: 28 }}>{t("home.cta_desc")}</p>
          <AuthAwareLink intent="start" className="site-button site-cta-button">
            {t("home.cta_button")}
          </AuthAwareLink>
        </section>
      </div>
    </>
  );
}
