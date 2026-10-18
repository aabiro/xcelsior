"use client";

import { useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { AuthAwareLink } from "@/components/marketing/auth-aware-link";
import posthog from "posthog-js";
import { SITE_ASSETS, siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";
import { gpuTierBadge, marketingGpuLabel } from "@/lib/marketing-gpu";

const SavingsCalculator = dynamic(
  () => import("./calculator").then((mod) => mod.SavingsCalculator),
  { loading: () => <div className="site-table-wrap" style={{ height: 220 }} aria-hidden /> },
);

interface GpuRow {
  model: string;
  vram: number;
  onDemand: number;
  spot: number;
  reserved1m: number;
  reserved1y: number;
}

const plans = [
  {
    icon: "bolt",
    title: "pricing.tier_ondemand_title",
    desc: "pricing.tier_ondemand_desc",
    price: "pricing.tier_ondemand_from",
    unit: "pricing.tier_ondemand_unit",
    features: ["pricing.tier_ondemand_i1", "pricing.tier_ondemand_i2", "pricing.tier_ondemand_i3", "pricing.tier_ondemand_i4"],
    featured: false,
  },
  {
    icon: "star",
    title: "pricing.tier_reserved_title",
    desc: "pricing.tier_reserved_desc",
    price: "pricing.tier_reserved_from",
    unit: "pricing.tier_ondemand_unit",
    features: ["pricing.tier_reserved_i1", "pricing.tier_reserved_i2", "pricing.tier_reserved_i3", "pricing.tier_reserved_i4"],
    featured: true,
  },
  {
    icon: "shield-check",
    title: "pricing.tier_enterprise_title",
    desc: "pricing.tier_enterprise_desc",
    price: "pricing.tier_enterprise_from",
    unit: "",
    features: ["pricing.tier_enterprise_i1", "pricing.tier_enterprise_i2", "pricing.tier_enterprise_i3", "pricing.tier_enterprise_i4"],
    featured: false,
  },
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

export function PricingContent({ gpus }: { gpus: GpuRow[] }) {
  const { t } = useLocale();

  useEffect(() => {
    posthog.capture("pricing_page_viewed", { gpu_count: gpus.length });
  }, [gpus.length]);

  const cheapestSpot = gpus.length ? Math.min(...gpus.map((gpu) => gpu.spot)) : 0.3;
  const maxSpotSaving = gpus.reduce((best, gpu) => {
    if (!gpu.onDemand) return best;
    return Math.max(best, Math.round((1 - gpu.spot / gpu.onDemand) * 100));
  }, 0);
  const maxReservedSaving = gpus.reduce((best, gpu) => {
    if (!gpu.onDemand) return best;
    return Math.max(best, Math.round((1 - gpu.reserved1y / gpu.onDemand) * 100));
  }, 0);
  const bestGpu = gpus.length
    ? gpus.reduce((a, b) => (a.spot <= b.spot ? a : b))
    : null;
  const rows = useMemo(
    () => [...gpus].sort((a, b) => a.spot - b.spot || a.model.localeCompare(b.model)),
    [gpus],
  );

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails">
            <div style={{ animation: "heroUp .7s ease both" }}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{t("pricing.badge")}</span>
              </div>
              <h1 className="site-hero-title">
                {t("pricing.title")} <span className="site-gradient-text">{t("pricing.col_spot")}</span>
              </h1>
              <p className="site-hero-copy">{t("pricing.subtitle")}</p>
              <div className="site-hero-actions">
                <AuthAwareLink intent="launch" className="site-button site-button-primary">
                  {t("pricing.cta_start")}
                </AuthAwareLink>
                <Link href="#calculator" className="site-button site-button-ghost">
                  {t("pricing.savings_calculator")}
                </Link>
              </div>
            </div>

            <div className="site-telemetry-wrap" aria-label="Live pricing preview">
              <div className="site-telemetry-card">
                <div className="site-telemetry-head">
                  <div className="site-telemetry-model">
                    <span className="site-telemetry-mark">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={SITE_ASSETS.iconGradient} style={{ width: 20, height: 20 }} alt="" aria-hidden />
                    </span>
                    <div>
                      <div className="site-mono" style={{ color: "var(--text)", fontSize: 13, fontWeight: 600 }}>
                        {bestGpu ? marketingGpuLabel(bestGpu.model) : "RTX 4090"}
                      </div>
                      <div className="site-mono" style={{ color: "var(--text-4)", fontSize: 11 }}>
                        {bestGpu ? `${bestGpu.vram} GB VRAM` : "Best value"}
                      </div>
                    </div>
                  </div>
                  <span className="site-live-badge">
                    <span className="site-live-dot" />
                    Live
                  </span>
                </div>

                {[
                  [t("pricing.col_spot"), `$${cheapestSpot.toFixed(2)}/hr`, "100%"],
                  [t("pricing.stat_spot_save"), `${maxSpotSaving || 70}%`, `${Math.min(maxSpotSaving || 70, 100)}%`],
                  [t("pricing.tier_reserved_title"), `${maxReservedSaving || 45}%`, `${Math.min(maxReservedSaving || 45, 100)}%`],
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
                  <span className="site-mono" style={{ color: "var(--text-4)", fontSize: 11, textTransform: "uppercase" }}>
                    {t("pricing.col_spot")}
                  </span>
                  <strong style={{ color: "var(--text)", fontSize: 25 }}>${cheapestSpot.toFixed(2)}/hr</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="site-rails site-kpi-strip">
            <div className="site-kpi">
              <div className="site-kpi-label">{t("pricing.col_spot")}</div>
              <div className="site-kpi-value">${cheapestSpot.toFixed(2)}</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">{t("pricing.stat_spot_save")}</div>
              <div className="site-kpi-value">{maxSpotSaving || 70}%</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">{t("pricing.tier_reserved_badge")}</div>
              <div className="site-kpi-value">{maxReservedSaving || 45}%</div>
            </div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="01" label={t("pricing.title")} />
          <p className="site-section-copy">{t("pricing.subtitle")}</p>
          <div className="site-pricing-cards site-section-flush">
            {plans.map((plan) => (
              <article key={plan.title} className={`site-plan-card ${plan.featured ? "site-plan-card-featured" : ""}`}>
                <div className="site-icon-box">
                  <ThemeIcon name={plan.icon} />
                </div>
                {plan.featured ? (
                  <div className="site-product-badge" style={{ color: "var(--gold)", marginBottom: 14 }}>
                    {t("pricing.tier_reserved_badge")} - {maxReservedSaving || 45}%
                  </div>
                ) : null}
                <h3 className="site-card-title">{t(plan.title)}</h3>
                <p className="site-card-copy">{t(plan.desc)}</p>
                <div className="site-plan-price">
                  <strong>{t(plan.price)}</strong>
                  <span style={{ color: "var(--text-4)" }}>{t(plan.unit)}</span>
                </div>
                <div className="site-plan-items">
                  {plan.features.map((feature) => (
                    <p key={feature} className="site-product-point">
                      <span style={{ color: "var(--green)" }}>+</span>
                      <span>{t(feature)}</span>
                    </p>
                  ))}
                </div>
                <AuthAwareLink intent="launch" className={`site-button ${plan.featured ? "site-button-primary" : "site-button-ghost"}`} style={{ marginTop: "auto", padding: "13px 18px" }}>
                  {t("pricing.cta_start")}
                </AuthAwareLink>
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="02" label={t("pricing.fleet_title")} />
          <p className="site-section-copy">{t("pricing.fleet_desc")}</p>
          <div className="site-table-wrap" style={{ marginTop: 36 }}>
            <table className="site-table" aria-label={t("pricing.table_label")}>
              <thead>
                <tr>
                  <th>{t("pricing.col_gpu")}</th>
                  <th>{t("pricing.col_vram")}</th>
                  <th>{t("pricing.col_spot")}</th>
                  <th>{t("pricing.col_ondemand")}</th>
                  <th>{t("pricing.col_reserved1")}</th>
                  <th>{t("pricing.col_reserved12")}</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((gpu) => (
                  <tr key={gpu.model}>
                    <td className="site-table-feature">
                      <span className="site-fleet-dot" style={{ color: tierColor(gpuTierBadge(gpu.model)) }} />
                      {marketingGpuLabel(gpu.model)}
                    </td>
                    <td>{gpu.vram} GB</td>
                    <td className="site-table-x">${gpu.spot.toFixed(2)}</td>
                    <td>${gpu.onDemand.toFixed(2)}</td>
                    <td>${gpu.reserved1m.toFixed(2)}</td>
                    <td>${gpu.reserved1y.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="calculator" className="site-rails site-section">
          <SectionMarker code="03" label={t("pricing.savings_calculator")} />
          <SavingsCalculator gpus={gpus.map((gpu) => ({ model: gpu.model, onDemand: gpu.onDemand }))} />
        </section>

        <section className="site-rails site-cta">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SITE_ASSETS.iconGradient} className="site-cta-mark" alt="" aria-hidden />
          <h2 className="site-cta-title">{t("pricing.cta_start")}</h2>
          <AuthAwareLink intent="launch" className="site-button site-button-primary" style={{ padding: "15px 28px" }}>
            {t("pricing.cta_start")}
          </AuthAwareLink>
        </section>
      </div>
    </>
  );
}

function tierColor(tier: ReturnType<typeof gpuTierBadge>) {
  switch (tier) {
    case "flagship":
      return "var(--gold)";
    case "datacenter":
      return "var(--cyan)";
    case "pro":
      return "var(--violet)";
    default:
      return "var(--green)";
  }
}
