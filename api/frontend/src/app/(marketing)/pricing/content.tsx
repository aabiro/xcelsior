"use client";

import { useEffect, useMemo } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { AuthAwareLink } from "@/components/marketing/auth-aware-link";
import posthog from "posthog-js";
import { SITE_ASSETS, siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";
import { gpuTierBadge, marketingGpuLabel } from "@/lib/marketing-gpu";
import { formatHourlyRate } from "@/lib/hourly-rate";

const SavingsCalculator = dynamic(
  () => import("./calculator").then((mod) => mod.SavingsCalculator),
  { loading: () => <div className="site-table-wrap" style={{ height: 220 }} aria-hidden /> },
);

interface GpuRow {
  model: string;
  vram: number;
  onDemand: number;
  spot: number | null;
  reserved1m: number | null;
  reserved1y: number | null;
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

  const minimum = (values: (number | null)[]) => {
    const known = values.filter((value): value is number => value !== null);
    return known.length ? Math.min(...known) : null;
  };
  const rateLabel = (value: number | null) => value === null ? "—" : formatHourlyRate(value);
  const cheapestSpot = minimum(gpus.map(gpu => gpu.spot));
  const maxSpotSaving = gpus.reduce((best, gpu) => {
    if (!gpu.onDemand || gpu.spot === null) return best;
    return Math.max(best, Math.round((1 - gpu.spot / gpu.onDemand) * 100));
  }, 0);
  const maxReservedSaving = gpus.reduce((best, gpu) => {
    if (!gpu.onDemand || gpu.reserved1y === null) return best;
    return Math.max(best, Math.round((1 - gpu.reserved1y / gpu.onDemand) * 100));
  }, 0);
  // Each plan's from-price is the cheapest row this page lists, never a fixed figure.
  const fromPrices: Record<string, number | null> = {
    "pricing.tier_ondemand_from": gpus.length ? Math.min(...gpus.map((gpu) => gpu.onDemand)) : null,
    "pricing.tier_reserved_from": minimum(gpus.map((gpu) => gpu.reserved1y)),
  };
  const spotSavingLabel = cheapestSpot === null ? "—" : `${maxSpotSaving}%`;
  const reservedSavingLabel = fromPrices["pricing.tier_reserved_from"] === null ? "—" : `${maxReservedSaving}%`;
  const bestGpu = [...gpus].filter(gpu => gpu.spot !== null).sort((a, b) => a.spot! - b.spot!)[0];
  const rows = useMemo(
    () => [...gpus].sort((a, b) => (a.spot ?? Infinity) - (b.spot ?? Infinity) || a.model.localeCompare(b.model)),
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
                        {bestGpu ? marketingGpuLabel(bestGpu.model) : t("pricing.rates_unavailable")}
                      </div>
                      <div className="site-mono" style={{ color: "var(--text-4)", fontSize: 11 }}>
                        {bestGpu ? `${bestGpu.vram} GB VRAM` : "—"}
                      </div>
                    </div>
                  </div>
                  <span className="site-live-badge">
                    <span className="site-live-dot" />
                    {gpus.length ? "Live" : "—"}
                  </span>
                </div>

                {[
                  [t("pricing.col_spot"), `${rateLabel(cheapestSpot)}/hr`, cheapestSpot === null ? "0%" : "100%"],
                  [t("pricing.stat_spot_save"), spotSavingLabel, `${Math.min(maxSpotSaving, 100)}%`],
                  [t("pricing.tier_reserved_title"), reservedSavingLabel, `${Math.min(maxReservedSaving, 100)}%`],
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
                  <strong style={{ color: "var(--text)", fontSize: 25 }}>{rateLabel(cheapestSpot)}/hr</strong>
                </div>
              </div>
            </div>
          </div>

          <div className="site-rails site-kpi-strip">
            <div className="site-kpi">
              <div className="site-kpi-label">{t("pricing.col_spot")}</div>
              <div className="site-kpi-value">{rateLabel(cheapestSpot)}</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">{t("pricing.stat_spot_save")}</div>
              <div className="site-kpi-value">{spotSavingLabel}</div>
            </div>
            <div className="site-kpi">
              <div className="site-kpi-label">{t("pricing.tier_reserved_badge")}</div>
              <div className="site-kpi-value">{reservedSavingLabel}</div>
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
                {/* Kept on every card, empty when there is nothing to say, so the
                    titles, prices and rules line up across the row. */}
                <div className="site-product-badge site-plan-badge" aria-hidden={!plan.featured}>
                  {plan.featured && fromPrices[plan.price] !== null ? t("pricing.tier_reserved_saving", { pct: maxReservedSaving }) : "\u00a0"}
                </div>
                <h3 className="site-card-title">{t(plan.title)}</h3>
                <p className="site-card-copy">{t(plan.desc)}</p>
                <div className="site-plan-price">
                  <strong>{fromPrices[plan.price] != null ? t(plan.price, { price: formatHourlyRate(fromPrices[plan.price]!) }) : "—"}</strong>
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
                {!rows.length && <tr><td colSpan={6}>{t("pricing.rates_unavailable")}</td></tr>}
                {rows.map((gpu) => (
                  <tr key={gpu.model}>
                    <td className="site-table-feature">
                      <span className="site-fleet-dot" style={{ color: tierColor(gpuTierBadge(gpu.model)) }} />
                      {marketingGpuLabel(gpu.model)}
                    </td>
                    <td>{gpu.vram} GB</td>
                    <td className="site-table-x">{rateLabel(gpu.spot)}</td>
                    <td>{formatHourlyRate(gpu.onDemand)}</td>
                    <td>{rateLabel(gpu.reserved1m)}</td>
                    <td>{rateLabel(gpu.reserved1y)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section id="calculator" className="site-rails site-section">
          <SectionMarker code="03" label={t("pricing.savings_calculator")} />
          <SavingsCalculator gpus={gpus.map((gpu) => ({ model: gpu.model, onDemand: gpu.onDemand, reserved: gpu.reserved1y }))} />
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
