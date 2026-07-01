"use client";

import Link from "next/link";
import { SITE_ASSETS, siteIcon } from "@/lib/brand-assets";
import { useLocale } from "@/lib/locale";

const values = [
  { icon: "shield", title: "about.val_sovereignty_title", desc: "about.val_sovereignty_desc" },
  { icon: "leaf", title: "about.val_green_title", desc: "about.val_green_desc" },
  { icon: "users", title: "about.val_community_title", desc: "about.val_community_desc" },
  { icon: "bolt", title: "about.val_access_title", desc: "about.val_access_desc" },
  { icon: "sparkle", title: "about.val_canada_title", desc: "about.val_canada_desc" },
  { icon: "globe", title: "about.val_local_title", desc: "about.val_local_desc" },
] as const;

const milestoneGroups = [
  { year: "about.journey_2024_title", events: ["about.journey_2024_p1", "about.journey_2024_p2"] },
  { year: "about.journey_2025_title", events: ["about.journey_2025_p1", "about.journey_2025_p2", "about.journey_2025_p3"] },
  { year: "about.journey_2026_title", events: ["about.journey_2026_p1"] },
];

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

export function AboutContent() {
  const { t } = useLocale();

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails" style={{ gridTemplateColumns: "1fr" }}>
            <div style={{ animation: "heroUp .7s ease both" }}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{t("about.badge")}</span>
              </div>
              <h1 className="site-hero-title">
                {t("about.title").split(/(open to the world|ouvert au monde)/i).map((part, i) =>
                  /open to the world|ouvert au monde/i.test(part) ? (
                    <span key={i} className="site-gradient-text">{part}</span>
                  ) : (
                    <span key={i}>{part}</span>
                  ),
                )}
              </h1>
              <p className="site-hero-copy" style={{ maxWidth: 640 }}>{t("about.subtitle")}</p>
            </div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section">
          <div className="site-callout">
            <h2 className="site-callout-title">{t("about.mission_title")}</h2>
            <p className="site-callout-copy">{t("about.mission_p1")}</p>
          </div>
        </section>

        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="01" label={t("about.values_title")} />
          <h2 className="site-section-heading">{t("about.values_title")}</h2>
          <div className="site-foundation-grid site-section-flush">
            {values.map((value) => (
              <article key={value.title} className="site-foundation-card">
                <div className="site-icon-box">
                  <ThemeIcon name={value.icon} />
                </div>
                <h3 className="site-card-title">{t(value.title)}</h3>
                <p className="site-card-copy">{t(value.desc)}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="02" label={t("about.journey_title")} />
          <h2 className="site-section-heading" style={{ marginBottom: 48 }}>{t("about.journey_title")}</h2>
          <div className="site-timeline">
            {milestoneGroups.flatMap((group) =>
              group.events.map((eventKey, ei) => (
                <div key={eventKey} className="site-timeline-item">
                  <span className="site-timeline-dot" />
                  {ei === 0 && <span className="site-timeline-year">{t(group.year)}</span>}
                  <p className="site-timeline-copy">{t(eventKey)}</p>
                </div>
              )),
            )}
          </div>
        </section>

        <section className="site-rails site-cta">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={SITE_ASSETS.iconGradient} className="site-cta-mark" alt="" aria-hidden />
          <h2 className="site-cta-title">{t("about.cta_title")}</h2>
          <p className="site-section-copy" style={{ marginBottom: 28 }}>{t("about.cta_desc")}</p>
          <Link href="/register" className="site-button site-button-primary" style={{ padding: "15px 28px" }}>
            {t("about.cta_button")}
          </Link>
        </section>
      </div>
    </>
  );
}
