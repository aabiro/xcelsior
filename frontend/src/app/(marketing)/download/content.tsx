"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Bell,
  ChevronRight,
  Cpu,
  Globe,
  Link2,
  Monitor,
  RefreshCw,
  Shield,
  Smartphone,
  Terminal,
} from "lucide-react";
import { m } from "@/components/marketing/motion";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.08, duration: 0.5, ease: "easeOut" as const },
  }),
};

type Platform = "macos" | "windows" | "linux" | "unknown";

function SectionMarker({ code, label }: { code: string; label: string }) {
  return (
    <div className="site-marker">
      <span className="site-marker-code">{code}</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
  );
}

function detectPlatform(): Platform {
  if (typeof navigator === "undefined") return "unknown";
  const ua = navigator.userAgent.toLowerCase();
  if (ua.includes("mac")) return "macos";
  if (ua.includes("win")) return "windows";
  if (ua.includes("linux")) return "linux";
  return "unknown";
}

const DOWNLOAD_BASE = "https://downloads.xcelsior.ca/desktop";

const PLATFORMS = [
  {
    id: "macos" as Platform,
    label: "macOS",
    desc: "Apple Silicon & Intel",
    file: "Xcelsior_0.1.0_aarch64.dmg",
    icon: Monitor,
  },
  {
    id: "windows" as Platform,
    label: "Windows",
    desc: "Windows 10+",
    file: "Xcelsior_0.1.0_x64-setup.exe",
    icon: Monitor,
  },
  {
    id: "linux" as Platform,
    label: "Linux",
    desc: ".deb / .AppImage / .rpm",
    file: "Xcelsior_0.1.0_amd64.AppImage",
    icon: Terminal,
  },
];

const DESKTOP_FEATURES = [
  { icon: Bell, titleKey: "download.feature_tray_title", descKey: "download.feature_tray_desc" },
  { icon: RefreshCw, titleKey: "download.feature_updates_title", descKey: "download.feature_updates_desc" },
  { icon: Link2, titleKey: "download.feature_links_title", descKey: "download.feature_links_desc" },
  { icon: Shield, titleKey: "download.feature_single_title", descKey: "download.feature_single_desc" },
  { icon: Cpu, titleKey: "download.feature_control_title", descKey: "download.feature_control_desc" },
  { icon: Monitor, titleKey: "download.feature_login_title", descKey: "download.feature_login_desc" },
];

export function DownloadContent() {
  const [detected] = useState<Platform | null>(() =>
    typeof window === "undefined" ? null : detectPlatform(),
  );
  const { t } = useLocale();

  const primary =
    detected == null || detected === "unknown"
      ? PLATFORMS[0]
      : PLATFORMS.find((p) => p.id === detected) ?? PLATFORMS[0];
  const platformLabel = detected == null ? t("download.primary_generic") : primary.label;
  const others = PLATFORMS.filter((p) => p.id !== primary.id);
  const PrimaryIcon = primary.icon;

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails">
            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
              <div className="site-pill">
                <span className="site-live-dot" />
                <span>{platformLabel}</span>
              </div>
              <h1 className="site-hero-title">{t("download.hero_title")}</h1>
              <p className="site-hero-copy">{t("download.hero_subtitle")}</p>
            </m.div>

            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={1} className="site-telemetry-wrap">
              <a href={`${DOWNLOAD_BASE}/${primary.file}`} className="site-download-primary">
                <div className="site-icon-box">
                  <PrimaryIcon className="site-svg-icon" />
                </div>
                <div className="site-download-primary-copy">
                  <p className="site-card-title">
                    {detected == null
                      ? platformLabel
                      : t("download.primary_for", { platform: platformLabel })}
                  </p>
                  <p className="site-card-copy">{primary.desc}</p>
                </div>
                <span className="site-download-pill">{t("download.primary_cta")}</span>
              </a>
            </m.div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section">
          <SectionMarker code="01" label={t("download.primary_cta")} />
          <div className="site-download-grid">
            <div className="site-callout">
              <h2 className="site-callout-title">
                {detected == null
                  ? platformLabel
                  : t("download.primary_for", { platform: platformLabel })}
              </h2>
              <p className="site-callout-copy">{primary.desc}</p>
              <div className="site-hero-actions" style={{ marginTop: 28 }}>
                <a href={`${DOWNLOAD_BASE}/${primary.file}`} className="site-button site-button-primary">
                  {t("download.primary_cta")}
                </a>
              </div>
            </div>
            <div>
              <div className="site-download-secondary-grid">
                {others.map((platform) => {
                  const Icon = platform.icon;
                  return (
                    <a
                      key={platform.id}
                      href={`${DOWNLOAD_BASE}/${platform.file}`}
                      className="site-download-secondary-card"
                    >
                      <div className="site-icon-box">
                        <Icon className="site-svg-icon" />
                      </div>
                      <div>
                        <p className="site-card-title" style={{ marginBottom: 8, fontSize: 20 }}>{platform.label}</p>
                        <p className="site-card-copy">{platform.desc}</p>
                      </div>
                      <ChevronRight className="site-download-chevron" />
                    </a>
                  );
                })}
              </div>
              <p className="site-kpi-label" style={{ marginTop: 18 }}>{t("download.build_note")}</p>
            </div>
          </div>
        </section>

        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="02" label={t("download.section_what_adds")} />
          <h2 className="site-section-heading">{t("download.section_what_adds")}</h2>
          <p className="site-section-copy">{t("download.section_what_adds_desc")}</p>
          <div className="site-feature-grid site-section-flush">
            {DESKTOP_FEATURES.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <m.article
                  key={feature.titleKey}
                  variants={fadeUp}
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  custom={index}
                  className="site-feature-card"
                >
                  <div className="site-icon-box">
                    <Icon className="site-svg-icon" />
                  </div>
                  <h3 className="site-card-title">{t(feature.titleKey)}</h3>
                  <p className="site-card-copy">{t(feature.descKey)}</p>
                </m.article>
              );
            })}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="03" label={t("download.mobile_title")} />
          <div className="site-split-panel">
            <div className="site-split-panel-media site-mobile-art-wrap">
              <div className="site-icon-box site-mobile-art-box">
                <Smartphone className="site-svg-icon" />
              </div>
            </div>
            <div className="site-split-panel-body">
              <h2 className="site-callout-title">{t("download.mobile_title")}</h2>
              <p className="site-callout-copy">{t("download.mobile_p1")}</p>
              <p className="site-callout-copy" style={{ marginTop: 18 }}>{t("download.mobile_p2")}</p>
              <div className="site-inline-points">
                <div className="site-inline-point"><Globe className="site-inline-icon" />{t("download.mobile_b1")}</div>
                <div className="site-inline-point"><Bell className="site-inline-icon" />{t("download.mobile_b2")}</div>
                <div className="site-inline-point"><Shield className="site-inline-icon" />{t("download.mobile_b3")}</div>
              </div>
            </div>
          </div>
        </section>

        <section className="site-rails site-cta">
          <h2 className="site-cta-title">{t("download.footer_note")}</h2>
          <div className="site-hero-actions">
            <Link href="/register" className="site-button site-button-primary">
              {t("download.footer_create")}
            </Link>
            <Link href="/login" className="site-button site-button-ghost">
              {t("download.footer_signin")}
            </Link>
          </div>
        </section>
      </div>
    </>
  );
}
