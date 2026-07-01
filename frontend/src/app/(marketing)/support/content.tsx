"use client";

import Link from "next/link";
import { BookOpen, Clock, Headphones, Mail, MessageCircle, Shield } from "lucide-react";
import { m } from "@/components/marketing/motion";
import { useLocale } from "@/lib/locale";

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.5, ease: "easeOut" as const },
  }),
};

const channels = [
  { icon: MessageCircle, titleKey: "support.chat_title", descKey: "support.chat_desc", actionKey: "support.chat_action", type: "chat" as const },
  { icon: Mail, titleKey: "support.email_title", descKey: "support.email_desc", actionKey: "support.email_action", type: "email" as const },
  { icon: BookOpen, titleKey: "support.docs_title", descKey: "support.docs_desc", actionKey: "support.docs_action", type: "docs" as const },
];

function SectionMarker({ code, label }: { code: string; label: string }) {
  return (
    <div className="site-marker">
      <span className="site-marker-code">[ {code} ]</span>
      <span className="site-marker-line" />
      <span>{label}</span>
    </div>
  );
}

export function SupportContent() {
  const { t } = useLocale();

  return (
    <>
      <section className="site-hero">
        <div className="site-grid-bg" aria-hidden />
        <div className="site-container">
          <div className="site-rails site-hero-rails" style={{ gridTemplateColumns: "1fr" }}>
            <m.div initial="hidden" animate="visible" variants={fadeUp} custom={0}>
              <div className="site-pill">
                <Headphones className="site-pill-icon" />
                <span>{t("support.badge")}</span>
              </div>
              <h1 className="site-hero-title">{t("support.title")}</h1>
              <p className="site-hero-copy" style={{ maxWidth: 760 }}>{t("support.subtitle")}</p>
            </m.div>
          </div>
        </div>
      </section>

      <div className="site-container">
        <section className="site-rails site-section" style={{ paddingBottom: 0 }}>
          <SectionMarker code="01" label={t("support.title")} />
          <h2 className="site-section-heading">{t("support.title")}</h2>
          <div className="site-support-grid site-section-flush">
            {channels.map((channel, index) => {
              const Icon = channel.icon;
              return (
                <m.article
                  key={channel.type}
                  variants={fadeUp}
                  initial="hidden"
                  whileInView="visible"
                  viewport={{ once: true }}
                  custom={index}
                  className="site-feature-card site-channel-card"
                >
                  <div className="site-icon-box">
                    <Icon className="site-svg-icon" />
                  </div>
                  <h3 className="site-card-title">{t(channel.titleKey)}</h3>
                  <p className="site-card-copy">{t(channel.descKey)}</p>
                  {channel.type === "chat" ? (
                    <button
                      type="button"
                      onClick={() => {
                        window.dispatchEvent(new CustomEvent("open-chat-widget"));
                      }}
                      className="site-button site-button-ghost"
                      style={{ marginTop: 24, width: "fit-content", padding: "13px 18px" }}
                    >
                      {t(channel.actionKey)}
                    </button>
                  ) : null}
                  {channel.type === "email" ? (
                    <a
                      href="mailto:support@xcelsior.ca"
                      className="site-button site-button-ghost"
                      style={{ marginTop: 24, width: "fit-content", padding: "13px 18px" }}
                    >
                      {t(channel.actionKey)}
                    </a>
                  ) : null}
                  {channel.type === "docs" ? (
                    <a
                      href="https://docs.xcelsior.ca"
                      className="site-button site-button-ghost"
                      style={{ marginTop: 24, width: "fit-content", padding: "13px 18px" }}
                    >
                      {t(channel.actionKey)}
                    </a>
                  ) : null}
                </m.article>
              );
            })}
          </div>
        </section>

        <section className="site-rails site-section">
          <SectionMarker code="02" label={t("support.hours_title")} />
          <div className="site-support-notes">
            <div className="site-callout">
              <div className="site-inline-heading">
                <Clock className="site-inline-icon" />
                <h2 className="site-callout-title">{t("support.hours_title")}</h2>
              </div>
              <p className="site-callout-copy">{t("support.hours_desc")}</p>
            </div>
            <div className="site-callout site-callout-alert">
              <div className="site-inline-heading">
                <Shield className="site-inline-icon" />
                <h2 className="site-callout-title">{t("support.security_title")}</h2>
              </div>
              <p className="site-callout-copy">{t("support.security_desc")}</p>
            </div>
          </div>
        </section>

        <section className="site-rails site-cta">
          <h2 className="site-cta-title">{t("support.cta")}</h2>
          <Link href="/pricing" className="site-button site-button-primary" style={{ padding: "15px 28px" }}>
            {t("support.cta_button")}
          </Link>
        </section>
      </div>
    </>
  );
}
