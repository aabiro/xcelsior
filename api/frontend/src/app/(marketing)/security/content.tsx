"use client";

import Link from "next/link";
import { ObfuscationSafeMailto } from "@/components/marketing/ObfuscationSafeMailto";
import { useLocale } from "@/lib/locale";

/**
 * Security posture page (adoption plan X6.29 / X7.36).
 *
 * Two jobs. It is a listing asset a directory reviewer opens, and it is the
 * page an enterprise security questionnaire gets answered from — so it states
 * the model concretely rather than in assurances, and leads with the
 * differentiator: destructive operations are bound to a server-side plan a
 * human approves, not to a flag the model sets.
 */
export function SecurityContent() {
  const { t } = useLocale();

  return (
    <div className="site-container">
      <div className="site-rails site-section site-legal-shell">
        <h1 className="site-section-heading site-legal-heading">{t("security.title")}</h1>
        <p className="site-legal-effective">{t("security.effective")}</p>

        <div className="site-legal-body">
          <Section title={t("security.headline_title")}>
            <p>{t("security.headline_p1")}</p>
            <p className="site-legal-paragraph-gap">{t("security.headline_p2")}</p>
            <p className="site-legal-paragraph-gap">{t("security.headline_p3")}</p>
          </Section>

          <Section title={`1. ${t("security.s1_title")}`}>
            <ul className="site-legal-list">
              <li>{t("security.s1_p1")}</li>
              <li>{t("security.s1_p2")}</li>
              <li>{t("security.s1_p3")}</li>
              <li>{t("security.s1_p4")}</li>
            </ul>
          </Section>

          <Section title={`2. ${t("security.s2_title")}`}>
            <ul className="site-legal-list">
              <li>{t("security.s2_p1")}</li>
              <li>{t("security.s2_p2")}</li>
              <li>{t("security.s2_p3")}</li>
              <li>{t("security.s2_p4")}</li>
            </ul>
            <p className="site-legal-paragraph-gap">
              {t("security.s2_scopes")}{" "}
              <Link href="https://docs.xcelsior.ca/authentication" className="site-inline-link">
                {t("security.s2_scopes_link")}
              </Link>
              .
            </p>
          </Section>

          <Section title={`3. ${t("security.s3_title")}`}>
            <ul className="site-legal-list">
              <li>{t("security.s3_p1")}</li>
              <li>{t("security.s3_p2")}</li>
              <li>{t("security.s3_p3")}</li>
              <li>{t("security.s3_p4")}</li>
            </ul>
          </Section>

          <Section title={`4. ${t("security.s4_title")}`}>
            <p>{t("security.s4_p1")}</p>
            <ul className="site-legal-list">
              <li>{t("security.s4_p2")}</li>
              <li>{t("security.s4_p3")}</li>
              <li>{t("security.s4_p4")}</li>
            </ul>
          </Section>

          <Section title={`5. ${t("security.s5_title")}`}>
            <ul className="site-legal-list">
              <li>{t("security.s5_p1")}</li>
              <li>{t("security.s5_p2")}</li>
              <li>{t("security.s5_p3")}</li>
            </ul>
            <p className="site-legal-paragraph-gap">
              {t("security.s5_privacy")}{" "}
              <Link href="/privacy" className="site-inline-link">
                {t("security.s5_privacy_link")}
              </Link>
              .
            </p>
          </Section>

          <Section title={`6. ${t("security.s6_title")}`}>
            <ul className="site-legal-list">
              <li>{t("security.s6_p1")}</li>
              <li>{t("security.s6_p2")}</li>
              <li>{t("security.s6_p3")}</li>
            </ul>
          </Section>

          <Section title={`7. ${t("security.s7_title")}`}>
            <p>{t("security.s7_p1")}</p>
            <p className="site-legal-paragraph-gap">
              {t("security.s7_p2")}
              <br />
              <ObfuscationSafeMailto href="mailto:security@xcelsior.ca" className="site-inline-link">
                {t("security.s7_email")}
              </ObfuscationSafeMailto>
            </p>
          </Section>
        </div>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="site-legal-section">
      <h2 className="site-legal-title">{title}</h2>
      {children}
    </section>
  );
}
