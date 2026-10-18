"use client";

import { ObfuscationSafeMailto } from "@/components/marketing/ObfuscationSafeMailto";
import { useLocale } from "@/lib/locale";

export function PrivacyContent() {
  const { t } = useLocale();

  return (
    <div className="site-container">
      <div className="site-rails site-section site-legal-shell">
        <h1 className="site-section-heading site-legal-heading">{t("privacy.title")}</h1>
        <p className="site-legal-effective">{t("privacy.effective")}</p>

        <div className="site-legal-body">
          <Section title={`1. ${t("privacy.s1_title")}`}>
            <p>
              {t("privacy.s1_p1")} {t("privacy.s1_p2")}
            </p>
          </Section>

          <Section title={`2. ${t("privacy.s2_title")}`}>
            <ul className="site-legal-list">
              <li>{t("privacy.s2_p1")}</li>
              <li>{t("privacy.s2_p2")}</li>
              <li>{t("privacy.s2_p3")}</li>
              <li>{t("privacy.s2_p4")}</li>
              <li>{t("privacy.s2_p5")}</li>
            </ul>
          </Section>

          <Section title={`3. ${t("privacy.s3_title")}`}>
            <ul className="site-legal-list">
              <li>{t("privacy.s3_p1")}</li>
              <li>{t("privacy.s3_p2")}</li>
              <li>{t("privacy.s3_p3")}</li>
              <li>{t("privacy.s3_p4")}</li>
              <li>{t("privacy.s3_p5")}</li>
              <li>{t("privacy.s3_p6")}</li>
              <li>{t("privacy.s3_p7")}</li>
            </ul>
          </Section>

          <Section title={`4. ${t("privacy.s4_title")}`}>
            <p>{t("privacy.s4_p1")}</p>
          </Section>

          <Section title={`5. ${t("privacy.s5_title")}`}>
            <p>{t("privacy.s5_p1")}</p>
            <p className="site-legal-paragraph-gap">{t("privacy.s5_p2")}</p>
          </Section>

          <Section title={`6. ${t("privacy.s6_title")}`}>
            <ul className="site-legal-list">
              <li>{t("privacy.s6_p1")}</li>
              <li>{t("privacy.s6_p2")}</li>
              <li>{t("privacy.s6_p3")}</li>
            </ul>
          </Section>

          <Section title={`7. ${t("privacy.s7_title")}`}>
            <ul className="site-legal-list">
              <li>{t("privacy.s7_p1")}</li>
              <li>{t("privacy.s7_p2")}</li>
              <li>{t("privacy.s7_p3")}</li>
              <li>{t("privacy.s7_p4")}</li>
              <li>{t("privacy.s7_p5")}</li>
            </ul>
          </Section>

          <Section title={`8. ${t("privacy.s8_title")}`}>
            <p>{t("privacy.s8_intro")}</p>
            <ul className="site-legal-list site-legal-list-gap">
              <li>{t("privacy.s8_p1")}</li>
              <li>{t("privacy.s8_p2")}</li>
              <li>{t("privacy.s8_p3")}</li>
              <li>{t("privacy.s8_p4")}</li>
              <li>{t("privacy.s8_p5")}</li>
            </ul>
            <p className="site-legal-paragraph-gap">
              {t("privacy.s8_contact")}{" "}
              <ObfuscationSafeMailto href="mailto:privacy@xcelsior.ca" className="site-inline-link">
                privacy@xcelsior.ca
              </ObfuscationSafeMailto>{" "}
              {t("privacy.s8_contact_suffix")}
            </p>
          </Section>

          <Section title={`9. ${t("privacy.s9_title")}`}>
            <ul className="site-legal-list">
              <li>{t("privacy.s9_p1")}</li>
              <li>{t("privacy.s9_p2")}</li>
              <li>{t("privacy.s9_p3")}</li>
              <li>{t("privacy.s9_p4")}</li>
            </ul>
            <p className="site-legal-paragraph-gap">{t("privacy.s9_p5")}</p>
          </Section>

          <Section title={`10. ${t("privacy.s10_title")}`}>
            <p>{t("privacy.s10_p1")}</p>
          </Section>

          <Section title={`11. ${t("privacy.s11_title")}`}>
            <p>{t("privacy.s11_p1")}</p>
          </Section>

          <Section title={`12. ${t("privacy.s12_title")}`}>
            <p>{t("privacy.s12_p1")}</p>
          </Section>

          <Section title={`13. ${t("privacy.s13_title")}`}>
            <p>
              {t("privacy.s13_p1")}
              <br />
              <ObfuscationSafeMailto href="mailto:privacy@xcelsior.ca" className="site-inline-link">
                {t("privacy.s13_email")}
              </ObfuscationSafeMailto>
              <br />
              {t("privacy.s13_p2")}
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
