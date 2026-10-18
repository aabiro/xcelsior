"use client";

import { ObfuscationSafeMailto } from "@/components/marketing/ObfuscationSafeMailto";
import { useLocale } from "@/lib/locale";

export function TermsContent() {
  const { t } = useLocale();

  return (
    <div className="site-container">
      <div className="site-rails site-section site-legal-shell">
        <h1 className="site-section-heading site-legal-heading">{t("terms.title")}</h1>
        <p className="site-legal-effective">{t("terms.effective")}</p>

        <div className="site-legal-body">
          <Section title={`1. ${t("terms.s1_title")}`}>
            <p>{t("terms.s1_p1")}</p>
          </Section>

          <Section title={`2. ${t("terms.s2_title")}`}>
            <ul className="site-legal-list">
              <li>{t("terms.s2_p1")}</li>
              <li>{t("terms.s2_p2")}</li>
              <li>{t("terms.s2_p3")}</li>
              <li>{t("terms.s2_p4")}</li>
              <li>{t("terms.s2_p5")}</li>
            </ul>
          </Section>

          <Section title={`3. ${t("terms.s3_title")}`}>
            <p>
              {t("terms.s3_p1")}{" "}
              <ObfuscationSafeMailto href="mailto:security@xcelsior.ca" className="site-inline-link">
                security@xcelsior.ca
              </ObfuscationSafeMailto>{" "}
              {t("terms.s3_p1_suffix")}
            </p>
          </Section>

          <Section title={`4. ${t("terms.s4_title")}`}>
            <p>{t("terms.s4_intro")}</p>
            <ul className="site-legal-list site-legal-list-gap">
              <li>{t("terms.s4_p1")}</li>
              <li>{t("terms.s4_p2")}</li>
              <li>{t("terms.s4_p3")}</li>
              <li>{t("terms.s4_p4")}</li>
              <li>{t("terms.s4_p5")}</li>
            </ul>
            <p className="site-legal-paragraph-gap">{t("terms.s4_note")}</p>
          </Section>

          <Section title={`5. ${t("terms.s5_title")}`}>
            <ul className="site-legal-list">
              <li>{t("terms.s5_p1")}</li>
              <li>{t("terms.s5_p2")}</li>
              <li>{t("terms.s5_p3")}</li>
              <li>{t("terms.s5_p4")}</li>
              <li>{t("terms.s5_p5")}</li>
              <li>{t("terms.s5_p6")}</li>
            </ul>
          </Section>

          <Section title={`6. ${t("terms.s6_title")}`}>
            <ul className="site-legal-list">
              <li>{t("terms.s6_p1")}</li>
              <li>{t("terms.s6_p2")}</li>
              <li>{t("terms.s6_p3")}</li>
              <li>{t("terms.s6_p4")}</li>
              <li>{t("terms.s6_p5")}</li>
            </ul>
          </Section>

          <Section title={`7. ${t("terms.s7_title")}`}>
            <p>{t("terms.s7_intro")}</p>
            <div className="site-table-wrap site-legal-table-wrap">
              <table className="site-table">
                <thead>
                  <tr>
                    <th>{t("terms.s7_col_tier")}</th>
                    <th>{t("terms.s7_col_uptime")}</th>
                    <th>{t("terms.s7_col_credit")}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="site-table-feature">{t("terms.s7_community")}</td>
                    <td>{t("terms.s7_community_up")}</td>
                    <td>{t("terms.s7_community_cr")}</td>
                  </tr>
                  <tr>
                    <td className="site-table-feature">{t("terms.s7_secure")}</td>
                    <td>{t("terms.s7_secure_up")}</td>
                    <td>{t("terms.s7_secure_cr")}</td>
                  </tr>
                  <tr>
                    <td className="site-table-feature">{t("terms.s7_enterprise")}</td>
                    <td>{t("terms.s7_enterprise_up")}</td>
                    <td>{t("terms.s7_enterprise_cr")}</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="site-legal-paragraph-gap">{t("terms.s7_note")}</p>
          </Section>

          <Section title={`8. ${t("terms.s8_title")}`}>
            <p>{t("terms.s8_p1")}</p>
          </Section>

          <Section title={`9. ${t("terms.s9_title")}`}>
            <p>
              {t("terms.s9_p1")}{" "}
              <a href="/privacy" className="site-inline-link">
                {t("footer.privacy")}
              </a>
            </p>
          </Section>

          <Section title={`10. ${t("terms.s10_title")}`}>
            <p>{t("terms.s10_p1")}</p>
          </Section>

          <Section title={`11. ${t("terms.s11_title")}`}>
            <p>{t("terms.s11_p1")}</p>
          </Section>

          <Section title={`12. ${t("terms.s12_title")}`}>
            <ul className="site-legal-list">
              <li>{t("terms.s12_p1")}</li>
              <li>{t("terms.s12_p2")}</li>
              <li>{t("terms.s12_p3")}</li>
            </ul>
          </Section>

          <Section title={`13. ${t("terms.s13_title")}`}>
            <p>{t("terms.s13_p1")}</p>
          </Section>

          <Section title={`14. ${t("terms.s14_title")}`}>
            <p>{t("terms.s14_p1")}</p>
          </Section>

          <Section title={`15. ${t("terms.s15_title")}`}>
            <p>
              {t("terms.s15_p1")}
              <br />
              <ObfuscationSafeMailto href="mailto:legal@xcelsior.ca" className="site-inline-link">
                {t("terms.s15_email")}
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
