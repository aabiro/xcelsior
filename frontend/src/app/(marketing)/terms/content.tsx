"use client";

import { useLocale } from "@/lib/locale";

export function TermsContent() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-4xl px-6 py-24">
      <h1 className="text-4xl font-bold mb-2">{t("terms.title")}</h1>
      <p className="text-sm text-text-muted mb-12">{t("terms.effective")}</p>

      <div className="prose-dark space-y-10 text-text-secondary leading-relaxed text-sm">
        <Section title={`1. ${t("terms.s1_title")}`}>
          <p>{t("terms.s1_p1")}</p>
        </Section>

        <Section title={`2. ${t("terms.s2_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
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
            <a href="mailto:security@xcelsior.ca" className="text-ice-blue hover:underline">
              security@xcelsior.ca
            </a>
          </p>
        </Section>

        <Section title={`4. ${t("terms.s4_title")}`}>
          <p>{t("terms.s4_intro")}</p>
          <ul className="list-disc pl-5 space-y-2 mt-2">
            <li>{t("terms.s4_p1")}</li>
            <li>{t("terms.s4_p2")}</li>
            <li>{t("terms.s4_p3")}</li>
            <li>{t("terms.s4_p4")}</li>
            <li>{t("terms.s4_p5")}</li>
          </ul>
          <p className="mt-3">{t("terms.s4_note")}</p>
        </Section>

        <Section title={`5. ${t("terms.s5_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
            <li>{t("terms.s5_p1")}</li>
            <li>{t("terms.s5_p2")}</li>
            <li>{t("terms.s5_p3")}</li>
            <li>{t("terms.s5_p4")}</li>
            <li>{t("terms.s5_p5")}</li>
            <li>{t("terms.s5_p6")}</li>
          </ul>
        </Section>

        <Section title={`6. ${t("terms.s6_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
            <li>{t("terms.s6_p1")}</li>
            <li>{t("terms.s6_p2")}</li>
            <li>{t("terms.s6_p3")}</li>
            <li>{t("terms.s6_p4")}</li>
            <li>{t("terms.s6_p5")}</li>
          </ul>
        </Section>

        <Section title={`7. ${t("terms.s7_title")}`}>
          <p>{t("terms.s7_intro")}</p>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-left">
                  <th className="py-2 pr-4 font-medium text-text-primary">{t("terms.s7_col_tier")}</th>
                  <th className="py-2 pr-4 font-medium text-text-primary">{t("terms.s7_col_uptime")}</th>
                  <th className="py-2 font-medium text-text-primary">{t("terms.s7_col_credit")}</th>
                </tr>
              </thead>
              <tbody className="text-text-secondary">
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-4">{t("terms.s7_community")}</td>
                  <td className="py-2 pr-4">{t("terms.s7_community_up")}</td>
                  <td className="py-2">{t("terms.s7_community_cr")}</td>
                </tr>
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-4">{t("terms.s7_secure")}</td>
                  <td className="py-2 pr-4">{t("terms.s7_secure_up")}</td>
                  <td className="py-2">{t("terms.s7_secure_cr")}</td>
                </tr>
                <tr className="border-b border-border/50">
                  <td className="py-2 pr-4">{t("terms.s7_sovereign")}</td>
                  <td className="py-2 pr-4">{t("terms.s7_sovereign_up")}</td>
                  <td className="py-2">{t("terms.s7_sovereign_cr")}</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3">{t("terms.s7_note")}</p>
        </Section>

        <Section title={`8. ${t("terms.s8_title")}`}>
          <p>{t("terms.s8_p1")}</p>
        </Section>

        <Section title={`9. ${t("terms.s9_title")}`}>
          <p>
            {t("terms.s9_p1")}{" "}
            <a href="/privacy" className="text-ice-blue hover:underline">
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
          <ul className="list-disc pl-5 space-y-2">
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
            <a href="mailto:legal@xcelsior.ca" className="text-ice-blue hover:underline">
              {t("terms.s15_email")}
            </a>
          </p>
        </Section>
      </div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="text-lg font-semibold text-text-primary mb-3">{title}</h2>
      {children}
    </section>
  );
}
