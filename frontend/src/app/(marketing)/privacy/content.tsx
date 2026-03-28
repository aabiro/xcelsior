"use client";

import { useLocale } from "@/lib/locale";

export function PrivacyContent() {
  const { t } = useLocale();

  return (
    <div className="mx-auto max-w-4xl px-6 py-24">
      <h1 className="text-4xl font-bold mb-2">{t("privacy.title")}</h1>
      <p className="text-sm text-text-muted mb-12">{t("privacy.effective")}</p>

      <div className="prose-dark space-y-10 text-text-secondary leading-relaxed text-sm">
        <Section title={`1. ${t("privacy.s1_title")}`}>
          <p>{t("privacy.s1_p1")} {t("privacy.s1_p2")}</p>
        </Section>

        <Section title={`2. ${t("privacy.s2_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
            <li>{t("privacy.s2_p1")}</li>
            <li>{t("privacy.s2_p2")}</li>
            <li>{t("privacy.s2_p3")}</li>
            <li>{t("privacy.s2_p4")}</li>
            <li>{t("privacy.s2_p5")}</li>
          </ul>
        </Section>

        <Section title={`3. ${t("privacy.s3_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
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
          <p className="mt-3">{t("privacy.s5_p2")}</p>
        </Section>

        <Section title={`6. ${t("privacy.s6_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
            <li>{t("privacy.s6_p1")}</li>
            <li>{t("privacy.s6_p2")}</li>
            <li>{t("privacy.s6_p3")}</li>
          </ul>
        </Section>

        <Section title={`7. ${t("privacy.s7_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
            <li>{t("privacy.s7_p1")}</li>
            <li>{t("privacy.s7_p2")}</li>
            <li>{t("privacy.s7_p3")}</li>
            <li>{t("privacy.s7_p4")}</li>
            <li>{t("privacy.s7_p5")}</li>
          </ul>
        </Section>

        <Section title={`8. ${t("privacy.s8_title")}`}>
          <p>{t("privacy.s8_intro")}</p>
          <ul className="list-disc pl-5 space-y-2 mt-2">
            <li>{t("privacy.s8_p1")}</li>
            <li>{t("privacy.s8_p2")}</li>
            <li>{t("privacy.s8_p3")}</li>
            <li>{t("privacy.s8_p4")}</li>
            <li>{t("privacy.s8_p5")}</li>
          </ul>
          <p className="mt-3">
            {t("privacy.s8_contact")}{" "}
            <a href="mailto:privacy@xcelsior.ca" className="text-ice-blue hover:underline">
              privacy@xcelsior.ca
            </a>
          </p>
        </Section>

        <Section title={`9. ${t("privacy.s9_title")}`}>
          <ul className="list-disc pl-5 space-y-2">
            <li>{t("privacy.s9_p1")}</li>
            <li>{t("privacy.s9_p2")}</li>
          </ul>
          <p className="mt-3">{t("privacy.s9_p3")}</p>
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
            <a href="mailto:privacy@xcelsior.ca" className="text-ice-blue hover:underline">
              {t("privacy.s13_email")}
            </a>
            <br />
            {t("privacy.s13_p2")}
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
