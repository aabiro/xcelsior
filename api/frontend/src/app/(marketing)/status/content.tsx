"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useLocale } from "@/lib/locale";

/**
 * Public status page (adoption plan X6.27).
 *
 * Deliberately honest about what it is: a **live** probe of each service plus
 * the objectives they are measured against. It does not claim historical
 * uptime, because we do not yet compute it — asserting "99.99% this month"
 * from a page that only knows the present would be the kind of status page an
 * enterprise buyer stops trusting the first time it is wrong.
 */

type ServiceState = "operational" | "degraded" | "down";

interface Service {
  name: string;
  state: ServiceState;
  detail: string;
  required: boolean;
}

interface StatusPayload {
  ok: boolean;
  verdict: "operational" | "degraded" | "blocked";
  services: Service[];
}

const REFRESH_MS = 60_000;

export function StatusContent() {
  const { t } = useLocale();
  const [payload, setPayload] = useState<StatusPayload | null>(null);
  const [failed, setFailed] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);

  const load = useCallback(async () => {
    try {
      const response = await fetch("/api/status", { cache: "no-store" });
      if (!response.ok) throw new Error(String(response.status));
      setPayload((await response.json()) as StatusPayload);
      setFailed(false);
    } catch {
      // The status page failing to load is itself a status signal — say so
      // rather than rendering a stale "all systems operational".
      setFailed(true);
    } finally {
      setCheckedAt(new Date());
    }
  }, []);

  useEffect(() => {
    void load();
    const timer = setInterval(() => void load(), REFRESH_MS);
    return () => clearInterval(timer);
  }, [load]);

  const verdict = failed ? "unreachable" : payload?.verdict;

  return (
    <div className="site-container">
      <div className="site-rails site-section site-legal-shell">
        <h1 className="site-section-heading site-legal-heading">{t("status.title")}</h1>
        <p className="site-legal-effective">
          {checkedAt
            ? `${t("status.checked_at")} ${checkedAt.toLocaleTimeString()}`
            : t("status.checking")}
        </p>

        <div className="site-legal-body">
          <section className="site-legal-section">
            <h2 className="site-legal-title">
              {verdict === "operational"
                ? t("status.verdict_operational")
                : verdict === "unreachable"
                  ? t("status.verdict_unreachable")
                  : verdict === "blocked"
                    ? t("status.verdict_blocked")
                    : verdict === "degraded"
                      ? t("status.verdict_degraded")
                      : t("status.checking")}
            </h2>
            {failed ? (
              <p>{t("status.unreachable_body")}</p>
            ) : (
              <ul className="site-legal-list">
                {(payload?.services ?? []).map((service) => (
                  <li key={service.name}>
                    <strong>{service.name}</strong> — {t(`status.state_${service.state}`)}
                    {service.detail ? ` · ${service.detail}` : ""}
                  </li>
                ))}
              </ul>
            )}
          </section>

          <section className="site-legal-section">
            <h2 className="site-legal-title">{t("status.slo_title")}</h2>
            <p>{t("status.slo_intro")}</p>
            <ul className="site-legal-list">
              <li>{t("status.slo_connector")}</li>
              <li>{t("status.slo_discovery")}</li>
              <li>{t("status.slo_read_latency")}</li>
              <li>{t("status.slo_write_latency")}</li>
              <li>{t("status.slo_rate_limit")}</li>
            </ul>
          </section>

          <section className="site-legal-section">
            <h2 className="site-legal-title">{t("status.history_title")}</h2>
            <p>{t("status.history_body")}</p>
          </section>

          <section className="site-legal-section">
            <h2 className="site-legal-title">{t("status.incident_title")}</h2>
            <p>
              {t("status.incident_body")}{" "}
              <Link href="/support" className="site-inline-link">
                {t("status.incident_link")}
              </Link>
              .
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
