"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowLeft, CheckCircle, Mail } from "lucide-react";
import { requestPasswordReset } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { SiteAuthAlert, SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("");
  const [loading, setLoading] = useState(false);
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const { t } = useLocale();

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await requestPasswordReset(email);
      if (res.account_exists === false) {
        setError(t("auth.forgot_no_account"));
      } else {
        setSent(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send reset email");
    } finally {
      setLoading(false);
    }
  }

  return (
    <SiteAuthCard>
      {sent ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.forgot_success_title")} subtitle={t("auth.forgot_success_desc", { email })} />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
          <Link href="/login" className="site-button site-button-ghost site-auth-button">
            <ArrowLeft className="h-4 w-4" />
            {t("auth.forgot_back")}
          </Link>
        </div>
      ) : (
        <>
          <SiteAuthHeader title={t("auth.forgot_title")} subtitle={t("auth.forgot_subtitle")} />
          <form onSubmit={handleSubmit} className="site-auth-form site-auth-section">
            {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}
            <label htmlFor="email" className="site-field-wrap site-auth-field-wrap">
              <span className="site-auth-label">{t("auth.forgot_email_label")}</span>
              <input
                id="email"
                type="email"
                autoComplete="email"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="site-field site-auth-field"
              />
            </label>
            <button type="submit" className="site-button site-button-primary site-auth-button" disabled={loading}>
              <Mail className="h-4 w-4" />
              {loading ? t("auth.forgot_loading") : t("auth.forgot_button")}
            </button>
          </form>
          <p className="site-auth-footer">
            {t("auth.forgot_remember")}{" "}
            <Link href="/login" className="site-auth-link">
              {t("auth.forgot_signin_link")}
            </Link>
          </p>
        </>
      )}
    </SiteAuthCard>
  );
}
