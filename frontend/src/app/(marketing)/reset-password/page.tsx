"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { AlertTriangle, ArrowLeft, CheckCircle, Eye, EyeOff, Lock, Loader2 } from "lucide-react";
import { PasswordRequirements } from "@/components/auth/password-requirements";
import { SiteAuthAlert, SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { confirmPasswordReset } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import {
  PASSWORD_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  getPasswordValidation,
  passwordsMatch,
} from "@/lib/password-validation";

function ResetPasswordForm() {
  const { t } = useLocale();
  const searchParams = useSearchParams();
  const token = searchParams.get("token") || "";

  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState("");
  const passwordValidation = getPasswordValidation(password);
  const matchingPasswords = passwordsMatch(password, confirm);
  const valid = passwordValidation.isValid && matchingPasswords && !!token;
  const passwordRequirements = [
    { key: "length", label: t("auth.pw_min"), satisfied: passwordValidation.hasValidLength },
    { key: "letter", label: t("auth.pw_rule_letter"), satisfied: passwordValidation.hasLetter },
    { key: "number", label: t("auth.pw_rule_number"), satisfied: passwordValidation.hasNumber },
    { key: "symbol", label: t("auth.pw_rule_symbol"), satisfied: passwordValidation.hasSupportedSymbol },
    { key: "match", label: t("auth.pw_match"), satisfied: matchingPasswords },
  ];

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!valid) {
      setError(password !== confirm ? t("auth.reset_mismatch") : t("auth.pw_policy_error"));
      return;
    }
    setError("");
    setLoading(true);
    try {
      await confirmPasswordReset(token, password);
      setSuccess(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : t("auth.reset_error"));
    } finally {
      setLoading(false);
    }
  }

  if (!token) {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.reset_invalid_title")} subtitle={t("auth.reset_invalid_desc")} />
          <div className="site-auth-status-icon" data-tone="error" style={{ margin: "0 auto" }}>
            <AlertTriangle className="h-7 w-7" />
          </div>
          <Link href="/forgot-password" className="site-button site-button-ghost site-auth-button">
            {t("auth.reset_request_new")}
          </Link>
        </div>
      </SiteAuthCard>
    );
  }

  return (
    <SiteAuthCard>
      {success ? (
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.reset_success_title")} subtitle={t("auth.reset_success_desc")} />
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
          <Link href="/login" className="site-button site-button-primary site-auth-button">
            {t("auth.reset_signin")}
          </Link>
        </div>
      ) : (
        <>
          <SiteAuthHeader title={t("auth.reset_title")} subtitle={t("auth.reset_subtitle")} />
          <form onSubmit={handleSubmit} className="site-auth-form site-auth-section">
            {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}

            <label htmlFor="password" className="site-field-wrap site-auth-field-wrap">
              <span className="site-auth-label">{t("auth.reset_new_pw")}</span>
              <div className="site-auth-input-wrap">
                <input
                  id="password"
                  type={showPw ? "text" : "password"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder={t("auth.pw_min")}
                  required
                  minLength={PASSWORD_MIN_LENGTH}
                  maxLength={PASSWORD_MAX_LENGTH}
                  autoComplete="new-password"
                  className="site-field site-auth-field"
                />
                <button
                  type="button"
                  onClick={() => setShowPw(!showPw)}
                  className="site-auth-input-action"
                  tabIndex={-1}
                >
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </label>

            <label htmlFor="confirm" className="site-field-wrap site-auth-field-wrap">
              <span className="site-auth-label">{t("auth.reset_confirm_pw")}</span>
              <div className="site-auth-input-wrap">
                <input
                  id="confirm"
                  type={showConfirm ? "text" : "password"}
                  value={confirm}
                  onChange={(e) => setConfirm(e.target.value)}
                  required
                  maxLength={PASSWORD_MAX_LENGTH}
                  autoComplete="new-password"
                  className="site-field site-auth-field"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirm(!showConfirm)}
                  className="site-auth-input-action"
                  tabIndex={-1}
                >
                  {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </label>

            <PasswordRequirements items={passwordRequirements} className="site-auth-requirements" />

            <button type="submit" className="site-button site-button-primary site-auth-button" disabled={loading || !valid}>
              <Lock className="h-4 w-4" />
              {loading ? t("auth.reset_loading") : t("auth.reset_button")}
            </button>
          </form>

          <p className="site-auth-footer">
            <Link href="/login" className="site-auth-link">
              <ArrowLeft className="inline h-3 w-3" /> {t("auth.forgot_back")}
            </Link>
          </p>
        </>
      )}
    </SiteAuthCard>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={<div className="site-auth-loading"><Loader2 className="site-auth-spinner h-8 w-8 animate-spin" /></div>}>
      <ResetPasswordForm />
    </Suspense>
  );
}
