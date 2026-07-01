"use client";

import { Suspense, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { CheckCircle, Eye, EyeOff, Mail } from "lucide-react";
import { PasswordRequirements } from "@/components/auth/password-requirements";
import { SiteAuthAlert, SiteAuthCard, SiteAuthDivider, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { normalizeAuthRedirectPath, oauthInitiate, register as apiRegister, resendVerification } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import {
  PASSWORD_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  getPasswordValidation,
  passwordsMatch,
} from "@/lib/password-validation";
import posthog from "posthog-js";

const OAUTH_PROVIDERS = ["github", "google", "huggingface"] as const;

function RegisterPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { login } = useAuth();
  const { t } = useLocale();
  const redirectTarget = normalizeAuthRedirectPath(searchParams.get("redirect"), "/dashboard");
  const inviteToken = searchParams.get("invite") ?? "";
  const [name, setName] = useState("");
  const [email, setEmail] = useState(searchParams.get("email") ?? "");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);
  const [agreedToTerms, setAgreedToTerms] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [verificationSent, setVerificationSent] = useState(false);
  const [resending, setResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);
  const passwordValidation = getPasswordValidation(password);
  const matchingPasswords = passwordsMatch(password, confirmPassword);
  const canSubmit = Boolean(email) && passwordValidation.isValid && matchingPasswords && agreedToTerms;
  const passwordRequirements = [
    { key: "length", label: t("auth.pw_min"), satisfied: passwordValidation.hasValidLength },
    { key: "letter", label: t("auth.pw_rule_letter"), satisfied: passwordValidation.hasLetter },
    { key: "number", label: t("auth.pw_rule_number"), satisfied: passwordValidation.hasNumber },
    { key: "symbol", label: t("auth.pw_rule_symbol"), satisfied: passwordValidation.hasSupportedSymbol },
    { key: "match", label: t("auth.pw_match"), satisfied: matchingPasswords },
  ];

  async function completeBrowserLogin() {
    await login().catch(() => {});
    if (inviteToken) {
      router.replace(`/accept-invite?token=${encodeURIComponent(inviteToken)}`);
    } else {
      router.replace(redirectTarget);
    }
  }

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    if (!passwordValidation.isValid || !matchingPasswords) {
      setError(password !== confirmPassword ? t("auth.reset_mismatch") : t("auth.pw_policy_error"));
      return;
    }
    if (!agreedToTerms) {
      setError(t("auth.register_agree_error"));
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await apiRegister(email, password, name || undefined);
      if (res.email_verification_required) {
        posthog.capture("user_registered", {
          method: "email",
          email_verification_required: true,
          has_name: Boolean(name),
          has_invite: Boolean(inviteToken),
        });
        setVerificationSent(true);
        return;
      }
      posthog.capture("user_registered", {
        method: "email",
        email_verification_required: false,
        has_name: Boolean(name),
        has_invite: Boolean(inviteToken),
      });
      await completeBrowserLogin();
    } catch (err) {
      posthog.captureException(err instanceof Error ? err : new Error(String(err)));
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleResend() {
    setResending(true);
    setResendSuccess(false);
    try {
      await resendVerification(email);
      setResendSuccess(true);
    } catch {
      // silent — don't reveal if email exists
    } finally {
      setResending(false);
    }
  }

  async function handleOAuth(provider: string) {
    posthog.capture("oauth_initiated", { provider, context: "register" });
    try {
      const res = await oauthInitiate(provider, redirectTarget);
      window.location.href = res.auth_url;
    } catch (err) {
      posthog.captureException(err instanceof Error ? err : new Error(String(err)));
      setError(err instanceof Error ? err.message : "OAuth failed");
    }
  }

  if (verificationSent) {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader
            title={t("auth.verify_check_email")}
            subtitle={
              <>
                {t("auth.verify_sent_to")} <span className="site-auth-emphasis">{email}</span>
              </>
            }
          />
          <div className="site-auth-status-icon" data-tone="info" style={{ margin: "0 auto" }}>
            <Mail className="h-7 w-7" />
          </div>
          <p className="site-auth-footer" style={{ marginTop: 0 }}>{t("auth.verify_instructions")}</p>
          <div className="site-auth-actions">
            <button type="button" className="site-button site-button-ghost site-auth-button" onClick={handleResend} disabled={resending}>
              {resending ? t("auth.verify_resending") : t("auth.verify_resend")}
            </button>
            {resendSuccess ? <SiteAuthAlert tone="success">{t("auth.verify_resent")}</SiteAuthAlert> : null}
          </div>
          <p className="site-auth-footer">
            {t("auth.register_signin")} <Link href="/login" className="site-auth-link">{t("auth.register_signin_link")}</Link>
          </p>
        </div>
      </SiteAuthCard>
    );
  }

  return (
    <SiteAuthCard>
      <SiteAuthHeader title={t("auth.register_title")} subtitle={t("auth.register_subtitle")} />
      <div className="site-auth-section">
        <div className="site-auth-provider-list">
          {OAUTH_PROVIDERS.map((provider) => {
            const labels = { github: t("auth.github"), google: t("auth.google"), huggingface: t("auth.huggingface") };
            return (
              <button key={provider} type="button" className="site-auth-provider" onClick={() => handleOAuth(provider)}>
                <ProviderLogo provider={provider} framed size={22} className="site-auth-provider-logo" />
                <span className="site-auth-provider-label">{labels[provider]}</span>
              </button>
            );
          })}
        </div>

        <SiteAuthDivider label={t("auth.or")} />

        <form onSubmit={handleRegister} className="site-auth-form">
          {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}

          <label htmlFor="name" className="site-field-wrap site-auth-field-wrap">
            <span className="site-auth-label">{t("auth.name")}</span>
            <input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t("auth.name_placeholder")}
              autoComplete="name"
              className="site-field site-auth-field"
            />
          </label>

          <label htmlFor="email" className="site-field-wrap site-auth-field-wrap">
            <span className="site-auth-label">{t("auth.email")}</span>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder={t("auth.email_placeholder")}
              autoComplete="email"
              className="site-field site-auth-field"
            />
          </label>

          <label htmlFor="password" className="site-field-wrap site-auth-field-wrap">
            <span className="site-auth-label">{t("auth.password")}</span>
            <div className="site-auth-input-wrap">
              <input
                id="password"
                type={showPw ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={PASSWORD_MIN_LENGTH}
                maxLength={PASSWORD_MAX_LENGTH}
                placeholder={t("auth.pw_min")}
                autoComplete="new-password"
                className="site-field site-auth-field"
              />
              <button type="button" onClick={() => setShowPw(!showPw)} className="site-auth-input-action" tabIndex={-1}>
                {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </label>

          <label htmlFor="confirm-password" className="site-field-wrap site-auth-field-wrap">
            <span className="site-auth-label">{t("auth.confirm_password")}</span>
            <div className="site-auth-input-wrap">
              <input
                id="confirm-password"
                type={showConfirmPw ? "text" : "password"}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                maxLength={PASSWORD_MAX_LENGTH}
                autoComplete="new-password"
                className="site-field site-auth-field"
              />
              <button type="button" onClick={() => setShowConfirmPw(!showConfirmPw)} className="site-auth-input-action" tabIndex={-1}>
                {showConfirmPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </label>

          <PasswordRequirements items={passwordRequirements} className="site-auth-requirements" />

          <label htmlFor="agree-terms" className="site-auth-checkbox-row">
            <span className="site-auth-checkbox">
              <input
                id="agree-terms"
                type="checkbox"
                required
                checked={agreedToTerms}
                onChange={(e) => setAgreedToTerms(e.target.checked)}
              />
              <span className="site-auth-checkbox-box" />
            </span>
            <span>
              {t("auth.register_agree_prefix")}{" "}
              <Link href="/terms" target="_blank" className="site-auth-link">
                {t("footer.terms")}
              </Link>{" "}
              {t("auth.register_agree_and")}{" "}
              <Link href="/privacy" target="_blank" className="site-auth-link">
                {t("footer.privacy")}
              </Link>
            </span>
          </label>

          <button type="submit" className="site-button site-button-primary site-auth-button" disabled={loading || !canSubmit}>
            {loading ? t("auth.register_loading") : t("auth.register_button")}
          </button>
        </form>

        <p className="site-auth-footer">
          {t("auth.register_signin")} <Link href="/login" className="site-auth-link">{t("auth.register_signin_link")}</Link>
        </p>
      </div>
    </SiteAuthCard>
  );
}

export default function RegisterPage() {
  return (
    <Suspense fallback={null}>
      <RegisterPageContent />
    </Suspense>
  );
}
