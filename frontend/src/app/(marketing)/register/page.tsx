"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { PasswordRequirements } from "@/components/auth/password-requirements";
import { Eye, EyeOff, Mail, CheckCircle } from "lucide-react";
import { register as apiRegister, oauthInitiate, resendVerification } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import {
  PASSWORD_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  getPasswordValidation,
  passwordsMatch,
} from "@/lib/password-validation";

const OAUTH_PROVIDERS = ["github", "google", "huggingface"] as const;

export default function RegisterPage() {
  const router = useRouter();
  const { login } = useAuth();
  const { t } = useLocale();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [verificationSent, setVerificationSent] = useState(false);
  const [resending, setResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);
  const passwordValidation = getPasswordValidation(password);
  const matchingPasswords = passwordsMatch(password, confirmPassword);
  const canSubmit = Boolean(email) && passwordValidation.isValid && matchingPasswords;
  const passwordRequirements = [
    { key: "length", label: t("auth.pw_min"), satisfied: passwordValidation.hasValidLength },
    { key: "letter", label: t("auth.pw_rule_letter"), satisfied: passwordValidation.hasLetter },
    { key: "number", label: t("auth.pw_rule_number"), satisfied: passwordValidation.hasNumber },
    { key: "symbol", label: t("auth.pw_rule_symbol"), satisfied: passwordValidation.hasSupportedSymbol },
    { key: "match", label: t("auth.pw_match"), satisfied: matchingPasswords },
  ];

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
    if (!passwordValidation.isValid || !matchingPasswords) {
      setError(password !== confirmPassword ? t("auth.reset_mismatch") : t("auth.pw_policy_error"));
      return;
    }
    setError("");
    setLoading(true);
    try {
      const res = await apiRegister(email, password, name || undefined);
      if (res.email_verification_required) {
        setVerificationSent(true);
        return;
      }
      // Fallback: if server returns a token directly (e.g. test mode)
      await login();
      router.push("/dashboard");
    } catch (err) {
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
    try {
      const res = await oauthInitiate(provider);
      window.location.href = res.auth_url;
    } catch (err) {
      setError(err instanceof Error ? err.message : "OAuth failed");
    }
  }

  // Email verification sent screen
  if (verificationSent) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
        <Card className="w-full max-w-md p-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-ice-blue/10">
            <Mail className="h-8 w-8 text-ice-blue" />
          </div>
          <h1 className="text-2xl font-bold mb-2">{t("auth.verify_check_email")}</h1>
          <p className="text-text-secondary mb-6">
            {t("auth.verify_sent_to")} <strong className="text-text-primary">{email}</strong>
          </p>
          <p className="text-sm text-text-muted mb-6">
            {t("auth.verify_instructions")}
          </p>
          <div className="space-y-3">
            <Button
              variant="outline"
              className="w-full"
              onClick={handleResend}
              disabled={resending}
            >
              {resending ? t("auth.verify_resending") : t("auth.verify_resend")}
            </Button>
            {resendSuccess && (
              <div className="flex items-center justify-center gap-2 text-sm text-green-400">
                <CheckCircle className="h-4 w-4" />
                {t("auth.verify_resent")}
              </div>
            )}
          </div>
          <p className="mt-6 text-center text-sm text-text-secondary">
            {t("auth.register_signin")}{" "}
            <Link href="/login" className="text-ice-blue hover:underline">{t("auth.register_signin_link")}</Link>
          </p>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8">
        <div className="mb-8 text-center">
          <Image src="/xcelsior-logo.svg" alt="Xcelsior" width={48} height={48} className="mx-auto mb-4" />
          <h1 className="text-2xl font-bold">{t("auth.register_title")}</h1>
          <p className="mt-1 text-sm text-text-secondary">
            {t("auth.register_subtitle")}
          </p>
        </div>

        {/* OAuth */}
        <div className="space-y-3 mb-6">
          {OAUTH_PROVIDERS.map((provider) => {
            const labels = { github: t("auth.github"), google: t("auth.google"), huggingface: t("auth.huggingface") };
            return (
              <Button
                key={provider}
                variant="outline"
                className="h-auto w-full justify-start gap-3 rounded-xl border-border/70 bg-background/40 px-4 py-3.5 hover:bg-surface-hover/70"
                onClick={() => handleOAuth(provider)}
              >
                <ProviderLogo
                  provider={provider}
                  framed
                  size={40}
                  className="rounded-xl border-border/60 bg-navy/70 shadow-none"
                />
                <span className="flex-1 text-left">{labels[provider]}</span>
              </Button>
            );
          })}
        </div>

        <div className="relative mb-6">
          <div className="absolute inset-0 flex items-center">
            <div className="w-full border-t border-border" />
          </div>
          <div className="relative flex justify-center text-xs">
            <span className="bg-surface px-2 text-text-muted">{t("auth.or")}</span>
          </div>
        </div>

        <form onSubmit={handleRegister} className="space-y-4">
          {error && (
            <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red">
              {error}
            </div>
          )}
          <div className="space-y-2">
            <Label htmlFor="name">{t("auth.name")}</Label>
            <Input
              id="name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={t("auth.name_placeholder")}
              autoComplete="name"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="email">{t("auth.email")}</Label>
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              placeholder={t("auth.email_placeholder")}
              autoComplete="email"
            />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">{t("auth.password")}</Label>
            <div className="relative">
              <Input
                id="password"
                type={showPw ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={PASSWORD_MIN_LENGTH}
                maxLength={PASSWORD_MAX_LENGTH}
                placeholder={t("auth.pw_min")}
                autoComplete="new-password"
                className="pr-16"
              />
              {passwordValidation.isValid && (
                <CheckCircle className="pointer-events-none absolute right-10 top-1/2 h-4 w-4 -translate-y-1/2 text-emerald" />
              )}
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
                tabIndex={-1}
              >
                {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
          <div className="space-y-2">
            <Label htmlFor="confirm-password">{t("auth.confirm_password")}</Label>
            <div className="relative">
              <Input
                id="confirm-password"
                type={showConfirmPw ? "text" : "password"}
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                maxLength={PASSWORD_MAX_LENGTH}
                autoComplete="new-password"
                className="pr-16"
              />
              {matchingPasswords && (
                <CheckCircle className="pointer-events-none absolute right-10 top-1/2 h-4 w-4 -translate-y-1/2 text-emerald" />
              )}
              <button
                type="button"
                onClick={() => setShowConfirmPw(!showConfirmPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
                tabIndex={-1}
              >
                {showConfirmPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
          <PasswordRequirements items={passwordRequirements} />
          <Button type="submit" className="w-full" disabled={loading || !canSubmit}>
            {loading ? t("auth.register_loading") : t("auth.register_button")}
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-text-secondary">
          {t("auth.register_signin")}{" "}
          <Link href="/login" className="text-ice-blue hover:underline">{t("auth.register_signin_link")}</Link>
        </p>
      </Card>
    </div>
  );
}
