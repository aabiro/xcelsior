"use client";

import { useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { PasswordRequirements } from "@/components/auth/password-requirements";
import { ArrowLeft, Lock, CheckCircle, Eye, EyeOff, AlertTriangle } from "lucide-react";
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
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
        <Card className="w-full max-w-md p-8 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-red/20">
            <AlertTriangle className="h-6 w-6 text-accent-red" />
          </div>
          <h1 className="text-2xl font-bold">{t("auth.reset_invalid_title")}</h1>
          <p className="mt-2 text-sm text-text-secondary">{t("auth.reset_invalid_desc")}</p>
          <Link href="/forgot-password">
            <Button variant="outline" className="mt-6">
              {t("auth.reset_request_new")}
            </Button>
          </Link>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8">
        {success ? (
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-emerald/20">
              <CheckCircle className="h-6 w-6 text-emerald" />
            </div>
            <h1 className="text-2xl font-bold">{t("auth.reset_success_title")}</h1>
            <p className="mt-2 text-sm text-text-secondary">{t("auth.reset_success_desc")}</p>
            <Link href="/login">
              <Button className="mt-6 w-full">{t("auth.reset_signin")}</Button>
            </Link>
          </div>
        ) : (
          <>
            <div className="mb-8 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-red">
                <Lock className="h-5 w-5 text-white" />
              </div>
              <h1 className="text-2xl font-bold">{t("auth.reset_title")}</h1>
              <p className="mt-1 text-sm text-text-secondary">{t("auth.reset_subtitle")}</p>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {error && (
                <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red">
                  {error}
                </div>
              )}

              <div className="space-y-2">
                <Label htmlFor="password">{t("auth.reset_new_pw")}</Label>
                <div className="relative">
                  <Input
                    id="password"
                    type={showPw ? "text" : "password"}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder={t("auth.pw_min")}
                    required
                    minLength={PASSWORD_MIN_LENGTH}
                    maxLength={PASSWORD_MAX_LENGTH}
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
                <Label htmlFor="confirm">{t("auth.reset_confirm_pw")}</Label>
                <div className="relative">
                  <Input
                    id="confirm"
                    type={showConfirm ? "text" : "password"}
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
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
                    onClick={() => setShowConfirm(!showConfirm)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
                    tabIndex={-1}
                  >
                    {showConfirm ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>
              <PasswordRequirements items={passwordRequirements} />

              <Button type="submit" className="w-full" disabled={loading || !valid}>
                {loading ? t("auth.reset_loading") : t("auth.reset_button")}
              </Button>
            </form>

            <p className="mt-6 text-center text-sm text-text-secondary">
              <Link href="/login" className="text-ice-blue hover:underline">
                <ArrowLeft className="inline h-3 w-3 mr-1" />
                {t("auth.forgot_back")}
              </Link>
            </p>
          </>
        )}
      </Card>
    </div>
  );
}

export default function ResetPasswordPage() {
  return (
    <Suspense fallback={
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-accent-red border-t-transparent" />
      </div>
    }>
      <ResetPasswordForm />
    </Suspense>
  );
}
