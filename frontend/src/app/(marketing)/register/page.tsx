"use client";

import { useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { Eye, EyeOff, Mail, CheckCircle } from "lucide-react";
import { register as apiRegister, oauthInitiate, resendVerification } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";

export default function RegisterPage() {
  const router = useRouter();
  const { login } = useAuth();
  const { t } = useLocale();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [verificationSent, setVerificationSent] = useState(false);
  const [resending, setResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);

  async function handleRegister(e: React.FormEvent) {
    e.preventDefault();
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
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-accent-red">
            <span className="text-2xl font-bold text-white">X</span>
          </div>
          <h1 className="text-2xl font-bold">{t("auth.register_title")}</h1>
          <p className="mt-1 text-sm text-text-secondary">
            {t("auth.register_subtitle")}
          </p>
        </div>

        {/* OAuth */}
        <div className="space-y-3 mb-6">
          <Button variant="outline" className="w-full" onClick={() => handleOAuth("github")}>
            <GitHubIcon />
            {t("auth.github")}
          </Button>
          <Button variant="outline" className="w-full" onClick={() => handleOAuth("google")}>
            <GoogleIcon />
            {t("auth.google")}
          </Button>
          <Button variant="outline" className="w-full" onClick={() => handleOAuth("huggingface")}>
            <HuggingFaceIcon />
            {t("auth.huggingface")}
          </Button>
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
            <Input id="name" value={name} onChange={(e) => setName(e.target.value)} placeholder={t("auth.name_placeholder")} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="email">{t("auth.email")}</Label>
            <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder={t("auth.email_placeholder")} />
          </div>
          <div className="space-y-2">
            <Label htmlFor="password">{t("auth.password")}</Label>
            <div className="relative">
              <Input id="password" type={showPw ? "text" : "password"} value={password} onChange={(e) => setPassword(e.target.value)} required minLength={8} placeholder={t("auth.pw_min")} />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
                tabIndex={-1}
              >
                {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            <p className="text-xs text-text-muted">{t("auth.pw_min")}</p>
          </div>
          <Button type="submit" className="w-full" disabled={loading}>
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

function GitHubIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
    </svg>
  );
}

function GoogleIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 24 24">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" />
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
    </svg>
  );
}

function HuggingFaceIcon() {
  return (
    <svg className="h-4 w-4" viewBox="0 0 95 88">
      <path fill="#FFD21E" d="M47.2 0C26.3 0 9.4 16.9 9.4 37.8c0 4.5.8 8.8 2.2 12.8h-.1c-3.4 0-6.5 1.8-8.2 4.7-1.7 2.9-1.8 6.5-.2 9.5l5.3 10c1.4 2.6 3.9 4.4 6.8 4.9l.5.1c3.6 5 9.4 8.2 15.9 8.2h31.7c6.5 0 12.3-3.3 15.9-8.2l.5-.1c2.9-.5 5.4-2.3 6.8-4.9l5.3-10c1.6-3 1.5-6.6-.2-9.5-1.7-2.9-4.8-4.7-8.2-4.7h-.1c1.5-4 2.2-8.3 2.2-12.8C85.1 16.9 68.2 0 47.2 0z" />
      <path fill="#FF9D0B" d="M28.8 54.3c-4 0-7.2-3.2-7.2-7.2s3.2-7.2 7.2-7.2 7.2 3.2 7.2 7.2-3.2 7.2-7.2 7.2zm36.8 0c-4 0-7.2-3.2-7.2-7.2s3.2-7.2 7.2-7.2 7.2 3.2 7.2 7.2-3.2 7.2-7.2 7.2z" />
      <path fill="#3A3B45" d="M31 47.1c0-2.3 1.9-4.2 4.2-4.2s4.2 1.9 4.2 4.2-1.9 4.2-4.2 4.2-4.2-1.9-4.2-4.2zm30.4 0c0-2.3 1.9-4.2 4.2-4.2s4.2 1.9 4.2 4.2-1.9 4.2-4.2 4.2-4.2-1.9-4.2-4.2z" />
      <path fill="#FF9D0B" d="M47.2 67.5c-6.1 0-11.5-3.1-14.6-7.8-.6-.9-.3-2.1.6-2.7.9-.6 2.1-.3 2.7.6 2.5 3.7 6.7 6 11.3 6s8.8-2.3 11.3-6c.6-.9 1.8-1.2 2.7-.6.9.6 1.2 1.8.6 2.7-3.1 4.7-8.5 7.8-14.6 7.8z" />
    </svg>
  );
}
