"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Card } from "@/components/ui/card";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { Eye, EyeOff, Loader2, Shield, Key } from "lucide-react";
import { login as apiLogin, oauthInitiate, verifyMfaLogin, sendMfaSms, passkeyAuthenticateOptions, passkeyAuthenticateComplete, resendVerification, ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";

const OAUTH_PROVIDERS = ["github", "google", "huggingface"] as const;

export default function LoginPage() {
  const router = useRouter();
  const { user, loading: authLoading, login } = useAuth();
  const { t } = useLocale();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  // MFA state
  const [mfaRequired, setMfaRequired] = useState(false);
  const [mfaChallengeId, setMfaChallengeId] = useState("");
  const [mfaMethods, setMfaMethods] = useState<string[]>([]);
  const [mfaMethod, setMfaMethod] = useState<"totp" | "sms" | "backup" | "passkey">("totp");
  const [mfaCode, setMfaCode] = useState("");
  const [mfaVerifying, setMfaVerifying] = useState(false);
  const [smsSent, setSmsSent] = useState(false);
  const [passkeyAuthenticating, setPasskeyAuthenticating] = useState(false);

  // Email verification state
  const [emailNotVerified, setEmailNotVerified] = useState(false);
  const [unverifiedEmail, setUnverifiedEmail] = useState("");
  const [resending, setResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);

  // Track last-used OAuth provider (read from cookie set by API on OAuth login)
  const [lastOAuth, setLastOAuth] = useState<string | null>(null);
  useEffect(() => {
    const match = document.cookie.match(/(?:^|;\s*)xcelsior_last_oauth=([^;]*)/);
    setLastOAuth(match ? decodeURIComponent(match[1]) : null);
  }, []);

  // If already authenticated, redirect to dashboard
  if (!authLoading && user) {
    router.replace("/dashboard");
    return null;
  }

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await apiLogin(email, password);
      // Check if MFA is required
      if (res.mfa_required) {
        setMfaRequired(true);
        setMfaChallengeId(res.challenge_id || "");
        setMfaMethods(res.methods || []);
        // Default to first available method
        if (res.methods?.includes("passkey")) setMfaMethod("passkey");
        else if (res.methods?.includes("totp")) setMfaMethod("totp");
        else if (res.methods?.includes("sms")) setMfaMethod("sms");
        else setMfaMethod("backup");
        return;
      }
      await login();
      router.push("/dashboard");
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        const body = err.body as { email_verification_required?: boolean; email?: string } | undefined;
        if (body?.email_verification_required) {
          setEmailNotVerified(true);
          setUnverifiedEmail(body.email || email);
          return;
        }
      }
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleMfaVerify(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setMfaVerifying(true);
    try {
      await verifyMfaLogin(mfaChallengeId, mfaMethod, mfaCode);
      await login();
      router.push("/dashboard");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setMfaVerifying(false);
    }
  }

  async function handleSendSms() {
    try {
      await sendMfaSms(mfaChallengeId);
      setSmsSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send SMS");
    }
  }

  function bufferToBase64url(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    bytes.forEach((b) => { binary += String.fromCharCode(b); });
    return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
  }
  function base64urlToBuffer(b64: string): ArrayBuffer {
    const padded = b64.replace(/-/g, "+").replace(/_/g, "/");
    const binary = atob(padded);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }

  async function handlePasskeyAuth() {
    setError("");
    setPasskeyAuthenticating(true);
    try {
      const optRes = await passkeyAuthenticateOptions(mfaChallengeId);
      const publicKey = optRes.options.publicKey as Record<string, unknown>;

      const getOptions: CredentialRequestOptions = {
        publicKey: {
          ...publicKey,
          challenge: base64urlToBuffer(publicKey.challenge as string),
          allowCredentials: ((publicKey.allowCredentials as Array<Record<string, string>>) || []).map((c) => ({
            ...c,
            id: base64urlToBuffer(c.id),
          })),
        } as PublicKeyCredentialRequestOptions,
      };

      const assertion = await navigator.credentials.get(getOptions) as PublicKeyCredential;
      if (!assertion) throw new Error("No credential returned");

      const assertionResponse = assertion.response as AuthenticatorAssertionResponse;
      const result = await passkeyAuthenticateComplete(optRes.state_id, {
        id: assertion.id,
        rawId: bufferToBase64url(assertion.rawId),
        type: assertion.type,
        response: {
          authenticatorData: bufferToBase64url(assertionResponse.authenticatorData),
          clientDataJSON: bufferToBase64url(assertionResponse.clientDataJSON),
          signature: bufferToBase64url(assertionResponse.signature),
          userHandle: assertionResponse.userHandle ? bufferToBase64url(assertionResponse.userHandle) : null,
        },
      });
      if (result.ok) {
        await login();
        router.push("/dashboard");
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError("Passkey authentication was cancelled");
      } else {
        setError(err instanceof Error ? err.message : t("auth.mfa_passkey_error"));
      }
    } finally {
      setPasskeyAuthenticating(false);
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

  async function handleResendVerification() {
    setResending(true);
    setResendSuccess(false);
    try {
      await resendVerification(unverifiedEmail);
      setResendSuccess(true);
    } catch {
      // Don't reveal error
    } finally {
      setResending(false);
    }
  }

  // Email not verified screen
  if (emailNotVerified) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
        <Card className="w-full max-w-md p-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-amber-500/10">
            <Shield className="h-8 w-8 text-amber-400" />
          </div>
          <h1 className="text-2xl font-bold mb-2">{t("auth.verify_required")}</h1>
          <p className="text-text-secondary mb-6">
            {t("auth.verify_required_desc")} <strong className="text-text-primary">{unverifiedEmail}</strong>
          </p>
          <div className="space-y-3">
            <Button
              variant="outline"
              className="w-full"
              onClick={handleResendVerification}
              disabled={resending}
            >
              {resending ? t("auth.verify_resending") : t("auth.verify_resend")}
            </Button>
            {resendSuccess && (
              <p className="text-sm text-green-400">{t("auth.verify_resent")}</p>
            )}
            <Button
              variant="ghost"
              className="w-full"
              onClick={() => { setEmailNotVerified(false); setError(""); }}
            >
              {t("auth.verify_back")}
            </Button>
          </div>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8">
        {mfaRequired ? (
          /* ── MFA Challenge ── */
          <>
            <div className="mb-8 text-center">
              <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-ice-blue/10">
                <Shield className="h-6 w-6 text-ice-blue" />
              </div>
              <h1 className="text-2xl font-bold">{t("auth.mfa_title")}</h1>
              <p className="mt-1 text-sm text-text-secondary">
                {t("auth.mfa_subtitle")}
              </p>
            </div>

            {/* Method selector tabs */}
            <div className="flex gap-1 mb-6 rounded-lg bg-surface p-1">
              {mfaMethods.includes("passkey") && (
                <button
                  onClick={() => { setMfaMethod("passkey"); setMfaCode(""); setError(""); }}
                  className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${mfaMethod === "passkey" ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"}`}
                >
                  {t("auth.mfa_passkey_label")}
                </button>
              )}
              {mfaMethods.includes("totp") && (
                <button
                  onClick={() => { setMfaMethod("totp"); setMfaCode(""); setError(""); }}
                  className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${mfaMethod === "totp" ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"}`}
                >
                  {t("auth.mfa_totp_label")}
                </button>
              )}
              {mfaMethods.includes("sms") && (
                <button
                  onClick={() => { setMfaMethod("sms"); setMfaCode(""); setError(""); setSmsSent(false); }}
                  className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${mfaMethod === "sms" ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"}`}
                >
                  {t("auth.mfa_sms_label")}
                </button>
              )}
              <button
                onClick={() => { setMfaMethod("backup"); setMfaCode(""); setError(""); }}
                className={`flex-1 rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${mfaMethod === "backup" ? "bg-card text-text-primary shadow-sm" : "text-text-muted hover:text-text-primary"}`}
              >
                {t("auth.mfa_backup_label")}
              </button>
            </div>

            {mfaMethod === "passkey" ? (
              <div className="space-y-4">
                {error && (
                  <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red">
                    {error}
                  </div>
                )}
                <p className="text-sm text-center text-text-secondary">{t("auth.mfa_passkey_prompt")}</p>
                <Button className="w-full" onClick={handlePasskeyAuth} disabled={passkeyAuthenticating}>
                  {passkeyAuthenticating ? <><Loader2 className="h-4 w-4 animate-spin" /> {t("auth.mfa_passkey_authenticating")}</> : <><Key className="h-4 w-4" /> {t("auth.mfa_verify")}</>}
                </Button>
              </div>
            ) : (
            <form onSubmit={handleMfaVerify} className="space-y-4">
              {error && (
                <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red">
                  {error}
                </div>
              )}

              {mfaMethod === "sms" && !smsSent && (
                <Button type="button" variant="outline" className="w-full" onClick={handleSendSms}>
                  {t("auth.mfa_send_sms")}
                </Button>
              )}

              <div className="space-y-2">
                <Label>{mfaMethod === "backup" ? t("auth.mfa_backup_label") : t("auth.mfa_code_placeholder")}</Label>
                <Input
                  value={mfaCode}
                  onChange={(e) => setMfaCode(mfaMethod === "backup" ? e.target.value.toUpperCase() : e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder={mfaMethod === "backup" ? "XXXX-XXXX" : "000000"}
                  className="font-mono text-center text-lg tracking-widest"
                  maxLength={mfaMethod === "backup" ? 9 : 6}
                  autoFocus
                />
              </div>

              <Button type="submit" className="w-full" disabled={mfaVerifying || !mfaCode}>
                {mfaVerifying ? <><Loader2 className="h-4 w-4 animate-spin" /> {t("auth.mfa_verifying")}</> : t("auth.mfa_verify")}
              </Button>
            </form>
            )}

            <button
              onClick={() => { setMfaRequired(false); setMfaCode(""); setError(""); }}
              className="mt-4 block w-full text-center text-sm text-text-muted hover:text-text-primary"
            >
              ← {t("auth.login_title")}
            </button>
          </>
        ) : (
        /* ── Normal Login ── */
        <>
        <div className="mb-8 text-center">
          <Image src="/xcelsior-logo.svg" alt="Xcelsior" width={48} height={48} className="mx-auto mb-4" />
          <h1 className="text-2xl font-bold">{t("auth.login_title")}</h1>
          <p className="mt-1 text-sm text-text-secondary">
            {t("auth.login_subtitle")}
          </p>
        </div>

        {/* OAuth Buttons */}
        <div className="space-y-3 mb-6">
          {OAUTH_PROVIDERS.map((provider) => {
            const labels = { github: t("auth.github"), google: t("auth.google"), huggingface: t("auth.huggingface") };
            return (
              <Button
                key={provider}
                variant="outline"
                className="relative h-auto w-full justify-start gap-3 rounded-xl border-border/70 bg-background/40 px-4 py-3.5 hover:bg-surface-hover/70"
                onClick={() => handleOAuth(provider)}
              >
                <ProviderLogo
                  provider={provider}
                  framed
                  size={40}
                  className="rounded-xl border-border/60 bg-navy/70 shadow-none"
                />
                <span className="flex-1 text-left">{labels[provider]}</span>
                {lastOAuth === provider && (
                  <span className="absolute right-3 rounded-full bg-accent-cyan/10 border border-accent-cyan/30 px-2 py-0.5 text-[10px] font-medium text-accent-cyan">
                    Last used
                  </span>
                )}
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

        {/* Email/Password */}
        <form onSubmit={handleLogin} className="space-y-4">
          {error && (
            <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red">
              {error}
            </div>
          )}
          <div className="space-y-2">
            <Label htmlFor="email">{t("auth.email")}</Label>
            <Input
              id="email"
              type="email"
              autoComplete="email"
              placeholder={t("auth.email_placeholder")}
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="password">{t("auth.password")}</Label>
              <Link
                href="/forgot-password"
                className="text-xs text-ice-blue hover:underline"
              >
                {t("auth.forgot_link")}
              </Link>
            </div>
            <div className="relative">
              <Input
                id="password"
                type={showPw ? "text" : "password"}
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
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
          <Button type="submit" className="w-full" disabled={loading}>
            {loading ? t("auth.login_loading") : t("auth.login_button")}
          </Button>
        </form>

        <p className="mt-6 text-center text-sm text-text-secondary">
          {t("auth.login_signup")}{" "}
          <Link href="/register" className="text-ice-blue hover:underline">
            {t("auth.login_signup_link")}
          </Link>
        </p>
        </>
        )}
      </Card>
    </div>
  );
}
