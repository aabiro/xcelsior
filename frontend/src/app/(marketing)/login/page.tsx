"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Eye, EyeOff, Key, Loader2, Shield } from "lucide-react";
import { ProviderLogo } from "@/components/ui/provider-logo";
import { SiteAuthAlert, SiteAuthCard, SiteAuthDivider, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import {
  ApiError,
  login as apiLogin,
  normalizeAuthRedirectPath,
  oauthInitiate,
  passkeyAuthenticateComplete,
  passkeyAuthenticateOptions,
  resendVerification,
  sendMfaSms,
  verifyMfaLogin,
} from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import posthog from "posthog-js";

const OAUTH_PROVIDERS = ["github", "google", "huggingface"] as const;

function LoginPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, loading: authLoading, login } = useAuth();
  const { t } = useLocale();
  const redirectTarget = normalizeAuthRedirectPath(searchParams.get("redirect"), "/dashboard");
  const oauthErrorCode = searchParams.get("error");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState(() => {
    if (oauthErrorCode?.startsWith("oauth_")) {
      return "Sign-in was interrupted. Please try again — it should work on the first attempt now.";
    }
    return "";
  });
  const [loading, setLoading] = useState(false);

  const [mfaRequired, setMfaRequired] = useState(false);
  const [mfaChallengeId, setMfaChallengeId] = useState("");
  const [mfaMethods, setMfaMethods] = useState<string[]>([]);
  const [mfaMethod, setMfaMethod] = useState<"totp" | "sms" | "backup" | "passkey">("totp");
  const [mfaCode, setMfaCode] = useState("");
  const [mfaVerifying, setMfaVerifying] = useState(false);
  const [smsSent, setSmsSent] = useState(false);
  const [passkeyAuthenticating, setPasskeyAuthenticating] = useState(false);
  const [passkeyAvailable, setPasskeyAvailable] = useState(false);
  const mfaAutoSubmitted = useRef(false);

  useEffect(() => {
    if (window.PublicKeyCredential?.isUserVerifyingPlatformAuthenticatorAvailable) {
      window.PublicKeyCredential.isUserVerifyingPlatformAuthenticatorAvailable()
        .then(setPasskeyAvailable)
        .catch(() => setPasskeyAvailable(false));
    }
  }, []);

  const [emailNotVerified, setEmailNotVerified] = useState(false);
  const [unverifiedEmail, setUnverifiedEmail] = useState("");
  const [resending, setResending] = useState(false);
  const [resendSuccess, setResendSuccess] = useState(false);

  const [lastOAuth, setLastOAuth] = useState<string | null>(null);
  useEffect(() => {
    const match = document.cookie.match(/(?:^|;\s*)xcelsior_last_oauth=([^;]*)/);
    setLastOAuth(match ? decodeURIComponent(match[1]) : null);
  }, []);

  if (!authLoading && user) {
    router.replace(redirectTarget);
    return null;
  }

  if (authLoading) {
    return (
      <div className="site-auth-loading">
        <Loader2 className="site-auth-spinner h-8 w-8 animate-spin" />
      </div>
    );
  }

  async function completeBrowserLogin(method: string = "password") {
    const loggedIn = await login().catch(() => false);
    if (loggedIn) {
      posthog.capture("user_logged_in", { method });
    }
    router.replace(redirectTarget);
  }

  async function handleLogin(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      const res = await apiLogin(email, password);
      if (res.mfa_required) {
        setMfaRequired(true);
        setMfaChallengeId(res.challenge_id || "");
        setMfaMethods(res.methods || []);
        if (res.methods?.includes("passkey") && passkeyAvailable) setMfaMethod("passkey");
        else if (res.methods?.includes("totp")) setMfaMethod("totp");
        else if (res.methods?.includes("sms")) setMfaMethod("sms");
        else setMfaMethod("backup");
        posthog.capture("mfa_challenged", {
          methods: res.methods,
          passkey_available: passkeyAvailable,
        });
        return;
      }
      await completeBrowserLogin();
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) {
        const body = err.body as
          | { email_verification_required?: boolean; email?: string; oauth_account?: boolean; error?: { message?: string } }
          | undefined;
        if (body?.email_verification_required) {
          setEmailNotVerified(true);
          setUnverifiedEmail(body.email || email);
          return;
        }
        if (body?.oauth_account) {
          setError(body.error?.message || "This account uses OAuth. Please sign in with your OAuth provider or use 'Forgot password' to set a password.");
          return;
        }
      }
      posthog.captureException(err instanceof Error ? err : new Error(String(err)));
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
      await completeBrowserLogin("mfa_" + mfaMethod);
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
    bytes.forEach((b) => {
      binary += String.fromCharCode(b);
    });
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

      const assertion = (await navigator.credentials.get(getOptions)) as PublicKeyCredential;
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
        await completeBrowserLogin("mfa_passkey");
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
    posthog.capture("oauth_initiated", { provider, context: "login" });
    setError("");
    setLoading(true);
    try {
      const res = await oauthInitiate(provider, redirectTarget);
      window.location.href = res.auth_url;
    } catch (err) {
      posthog.captureException(err instanceof Error ? err : new Error(String(err)));
      setError(err instanceof Error ? err.message : "OAuth failed");
      setLoading(false);
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

  if (emailNotVerified) {
    return (
      <SiteAuthCard>
        <div className="site-auth-section site-auth-stack">
          <SiteAuthHeader title={t("auth.verify_required")} subtitle={<>{t("auth.verify_required_desc")} <span className="site-auth-emphasis">{unverifiedEmail}</span></>} />
          <div className="site-auth-status-icon" data-tone="warn" style={{ margin: "0 auto" }}>
            <Shield className="h-7 w-7" />
          </div>
          <div className="site-auth-actions">
            <button
              type="button"
              className="site-button site-button-ghost site-auth-button"
              onClick={handleResendVerification}
              disabled={resending}
            >
              {resending ? t("auth.verify_resending") : t("auth.verify_resend")}
            </button>
            {resendSuccess ? <SiteAuthAlert tone="success">{t("auth.verify_resent")}</SiteAuthAlert> : null}
            <button
              type="button"
              className="site-button site-button-ghost site-auth-button"
              onClick={() => {
                setEmailNotVerified(false);
                setError("");
              }}
            >
              {t("auth.verify_back")}
            </button>
          </div>
        </div>
      </SiteAuthCard>
    );
  }

  return (
    <SiteAuthCard>
      {mfaRequired ? (
        <>
          <SiteAuthHeader title={t("auth.mfa_title")} subtitle={t("auth.mfa_subtitle")} />
          <div className="site-auth-section site-auth-stack">
            <div className="site-auth-tabs">
              {mfaMethods.includes("passkey") && passkeyAvailable ? (
                <button
                  type="button"
                  onClick={() => {
                    setMfaMethod("passkey");
                    setMfaCode("");
                    setError("");
                  }}
                  className="site-auth-tab"
                  data-active={mfaMethod === "passkey"}
                >
                  {t("auth.mfa_passkey_label")}
                </button>
              ) : null}
              {mfaMethods.includes("totp") ? (
                <button
                  type="button"
                  onClick={() => {
                    setMfaMethod("totp");
                    setMfaCode("");
                    setError("");
                  }}
                  className="site-auth-tab"
                  data-active={mfaMethod === "totp"}
                >
                  {t("auth.mfa_totp_label")}
                </button>
              ) : null}
              {mfaMethods.includes("sms") ? (
                <button
                  type="button"
                  onClick={() => {
                    setMfaMethod("sms");
                    setMfaCode("");
                    setError("");
                    setSmsSent(false);
                  }}
                  className="site-auth-tab"
                  data-active={mfaMethod === "sms"}
                >
                  {t("auth.mfa_sms_label")}
                </button>
              ) : null}
              <button
                type="button"
                onClick={() => {
                  setMfaMethod("backup");
                  setMfaCode("");
                  setError("");
                }}
                className="site-auth-tab"
                data-active={mfaMethod === "backup"}
              >
                {t("auth.mfa_backup_label")}
              </button>
            </div>

            {mfaMethod === "passkey" ? (
              <div className="site-auth-stack">
                {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}
                <div className="site-auth-status-icon" data-tone="info" style={{ margin: "0 auto" }}>
                  <Key className="h-7 w-7" />
                </div>
                <p className="site-auth-footer" style={{ marginTop: 0 }}>{t("auth.mfa_passkey_prompt")}</p>
                <button
                  type="button"
                  className="site-button site-button-primary site-auth-button"
                  onClick={handlePasskeyAuth}
                  disabled={passkeyAuthenticating}
                >
                  {passkeyAuthenticating ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" /> {t("auth.mfa_passkey_authenticating")}
                    </>
                  ) : (
                    <>
                      <Key className="h-4 w-4" /> {t("auth.mfa_verify")}
                    </>
                  )}
                </button>
              </div>
            ) : (
              <form onSubmit={handleMfaVerify} className="site-auth-form">
                {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}

                {mfaMethod === "sms" && !smsSent ? (
                  <button type="button" className="site-button site-button-ghost site-auth-button" onClick={handleSendSms}>
                    {t("auth.mfa_send_sms")}
                  </button>
                ) : null}

                <label className="site-field-wrap site-auth-field-wrap">
                  <span className="site-auth-label">{mfaMethod === "backup" ? t("auth.mfa_backup_label") : t("auth.mfa_code_placeholder")}</span>
                  <input
                    value={mfaCode}
                    onChange={(e) => {
                      const val = mfaMethod === "backup" ? e.target.value.toUpperCase() : e.target.value.replace(/\D/g, "").slice(0, 6);
                      setMfaCode(val);
                      if (mfaMethod !== "backup" && val.length === 6 && !mfaAutoSubmitted.current) {
                        mfaAutoSubmitted.current = true;
                        setMfaVerifying(true);
                        verifyMfaLogin(mfaChallengeId, mfaMethod, val)
                          .then(() => completeBrowserLogin("mfa_" + mfaMethod))
                          .catch((err) => setError(err instanceof Error ? err.message : "Verification failed"))
                          .finally(() => {
                            mfaAutoSubmitted.current = false;
                            setMfaVerifying(false);
                          });
                      }
                    }}
                    placeholder={mfaMethod === "backup" ? "XXXX-XXXX" : "000000"}
                    className={`site-field site-auth-field ${mfaMethod === "backup" ? "" : "site-auth-otp"}`}
                    maxLength={mfaMethod === "backup" ? 9 : 6}
                    autoFocus
                  />
                </label>

                <button type="submit" className="site-button site-button-primary site-auth-button" disabled={mfaVerifying || !mfaCode}>
                  {mfaVerifying ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" /> {t("auth.mfa_verifying")}
                    </>
                  ) : (
                    t("auth.mfa_verify")
                  )}
                </button>
              </form>
            )}

            <button
              type="button"
              onClick={() => {
                setMfaRequired(false);
                setMfaCode("");
                setError("");
              }}
              className="site-auth-text-button site-auth-centered-copy"
            >
              ← {t("auth.login_title")}
            </button>
          </div>
        </>
      ) : (
        <>
          <SiteAuthHeader title={t("auth.login_title")} subtitle={t("auth.login_subtitle")} />
          <div className="site-auth-section">
            <div className="site-auth-provider-list">
              {OAUTH_PROVIDERS.map((provider) => {
                const labels = { github: t("auth.github"), google: t("auth.google"), huggingface: t("auth.huggingface") };
                return (
                  <button key={provider} type="button" className="site-auth-provider" onClick={() => handleOAuth(provider)}>
                    <ProviderLogo provider={provider} framed size={22} className="site-auth-provider-logo" />
                    <span className="site-auth-provider-label">{labels[provider]}</span>
                    {lastOAuth === provider ? <span className="site-auth-provider-badge">Last used</span> : null}
                  </button>
                );
              })}
            </div>

            <SiteAuthDivider label={t("auth.or")} />

            <form onSubmit={handleLogin} className="site-auth-form">
              {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}
              <label htmlFor="email" className="site-field-wrap site-auth-field-wrap">
                <span className="site-auth-label">{t("auth.email")}</span>
                <input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder={t("auth.email_placeholder")}
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className="site-field site-auth-field"
                />
              </label>
              <label htmlFor="password" className="site-field-wrap site-auth-field-wrap">
                <div className="site-auth-field-row">
                  <span className="site-auth-label">{t("auth.password")}</span>
                  <Link href="/forgot-password" className="site-auth-helper-link">
                    {t("auth.forgot_link")}
                  </Link>
                </div>
                <div className="site-auth-input-wrap">
                  <input
                    id="password"
                    type={showPw ? "text" : "password"}
                    autoComplete="current-password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    className="site-field site-auth-field"
                  />
                  <button type="button" onClick={() => setShowPw(!showPw)} className="site-auth-input-action" tabIndex={-1}>
                    {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </label>
              <button type="submit" className="site-button site-button-primary site-auth-button" disabled={loading}>
                {loading ? t("auth.login_loading") : t("auth.login_button")}
              </button>
            </form>

            <p className="site-auth-footer">
              {t("auth.login_signup")} <Link href="/register" className="site-auth-link">{t("auth.login_signup_link")}</Link>
            </p>
          </div>
        </>
      )}
    </SiteAuthCard>
  );
}

export default function LoginPage() {
  return (
    <Suspense fallback={null}>
      <LoginPageContent />
    </Suspense>
  );
}
