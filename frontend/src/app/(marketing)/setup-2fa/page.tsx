"use client";

import { Suspense, useCallback, useEffect, useState } from "react";
import Image from "next/image";
import { useRouter, useSearchParams } from "next/navigation";
import {
  AlertTriangle,
  CheckCircle,
  Copy,
  Download,
  Key,
  Loader2,
  Shield,
  Smartphone,
} from "lucide-react";
import QRCode from "qrcode";
import { toast } from "sonner";
import { SiteAuthAlert, SiteAuthCard, SiteAuthHeader } from "@/components/marketing/SiteAuthShell";
import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import { describePasskeyRegistrationError } from "@/lib/passkeys";

function Setup2FAPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, loading: authLoading } = useAuth();
  const { t } = useLocale();
  const redirectTarget = searchParams.get("redirect") || "/dashboard";

  const [checkingMfa, setCheckingMfa] = useState(true);
  const [mfaEnabled, setMfaEnabled] = useState(false);
  const [backupCodes, setBackupCodes] = useState<string[] | null>(null);
  const [activeMethod, setActiveMethod] = useState<"passkey" | "totp" | "sms" | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [totpSetup, setTotpSetup] = useState<{ secret: string; uri: string; methodId: number } | null>(null);
  const [totpQrDataUrl, setTotpQrDataUrl] = useState<string | null>(null);
  const [totpCode, setTotpCode] = useState("");

  const [smsPhone, setSmsPhone] = useState("");
  const [smsCode, setSmsCode] = useState("");
  const [smsCodeSent, setSmsCodeSent] = useState(false);

  const [passkeyName, setPasskeyName] = useState("");

  const loadMfa = useCallback(async () => {
    try {
      const res = await api.fetchMfaMethods();
      const hasEnabledMethod = (res.methods || []).some((method) => method.enabled);
      if (res.mfa_enabled || hasEnabledMethod) {
        router.replace("/dashboard/settings#security");
        return;
      }
    } catch {
      // Ignore errors, assume not enabled
    } finally {
      setCheckingMfa(false);
    }
  }, [router]);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    } else if (user) {
      loadMfa();
    }
  }, [user, authLoading, router, loadMfa]);

  const handleSkip = () => {
    router.replace(redirectTarget);
  };

  const resetState = () => {
    setActiveMethod(null);
    setError("");
    setLoading(false);
    setTotpCode("");
    setSmsCode("");
  };

  const handleTotpSetup = async () => {
    setLoading(true);
    setError("");
    try {
      const res = await api.setupTotp();
      setTotpSetup({ secret: res.secret, uri: res.provisioning_uri, methodId: res.method_id });
      const dataUrl = await QRCode.toDataURL(res.provisioning_uri, { width: 200, margin: 1 });
      setTotpQrDataUrl(dataUrl);
      setActiveMethod("totp");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to set up TOTP");
    } finally {
      setLoading(false);
    }
  };

  const handleTotpVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (totpCode.length !== 6) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.verifyTotp(totpCode, totpSetup?.methodId);
      if (res.backup_codes) setBackupCodes(res.backup_codes);
      setMfaEnabled(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid code");
    } finally {
      setLoading(false);
    }
  };

  const handleSmsSetup = async (e: React.FormEvent) => {
    e.preventDefault();
    const phone = `+1${smsPhone.replace(/\D/g, "")}`;
    if (phone.length < 8) {
      setError("Enter a valid phone number");
      return;
    }
    setLoading(true);
    setError("");
    try {
      await api.setupSms(phone);
      setSmsCodeSent(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send code");
    } finally {
      setLoading(false);
    }
  };

  const handleSmsVerify = async (e: React.FormEvent) => {
    e.preventDefault();
    if (smsCode.length !== 6) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.verifySms(smsCode);
      if (res.backup_codes) setBackupCodes(res.backup_codes);
      setMfaEnabled(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Invalid code");
    } finally {
      setLoading(false);
    }
  };

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

  const handlePasskeyAdd = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!window.PublicKeyCredential) {
      setError(t("dash.settings.mfa_passkey_unsupported"));
      return;
    }
    setLoading(true);
    setError("");
    try {
      const optRes = await api.passkeyRegisterOptions(passkeyName || "Security Key");
      const publicKey = optRes.options.publicKey as Record<string, unknown>;

      const createOptions: CredentialCreationOptions = {
        publicKey: {
          ...publicKey,
          challenge: base64urlToBuffer(publicKey.challenge as string),
          user: {
            ...(publicKey.user as Record<string, unknown>),
            id: base64urlToBuffer((publicKey.user as Record<string, string>).id),
          },
          excludeCredentials: ((publicKey.excludeCredentials as Array<Record<string, string>>) || []).map((c) => ({
            ...c,
            id: base64urlToBuffer(c.id),
          })),
        } as PublicKeyCredentialCreationOptions,
      };

      const credential = (await navigator.credentials.create(createOptions)) as PublicKeyCredential;
      if (!credential) throw new Error("No credential returned");

      const attestation = credential.response as AuthenticatorAttestationResponse;
      const completeRes = await api.passkeyRegisterComplete(optRes.state_id, {
        id: credential.id,
        rawId: bufferToBase64url(credential.rawId),
        type: credential.type,
        response: {
          clientDataJSON: bufferToBase64url(attestation.clientDataJSON),
          attestationObject: bufferToBase64url(attestation.attestationObject),
        },
      });
      if (completeRes.backup_codes) setBackupCodes(completeRes.backup_codes);
      setMfaEnabled(true);
    } catch (err) {
      const message = describePasskeyRegistrationError(err);
      setError(message);
      if (message.includes("already added")) {
        void loadMfa();
      }
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadBackupCodes = () => {
    if (!backupCodes) return;
    const text = `Xcelsior Backup Codes\n\n${backupCodes.join("\n")}\n\nKeep these safe. Each code can only be used once.\nGenerated on: ${new Date().toLocaleDateString()}`;
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "xcelsior-backup-codes.txt";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  };

  if (authLoading || checkingMfa) {
    return (
      <div className="site-auth-loading">
        <Loader2 className="site-auth-spinner h-8 w-8 animate-spin" />
      </div>
    );
  }

  if (mfaEnabled && backupCodes) {
    return (
      <SiteAuthCard className="site-auth-card-wide">
        <SiteAuthHeader
          title="2FA Enabled Successfully"
          subtitle="Your account is now secure. Please save these backup codes in a safe place."
        />
        <div className="site-auth-section site-auth-stack">
          <div className="site-auth-status-icon" data-tone="success" style={{ margin: "0 auto" }}>
            <CheckCircle className="h-7 w-7" />
          </div>
          <SiteAuthAlert tone="warn">
            <strong>Save these backup codes</strong>
            <br />
            If you lose your device, these codes are the only way to access your account. Each code can only be used once.
          </SiteAuthAlert>
          <div className="site-auth-code-grid">
            {backupCodes.map((code) => (
              <div key={code} className="site-auth-code">
                {code}
              </div>
            ))}
          </div>
          <div className="site-auth-button-row">
            <button
              type="button"
              className="site-button site-button-ghost site-auth-button"
              onClick={() => {
                navigator.clipboard.writeText(backupCodes.join("\n"));
                toast.success("Copied to clipboard");
              }}
            >
              <Copy className="h-4 w-4" />
              Copy
            </button>
            <button type="button" className="site-button site-button-ghost site-auth-button" onClick={handleDownloadBackupCodes}>
              <Download className="h-4 w-4" />
              Download
            </button>
          </div>
          <button type="button" className="site-button site-button-primary site-auth-button" onClick={() => router.replace(redirectTarget)}>
            I have saved these codes — Continue
          </button>
        </div>
      </SiteAuthCard>
    );
  }

  return (
    <SiteAuthCard className="site-auth-card-wide">
      <SiteAuthHeader title="Secure Your Account" subtitle="Protect your account with Two-Factor Authentication (2FA)." />
      <div className="site-auth-section site-auth-stack">
        {error ? <SiteAuthAlert>{error}</SiteAuthAlert> : null}

        {!activeMethod ? (
          <>
            <div className="site-auth-option-grid">
              <button type="button" onClick={() => setActiveMethod("passkey")} className="site-auth-option">
                <span className="site-auth-option-icon" style={{ color: "var(--cyan)" }}>
                  <Key className="h-5 w-5" />
                </span>
                <span>
                  <strong className="site-auth-emphasis">Passkey</strong>
                  <span className="site-auth-option-copy">Use Face ID, Touch ID, or a security key. (Recommended)</span>
                </span>
              </button>

              <button type="button" onClick={handleTotpSetup} disabled={loading} className="site-auth-option">
                <span className="site-auth-option-icon" style={{ color: "var(--cyan)" }}>
                  <Smartphone className="h-5 w-5" />
                </span>
                <span>
                  <strong className="site-auth-emphasis">Authenticator App</strong>
                  <span className="site-auth-option-copy">Use an app like Google Authenticator or Authy.</span>
                </span>
              </button>

              <button type="button" onClick={() => setActiveMethod("sms")} className="site-auth-option">
                <span className="site-auth-option-icon" style={{ color: "var(--cyan)" }}>
                  <Shield className="h-5 w-5" />
                </span>
                <span>
                  <strong className="site-auth-emphasis">SMS Verification</strong>
                  <span className="site-auth-option-copy">Receive a code via text message.</span>
                </span>
              </button>
            </div>
            <button type="button" onClick={handleSkip} className="site-auth-text-button site-auth-centered-copy">
              Skip for now — I&apos;ll do this later
            </button>
          </>
        ) : null}

        {activeMethod === "passkey" ? (
          <form onSubmit={handlePasskeyAdd} className="site-auth-form">
            <label className="site-field-wrap site-auth-field-wrap">
              <span className="site-auth-label">Name your passkey</span>
              <input
                value={passkeyName}
                onChange={(e) => setPasskeyName(e.target.value)}
                placeholder="e.g. MacBook Touch ID"
                className="site-field site-auth-field"
                autoFocus
              />
            </label>
            <div className="site-auth-button-row">
              <button type="button" className="site-button site-button-ghost site-auth-button" onClick={resetState}>
                Back
              </button>
              <button type="submit" className="site-button site-button-primary site-auth-button" disabled={loading || !passkeyName}>
                {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Registering</> : "Register Passkey"}
              </button>
            </div>
          </form>
        ) : null}

        {activeMethod === "totp" ? (
          <form onSubmit={handleTotpVerify} className="site-auth-form">
            <div className="site-auth-qr-card">
              {totpQrDataUrl ? (
                <Image src={totpQrDataUrl} alt="QR Code" width={160} height={160} className="site-auth-qr-image" />
              ) : (
                <div className="site-auth-qr-placeholder" />
              )}
              <p className="site-auth-emphasis" style={{ marginTop: 16 }}>Scan this QR code</p>
              <p className="site-auth-note">
                Or enter key manually: <span className="site-auth-code-inline">{totpSetup?.secret}</span>
              </p>
            </div>
            <label className="site-field-wrap site-auth-field-wrap">
              <span className="site-auth-label">Enter 6-digit code</span>
              <input
                value={totpCode}
                onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="000000"
                className="site-field site-auth-field site-auth-otp"
                maxLength={6}
                autoFocus
              />
            </label>
            <div className="site-auth-button-row">
              <button type="button" className="site-button site-button-ghost site-auth-button" onClick={resetState}>
                Back
              </button>
              <button type="submit" className="site-button site-button-primary site-auth-button" disabled={loading || totpCode.length !== 6}>
                {loading ? <><Loader2 className="h-4 w-4 animate-spin" /> Verifying</> : "Verify & Enable"}
              </button>
            </div>
          </form>
        ) : null}

        {activeMethod === "sms" ? (
          <form onSubmit={smsCodeSent ? handleSmsVerify : handleSmsSetup} className="site-auth-form">
            {!smsCodeSent ? (
              <label className="site-field-wrap site-auth-field-wrap">
                <span className="site-auth-label">Phone Number</span>
                <div className="site-auth-phone-row">
                  <span className="site-auth-prefix">+1</span>
                  <input
                    value={smsPhone}
                    onChange={(e) => setSmsPhone(e.target.value)}
                    placeholder="(555) 123-4567"
                    className="site-field site-auth-field"
                    type="tel"
                    autoFocus
                  />
                </div>
              </label>
            ) : (
              <label className="site-field-wrap site-auth-field-wrap">
                <div className="site-auth-field-row">
                  <span className="site-auth-label">Enter verification code</span>
                  <span className="site-auth-note">Sent to +1 {smsPhone}</span>
                </div>
                <input
                  value={smsCode}
                  onChange={(e) => setSmsCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="000000"
                  className="site-field site-auth-field site-auth-otp"
                  maxLength={6}
                  autoFocus
                />
              </label>
            )}
            <div className="site-auth-button-row">
              <button
                type="button"
                className="site-button site-button-ghost site-auth-button"
                onClick={() => {
                  if (smsCodeSent) {
                    setSmsCodeSent(false);
                    setSmsCode("");
                    setError("");
                  } else {
                    resetState();
                  }
                }}
              >
                Back
              </button>
              <button
                type="submit"
                className="site-button site-button-primary site-auth-button"
                disabled={loading || (!smsCodeSent && !smsPhone) || (smsCodeSent && smsCode.length !== 6)}
              >
                {loading ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> {smsCodeSent ? "Verifying" : "Sending"}</>
                ) : smsCodeSent ? (
                  "Verify & Enable"
                ) : (
                  "Send Code"
                )}
              </button>
            </div>
          </form>
        ) : null}
      </div>
    </SiteAuthCard>
  );
}

export default function Setup2FAPage() {
  return (
    <Suspense fallback={<div className="site-auth-loading"><Loader2 className="site-auth-spinner h-8 w-8 animate-spin" /></div>}>
      <Setup2FAPageContent />
    </Suspense>
  );
}
