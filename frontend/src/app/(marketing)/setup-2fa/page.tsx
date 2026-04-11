"use client";

import { Suspense, useEffect, useState, useCallback } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { CheckCircle, Shield, Key, Smartphone, AlertTriangle, Loader2, Copy, Download } from "lucide-react";
import QRCode from "qrcode";
import { toast } from "sonner";
import * as api from "@/lib/api";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";

function Setup2FAPageContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { user, loading: authLoading } = useAuth();
  const { t } = useLocale();
  const redirectTarget = searchParams.get("redirect") || "/dashboard";

  const [checkingMfa, setCheckingMfa] = useState(true);
  const [mfaEnabled, setMfaEnabled] = useState(false);
  const [mfaMethods, setMfaMethods] = useState<api.MfaMethod[]>([]);
  const [backupCodes, setBackupCodes] = useState<string[] | null>(null);

  const [activeMethod, setActiveMethod] = useState<"passkey" | "totp" | "sms" | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // TOTP state
  const [totpSetup, setTotpSetup] = useState<{ secret: string; uri: string; methodId: number } | null>(null);
  const [totpQrDataUrl, setTotpQrDataUrl] = useState<string | null>(null);
  const [totpCode, setTotpCode] = useState("");

  // SMS state
  const [smsPhone, setSmsPhone] = useState("");
  const [smsCode, setSmsCode] = useState("");
  const [smsCodeSent, setSmsCodeSent] = useState(false);

  // Passkey state
  const [passkeyName, setPasskeyName] = useState("");

  const loadMfa = useCallback(async () => {
    try {
      const res = await api.fetchMfaMethods();
      if (res.mfa_enabled) {
        // If already set up, redirect immediately
        router.replace(redirectTarget);
        return;
      }
      setMfaEnabled(false);
      setMfaMethods(res.methods || []);
    } catch {
      // Ignore errors, assume not enabled
    } finally {
      setCheckingMfa(false);
    }
  }, [router, redirectTarget]);

  useEffect(() => {
    if (!authLoading && !user) {
      router.replace("/login");
    } else if (user) {
      loadMfa();
    }
  }, [user, authLoading, router, loadMfa]);

  if (authLoading || checkingMfa) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4">
        <Loader2 className="h-8 w-8 animate-spin text-ice-blue" />
      </div>
    );
  }

  const handleSkip = () => {
    router.replace(redirectTarget);
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
    } catch (err) { setError(err instanceof Error ? err.message : "Failed to set up TOTP"); }
    finally { setLoading(false); }
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
    } catch (err) { setError(err instanceof Error ? err.message : "Invalid code"); }
    finally { setLoading(false); }
  };

  const handleSmsSetup = async (e: React.FormEvent) => {
    e.preventDefault();
    const phone = `+1${smsPhone.replace(/\D/g, "")}`;
    if (phone.length < 8) { setError("Enter a valid phone number"); return; }
    setLoading(true);
    setError("");
    try {
      await api.setupSms(phone);
      setSmsCodeSent(true);
    } catch (err) { setError(err instanceof Error ? err.message : "Failed to send code"); }
    finally { setLoading(false); }
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
    } catch (err) { setError(err instanceof Error ? err.message : "Invalid code"); }
    finally { setLoading(false); }
  };

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

      const credential = await navigator.credentials.create(createOptions) as PublicKeyCredential;
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
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setError("Passkey registration cancelled");
      } else {
        setError(err instanceof Error ? err.message : "Failed to register passkey");
      }
    } finally { setLoading(false); }
  };

  const handleDownloadBackupCodes = () => {
    if (!backupCodes) return;
    const text = `Xcelsior Backup Codes\n\n${backupCodes.join("\n")}\n\nKeep these safe. Each code can only be used once.\nGenerated on: ${new Date().toLocaleDateString()}`;
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "xcelsior-backup-codes.txt";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // If successfully enabled and showing backup codes
  if (mfaEnabled && backupCodes) {
    return (
      <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
        <Card className="w-full max-w-lg p-8">
          <div className="mb-6 text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500/10">
              <CheckCircle className="h-8 w-8 text-emerald-500" />
            </div>
            <h1 className="text-2xl font-bold">2FA Enabled Successfully</h1>
            <p className="mt-2 text-text-secondary">
              Your account is now secure. Please save these backup codes in a safe place.
            </p>
          </div>

          <div className="rounded-xl bg-accent-red/10 border border-accent-red/30 p-4 mb-6">
            <div className="flex items-start gap-3">
              <AlertTriangle className="h-5 w-5 text-accent-red shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="font-medium text-accent-red mb-1">Save these backup codes</p>
                <p className="text-accent-red/80">
                  If you lose your device, these codes are the only way to access your account.
                  Each code can only be used once.
                </p>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 mb-6">
            {backupCodes.map((code, i) => (
              <div key={i} className="rounded-lg bg-surface border border-border p-3 text-center font-mono text-sm tracking-widest select-all">
                {code}
              </div>
            ))}
          </div>

          <div className="flex gap-3 mb-8">
            <Button variant="outline" className="flex-1" onClick={() => { navigator.clipboard.writeText(backupCodes.join("\n")); toast.success("Copied to clipboard"); }}>
              <Copy className="h-4 w-4 mr-2" /> Copy
            </Button>
            <Button variant="outline" className="flex-1" onClick={handleDownloadBackupCodes}>
              <Download className="h-4 w-4 mr-2" /> Download
            </Button>
          </div>

          <Button className="w-full" onClick={() => router.replace(redirectTarget)}>
            I have saved these codes — Continue
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-4rem)] items-center justify-center px-4 py-12">
      <Card className="w-full max-w-md p-8">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-ice-blue/10">
            <Shield className="h-6 w-6 text-ice-blue" />
          </div>
          <h1 className="text-2xl font-bold">Secure Your Account</h1>
          <p className="mt-1 text-sm text-text-secondary">
            Protect your account with Two-Factor Authentication (2FA).
          </p>
        </div>

        {error && (
          <div className="rounded-lg bg-accent-red/10 border border-accent-red/30 p-3 text-sm text-accent-red mb-6">
            {error}
          </div>
        )}

        {!activeMethod ? (
          <div className="space-y-4">
            <button
              onClick={() => setActiveMethod("passkey")}
              className="flex w-full items-start gap-4 rounded-xl border border-border bg-surface p-4 text-left transition-colors hover:border-ice-blue/50 hover:bg-surface-hover"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-ice-blue/10">
                <Key className="h-5 w-5 text-ice-blue" />
              </div>
              <div>
                <p className="font-semibold text-text-primary">Passkey</p>
                <p className="text-sm text-text-secondary">Use Face ID, Touch ID, or a security key. (Recommended)</p>
              </div>
            </button>

            <button
              onClick={handleTotpSetup}
              disabled={loading}
              className="flex w-full items-start gap-4 rounded-xl border border-border bg-surface p-4 text-left transition-colors hover:border-ice-blue/50 hover:bg-surface-hover"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-ice-blue/10">
                <Smartphone className="h-5 w-5 text-ice-blue" />
              </div>
              <div>
                <p className="font-semibold text-text-primary">Authenticator App</p>
                <p className="text-sm text-text-secondary">Use an app like Google Authenticator or Authy.</p>
              </div>
            </button>

            <button
              onClick={() => setActiveMethod("sms")}
              className="flex w-full items-start gap-4 rounded-xl border border-border bg-surface p-4 text-left transition-colors hover:border-ice-blue/50 hover:bg-surface-hover"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-ice-blue/10">
                <Shield className="h-5 w-5 text-ice-blue" />
              </div>
              <div>
                <p className="font-semibold text-text-primary">SMS Verification</p>
                <p className="text-sm text-text-secondary">Receive a code via text message.</p>
              </div>
            </button>

            <div className="pt-4 text-center">
              <button
                onClick={handleSkip}
                className="text-sm text-text-muted hover:text-text-primary transition-colors"
              >
                Skip for now — I'll do this later
              </button>
            </div>
          </div>
        ) : activeMethod === "passkey" ? (
          <form onSubmit={handlePasskeyAdd} className="space-y-6">
            <div>
              <Label>Name your passkey</Label>
              <Input
                value={passkeyName}
                onChange={(e) => setPasskeyName(e.target.value)}
                placeholder="e.g. MacBook Touch ID"
                className="mt-2"
                autoFocus
              />
            </div>
            <div className="flex gap-3">
              <Button type="button" variant="outline" className="flex-1" onClick={() => setActiveMethod(null)}>
                Back
              </Button>
              <Button type="submit" className="flex-1" disabled={loading || !passkeyName}>
                {loading ? <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Registering</> : "Register Passkey"}
              </Button>
            </div>
          </form>
        ) : activeMethod === "totp" ? (
          <form onSubmit={handleTotpVerify} className="space-y-6">
            <div className="rounded-xl border border-border bg-surface p-6 text-center">
              {totpQrDataUrl ? (
                <Image src={totpQrDataUrl} alt="QR Code" width={160} height={160} className="mx-auto rounded-lg bg-white p-2" />
              ) : (
                <div className="mx-auto h-40 w-40 animate-pulse rounded-lg bg-border" />
              )}
              <p className="mt-4 text-sm font-medium">Scan this QR code</p>
              <p className="text-xs text-text-muted">Or enter key manually: <span className="font-mono text-text-primary">{totpSetup?.secret}</span></p>
            </div>
            <div>
              <Label>Enter 6-digit code</Label>
              <Input
                value={totpCode}
                onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                placeholder="000000"
                className="mt-2 text-center font-mono text-lg tracking-widest"
                maxLength={6}
                autoFocus
              />
            </div>
            <div className="flex gap-3">
              <Button type="button" variant="outline" className="flex-1" onClick={() => setActiveMethod(null)}>
                Back
              </Button>
              <Button type="submit" className="flex-1" disabled={loading || totpCode.length !== 6}>
                {loading ? <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Verifying</> : "Verify & Enable"}
              </Button>
            </div>
          </form>
        ) : (
          <form onSubmit={smsCodeSent ? handleSmsVerify : handleSmsSetup} className="space-y-6">
            {!smsCodeSent ? (
              <div>
                <Label>Phone Number</Label>
                <div className="mt-2 flex items-center gap-2">
                  <div className="flex h-10 w-12 shrink-0 items-center justify-center rounded-lg border border-border bg-surface text-sm font-medium text-text-secondary">
                    +1
                  </div>
                  <Input
                    value={smsPhone}
                    onChange={(e) => setSmsPhone(e.target.value)}
                    placeholder="(555) 123-4567"
                    className="flex-1"
                    type="tel"
                    autoFocus
                  />
                </div>
              </div>
            ) : (
              <div>
                <Label>Enter verification code</Label>
                <p className="mb-4 text-xs text-text-muted">Sent to +1 {smsPhone}</p>
                <Input
                  value={smsCode}
                  onChange={(e) => setSmsCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                  placeholder="000000"
                  className="mt-2 text-center font-mono text-lg tracking-widest"
                  maxLength={6}
                  autoFocus
                />
              </div>
            )}
            <div className="flex gap-3">
              <Button
                type="button"
                variant="outline"
                className="flex-1"
                onClick={() => smsCodeSent ? setSmsCodeSent(false) : setActiveMethod(null)}
              >
                Back
              </Button>
              <Button type="submit" className="flex-1" disabled={loading || (!smsCodeSent && !smsPhone) || (smsCodeSent && smsCode.length !== 6)}>
                {loading ? <><Loader2 className="h-4 w-4 animate-spin mr-2" /> {smsCodeSent ? "Verifying" : "Sending"}</> : smsCodeSent ? "Verify & Enable" : "Send Code"}
              </Button>
            </div>
          </form>
        )}
      </Card>
    </div>
  );
}

export default function Setup2FAPage() {
  return (
    <Suspense fallback={null}>
      <Setup2FAPageContent />
    </Suspense>
  );
}
