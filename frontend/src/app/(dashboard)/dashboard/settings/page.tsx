"use client";

import { useEffect, useState, useCallback } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import {
  Settings as SettingsIcon, Save, Shield, Bell, Globe, Key, Terminal,
  Lock, Trash2, Download, Eye, EyeOff, Copy, Plus, AlertTriangle,
  CheckCircle, Loader2, ShieldCheck, X, Users, UserPlus, UserMinus,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { ApiKeyInfo, ConsentRecord, TeamInfo, TeamMember, UserSshKey } from "@/lib/api";
import QRCode from "qrcode";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { COUNTRY_CODES } from "@/lib/country-codes";

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const { t } = useLocale();
  const userId = user?.customer_id || user?.user_id || "";

  // Profile
  const [name, setName] = useState(user?.name || "");
  const [email] = useState(user?.email || "");
  const [canadaOnly, setCanadaOnly] = useState(false);
  const [notifications, setNotifications] = useState(true);
  const [saving, setSaving] = useState(false);

  // API Keys
  const [apiKeys, setApiKeys] = useState<ApiKeyInfo[]>([]);
  const [newKeyName, setNewKeyName] = useState("");
  const [newKeyScope, setNewKeyScope] = useState<"full-access" | "read-only">("full-access");
  const [generatedKey, setGeneratedKey] = useState<string | null>(null);
  const [generatingKey, setGeneratingKey] = useState(false);

  // SSH
  const [sshPubKey, setSshPubKey] = useState("");
  const [generatingSsh, setGeneratingSsh] = useState(false);
  const [userSshKeys, setUserSshKeys] = useState<UserSshKey[]>([]);
  const [newSshKeyName, setNewSshKeyName] = useState("");
  const [newSshKeyValue, setNewSshKeyValue] = useState("");
  const [addingSshKey, setAddingSshKey] = useState(false);

  // Password
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [changingPw, setChangingPw] = useState(false);
  const [showPw, setShowPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);

  // MFA
  const [mfaEnabled, setMfaEnabled] = useState(false);
  const [mfaMethods, setMfaMethods] = useState<api.MfaMethod[]>([]);
  const [mfaBackupRemaining, setMfaBackupRemaining] = useState(0);
  const [mfaLoading, setMfaLoading] = useState(false);
  // TOTP setup
  const [totpSetup, setTotpSetup] = useState<{ secret: string; uri: string; methodId: number } | null>(null);
  const [totpQrDataUrl, setTotpQrDataUrl] = useState<string | null>(null);
  const [totpCode, setTotpCode] = useState("");
  const [totpVerifying, setTotpVerifying] = useState(false);
  const [backupCodes, setBackupCodes] = useState<string[] | null>(null);
  // SMS setup
  const [smsSetup, setSmsSetup] = useState(false);
  const [smsPhone, setSmsPhone] = useState("");
  const [smsCountryCode, setSmsCountryCode] = useState("+1");
  const [smsCode, setSmsCode] = useState("");
  const [smsSending, setSmsSending] = useState(false);
  const [smsVerifying, setSmsVerifying] = useState(false);
  const [smsCodeSent, setSmsCodeSent] = useState(false);
  // Passkey setup
  const [passkeyName, setPasskeyName] = useState("");
  const [passkeyRegistering, setPasskeyRegistering] = useState(false);
  const [passkeyRemoving, setPasskeyRemoving] = useState<number | null>(null);

  // Consent
  const [consents, setConsents] = useState<ConsentRecord[]>([]);

  // Deletion
  const [deleteConfirm, setDeleteConfirm] = useState("");
  const [deleting, setDeleting] = useState(false);

  // Teams
  const [teams, setTeams] = useState<TeamInfo[]>([]);
  const [activeTeam, setActiveTeam] = useState<TeamInfo | null>(null);
  const [teamMembers, setTeamMembers] = useState<TeamMember[]>([]);
  const [newTeamName, setNewTeamName] = useState("");
  const [inviteEmail, setInviteEmail] = useState("");
  const [inviteRole, setInviteRole] = useState("member");
  const [creatingTeam, setCreatingTeam] = useState(false);
  const [inviting, setInviting] = useState(false);
  const [removeTarget, setRemoveTarget] = useState<string | null>(null);
  const [deletingTeam, setDeletingTeam] = useState(false);
  const [deleteTeamConfirm, setDeleteTeamConfirm] = useState(false);

  // Sessions
  const [sessions, setSessions] = useState<api.SessionInfo[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(false);
  const [revokingSession, setRevokingSession] = useState<string | null>(null);

  const loadTeams = useCallback(async () => {
    try {
      const res = await api.fetchMyTeams();
      setTeams(res.teams || []);
      if ((res.teams || []).length > 0 && !activeTeam) {
        const first = res.teams[0];
        setActiveTeam(first);
        const detail = await api.fetchTeam(first.team_id);
        setTeamMembers(detail.members || []);
      }
    } catch { /* no teams yet */ }
  }, [activeTeam]);

  const selectTeam = async (team: TeamInfo) => {
    setActiveTeam(team);
    try {
      const detail = await api.fetchTeam(team.team_id);
      setTeamMembers(detail.members || []);
    } catch { toast.error("Failed to load team"); }
  };

  const handleCreateTeam = async () => {
    if (!newTeamName.trim()) return;
    setCreatingTeam(true);
    try {
      const res = await api.createTeam({ name: newTeamName.trim() });
      toast.success(`Team "${res.name}" created`);
      setNewTeamName("");
      await loadTeams();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create team");
    } finally { setCreatingTeam(false); }
  };

  const handleInvite = async () => {
    if (!inviteEmail.trim() || !activeTeam) return;
    setInviting(true);
    try {
      await api.addTeamMember(activeTeam.team_id, { email: inviteEmail.trim(), role: inviteRole });
      toast.success(`${inviteEmail.trim()} invited`);
      setInviteEmail("");
      const detail = await api.fetchTeam(activeTeam.team_id);
      setTeamMembers(detail.members || []);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to invite");
    } finally { setInviting(false); }
  };

  const handleRemoveMember = async (email: string) => {
    setRemoveTarget(null);
    if (!activeTeam) return;
    try {
      await api.removeTeamMember(activeTeam.team_id, email);
      toast.success(`${email} removed`);
      setTeamMembers((prev) => prev.filter((m) => m.email !== email));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to remove");
    }
  };

  const handleDeleteTeam = async () => {
    if (!activeTeam) return;
    setDeletingTeam(true);
    try {
      await api.deleteTeam(activeTeam.team_id);
      toast.success(`Team "${activeTeam.name}" deleted`);
      setActiveTeam(null);
      setTeamMembers([]);
      setDeleteTeamConfirm(false);
      await loadTeams();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete team");
    } finally { setDeletingTeam(false); }
  };

  const handleRoleChange = async (memberEmail: string, newRole: string) => {
    if (!activeTeam) return;
    try {
      await api.updateMemberRole(activeTeam.team_id, memberEmail, newRole);
      toast.success(`${memberEmail} role updated to ${newRole}`);
      setTeamMembers((prev) =>
        prev.map((m) => m.email === memberEmail ? { ...m, role: newRole } : m)
      );
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to update role");
    }
  };

  useEffect(() => {
    // Load preferences
    fetch("/api/users/me/preferences", { credentials: "include" })
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then((prefs) => {
        setCanadaOnly(prefs.canada_only_routing ?? false);
        setNotifications(prefs.notifications ?? true);
      })
      .catch((e) => console.error("Failed to load preferences", e));

    // Load API keys
    api.fetchApiKeys()
      .then((res) => setApiKeys(res.keys || []))
      .catch((e) => console.error("Failed to load API keys", e));

    // Load SSH public key
    api.fetchSshPubKey()
      .then((res) => setSshPubKey(res.public_key || ""))
      .catch((e) => console.error("Failed to load SSH key", e));

    // Load user SSH keys
    api.listSshKeys()
      .then((res) => setUserSshKeys(res.keys || []))
      .catch((e) => console.error("Failed to load user SSH keys", e));

    // Load consent records
    if (userId) {
      api.fetchConsent(userId)
        .then((res) => setConsents(res.consents || []))
        .catch((e) => console.error("Failed to load consent records", e));
    }

    // Load teams
    loadTeams();

    // Load MFA status
    loadMfa();

    // Load sessions
    loadSessions();
  }, [userId, loadTeams]);

  const loadMfa = useCallback(async () => {
    try {
      const res = await api.fetchMfaMethods();
      setMfaEnabled(res.mfa_enabled);
      setMfaMethods(res.methods || []);
      setMfaBackupRemaining(res.backup_codes_remaining || 0);
    } catch { /* MFA not available yet */ }
  }, []);

  const loadSessions = useCallback(async () => {
    setSessionsLoading(true);
    try {
      const res = await api.fetchSessions();
      setSessions(res.sessions || []);
    } catch { /* sessions not available yet */ }
    finally { setSessionsLoading(false); }
  }, []);

  const handleRevokeSession = async (tokenPrefix: string) => {
    setRevokingSession(tokenPrefix);
    try {
      await api.revokeSession(tokenPrefix);
      toast.success(t("dash.settings.session_revoked"));
      loadSessions();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to revoke session");
    } finally {
      setRevokingSession(null);
    }
  };

  // ── Profile Save ──
  const handleSave = async () => {
    setSaving(true);
    try {
      await fetch("/api/users/me/preferences", {
        method: "PUT", credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ canada_only_routing: canadaOnly, notifications }),
      });
      toast.success("Settings saved");
    } catch { toast.error("Failed to save settings"); }
    finally { setSaving(false); }
  };

  // ── API Keys ──
  const handleGenerateKey = async () => {
    if (!newKeyName.trim()) { toast.error("Enter a key name"); return; }
    setGeneratingKey(true);
    try {
      const res = await api.generateApiKey(newKeyName.trim(), newKeyScope);
      setGeneratedKey(res.key);
      setNewKeyName("");
      api.fetchApiKeys().then((r) => setApiKeys(r.keys || []));
      toast.success("API key generated — copy it now, it won't be shown again");
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to generate key"); }
    finally { setGeneratingKey(false); }
  };

  const handleRevokeKey = async (preview: string) => {
    try {
      await api.revokeApiKey(preview);
      setApiKeys((keys) => keys.filter((k) => k.preview !== preview));
      toast.success("Key revoked");
    } catch { toast.error("Failed to revoke key"); }
  };

  // ── SSH Key ──
  const handleGenerateSsh = async () => {
    setGeneratingSsh(true);
    try {
      const res = await api.generateSshKey();
      setSshPubKey(res.public_key);
      toast.success("SSH keypair generated");
    } catch { toast.error("Failed to generate SSH key"); }
    finally { setGeneratingSsh(false); }
  };

  const handleAddSshKey = async () => {
    if (!newSshKeyValue.trim()) { toast.error("Paste your SSH public key"); return; }
    setAddingSshKey(true);
    try {
      await api.uploadSshKey(newSshKeyName.trim() || "default", newSshKeyValue.trim());
      toast.success("SSH key added");
      setNewSshKeyName("");
      setNewSshKeyValue("");
      const res = await api.listSshKeys();
      setUserSshKeys(res.keys || []);
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to add SSH key"); }
    finally { setAddingSshKey(false); }
  };

  const handleDeleteSshKey = async (keyId: string) => {
    try {
      await api.deleteSshKey(keyId);
      setUserSshKeys((keys) => keys.filter((k) => k.id !== keyId));
      toast.success("SSH key removed");
    } catch { toast.error("Failed to remove SSH key"); }
  };

  // ── Change Password ──
  // ── MFA Handlers ──
  const handleTotpSetup = async () => {
    setMfaLoading(true);
    try {
      const res = await api.setupTotp();
      setTotpSetup({ secret: res.secret, uri: res.provisioning_uri, methodId: res.method_id });
      // Generate QR code locally — never send secret to external services
      const dataUrl = await QRCode.toDataURL(res.provisioning_uri, { width: 200, margin: 1 });
      setTotpQrDataUrl(dataUrl);
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to set up TOTP"); }
    finally { setMfaLoading(false); }
  };

  const handleTotpVerify = async () => {
    if (totpCode.length !== 6) return;
    setTotpVerifying(true);
    try {
      const res = await api.verifyTotp(totpCode, totpSetup?.methodId);
      toast.success("Authenticator app enabled");
      if (res.backup_codes) setBackupCodes(res.backup_codes);
      setTotpSetup(null);
      setTotpQrDataUrl(null);
      setTotpCode("");
      loadMfa();
    } catch (err) { toast.error(err instanceof Error ? err.message : "Invalid code"); }
    finally { setTotpVerifying(false); }
  };

  const handleTotpDisable = async () => {
    if (!window.confirm("Are you sure you want to disable your authenticator app?")) return;
    try {
      await api.disableTotp();
      toast.success("Authenticator app disabled");
      loadMfa();
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to disable TOTP"); }
  };

  const handleSmsSetup = async () => {
    const phone = `${smsCountryCode}${smsPhone.replace(/\D/g, "")}`;
    if (phone.length < 8) { toast.error("Enter a valid phone number"); return; }
    setSmsSending(true);
    try {
      await api.setupSms(phone);
      setSmsCodeSent(true);
      toast.success("Verification code sent");
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to send code"); }
    finally { setSmsSending(false); }
  };

  const handleSmsVerify = async () => {
    if (smsCode.length !== 6) return;
    setSmsVerifying(true);
    try {
      const res = await api.verifySms(smsCode);
      toast.success("SMS verification enabled");
      if (res.backup_codes) setBackupCodes(res.backup_codes);
      setSmsSetup(false);
      setSmsCode("");
      setSmsPhone("");
      setSmsCodeSent(false);
      loadMfa();
    } catch (err) { toast.error(err instanceof Error ? err.message : "Invalid code"); }
    finally { setSmsVerifying(false); }
  };

  const handleSmsDisable = async () => {
    if (!window.confirm("Are you sure you want to disable SMS verification?")) return;
    try {
      await api.disableSms();
      toast.success("SMS verification disabled");
      loadMfa();
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to disable SMS"); }
  };

  // ── Passkey helpers ──
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

  const handlePasskeyAdd = async () => {
    if (!window.PublicKeyCredential) {
      toast.error(t("dash.settings.mfa_passkey_unsupported"));
      return;
    }
    setPasskeyRegistering(true);
    try {
      const optRes = await api.passkeyRegisterOptions(passkeyName || "Security Key");
      const publicKey = optRes.options.publicKey as Record<string, unknown>;

      // Convert base64url strings to ArrayBuffer for browser API
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
      toast.success(t("dash.settings.mfa_passkey_success"));
      if (completeRes.backup_codes) setBackupCodes(completeRes.backup_codes);
      setPasskeyName("");
      loadMfa();
    } catch (err) {
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        toast.error("Passkey registration cancelled");
      } else {
        toast.error(err instanceof Error ? err.message : "Failed to register passkey");
      }
    } finally { setPasskeyRegistering(false); }
  };

  const handlePasskeyDelete = async (methodId: number) => {
    setPasskeyRemoving(methodId);
    try {
      await api.deletePasskey(methodId);
      toast.success("Passkey removed");
      loadMfa();
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to remove passkey"); }
    finally { setPasskeyRemoving(null); }
  };

  const handleRegenerateBackupCodes = async () => {
    try {
      const res = await api.regenerateBackupCodes();
      setBackupCodes(res.backup_codes);
      loadMfa();
      toast.success("New backup codes generated");
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to regenerate codes"); }
  };

  const handleDisableAllMfa = async () => {
    if (!window.confirm("This will remove ALL two-factor authentication methods and backup codes. Are you sure?")) return;
    try {
      await api.disableAllMfa();
      toast.success("All 2FA methods disabled");
      setBackupCodes(null);
      loadMfa();
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to disable 2FA"); }
  };

  const handleChangePassword = async () => {
    if (newPw.length < 8) { toast.error("Password must be at least 8 characters"); return; }
    if (newPw !== confirmPw) { toast.error("Passwords do not match"); return; }
    setChangingPw(true);
    try {
      await api.changePassword(currentPw, newPw);
      toast.success("Password changed");
      setCurrentPw(""); setNewPw(""); setConfirmPw("");
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to change password"); }
    finally { setChangingPw(false); }
  };

  // ── Consent Toggles ──
  const consentTypes = ["cross_border", "telemetry", "profiling", "data_collection"];
  const consentLabels: Record<string, { label: string; desc: string }> = {
    cross_border: { label: "Cross-Border Transfer", desc: "Allow data processing outside Canada when needed" },
    telemetry: { label: "Telemetry Collection", desc: "Platform usage analytics to improve service quality" },
    profiling: { label: "Profiling", desc: "Usage patterns for personalized recommendations" },
    data_collection: { label: "Data Collection", desc: "General data collection for service operation" },
  };

  const hasConsent = (type: string) => consents.some((c) => c.consent_type === type);

  const toggleConsent = async (type: string) => {
    if (!userId) return;
    try {
      if (hasConsent(type)) {
        await api.revokeConsent(userId, type);
        setConsents((prev) => prev.filter((c) => c.consent_type !== type));
        toast.success(`${consentLabels[type]?.label || type} consent revoked`);
      } else {
        await api.recordConsent(userId, type);
        setConsents((prev) => [...prev, { consent_id: "", entity_id: userId, consent_type: type, granted_at: new Date().toISOString() }]);
        toast.success(`${consentLabels[type]?.label || type} consent granted`);
      }
    } catch { toast.error("Failed to update consent"); }
  };

  // ── Delete Account ──
  const handleDeleteAccount = async () => {
    if (deleteConfirm !== "DELETE") return;
    setDeleting(true);
    try {
      await api.deleteAccount();
      toast.success("Account deleted");
      logout();
    } catch { toast.error("Failed to delete account"); }
    finally { setDeleting(false); }
  };

  const Toggle = ({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) => (
    <button
      onClick={onToggle}
      className={`relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors ${enabled ? "bg-emerald" : "bg-border"}`}
    >
      <span className={`inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${enabled ? "translate-x-6" : "translate-x-1"}`} />
    </button>
  );

  return (
    <div className="space-y-6 max-w-2xl">
      <h1 className="text-2xl font-bold">{t("dash.settings.title")}</h1>

      {/* ── Profile ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><SettingsIcon className="h-4 w-4" /> {t("dash.settings.profile")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>{t("dash.settings.name")}</Label>
            <Input value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="space-y-2">
            <Label>{t("dash.settings.email")}</Label>
            <Input value={email} disabled className="opacity-60" />
            <p className="text-xs text-text-muted">{t("dash.settings.email_note")}</p>
          </div>
        </CardContent>
      </Card>

      {/* ── Jurisdiction ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Globe className="h-4 w-4" /> {t("dash.settings.jurisdiction")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{t("dash.settings.canada_only")}</p>
              <p className="text-xs text-text-secondary">{t("dash.settings.canada_only_desc")}</p>
            </div>
            <Toggle enabled={canadaOnly} onToggle={() => setCanadaOnly(!canadaOnly)} />
          </div>
        </CardContent>
      </Card>

      {/* ── Notifications ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Bell className="h-4 w-4" /> {t("dash.settings.notifications")}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium">{t("dash.settings.email_notif")}</p>
              <p className="text-xs text-text-secondary">{t("dash.settings.email_notif_desc")}</p>
            </div>
            <Toggle enabled={notifications} onToggle={() => setNotifications(!notifications)} />
          </div>
        </CardContent>
      </Card>

      <Button onClick={handleSave} disabled={saving}>
        {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
        {saving ? "Saving..." : t("dash.settings.save")}
      </Button>

      {/* ── API Keys ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Key className="h-4 w-4" /> {t("dash.settings.api_keys")}</CardTitle>
          <CardDescription>{t("dash.settings.api_keys_desc")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Generate new key */}
          <div className="flex flex-col sm:flex-row gap-2">
            <Input
              placeholder={t("dash.settings.key_name_placeholder")}
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              className="flex-1 min-w-0"
            />
            <div className="flex gap-2">
              <Select value={newKeyScope} onChange={(e) => setNewKeyScope(e.target.value as "full-access" | "read-only")}>
                <option value="full-access">{t("dash.settings.full_access")}</option>
                <option value="read-only">{t("dash.settings.read_only")}</option>
              </Select>
              <Button onClick={handleGenerateKey} disabled={generatingKey} size="sm">
                {generatingKey ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                {t("dash.settings.generate")}
              </Button>
            </div>
          </div>

          {/* Show newly generated key */}
          {generatedKey && (
            <div className="rounded-lg border border-gold/30 bg-gold/5 p-3">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-gold" />
                <span className="text-sm font-medium text-gold">Copy your key now — it won&apos;t be shown again</span>
              </div>
              <div className="flex items-center gap-2">
                <code className="flex-1 rounded bg-background px-3 py-2 font-mono text-xs break-all">{generatedKey}</code>
                <Button
                  variant="outline" size="icon"
                  onClick={() => { navigator.clipboard.writeText(generatedKey); toast.success("Copied"); }}
                >
                  <Copy className="h-3.5 w-3.5" />
                </Button>
              </div>
              <Button variant="ghost" size="sm" className="mt-2 text-xs" onClick={() => setGeneratedKey(null)}>
                Dismiss
              </Button>
            </div>
          )}

          {/* Existing keys */}
          {apiKeys.length === 0 ? (
            <p className="text-sm text-text-muted">{t("dash.settings.no_keys")}</p>
          ) : (
            <div className="space-y-2">
              {apiKeys.map((k) => (
                <div key={k.preview} className="flex items-center justify-between rounded-lg border border-border p-3">
                  <div>
                    <p className="text-sm font-medium">{k.name}</p>
                    <p className="text-xs text-text-muted font-mono">{k.preview}… · {k.scope}</p>
                  </div>
                  <Button variant="ghost" size="sm" onClick={() => handleRevokeKey(k.preview)} className="text-accent-red hover:text-accent-red">
                    <Trash2 className="h-3.5 w-3.5" /> Revoke
                  </Button>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── SSH Keys ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Terminal className="h-4 w-4" /> {t("dash.settings.ssh_keys")}</CardTitle>
          <CardDescription>{t("dash.settings.ssh_desc")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* User SSH Keys */}
          <div className="space-y-3">
            <Label className="text-sm font-medium">Your SSH Public Keys</Label>
            <p className="text-xs text-text-muted">Add your SSH public keys for secure access to GPU hosts. Paste the contents of your <code className="bg-background px-1 rounded">~/.ssh/id_ed25519.pub</code> or <code className="bg-background px-1 rounded">~/.ssh/id_rsa.pub</code> file.</p>

            {/* Add new key form */}
            <div className="space-y-2 rounded-lg border border-border p-3">
              <Input
                placeholder="Key name (e.g. Work Laptop, Home Desktop)"
                value={newSshKeyName}
                onChange={(e) => setNewSshKeyName(e.target.value)}
              />
              <textarea
                className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs placeholder:text-text-muted min-h-[80px] resize-y focus:outline-none focus:ring-2 focus:ring-ring"
                placeholder="ssh-ed25519 AAAA... user@host"
                value={newSshKeyValue}
                onChange={(e) => setNewSshKeyValue(e.target.value)}
              />
              <Button variant="outline" size="sm" onClick={handleAddSshKey} disabled={addingSshKey}>
                {addingSshKey ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                Add SSH Key
              </Button>
            </div>

            {/* Existing user keys */}
            {userSshKeys.length > 0 && (
              <div className="space-y-2">
                {userSshKeys.map((k) => (
                  <div key={k.id} className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-medium">{k.name}</p>
                      <p className="text-xs text-text-muted font-mono truncate">{k.fingerprint}</p>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => handleDeleteSshKey(k.id)} className="text-accent-red hover:text-accent-red shrink-0 ml-2">
                      <Trash2 className="h-3.5 w-3.5" /> Remove
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Platform SSH Key (generate) */}
          <div className="space-y-3 border-t border-border pt-4">
            <Label className="text-sm font-medium">Platform SSH Key</Label>
            <p className="text-xs text-text-muted">Generate a platform keypair for infrastructure access.</p>
            {sshPubKey ? (
              <div>
                <div className="flex items-start gap-2">
                  <code className="flex-1 rounded bg-background px-3 py-2 font-mono text-xs break-all max-h-20 overflow-y-auto">
                    {sshPubKey}
                  </code>
                  <Button
                    variant="outline" size="icon"
                    onClick={() => { navigator.clipboard.writeText(sshPubKey); toast.success("Copied to clipboard"); }}
                  >
                    <Copy className="h-3.5 w-3.5" />
                  </Button>
                </div>
                <p className="text-xs text-text-muted mt-2">{t("dash.settings.ssh_note")}</p>
              </div>
            ) : (
              <p className="text-sm text-text-muted">No platform keypair generated yet</p>
            )}
            <Button variant="outline" size="sm" onClick={handleGenerateSsh} disabled={generatingSsh}>
              {generatingSsh ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
              {sshPubKey ? t("dash.settings.regen_keypair") : t("dash.settings.gen_keypair")}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* ── Change Password ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Lock className="h-4 w-4" /> {t("dash.settings.change_pw")}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label>{t("dash.settings.current_pw")}</Label>
            <div className="relative">
              <Input
                type={showPw ? "text" : "password"}
                value={currentPw}
                onChange={(e) => setCurrentPw(e.target.value)}
              />
              <button
                type="button"
                onClick={() => setShowPw(!showPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
              >
                {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
          <div className="space-y-2">
            <Label>{t("dash.settings.new_pw")}</Label>
            <div className="relative">
              <Input type={showNewPw ? "text" : "password"} value={newPw} onChange={(e) => setNewPw(e.target.value)} placeholder={t("dash.settings.pw_min")} />
              <button
                type="button"
                onClick={() => setShowNewPw(!showNewPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
              >
                {showNewPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
          </div>
          <div className="space-y-2">
            <Label>{t("dash.settings.confirm_pw")}</Label>
            <div className="relative">
              <Input type={showConfirmPw ? "text" : "password"} value={confirmPw} onChange={(e) => setConfirmPw(e.target.value)} />
              <button
                type="button"
                onClick={() => setShowConfirmPw(!showConfirmPw)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
              >
                {showConfirmPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              </button>
            </div>
            {confirmPw && newPw !== confirmPw && (
              <p className="text-xs text-accent-red">{t("dash.settings.pw_mismatch")}</p>
            )}
          </div>
          <Button
            onClick={handleChangePassword}
            disabled={changingPw || !currentPw || newPw.length < 8 || newPw !== confirmPw}
          >
            {changingPw ? <><Loader2 className="h-4 w-4 animate-spin" /> Changing…</> : t("dash.settings.change_pw")}
          </Button>
        </CardContent>
      </Card>

      {/* ── Two-Factor Authentication ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="h-4 w-4" /> {t("dash.settings.mfa_title")}
          </CardTitle>
          <CardDescription>{t("dash.settings.mfa_desc")}</CardDescription>
          {mfaEnabled && (
            <div className="flex items-center gap-2 mt-2">
              <CheckCircle className="h-4 w-4 text-emerald" />
              <span className="text-xs text-emerald font-medium">{t("dash.settings.mfa_enabled")}</span>
            </div>
          )}
        </CardHeader>
        <CardContent className="space-y-6">
          {/* ── TOTP Authenticator ── */}
          <div className="rounded-lg border border-border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium flex items-center gap-2">
                  <Key className="h-3.5 w-3.5 text-ice-blue" />
                  {t("dash.settings.mfa_totp_title")}
                </p>
                <p className="text-xs text-text-muted mt-0.5">{t("dash.settings.mfa_totp_desc")}</p>
              </div>
              {mfaMethods.some((m) => m.type === "totp" && m.enabled) ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-emerald font-medium flex items-center gap-1">
                    <CheckCircle className="h-3 w-3" /> Active
                  </span>
                  <Button variant="outline" size="sm" onClick={handleTotpDisable} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                    {t("dash.settings.mfa_disable")}
                  </Button>
                </div>
              ) : (
                <Button variant="outline" size="sm" onClick={handleTotpSetup} disabled={mfaLoading}>
                  {mfaLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                  {t("dash.settings.mfa_totp_setup")}
                </Button>
              )}
            </div>
            {totpSetup && (
              <div className="mt-3 space-y-3 rounded-lg bg-surface p-4">
                <p className="text-sm text-text-secondary">{t("dash.settings.mfa_totp_scan")}</p>
                {/* QR Code generated locally — secret never leaves browser */}
                <div className="flex justify-center">
                  <div className="rounded-lg bg-white p-3">
                    {totpQrDataUrl ? (
                      /* eslint-disable-next-line @next/next/no-img-element */
                      <img
                        src={totpQrDataUrl}
                        alt="TOTP QR Code"
                        width={200}
                        height={200}
                        className="rounded"
                      />
                    ) : (
                      <div className="h-[200px] w-[200px] flex items-center justify-center">
                        <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
                      </div>
                    )}
                  </div>
                </div>
                <div className="space-y-1">
                  <p className="text-xs text-text-muted">{t("dash.settings.mfa_totp_manual")}</p>
                  <div className="flex items-center gap-2">
                    <code className="flex-1 text-xs font-mono bg-background rounded px-2 py-1.5 select-all border border-border break-all">
                      {totpSetup.secret}
                    </code>
                    <button
                      onClick={() => { navigator.clipboard.writeText(totpSetup.secret); toast.success("Copied"); }}
                      className="text-text-muted hover:text-text-primary"
                    >
                      <Copy className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label>{t("dash.settings.mfa_totp_verify")}</Label>
                  <div className="flex gap-2">
                    <Input
                      value={totpCode}
                      onChange={(e) => setTotpCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                      placeholder={t("dash.settings.mfa_code_placeholder")}
                      className="font-mono w-32 text-center tracking-widest"
                      maxLength={6}
                    />
                    <Button onClick={handleTotpVerify} disabled={totpVerifying || totpCode.length !== 6}>
                      {totpVerifying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                      {t("dash.settings.mfa_verify")}
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* ── SMS Verification ── */}
          <div className="rounded-lg border border-border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium flex items-center gap-2">
                  <Bell className="h-3.5 w-3.5 text-accent-gold" />
                  {t("dash.settings.mfa_sms_title")}
                </p>
                <p className="text-xs text-text-muted mt-0.5">{t("dash.settings.mfa_sms_desc")}</p>
              </div>
              {mfaMethods.some((m) => m.type === "sms" && m.enabled) ? (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-emerald font-medium flex items-center gap-1">
                    <CheckCircle className="h-3 w-3" /> Active
                  </span>
                  <span className="text-xs text-text-muted">
                    •••• {mfaMethods.find((m) => m.type === "sms")?.phone_number}
                  </span>
                  <Button variant="outline" size="sm" onClick={handleSmsDisable} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                    {t("dash.settings.mfa_disable")}
                  </Button>
                </div>
              ) : (
                <Button variant="outline" size="sm" onClick={() => setSmsSetup(true)}>
                  {t("dash.settings.mfa_sms_setup")}
                </Button>
              )}
            </div>
            {smsSetup && !mfaMethods.some((m) => m.type === "sms" && m.enabled) && (
              <div className="mt-3 space-y-3 rounded-lg bg-surface p-4">
                {!smsCodeSent ? (
                  <div className="space-y-2">
                    <Label>{t("dash.settings.mfa_sms_phone")}</Label>
                    <div className="flex gap-2">
                      <select
                        value={smsCountryCode}
                        onChange={(e) => setSmsCountryCode(e.target.value)}
                        className="rounded-md border border-border bg-background px-2 py-1.5 text-sm min-w-[100px]"
                      >
                        {COUNTRY_CODES.map((c) => (
                          <option key={`${c.code}-${c.name}`} value={c.code}>
                            {c.flag} {c.code}
                          </option>
                        ))}
                      </select>
                      <Input
                        type="tel"
                        value={smsPhone}
                        onChange={(e) => setSmsPhone(e.target.value)}
                        placeholder="4165551234"
                        className="flex-1 font-mono"
                      />
                      <Button onClick={handleSmsSetup} disabled={smsSending}>
                        {smsSending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                        Send Code
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Label>{t("dash.settings.mfa_sms_verify")}</Label>
                    <div className="flex gap-2">
                      <Input
                        value={smsCode}
                        onChange={(e) => setSmsCode(e.target.value.replace(/\D/g, "").slice(0, 6))}
                        placeholder={t("dash.settings.mfa_code_placeholder")}
                        className="font-mono w-32 text-center tracking-widest"
                        maxLength={6}
                      />
                      <Button onClick={handleSmsVerify} disabled={smsVerifying || smsCode.length !== 6}>
                        {smsVerifying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                        {t("dash.settings.mfa_verify")}
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* ── Passkeys ── */}
          <div className="rounded-lg border border-border p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium flex items-center gap-2"><Key className="h-4 w-4" /> {t("dash.settings.mfa_passkey_title")}</p>
                <p className="text-xs text-text-muted mt-0.5">{t("dash.settings.mfa_passkey_desc")}</p>
              </div>
            </div>

            {/* Existing passkeys */}
            {mfaMethods.filter((m) => m.type === "passkey" && m.enabled).map((pk) => (
              <div key={pk.id} className="flex items-center justify-between rounded-md bg-surface px-3 py-2">
                <div className="flex items-center gap-2">
                  <Key className="h-3.5 w-3.5 text-emerald" />
                  <span className="text-sm">{pk.device_name || "Security Key"}</span>
                  <span className="text-xs text-text-muted">{new Date(pk.created_at * 1000).toLocaleDateString()}</span>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  className="text-accent-red border-accent-red/30 hover:bg-accent-red/10"
                  disabled={passkeyRemoving === pk.id}
                  onClick={() => {
                    if (!window.confirm(t("dash.settings.mfa_passkey_remove_desc", { name: pk.device_name || "Security Key" }))) return;
                    handlePasskeyDelete(pk.id);
                  }}
                >
                  {passkeyRemoving === pk.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                </Button>
              </div>
            ))}

            {/* Add new passkey */}
            <div className="flex gap-2">
              <Input
                value={passkeyName}
                onChange={(e) => setPasskeyName(e.target.value)}
                placeholder={t("dash.settings.mfa_passkey_name_placeholder")}
                className="flex-1"
              />
              <Button onClick={handlePasskeyAdd} disabled={passkeyRegistering}>
                {passkeyRegistering ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                {passkeyRegistering ? t("dash.settings.mfa_passkey_registering") : t("dash.settings.mfa_passkey_add")}
              </Button>
            </div>
          </div>

          {/* ── Backup Codes ── */}
          {mfaEnabled && (
            <div className="rounded-lg border border-border p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium">{t("dash.settings.mfa_backup_title")}</p>
                  <p className="text-xs text-text-muted mt-0.5">
                    {t("dash.settings.mfa_backup_remaining", { count: String(mfaBackupRemaining) })}
                  </p>
                </div>
                <Button variant="outline" size="sm" onClick={handleRegenerateBackupCodes}>
                  {t("dash.settings.mfa_backup_regenerate")}
                </Button>
              </div>
              {backupCodes && (
                <div className="space-y-2">
                  <p className="text-xs text-accent-gold flex items-center gap-1">
                    <AlertTriangle className="h-3 w-3" /> {t("dash.settings.mfa_backup_desc")}
                  </p>
                  <div className="grid grid-cols-2 gap-1.5 rounded-lg bg-background border border-border p-3">
                    {backupCodes.map((code, i) => (
                      <code key={i} className="text-xs font-mono text-text-primary select-all">{code}</code>
                    ))}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      navigator.clipboard.writeText(backupCodes.join("\n"));
                      toast.success("Backup codes copied");
                    }}
                  >
                    <Copy className="h-3.5 w-3.5" /> Copy All
                  </Button>
                </div>
              )}
            </div>
          )}

          {/* Disable all */}
          {mfaEnabled && (
            <div className="pt-2 border-t border-border">
              <Button variant="outline" size="sm" onClick={handleDisableAllMfa} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                <AlertTriangle className="h-3.5 w-3.5" /> {t("dash.settings.mfa_disable_all")}
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── PIPEDA Consent ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><ShieldCheck className="h-4 w-4" /> {t("dash.settings.pipeda_title")}</CardTitle>
          <CardDescription>{t("dash.settings.pipeda_desc")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {consentTypes.map((type) => {
            const info = consentLabels[type];
            const enabled = hasConsent(type);
            return (
              <div key={type} className="flex items-center justify-between rounded-lg border border-border p-3">
                <div>
                  <p className="text-sm font-medium">{info?.label || type}</p>
                  <p className="text-xs text-text-secondary">{info?.desc || ""}</p>
                </div>
                <Toggle enabled={enabled} onToggle={() => toggleConsent(type)} />
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* ── Team Management ── */}
      <Card id="team">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Users className="h-4 w-4" /> {t("dash.settings.team")}</CardTitle>
          <CardDescription>{t("dash.settings.team_desc")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {teams.length === 0 ? (
            <div className="space-y-3">
              <p className="text-sm text-text-muted">You don&apos;t belong to any teams yet.</p>
              <div className="flex gap-2">
                <Input
                  placeholder="Team name"
                  value={newTeamName}
                  onChange={(e) => setNewTeamName(e.target.value)}
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleCreateTeam}
                  disabled={!newTeamName.trim() || creatingTeam}
                >
                  {creatingTeam ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                  Create Team
                </Button>
              </div>
            </div>
          ) : (
            <>
              {/* Team selector (if multiple) */}
              {teams.length > 1 && (
                <div className="flex gap-1.5 flex-wrap">
                  {teams.map((t) => (
                    <button
                      key={t.team_id}
                      onClick={() => selectTeam(t)}
                      className={`rounded-md px-3 py-1 text-xs font-medium transition-colors ${
                        activeTeam?.team_id === t.team_id
                          ? "bg-card text-text-primary shadow-sm"
                          : "bg-surface text-text-muted hover:text-text-primary"
                      }`}
                    >
                      {t.name}
                    </button>
                  ))}
                </div>
              )}

              {activeTeam && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between rounded-lg border border-border p-3">
                    <div>
                      <p className="text-sm font-medium">{activeTeam.name}</p>
                      <p className="text-xs text-text-muted">
                        {activeTeam.plan} plan &middot; {teamMembers.length}/{activeTeam.max_members} members
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-text-muted">{activeTeam.team_id}</span>
                      {activeTeam.owner_email === email && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setDeleteTeamConfirm(true)}
                          className="text-accent-red hover:text-accent-red"
                          title="Delete team"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      )}
                    </div>
                  </div>

                  {/* Members list */}
                  <div className="space-y-2">
                    <p className="text-xs font-medium text-text-muted uppercase tracking-wider">Members</p>
                    {teamMembers.map((m) => {
                      const isOwner = m.email === activeTeam.owner_email;
                      const iAmAdmin = teamMembers.find((x) => x.email === email)?.role === "admin";
                      return (
                        <div key={m.email} className="flex items-center justify-between rounded-lg border border-border p-3">
                          <div className="flex items-center gap-3">
                            <div className="flex h-8 w-8 items-center justify-center rounded-full bg-ice-blue/10 text-xs font-bold text-ice-blue">
                              {m.email.charAt(0).toUpperCase()}
                            </div>
                            <div>
                              <p className="text-sm">{m.email}{isOwner && <span className="ml-1.5 text-[10px] text-accent-gold font-medium">OWNER</span>}</p>
                              {iAmAdmin && !isOwner ? (
                                <Select
                                  value={m.role}
                                  onChange={(e) => handleRoleChange(m.email, e.target.value)}
                                  className="mt-0.5 h-6 w-24 text-xs py-0"
                                >
                                  <option value="admin">Admin</option>
                                  <option value="member">Member</option>
                                  <option value="viewer">Viewer</option>
                                </Select>
                              ) : (
                                <p className="text-xs text-text-muted capitalize">{m.role}</p>
                              )}
                            </div>
                          </div>
                          {iAmAdmin && m.email !== email && !isOwner && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setRemoveTarget(m.email)}
                              className="text-accent-red hover:text-accent-red"
                            >
                              <UserMinus className="h-3.5 w-3.5" />
                            </Button>
                          )}
                        </div>
                      );
                    })}
                  </div>

                  {/* Invite member */}
                  <div className="flex gap-2">
                    <Input
                      type="email"
                      placeholder="Email address"
                      value={inviteEmail}
                      onChange={(e) => setInviteEmail(e.target.value)}
                      className="flex-1"
                    />
                    <Select value={inviteRole} onChange={(e) => setInviteRole(e.target.value)} className="w-28">
                      <option value="member">Member</option>
                      <option value="admin">Admin</option>
                      <option value="viewer">Viewer</option>
                    </Select>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleInvite}
                      disabled={!inviteEmail.trim() || inviting}
                    >
                      {inviting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <UserPlus className="h-3.5 w-3.5" />}
                      Invite
                    </Button>
                  </div>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* ── Active Sessions ── */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Globe className="h-4 w-4" /> {t("dash.settings.sessions_title")}</CardTitle>
          <CardDescription>{t("dash.settings.sessions_desc")}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          {sessionsLoading ? (
            <div className="flex items-center justify-center py-6"><Loader2 className="h-5 w-5 animate-spin" /></div>
          ) : sessions.length === 0 ? (
            <p className="text-sm text-text-muted">{t("dash.settings.sessions_none")}</p>
          ) : (
            sessions.map((s) => {
              const ua = s.user_agent || "";
              const browser = ua.match(/Chrome|Firefox|Safari|Edge|Opera/)?.[0] || "Unknown browser";
              const os = ua.match(/Windows|Mac OS|Linux|Android|iOS/)?.[0] || "Unknown OS";
              const lastActive = s.last_active ? new Date(s.last_active * 1000).toLocaleString() : "—";
              return (
                <div key={s.token_prefix} className="flex items-center justify-between rounded-lg border border-border p-3">
                  <div className="space-y-0.5 min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium truncate">{browser} on {os}</p>
                      {s.is_current && (
                        <span className="shrink-0 rounded bg-green-500/20 px-1.5 py-0.5 text-[10px] font-medium text-green-400">
                          {t("dash.settings.sessions_current")}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-text-muted">
                      {s.ip_address || "Unknown IP"} · {t("dash.settings.sessions_last_active")} {lastActive}
                    </p>
                  </div>
                  {!s.is_current && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="ml-3 shrink-0 text-accent-red border-accent-red/30 hover:bg-accent-red/10"
                      onClick={() => handleRevokeSession(s.token_prefix)}
                      disabled={revokingSession === s.token_prefix}
                    >
                      {revokingSession === s.token_prefix ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        t("dash.settings.sessions_revoke")
                      )}
                    </Button>
                  )}
                </div>
              );
            })
          )}
        </CardContent>
      </Card>

      {/* ── Data Export & Deletion ── */}
      <Card className="border-accent-red/20">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-accent-red"><Trash2 className="h-4 w-4" /> Danger Zone</CardTitle>
          <CardDescription>Export your data or permanently delete your account</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between rounded-lg border border-border p-3">
            <div>
              <p className="text-sm font-medium">Export Data</p>
              <p className="text-xs text-text-secondary">Download all your billing data as CSV</p>
            </div>
            <Button variant="outline" size="sm" onClick={async () => {
              if (!confirm("Export all your billing data as CSV?")) return;
              try {
                const now = Math.floor(Date.now() / 1000);
                const blob = await api.downloadInvoice(userId, "csv", now - 365 * 86400, now);
                const url = URL.createObjectURL(blob);
                const a = document.createElement("a"); a.href = url; a.download = "xcelsior-data-export.csv"; a.click();
                URL.revokeObjectURL(url);
                toast.success("Data exported");
              } catch { toast.error("Export failed"); }
            }}>
              <Download className="h-3.5 w-3.5" /> Export
            </Button>
          </div>

          <div className="rounded-lg border border-accent-red/30 bg-accent-red/5 p-4">
            <p className="text-sm font-medium text-accent-red mb-2">Delete Account</p>
            <p className="text-xs text-text-secondary mb-3">
              This will permanently delete your account, all sessions, and related data. This action cannot be undone.
            </p>
            <div className="flex gap-2">
              <Input
                placeholder='Type "DELETE" to confirm'
                value={deleteConfirm}
                onChange={(e) => setDeleteConfirm(e.target.value)}
                className="flex-1"
              />
              <Button
                variant="outline"
                onClick={handleDeleteAccount}
                disabled={deleteConfirm !== "DELETE" || deleting}
                className="text-accent-red border-accent-red/30 hover:bg-accent-red/10"
              >
                {deleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                Delete
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      <ConfirmDialog
        open={removeTarget !== null}
        title={t("dash.settings.remove_member_title")}
        description={t("dash.settings.remove_member_desc", { email: removeTarget || "" })}
        confirmLabel={t("dash.settings.remove_member_confirm")}
        cancelLabel={t("common.cancel")}
        variant="danger"
        onConfirm={() => removeTarget && handleRemoveMember(removeTarget)}
        onCancel={() => setRemoveTarget(null)}
      />
      <ConfirmDialog
        open={deleteTeamConfirm}
        title="Delete Team"
        description={`Are you sure you want to delete "${activeTeam?.name}"? All members will be removed and this cannot be undone.`}
        confirmLabel={deletingTeam ? "Deleting…" : "Delete Team"}
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={handleDeleteTeam}
        onCancel={() => setDeleteTeamConfirm(false)}
      />
    </div>
  );
}
