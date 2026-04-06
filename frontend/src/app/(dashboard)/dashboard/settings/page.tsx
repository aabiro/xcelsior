"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input, Label, Select } from "@/components/ui/input";
import { PasswordRequirements } from "@/components/auth/password-requirements";
import {
  Settings as SettingsIcon, Save, Shield, Bell, Globe, Key, Terminal,
  Lock, Trash2, Download, Eye, EyeOff, Copy, Plus, AlertTriangle,
  CheckCircle, Loader2, ShieldCheck, X, Users, UserPlus, UserMinus,
  User, Fingerprint, MonitorSmartphone, KeyRound,
} from "lucide-react";
import { useAuth } from "@/lib/auth";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { ApiKeyInfo, ConsentRecord, TeamInfo, TeamMember, UserSshKey } from "@/lib/api";
import QRCode from "qrcode";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { COUNTRY_CODES } from "@/lib/country-codes";
import { FadeIn, StaggerList, StaggerItem } from "@/components/ui/motion";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";
import {
  PASSWORD_MAX_LENGTH,
  PASSWORD_MIN_LENGTH,
  getPasswordValidation,
  passwordsMatch,
} from "@/lib/password-validation";

// ── Tab definition ──────────────────────────────────────────────────

const TABS = [
  { id: "profile", labelKey: "dash.settings.tab_profile", fallback: "Profile", icon: User, color: "text-accent-cyan" },
  { id: "security", labelKey: "dash.settings.tab_security", fallback: "Security", icon: Shield, color: "text-accent-violet" },
  { id: "api-keys", labelKey: "dash.settings.tab_api", fallback: "API & SSH", icon: Key, color: "text-accent-gold" },
  { id: "team", labelKey: "dash.settings.tab_team", fallback: "Team", icon: Users, color: "text-emerald" },
  { id: "privacy", labelKey: "dash.settings.tab_privacy", fallback: "Privacy", icon: ShieldCheck, color: "text-accent-cyan" },
] as const;

type TabId = (typeof TABS)[number]["id"];

// ── Icon badge helper ────���──────────────────────────────────────────

function IconBadge({ icon: Icon, color, bg }: { icon: typeof User; color: string; bg: string }) {
  return (
    <div className={cn("flex h-8 w-8 items-center justify-center rounded-lg", bg)}>
      <Icon className={cn("h-4 w-4", color)} />
    </div>
  );
}

// ── Toggle ──────────────────────────────────────────────────────────

function Toggle({ enabled, onToggle }: { enabled: boolean; onToggle: () => void }) {
  return (
    <button
      onClick={onToggle}
      className={cn(
        "relative inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200",
        enabled ? "bg-emerald" : "bg-border",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200",
          enabled ? "translate-x-6" : "translate-x-1",
        )}
      />
    </button>
  );
}

// ── Main Page ─────────────────────────────────────────���─────────────

export default function SettingsPage() {
  const { user, logout } = useAuth();
  const { t } = useLocale();
  const userId = user?.customer_id || user?.user_id || "";

  // Tab
  const [activeTab, setActiveTab] = useState<TabId>("profile");
  const tabRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [indicatorStyle, setIndicatorStyle] = useState({ left: 0, width: 0 });

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

  // ── Hash-based tab routing ──
  useEffect(() => {
    const hash = window.location.hash.replace("#", "");
    if (TABS.some((tab) => tab.id === hash)) setActiveTab(hash as TabId);
  }, []);

  useEffect(() => {
    window.location.hash = activeTab;
  }, [activeTab]);

  // ── Animated tab indicator ──
  useEffect(() => {
    const el = tabRefs.current[activeTab];
    if (el) {
      setIndicatorStyle({ left: el.offsetLeft, width: el.offsetWidth });
    }
  }, [activeTab]);

  // ── Data loading ──────────────────────────────────────────────────

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

  useEffect(() => {
    fetch("/api/users/me/preferences", { credentials: "include" })
      .then((r) => r.ok ? r.json() : Promise.reject())
      .then((prefs) => {
        setCanadaOnly(prefs.canada_only_routing ?? false);
        setNotifications(prefs.notifications ?? true);
      })
      .catch((e) => console.error("Failed to load preferences", e));

    api.fetchApiKeys()
      .then((res) => setApiKeys(res.keys || []))
      .catch((e) => console.error("Failed to load API keys", e));

    api.fetchSshPubKey()
      .then((res) => setSshPubKey(res.public_key || ""))
      .catch((e) => console.error("Failed to load SSH key", e));

    api.listSshKeys()
      .then((res) => setUserSshKeys(res.keys || []))
      .catch((e) => console.error("Failed to load user SSH keys", e));

    if (userId) {
      api.fetchConsent(userId)
        .then((res) => setConsents(res.consents || []))
        .catch((e) => console.error("Failed to load consent records", e));
    }

    loadTeams();
    loadMfa();
    loadSessions();
  }, [userId, loadTeams, loadMfa, loadSessions]);

  // ── Handlers ──────────────────────────────────────────────────────

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

  const handleTotpSetup = async () => {
    setMfaLoading(true);
    try {
      const res = await api.setupTotp();
      setTotpSetup({ secret: res.secret, uri: res.provisioning_uri, methodId: res.method_id });
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
    if (!getPasswordValidation(newPw).isValid) { toast.error(t("auth.pw_policy_error")); return; }
    if (!passwordsMatch(newPw, confirmPw)) { toast.error(t("dash.settings.pw_mismatch")); return; }
    setChangingPw(true);
    try {
      await api.changePassword(currentPw, newPw);
      toast.success("Password changed");
      setCurrentPw(""); setNewPw(""); setConfirmPw("");
    } catch (err) { toast.error(err instanceof Error ? err.message : "Failed to change password"); }
    finally { setChangingPw(false); }
  };

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

  const handleRemoveMember = async (memberEmail: string) => {
    setRemoveTarget(null);
    if (!activeTeam) return;
    try {
      await api.removeTeamMember(activeTeam.team_id, memberEmail);
      toast.success(`${memberEmail} removed`);
      setTeamMembers((prev) => prev.filter((m) => m.email !== memberEmail));
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

  // ── Render ────────────────────────────────────────────────────────

  return (
    <div className="max-w-3xl space-y-6">
      {/* ── Page Header ── */}
      <FadeIn>
        <div className="flex items-start gap-4">
          <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-xl bg-accent-cyan/10 ring-1 ring-accent-cyan/20">
            <SettingsIcon className="h-6 w-6 text-accent-cyan" />
          </div>
          <div>
            <h1 className="text-2xl font-bold">{t("dash.settings.title")}</h1>
            <p className="text-sm text-text-secondary mt-0.5">
              {t("dash.settings.subtitle") !== "dash.settings.subtitle"
                ? t("dash.settings.subtitle")
                : "Manage your account, security, and preferences"}
            </p>
          </div>
        </div>
      </FadeIn>

      {/* ── Tab Bar ── */}
      <FadeIn delay={0.08}>
        <div className="relative">
          <div className="flex gap-1 rounded-xl border border-border/60 bg-surface/50 p-1 backdrop-blur-sm overflow-x-auto">
            {TABS.map((tab) => {
              const active = activeTab === tab.id;
              const label = t(tab.labelKey) !== tab.labelKey ? t(tab.labelKey) : tab.fallback;
              return (
                <button
                  key={tab.id}
                  ref={(el) => { tabRefs.current[tab.id] = el; }}
                  onClick={() => setActiveTab(tab.id)}
                  className={cn(
                    "relative flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-colors whitespace-nowrap",
                    active
                      ? "text-text-primary"
                      : "text-text-muted hover:text-text-secondary",
                  )}
                >
                  <tab.icon className={cn("h-3.5 w-3.5", active ? tab.color : "")} />
                  {label}
                </button>
              );
            })}
            {/* Animated indicator */}
            <motion.div
              className="absolute bottom-1 h-[calc(100%-8px)] rounded-lg bg-surface border border-border/60 shadow-sm -z-10"
              animate={{ left: indicatorStyle.left, width: indicatorStyle.width }}
              transition={{ type: "spring", stiffness: 380, damping: 30, mass: 0.8 }}
            />
          </div>
          <div className="brand-line mt-3" />
        </div>
      </FadeIn>

      {/* ── Tab Content ── */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
        >
          {activeTab === "profile" && (
            <ProfileTab
              t={t} name={name} setName={setName} email={email}
              canadaOnly={canadaOnly} setCanadaOnly={setCanadaOnly}
              notifications={notifications} setNotifications={setNotifications}
              saving={saving} onSave={handleSave}
            />
          )}
          {activeTab === "security" && (
            <SecurityTab
              t={t}
              currentPw={currentPw} setCurrentPw={setCurrentPw}
              newPw={newPw} setNewPw={setNewPw}
              confirmPw={confirmPw} setConfirmPw={setConfirmPw}
              changingPw={changingPw} onChangePassword={handleChangePassword}
              showPw={showPw} setShowPw={setShowPw}
              showNewPw={showNewPw} setShowNewPw={setShowNewPw}
              showConfirmPw={showConfirmPw} setShowConfirmPw={setShowConfirmPw}
              mfaEnabled={mfaEnabled} mfaMethods={mfaMethods}
              mfaBackupRemaining={mfaBackupRemaining} mfaLoading={mfaLoading}
              totpSetup={totpSetup} totpQrDataUrl={totpQrDataUrl}
              totpCode={totpCode} setTotpCode={setTotpCode}
              totpVerifying={totpVerifying} backupCodes={backupCodes}
              onTotpSetup={handleTotpSetup} onTotpVerify={handleTotpVerify}
              onTotpDisable={handleTotpDisable}
              smsSetup={smsSetup} setSmsSetup={setSmsSetup}
              smsPhone={smsPhone} setSmsPhone={setSmsPhone}
              smsCountryCode={smsCountryCode} setSmsCountryCode={setSmsCountryCode}
              smsCode={smsCode} setSmsCode={setSmsCode}
              smsSending={smsSending} smsVerifying={smsVerifying}
              smsCodeSent={smsCodeSent}
              onSmsSetup={handleSmsSetup} onSmsVerify={handleSmsVerify}
              onSmsDisable={handleSmsDisable}
              passkeyName={passkeyName} setPasskeyName={setPasskeyName}
              passkeyRegistering={passkeyRegistering}
              passkeyRemoving={passkeyRemoving}
              onPasskeyAdd={handlePasskeyAdd} onPasskeyDelete={handlePasskeyDelete}
              onRegenerateBackupCodes={handleRegenerateBackupCodes}
              onDisableAllMfa={handleDisableAllMfa}
              sessions={sessions} sessionsLoading={sessionsLoading}
              revokingSession={revokingSession} onRevokeSession={handleRevokeSession}
            />
          )}
          {activeTab === "api-keys" && (
            <ApiKeysTab
              t={t}
              apiKeys={apiKeys} newKeyName={newKeyName} setNewKeyName={setNewKeyName}
              newKeyScope={newKeyScope} setNewKeyScope={setNewKeyScope}
              generatedKey={generatedKey} setGeneratedKey={setGeneratedKey}
              generatingKey={generatingKey}
              onGenerateKey={handleGenerateKey} onRevokeKey={handleRevokeKey}
              sshPubKey={sshPubKey} generatingSsh={generatingSsh}
              userSshKeys={userSshKeys}
              newSshKeyName={newSshKeyName} setNewSshKeyName={setNewSshKeyName}
              newSshKeyValue={newSshKeyValue} setNewSshKeyValue={setNewSshKeyValue}
              addingSshKey={addingSshKey}
              onGenerateSsh={handleGenerateSsh} onAddSshKey={handleAddSshKey}
              onDeleteSshKey={handleDeleteSshKey}
            />
          )}
          {activeTab === "team" && (
            <TeamTab
              t={t} email={email}
              teams={teams} activeTeam={activeTeam} teamMembers={teamMembers}
              newTeamName={newTeamName} setNewTeamName={setNewTeamName}
              inviteEmail={inviteEmail} setInviteEmail={setInviteEmail}
              inviteRole={inviteRole} setInviteRole={setInviteRole}
              creatingTeam={creatingTeam} inviting={inviting}
              removeTarget={removeTarget} setRemoveTarget={setRemoveTarget}
              deletingTeam={deletingTeam}
              deleteTeamConfirm={deleteTeamConfirm} setDeleteTeamConfirm={setDeleteTeamConfirm}
              onSelectTeam={selectTeam} onCreateTeam={handleCreateTeam}
              onInvite={handleInvite} onRemoveMember={handleRemoveMember}
              onDeleteTeam={handleDeleteTeam} onRoleChange={handleRoleChange}
            />
          )}
          {activeTab === "privacy" && (
            <PrivacyTab
              t={t}
              consentTypes={consentTypes} consentLabels={consentLabels}
              hasConsent={hasConsent} toggleConsent={toggleConsent}
              deleteConfirm={deleteConfirm} setDeleteConfirm={setDeleteConfirm}
              deleting={deleting} onDeleteAccount={handleDeleteAccount}
              userId={userId}
            />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

// ════════════════════════════════════════════════════════════════════
// ── Profile Tab ─────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════════════════

function ProfileTab({
  t, name, setName, email, canadaOnly, setCanadaOnly,
  notifications, setNotifications, saving, onSave,
}: {
  t: (k: string) => string;
  name: string; setName: (v: string) => void;
  email: string;
  canadaOnly: boolean; setCanadaOnly: (v: boolean) => void;
  notifications: boolean; setNotifications: (v: boolean) => void;
  saving: boolean; onSave: () => void;
}) {
  return (
    <StaggerList className="space-y-5">
      {/* Profile */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface brand-top-accent">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={User} color="text-accent-cyan" bg="bg-accent-cyan/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.profile")}</h3>
                <p className="text-xs text-text-muted">Your public identity on the platform</p>
              </div>
            </div>
          </div>
          <div className="p-5 space-y-4">
            <div className="space-y-2">
              <Label>{t("dash.settings.name")}</Label>
              <Input value={name} onChange={(e) => setName(e.target.value)} />
            </div>
            <div className="space-y-2">
              <Label>{t("dash.settings.email")}</Label>
              <Input value={email} disabled className="opacity-60" />
              <p className="text-xs text-text-muted">{t("dash.settings.email_note")}</p>
            </div>
          </div>
        </div>
      </StaggerItem>

      {/* Jurisdiction */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={Globe} color="text-accent-violet" bg="bg-accent-violet/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.jurisdiction")}</h3>
                <p className="text-xs text-text-muted">Data residency and routing preferences</p>
              </div>
            </div>
          </div>
          <div className="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t("dash.settings.canada_only")}</p>
                <p className="text-xs text-text-secondary">{t("dash.settings.canada_only_desc")}</p>
              </div>
              <Toggle enabled={canadaOnly} onToggle={() => setCanadaOnly(!canadaOnly)} />
            </div>
          </div>
        </div>
      </StaggerItem>

      {/* Notifications */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={Bell} color="text-accent-gold" bg="bg-accent-gold/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.notifications")}</h3>
                <p className="text-xs text-text-muted">Control how we reach you</p>
              </div>
            </div>
          </div>
          <div className="p-5">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium">{t("dash.settings.email_notif")}</p>
                <p className="text-xs text-text-secondary">{t("dash.settings.email_notif_desc")}</p>
              </div>
              <Toggle enabled={notifications} onToggle={() => setNotifications(!notifications)} />
            </div>
          </div>
        </div>
      </StaggerItem>

      {/* Save */}
      <StaggerItem>
        <Button onClick={onSave} disabled={saving} className="w-full sm:w-auto">
          {saving ? <Loader2 className="h-4 w-4 animate-spin" /> : <Save className="h-4 w-4" />}
          {saving ? "Saving..." : t("dash.settings.save")}
        </Button>
      </StaggerItem>
    </StaggerList>
  );
}

// ═════════════════════���══════════════════════════════════════════════
// ── Security Tab ────────────────────────────────────────────────────
// ════════════════════════════════════════════════════════��═══════════

function SecurityTab({
  t, currentPw, setCurrentPw, newPw, setNewPw, confirmPw, setConfirmPw,
  changingPw, onChangePassword, showPw, setShowPw, showNewPw, setShowNewPw,
  showConfirmPw, setShowConfirmPw,
  mfaEnabled, mfaMethods, mfaBackupRemaining, mfaLoading,
  totpSetup, totpQrDataUrl, totpCode, setTotpCode, totpVerifying, backupCodes,
  onTotpSetup, onTotpVerify, onTotpDisable,
  smsSetup, setSmsSetup, smsPhone, setSmsPhone, smsCountryCode, setSmsCountryCode,
  smsCode, setSmsCode, smsSending, smsVerifying, smsCodeSent,
  onSmsSetup, onSmsVerify, onSmsDisable,
  passkeyName, setPasskeyName, passkeyRegistering, passkeyRemoving,
  onPasskeyAdd, onPasskeyDelete,
  onRegenerateBackupCodes, onDisableAllMfa,
  sessions, sessionsLoading, revokingSession, onRevokeSession,
}: {
  t: (k: string, vars?: Record<string, string>) => string;
  currentPw: string; setCurrentPw: (v: string) => void;
  newPw: string; setNewPw: (v: string) => void;
  confirmPw: string; setConfirmPw: (v: string) => void;
  changingPw: boolean; onChangePassword: () => void;
  showPw: boolean; setShowPw: (v: boolean) => void;
  showNewPw: boolean; setShowNewPw: (v: boolean) => void;
  showConfirmPw: boolean; setShowConfirmPw: (v: boolean) => void;
  mfaEnabled: boolean; mfaMethods: api.MfaMethod[];
  mfaBackupRemaining: number; mfaLoading: boolean;
  totpSetup: { secret: string; uri: string; methodId: number } | null;
  totpQrDataUrl: string | null;
  totpCode: string; setTotpCode: (v: string) => void;
  totpVerifying: boolean; backupCodes: string[] | null;
  onTotpSetup: () => void; onTotpVerify: () => void; onTotpDisable: () => void;
  smsSetup: boolean; setSmsSetup: (v: boolean) => void;
  smsPhone: string; setSmsPhone: (v: string) => void;
  smsCountryCode: string; setSmsCountryCode: (v: string) => void;
  smsCode: string; setSmsCode: (v: string) => void;
  smsSending: boolean; smsVerifying: boolean; smsCodeSent: boolean;
  onSmsSetup: () => void; onSmsVerify: () => void; onSmsDisable: () => void;
  passkeyName: string; setPasskeyName: (v: string) => void;
  passkeyRegistering: boolean; passkeyRemoving: number | null;
  onPasskeyAdd: () => void; onPasskeyDelete: (id: number) => void;
  onRegenerateBackupCodes: () => void; onDisableAllMfa: () => void;
  sessions: api.SessionInfo[]; sessionsLoading: boolean;
  revokingSession: string | null; onRevokeSession: (p: string) => void;
}) {
  const newPasswordValidation = getPasswordValidation(newPw);
  const matchingPasswords = passwordsMatch(newPw, confirmPw);
  const passwordRequirements = [
    { key: "length", label: t("auth.pw_min"), satisfied: newPasswordValidation.hasValidLength },
    { key: "letter", label: t("auth.pw_rule_letter"), satisfied: newPasswordValidation.hasLetter },
    { key: "number", label: t("auth.pw_rule_number"), satisfied: newPasswordValidation.hasNumber },
    { key: "symbol", label: t("auth.pw_rule_symbol"), satisfied: newPasswordValidation.hasSupportedSymbol },
    { key: "match", label: t("auth.pw_match"), satisfied: matchingPasswords },
  ];

  return (
    <StaggerList className="space-y-5">
      {/* Change Password */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface brand-top-accent">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={Lock} color="text-accent-violet" bg="bg-accent-violet/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.change_pw")}</h3>
                <p className="text-xs text-text-muted">Update your account password</p>
              </div>
            </div>
          </div>
          <div className="p-5 space-y-4">
            <div className="space-y-2">
              <Label>{t("dash.settings.current_pw")}</Label>
              <div className="relative">
                <Input type={showPw ? "text" : "password"} value={currentPw} onChange={(e) => setCurrentPw(e.target.value)} />
                <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary">
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
            <div className="space-y-2">
              <Label>{t("dash.settings.new_pw")}</Label>
              <div className="relative">
                <Input
                  type={showNewPw ? "text" : "password"}
                  value={newPw}
                  onChange={(e) => setNewPw(e.target.value)}
                  placeholder={t("dash.settings.pw_min")}
                  minLength={PASSWORD_MIN_LENGTH}
                  maxLength={PASSWORD_MAX_LENGTH}
                  autoComplete="new-password"
                  className="pr-16"
                />
                {newPasswordValidation.isValid && (
                  <CheckCircle className="pointer-events-none absolute right-10 top-1/2 h-4 w-4 -translate-y-1/2 text-emerald" />
                )}
                <button type="button" onClick={() => setShowNewPw(!showNewPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary">
                  {showNewPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
            <div className="space-y-2">
              <Label>{t("dash.settings.confirm_pw")}</Label>
              <div className="relative">
                <Input
                  type={showConfirmPw ? "text" : "password"}
                  value={confirmPw}
                  onChange={(e) => setConfirmPw(e.target.value)}
                  maxLength={PASSWORD_MAX_LENGTH}
                  autoComplete="new-password"
                  className="pr-16"
                />
                {matchingPasswords && (
                  <CheckCircle className="pointer-events-none absolute right-10 top-1/2 h-4 w-4 -translate-y-1/2 text-emerald" />
                )}
                <button type="button" onClick={() => setShowConfirmPw(!showConfirmPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary">
                  {showConfirmPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>
            <PasswordRequirements items={passwordRequirements} />
            <Button onClick={onChangePassword} disabled={changingPw || !currentPw || !newPasswordValidation.isValid || !matchingPasswords}>
              {changingPw ? <><Loader2 className="h-4 w-4 animate-spin" /> Changing...</> : t("dash.settings.change_pw")}
            </Button>
          </div>
        </div>
      </StaggerItem>

      {/* Two-Factor Authentication */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <IconBadge icon={Shield} color="text-emerald" bg="bg-emerald/10" />
                <div>
                  <h3 className="text-sm font-semibold">{t("dash.settings.mfa_title")}</h3>
                  <p className="text-xs text-text-muted">{t("dash.settings.mfa_desc")}</p>
                </div>
              </div>
              {mfaEnabled && (
                <span className="flex items-center gap-1.5 rounded-full bg-emerald/10 px-2.5 py-1 text-xs font-medium text-emerald ring-1 ring-emerald/20">
                  <CheckCircle className="h-3 w-3" /> {t("dash.settings.mfa_enabled")}
                </span>
              )}
            </div>
          </div>
          <div className="p-5 space-y-4">
            {/* TOTP */}
            <div className="rounded-lg border border-border/60 bg-navy-light/30 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <KeyRound className="h-4 w-4 text-ice-blue" />
                  <div>
                    <p className="text-sm font-medium">{t("dash.settings.mfa_totp_title")}</p>
                    <p className="text-xs text-text-muted">{t("dash.settings.mfa_totp_desc")}</p>
                  </div>
                </div>
                {mfaMethods.some((m) => m.type === "totp" && m.enabled) ? (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-emerald font-medium flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" /> Active
                    </span>
                    <Button variant="outline" size="sm" onClick={onTotpDisable} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                      {t("dash.settings.mfa_disable")}
                    </Button>
                  </div>
                ) : (
                  <Button variant="outline" size="sm" onClick={onTotpSetup} disabled={mfaLoading}>
                    {mfaLoading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                    {t("dash.settings.mfa_totp_setup")}
                  </Button>
                )}
              </div>
              {totpSetup && (
                <div className="mt-3 space-y-3 rounded-lg bg-surface p-4 border border-border/40">
                  <p className="text-sm text-text-secondary">{t("dash.settings.mfa_totp_scan")}</p>
                  <div className="flex justify-center">
                    <div className="rounded-lg bg-white p-3">
                      {totpQrDataUrl ? (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img src={totpQrDataUrl} alt="TOTP QR Code" width={200} height={200} className="rounded" />
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
                      <button onClick={() => { navigator.clipboard.writeText(totpSetup.secret); toast.success("Copied"); }} className="text-text-muted hover:text-text-primary">
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
                      <Button onClick={onTotpVerify} disabled={totpVerifying || totpCode.length !== 6}>
                        {totpVerifying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                        {t("dash.settings.mfa_verify")}
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* SMS */}
            <div className="rounded-lg border border-border/60 bg-navy-light/30 p-4 space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2.5">
                  <MonitorSmartphone className="h-4 w-4 text-accent-gold" />
                  <div>
                    <p className="text-sm font-medium">{t("dash.settings.mfa_sms_title")}</p>
                    <p className="text-xs text-text-muted">{t("dash.settings.mfa_sms_desc")}</p>
                  </div>
                </div>
                {mfaMethods.some((m) => m.type === "sms" && m.enabled) ? (
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-emerald font-medium flex items-center gap-1">
                      <CheckCircle className="h-3 w-3" /> Active
                    </span>
                    <span className="text-xs text-text-muted">
                      {mfaMethods.find((m) => m.type === "sms")?.phone_number}
                    </span>
                    <Button variant="outline" size="sm" onClick={onSmsDisable} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
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
                <div className="mt-3 space-y-3 rounded-lg bg-surface p-4 border border-border/40">
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
                        <Input type="tel" value={smsPhone} onChange={(e) => setSmsPhone(e.target.value)} placeholder="4165551234" className="flex-1 font-mono" />
                        <Button onClick={onSmsSetup} disabled={smsSending}>
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
                        <Button onClick={onSmsVerify} disabled={smsVerifying || smsCode.length !== 6}>
                          {smsVerifying ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
                          {t("dash.settings.mfa_verify")}
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Passkeys */}
            <div className="rounded-lg border border-border/60 bg-navy-light/30 p-4 space-y-3">
              <div className="flex items-center gap-2.5">
                <Fingerprint className="h-4 w-4 text-accent-cyan" />
                <div>
                  <p className="text-sm font-medium">{t("dash.settings.mfa_passkey_title")}</p>
                  <p className="text-xs text-text-muted">{t("dash.settings.mfa_passkey_desc")}</p>
                </div>
              </div>

              {mfaMethods.filter((m) => m.type === "passkey" && m.enabled).map((pk) => (
                <div key={pk.id} className="flex items-center justify-between rounded-lg bg-surface border border-border/40 px-3 py-2.5">
                  <div className="flex items-center gap-2">
                    <Key className="h-3.5 w-3.5 text-emerald" />
                    <span className="text-sm">{pk.device_name || "Security Key"}</span>
                    <span className="text-xs text-text-muted">{new Date(pk.created_at * 1000).toLocaleDateString()}</span>
                  </div>
                  <Button
                    variant="outline" size="sm"
                    className="text-accent-red border-accent-red/30 hover:bg-accent-red/10"
                    disabled={passkeyRemoving === pk.id}
                    onClick={() => {
                      if (!window.confirm(t("dash.settings.mfa_passkey_remove_desc", { name: pk.device_name || "Security Key" }))) return;
                      onPasskeyDelete(pk.id);
                    }}
                  >
                    {passkeyRemoving === pk.id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                  </Button>
                </div>
              ))}

              <div className="flex gap-2">
                <Input value={passkeyName} onChange={(e) => setPasskeyName(e.target.value)} placeholder={t("dash.settings.mfa_passkey_name_placeholder")} className="flex-1" />
                <Button onClick={onPasskeyAdd} disabled={passkeyRegistering}>
                  {passkeyRegistering ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                  {passkeyRegistering ? t("dash.settings.mfa_passkey_registering") : t("dash.settings.mfa_passkey_add")}
                </Button>
              </div>
            </div>

            {/* Backup Codes */}
            {mfaEnabled && (
              <div className="rounded-lg border border-border/60 bg-navy-light/30 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium">{t("dash.settings.mfa_backup_title")}</p>
                    <p className="text-xs text-text-muted mt-0.5">
                      {t("dash.settings.mfa_backup_remaining", { count: String(mfaBackupRemaining) })}
                    </p>
                  </div>
                  <Button variant="outline" size="sm" onClick={onRegenerateBackupCodes}>
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
                    <Button variant="outline" size="sm" onClick={() => { navigator.clipboard.writeText(backupCodes.join("\n")); toast.success("Backup codes copied"); }}>
                      <Copy className="h-3.5 w-3.5" /> Copy All
                    </Button>
                  </div>
                )}
              </div>
            )}

            {/* Disable all MFA */}
            {mfaEnabled && (
              <div className="pt-3 border-t border-border/60">
                <Button variant="outline" size="sm" onClick={onDisableAllMfa} className="text-accent-red border-accent-red/30 hover:bg-accent-red/10">
                  <AlertTriangle className="h-3.5 w-3.5" /> {t("dash.settings.mfa_disable_all")}
                </Button>
              </div>
            )}
          </div>
        </div>
      </StaggerItem>

      {/* Active Sessions */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={Globe} color="text-accent-cyan" bg="bg-accent-cyan/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.sessions_title")}</h3>
                <p className="text-xs text-text-muted">{t("dash.settings.sessions_desc")}</p>
              </div>
            </div>
          </div>
          <div className="p-5 space-y-3">
            {sessionsLoading ? (
              <div className="flex items-center justify-center py-6"><Loader2 className="h-5 w-5 animate-spin text-text-muted" /></div>
            ) : sessions.length === 0 ? (
              <p className="text-sm text-text-muted">{t("dash.settings.sessions_none")}</p>
            ) : (
              sessions.map((s) => {
                const ua = s.user_agent || "";
                const browser = ua.match(/Chrome|Firefox|Safari|Edge|Opera/)?.[0] || "Unknown browser";
                const os = ua.match(/Windows|Mac OS|Linux|Android|iOS/)?.[0] || "Unknown OS";
                const lastActive = s.last_active ? new Date(s.last_active * 1000).toLocaleString() : "--";
                return (
                  <div key={s.token_prefix} className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/30 p-3 transition-colors hover:bg-surface-hover">
                    <div className="space-y-0.5 min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <p className="text-sm font-medium truncate">{browser} on {os}</p>
                        {s.is_current && (
                          <span className="shrink-0 rounded-full bg-emerald/15 px-2 py-0.5 text-[10px] font-medium text-emerald ring-1 ring-emerald/20">
                            {t("dash.settings.sessions_current")}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-text-muted">
                        {s.ip_address || "Unknown IP"} · {t("dash.settings.sessions_last_active")} {lastActive}
                      </p>
                    </div>
                    {!s.is_current && (
                      <Button variant="outline" size="sm" className="ml-3 shrink-0 text-accent-red border-accent-red/30 hover:bg-accent-red/10" onClick={() => onRevokeSession(s.token_prefix)} disabled={revokingSession === s.token_prefix}>
                        {revokingSession === s.token_prefix ? <Loader2 className="h-3 w-3 animate-spin" /> : t("dash.settings.sessions_revoke")}
                      </Button>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      </StaggerItem>
    </StaggerList>
  );
}

// ═════��══════════════════════════════════════════════════════════════
// ── API Keys Tab ────────────────────────────────────────────────────
// ═══════════════════════════════��════════════════════════════════════

function ApiKeysTab({
  t, apiKeys, newKeyName, setNewKeyName, newKeyScope, setNewKeyScope,
  generatedKey, setGeneratedKey, generatingKey, onGenerateKey, onRevokeKey,
  sshPubKey, generatingSsh, userSshKeys,
  newSshKeyName, setNewSshKeyName, newSshKeyValue, setNewSshKeyValue,
  addingSshKey, onGenerateSsh, onAddSshKey, onDeleteSshKey,
}: {
  t: (k: string) => string;
  apiKeys: ApiKeyInfo[]; newKeyName: string; setNewKeyName: (v: string) => void;
  newKeyScope: "full-access" | "read-only"; setNewKeyScope: (v: "full-access" | "read-only") => void;
  generatedKey: string | null; setGeneratedKey: (v: string | null) => void;
  generatingKey: boolean; onGenerateKey: () => void; onRevokeKey: (p: string) => void;
  sshPubKey: string; generatingSsh: boolean; userSshKeys: UserSshKey[];
  newSshKeyName: string; setNewSshKeyName: (v: string) => void;
  newSshKeyValue: string; setNewSshKeyValue: (v: string) => void;
  addingSshKey: boolean; onGenerateSsh: () => void; onAddSshKey: () => void;
  onDeleteSshKey: (id: string) => void;
}) {
  const [revealedKeys, setRevealedKeys] = useState<Set<string>>(new Set());
  const [showPlatformKey, setShowPlatformKey] = useState(false);
  const [showGenKey, setShowGenKey] = useState(true);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const toggleReveal = (preview: string) => {
    setRevealedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(preview)) next.delete(preview); else next.add(preview);
      return next;
    });
  };

  const copyWithFeedback = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success("Copied to clipboard");
    setTimeout(() => setCopiedId(null), 2000);
  };

  const maskKey = (preview: string) => `${preview}${"*".repeat(28)}`;

  return (
    <StaggerList className="space-y-5">
      {/* API Keys */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface brand-top-accent">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <IconBadge icon={Key} color="text-accent-gold" bg="bg-accent-gold/10" />
                <div>
                  <h3 className="text-sm font-semibold">{t("dash.settings.api_keys")}</h3>
                  <p className="text-xs text-text-muted">{t("dash.settings.api_keys_desc")}</p>
                </div>
              </div>
              {apiKeys.length > 0 && (
                <span className="rounded-full bg-accent-gold/10 px-2.5 py-0.5 text-[11px] font-medium text-accent-gold ring-1 ring-accent-gold/20">
                  {apiKeys.length} key{apiKeys.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>
          <div className="p-5 space-y-4">
            {/* Generate new key */}
            <div className="rounded-lg border border-dashed border-border/80 bg-navy-light/20 p-4 space-y-3">
              <p className="text-xs font-medium text-text-muted uppercase tracking-wider">Create New Key</p>
              <div className="flex flex-col sm:flex-row gap-2">
                <Input placeholder={t("dash.settings.key_name_placeholder")} value={newKeyName} onChange={(e) => setNewKeyName(e.target.value)} className="flex-1 min-w-0" />
                <div className="flex gap-2">
                  <Select value={newKeyScope} onChange={(e) => setNewKeyScope(e.target.value as "full-access" | "read-only")}>
                    <option value="full-access">{t("dash.settings.full_access")}</option>
                    <option value="read-only">{t("dash.settings.read_only")}</option>
                  </Select>
                  <Button onClick={onGenerateKey} disabled={generatingKey} size="sm">
                    {generatingKey ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                    {t("dash.settings.generate")}
                  </Button>
                </div>
              </div>
            </div>

            {/* Newly generated key — prominent reveal */}
            <AnimatePresence>
              {generatedKey && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.97, y: -4 }}
                  animate={{ opacity: 1, scale: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.97, y: -4 }}
                  className="rounded-xl border border-accent-gold/40 bg-gradient-to-b from-accent-gold/8 to-accent-gold/3 p-4 space-y-3"
                >
                  <div className="flex items-center gap-2">
                    <div className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-gold/20">
                      <AlertTriangle className="h-3.5 w-3.5 text-accent-gold" />
                    </div>
                    <span className="text-sm font-semibold text-accent-gold">Copy your key now — it won&apos;t be shown again</span>
                  </div>
                  <div className="flex items-center gap-2 rounded-lg bg-background/80 border border-border/60 p-1">
                    <code className="flex-1 px-3 py-2 font-mono text-xs break-all select-all">
                      {showGenKey ? generatedKey : generatedKey.slice(0, 8) + "*".repeat(generatedKey.length - 8)}
                    </code>
                    <div className="flex shrink-0 gap-1 pr-1">
                      <button
                        onClick={() => setShowGenKey(!showGenKey)}
                        className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-text-primary hover:bg-surface-hover transition-colors"
                        title={showGenKey ? "Hide" : "Reveal"}
                      >
                        {showGenKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                      <button
                        onClick={() => copyWithFeedback(generatedKey, "gen")}
                        className={cn(
                          "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
                          copiedId === "gen" ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
                        )}
                        title="Copy"
                      >
                        {copiedId === "gen" ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                  <button onClick={() => setGeneratedKey(null)} className="text-xs text-text-muted hover:text-text-secondary transition-colors">
                    Dismiss
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Existing keys */}
            {apiKeys.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface-hover mb-3">
                  <Key className="h-5 w-5 text-text-muted" />
                </div>
                <p className="text-sm text-text-muted">{t("dash.settings.no_keys")}</p>
                <p className="text-xs text-text-muted mt-1">Create your first API key above to get started</p>
              </div>
            ) : (
              <div className="space-y-2">
                {apiKeys.map((k) => {
                  const revealed = revealedKeys.has(k.preview);
                  return (
                    <div
                      key={k.preview}
                      className="group rounded-lg border border-border/60 bg-navy-light/30 overflow-hidden transition-all hover:border-border hover:bg-surface-hover"
                    >
                      <div className="flex items-center gap-3 p-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-gold/8 ring-1 ring-accent-gold/15">
                          <Key className="h-4 w-4 text-accent-gold" />
                        </div>
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-2">
                            <p className="text-sm font-medium truncate">{k.name}</p>
                            <span className={cn(
                              "shrink-0 rounded-full px-2 py-0.5 text-[10px] font-medium ring-1",
                              k.scope === "full-access"
                                ? "bg-accent-cyan/10 text-accent-cyan ring-accent-cyan/20"
                                : "bg-surface-hover text-text-muted ring-border",
                            )}>
                              {k.scope === "full-access" ? "Full Access" : "Read Only"}
                            </span>
                          </div>
                          <div className="flex items-center gap-1.5 mt-0.5">
                            <code className="text-xs text-text-muted font-mono">
                              {revealed ? k.preview : maskKey(k.preview.slice(0, 10))}
                            </code>
                            <button
                              onClick={() => toggleReveal(k.preview)}
                              className="text-text-muted hover:text-text-secondary transition-colors"
                              title={revealed ? "Hide" : "Reveal"}
                            >
                              {revealed ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                            </button>
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
                          <button
                            onClick={() => copyWithFeedback(k.preview, k.preview)}
                            className={cn(
                              "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
                              copiedId === k.preview ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
                            )}
                            title="Copy preview"
                          >
                            {copiedId === k.preview ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                          </button>
                          <button
                            onClick={() => onRevokeKey(k.preview)}
                            className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-accent-red hover:bg-accent-red/10 transition-colors"
                            title="Revoke key"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </StaggerItem>

      {/* SSH Keys */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <IconBadge icon={Terminal} color="text-emerald" bg="bg-emerald/10" />
                <div>
                  <h3 className="text-sm font-semibold">{t("dash.settings.ssh_keys")}</h3>
                  <p className="text-xs text-text-muted">{t("dash.settings.ssh_desc")}</p>
                </div>
              </div>
              {userSshKeys.length > 0 && (
                <span className="rounded-full bg-emerald/10 px-2.5 py-0.5 text-[11px] font-medium text-emerald ring-1 ring-emerald/20">
                  {userSshKeys.length} key{userSshKeys.length !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </div>
          <div className="p-5 space-y-6">
            {/* Your SSH Keys */}
            <div className="space-y-3">
              <p className="text-xs font-medium text-text-muted uppercase tracking-wider">Your SSH Public Keys</p>
              <p className="text-xs text-text-muted">
                Add your SSH public keys for secure access to GPU hosts. Paste the contents of your{" "}
                <code className="bg-background px-1 rounded text-accent-cyan">~/.ssh/id_ed25519.pub</code> or{" "}
                <code className="bg-background px-1 rounded text-accent-cyan">~/.ssh/id_rsa.pub</code> file.
              </p>

              <div className="rounded-lg border border-dashed border-border/80 bg-navy-light/20 p-4 space-y-3">
                <Input placeholder="Key name (e.g. Work Laptop, Home Desktop)" value={newSshKeyName} onChange={(e) => setNewSshKeyName(e.target.value)} />
                <textarea
                  className="w-full rounded-md border border-border bg-background px-3 py-2 font-mono text-xs placeholder:text-text-muted min-h-[80px] resize-y focus:outline-none focus:ring-2 focus:ring-ring transition-shadow"
                  placeholder="ssh-ed25519 AAAA... user@host"
                  value={newSshKeyValue}
                  onChange={(e) => setNewSshKeyValue(e.target.value)}
                />
                <Button variant="outline" size="sm" onClick={onAddSshKey} disabled={addingSshKey}>
                  {addingSshKey ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                  Add SSH Key
                </Button>
              </div>

              {userSshKeys.length > 0 && (
                <div className="space-y-2">
                  {userSshKeys.map((k) => (
                    <SshKeyCard
                      key={k.id}
                      keyData={k}
                      onDelete={onDeleteSshKey}
                      copiedId={copiedId}
                      onCopy={copyWithFeedback}
                    />
                  ))}
                </div>
              )}
            </div>

            {/* Platform SSH Key */}
            <div className="space-y-3 border-t border-border/60 pt-5">
              <div className="flex items-center gap-2">
                <p className="text-xs font-medium text-text-muted uppercase tracking-wider">Platform SSH Key</p>
                {sshPubKey && (
                  <span className="rounded-full bg-emerald/10 px-2 py-0.5 text-[10px] font-medium text-emerald ring-1 ring-emerald/20">
                    Generated
                  </span>
                )}
              </div>
              <p className="text-xs text-text-muted">Generate a platform keypair for infrastructure access.</p>
              {sshPubKey ? (
                <div className="rounded-lg border border-border/60 bg-navy-light/30 p-3">
                  <div className="flex items-center gap-2">
                    <div className="flex-1 min-w-0 rounded-lg bg-background/80 border border-border/40 px-3 py-2">
                      <code className="font-mono text-xs break-all block max-h-20 overflow-y-auto">
                        {showPlatformKey ? sshPubKey : sshPubKey.slice(0, 24) + " *** " + sshPubKey.slice(-16)}
                      </code>
                    </div>
                    <div className="flex flex-col gap-1 shrink-0">
                      <button
                        onClick={() => setShowPlatformKey(!showPlatformKey)}
                        className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-text-primary hover:bg-surface-hover transition-colors"
                        title={showPlatformKey ? "Hide" : "Reveal"}
                      >
                        {showPlatformKey ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                      </button>
                      <button
                        onClick={() => copyWithFeedback(sshPubKey, "platform")}
                        className={cn(
                          "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
                          copiedId === "platform" ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
                        )}
                        title="Copy"
                      >
                        {copiedId === "platform" ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                      </button>
                    </div>
                  </div>
                  <p className="text-xs text-text-muted mt-2">{t("dash.settings.ssh_note")}</p>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center py-6 text-center rounded-lg border border-dashed border-border/60 bg-navy-light/10">
                  <Terminal className="h-5 w-5 text-text-muted mb-2" />
                  <p className="text-sm text-text-muted">No platform keypair generated yet</p>
                </div>
              )}
              <Button variant="outline" size="sm" onClick={onGenerateSsh} disabled={generatingSsh}>
                {generatingSsh ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                {sshPubKey ? t("dash.settings.regen_keypair") : t("dash.settings.gen_keypair")}
              </Button>
            </div>
          </div>
        </div>
      </StaggerItem>
    </StaggerList>
  );
}

/** Individual SSH key card with reveal/copy/delete */
function SshKeyCard({
  keyData, onDelete, copiedId, onCopy,
}: {
  keyData: UserSshKey;
  onDelete: (id: string) => void;
  copiedId: string | null;
  onCopy: (text: string, id: string) => void;
}) {
  const [revealed, setRevealed] = useState(false);

  return (
    <div className="group rounded-lg border border-border/60 bg-navy-light/30 overflow-hidden transition-all hover:border-border hover:bg-surface-hover">
      <div className="flex items-center gap-3 p-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-emerald/8 ring-1 ring-emerald/15">
          <Terminal className="h-4 w-4 text-emerald" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium truncate">{keyData.name}</p>
          <div className="flex items-center gap-1.5 mt-0.5">
            <code className="text-xs text-text-muted font-mono truncate">
              {revealed ? keyData.fingerprint : keyData.fingerprint.slice(0, 12) + "***" + keyData.fingerprint.slice(-6)}
            </code>
            <button
              onClick={() => setRevealed(!revealed)}
              className="shrink-0 text-text-muted hover:text-text-secondary transition-colors"
              title={revealed ? "Hide" : "Reveal"}
            >
              {revealed ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
            </button>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1 opacity-60 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => onCopy(keyData.fingerprint, keyData.id)}
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
              copiedId === keyData.id ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
            )}
            title="Copy fingerprint"
          >
            {copiedId === keyData.id ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
          <button
            onClick={() => onDelete(keyData.id)}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-accent-red hover:bg-accent-red/10 transition-colors"
            title="Remove key"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}

// ═════════════════════════════════════���══════════════════════════════
// ── Team Tab ────────────────────────────────────────────────────────
// ══════════════════════════════════��═════════════════════════════════

function TeamTab({
  t, email, teams, activeTeam, teamMembers,
  newTeamName, setNewTeamName, inviteEmail, setInviteEmail,
  inviteRole, setInviteRole, creatingTeam, inviting,
  removeTarget, setRemoveTarget, deletingTeam,
  deleteTeamConfirm, setDeleteTeamConfirm,
  onSelectTeam, onCreateTeam, onInvite, onRemoveMember,
  onDeleteTeam, onRoleChange,
}: {
  t: (k: string, vars?: Record<string, string>) => string;
  email: string;
  teams: TeamInfo[]; activeTeam: TeamInfo | null; teamMembers: TeamMember[];
  newTeamName: string; setNewTeamName: (v: string) => void;
  inviteEmail: string; setInviteEmail: (v: string) => void;
  inviteRole: string; setInviteRole: (v: string) => void;
  creatingTeam: boolean; inviting: boolean;
  removeTarget: string | null; setRemoveTarget: (v: string | null) => void;
  deletingTeam: boolean;
  deleteTeamConfirm: boolean; setDeleteTeamConfirm: (v: boolean) => void;
  onSelectTeam: (t: TeamInfo) => void; onCreateTeam: () => void;
  onInvite: () => void; onRemoveMember: (e: string) => void;
  onDeleteTeam: () => void; onRoleChange: (e: string, r: string) => void;
}) {
  return (
    <StaggerList className="space-y-5">
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface brand-top-accent" id="team">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={Users} color="text-emerald" bg="bg-emerald/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.team")}</h3>
                <p className="text-xs text-text-muted">{t("dash.settings.team_desc")}</p>
              </div>
            </div>
          </div>
          <div className="p-5 space-y-4">
            {teams.length === 0 ? (
              <div className="space-y-3">
                <p className="text-sm text-text-muted">You don&apos;t belong to any teams yet.</p>
                <div className="flex gap-2">
                  <Input placeholder="Team name" value={newTeamName} onChange={(e) => setNewTeamName(e.target.value)} className="flex-1" />
                  <Button variant="outline" size="sm" onClick={onCreateTeam} disabled={!newTeamName.trim() || creatingTeam}>
                    {creatingTeam ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                    Create Team
                  </Button>
                </div>
              </div>
            ) : (
              <>
                {teams.length > 1 && (
                  <div className="flex gap-1.5 flex-wrap">
                    {teams.map((team) => (
                      <button
                        key={team.team_id}
                        onClick={() => onSelectTeam(team)}
                        className={cn(
                          "rounded-lg px-3 py-1.5 text-xs font-medium transition-colors",
                          activeTeam?.team_id === team.team_id
                            ? "bg-accent-cyan/10 text-accent-cyan ring-1 ring-accent-cyan/20"
                            : "bg-surface-hover text-text-muted hover:text-text-primary",
                        )}
                      >
                        {team.name}
                      </button>
                    ))}
                  </div>
                )}

                {activeTeam && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/30 p-3">
                      <div>
                        <p className="text-sm font-medium">{activeTeam.name}</p>
                        <p className="text-xs text-text-muted">
                          {activeTeam.plan} plan &middot; {teamMembers.length}/{activeTeam.max_members} members
                        </p>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-text-muted">{activeTeam.team_id}</span>
                        {activeTeam.owner_email === email && (
                          <Button variant="ghost" size="sm" onClick={() => setDeleteTeamConfirm(true)} className="text-accent-red hover:text-accent-red" title="Delete team">
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        )}
                      </div>
                    </div>

                    {/* Members */}
                    <div className="space-y-2">
                      <p className="text-xs font-medium text-text-muted uppercase tracking-wider">Members</p>
                      {teamMembers.map((m) => {
                        const isOwner = m.email === activeTeam.owner_email;
                        const iAmAdmin = teamMembers.find((x) => x.email === email)?.role === "admin";
                        return (
                          <div key={m.email} className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/30 p-3 transition-colors hover:bg-surface-hover">
                            <div className="flex items-center gap-3">
                              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-cyan/15 text-xs font-bold text-accent-cyan ring-1 ring-accent-cyan/20">
                                {m.email.charAt(0).toUpperCase()}
                              </div>
                              <div>
                                <p className="text-sm">{m.email}{isOwner && <span className="ml-1.5 text-[10px] text-accent-gold font-medium">OWNER</span>}</p>
                                {iAmAdmin && !isOwner ? (
                                  <Select value={m.role} onChange={(e) => onRoleChange(m.email, e.target.value)} className="mt-0.5 h-6 w-24 text-xs py-0">
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
                              <Button variant="ghost" size="sm" onClick={() => setRemoveTarget(m.email)} className="text-accent-red hover:text-accent-red">
                                <UserMinus className="h-3.5 w-3.5" />
                              </Button>
                            )}
                          </div>
                        );
                      })}
                    </div>

                    {/* Invite */}
                    <div className="flex gap-2">
                      <Input type="email" autoComplete="email" placeholder="Email address" value={inviteEmail} onChange={(e) => setInviteEmail(e.target.value)} className="flex-1" />
                      <Select value={inviteRole} onChange={(e) => setInviteRole(e.target.value)} className="w-28">
                        <option value="member">Member</option>
                        <option value="admin">Admin</option>
                        <option value="viewer">Viewer</option>
                      </Select>
                      <Button variant="outline" size="sm" onClick={onInvite} disabled={!inviteEmail.trim() || inviting}>
                        {inviting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <UserPlus className="h-3.5 w-3.5" />}
                        Invite
                      </Button>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </StaggerItem>

      <ConfirmDialog
        open={removeTarget !== null}
        title={t("dash.settings.remove_member_title")}
        description={t("dash.settings.remove_member_desc", { email: removeTarget || "" })}
        confirmLabel={t("dash.settings.remove_member_confirm")}
        cancelLabel={t("common.cancel")}
        variant="danger"
        onConfirm={() => removeTarget && onRemoveMember(removeTarget)}
        onCancel={() => setRemoveTarget(null)}
      />
      <ConfirmDialog
        open={deleteTeamConfirm}
        title="Delete Team"
        description={`Are you sure you want to delete "${activeTeam?.name}"? All members will be removed and this cannot be undone.`}
        confirmLabel={deletingTeam ? "Deleting..." : "Delete Team"}
        cancelLabel="Cancel"
        variant="danger"
        onConfirm={onDeleteTeam}
        onCancel={() => setDeleteTeamConfirm(false)}
      />
    </StaggerList>
  );
}

// ═════════════════════════════════��══════════════════════════════════
// ── Privacy Tab ─────────────────────────────────────────────────────
// ══════════════════════════��═════════════════════════════���═══════════

function PrivacyTab({
  t, consentTypes, consentLabels, hasConsent, toggleConsent,
  deleteConfirm, setDeleteConfirm, deleting, onDeleteAccount, userId,
}: {
  t: (k: string) => string;
  consentTypes: string[];
  consentLabels: Record<string, { label: string; desc: string }>;
  hasConsent: (type: string) => boolean;
  toggleConsent: (type: string) => void;
  deleteConfirm: string; setDeleteConfirm: (v: string) => void;
  deleting: boolean; onDeleteAccount: () => void;
  userId: string;
}) {
  return (
    <StaggerList className="space-y-5">
      {/* PIPEDA Consent */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-border bg-surface brand-top-accent">
          <div className="border-b border-border/60 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={ShieldCheck} color="text-accent-cyan" bg="bg-accent-cyan/10" />
              <div>
                <h3 className="text-sm font-semibold">{t("dash.settings.pipeda_title")}</h3>
                <p className="text-xs text-text-muted">{t("dash.settings.pipeda_desc")}</p>
              </div>
            </div>
          </div>
          <div className="p-5 space-y-3">
            {consentTypes.map((type) => {
              const info = consentLabels[type];
              const enabled = hasConsent(type);
              return (
                <div key={type} className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/30 p-3 transition-colors hover:bg-surface-hover">
                  <div>
                    <p className="text-sm font-medium">{info?.label || type}</p>
                    <p className="text-xs text-text-secondary">{info?.desc || ""}</p>
                  </div>
                  <Toggle enabled={enabled} onToggle={() => toggleConsent(type)} />
                </div>
              );
            })}
          </div>
        </div>
      </StaggerItem>

      {/* Danger Zone */}
      <StaggerItem>
        <div className="glow-card rounded-xl border border-accent-red/20 bg-surface">
          <div className="border-b border-accent-red/10 px-5 py-4">
            <div className="flex items-center gap-3">
              <IconBadge icon={Trash2} color="text-accent-red" bg="bg-accent-red/10" />
              <div>
                <h3 className="text-sm font-semibold text-accent-red">Danger Zone</h3>
                <p className="text-xs text-text-muted">Export your data or permanently delete your account</p>
              </div>
            </div>
          </div>
          <div className="p-5 space-y-4">
            <div className="flex items-center justify-between rounded-lg border border-border/60 bg-navy-light/30 p-3">
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
                  onClick={onDeleteAccount}
                  disabled={deleteConfirm !== "DELETE" || deleting}
                  className="text-accent-red border-accent-red/30 hover:bg-accent-red/10"
                >
                  {deleting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Trash2 className="h-3.5 w-3.5" />}
                  Delete
                </Button>
              </div>
            </div>
          </div>
        </div>
      </StaggerItem>
    </StaggerList>
  );
}
