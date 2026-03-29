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
import type { ApiKeyInfo, ConsentRecord, TeamInfo, TeamMember } from "@/lib/api";
import { toast } from "sonner";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

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

  // Password
  const [currentPw, setCurrentPw] = useState("");
  const [newPw, setNewPw] = useState("");
  const [confirmPw, setConfirmPw] = useState("");
  const [changingPw, setChangingPw] = useState(false);
  const [showPw, setShowPw] = useState(false);
  const [showNewPw, setShowNewPw] = useState(false);
  const [showConfirmPw, setShowConfirmPw] = useState(false);

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

    // Load consent records
    if (userId) {
      api.fetchConsent(userId)
        .then((res) => setConsents(res.consents || []))
        .catch((e) => console.error("Failed to load consent records", e));
    }

    // Load teams
    loadTeams();
  }, [userId, loadTeams]);

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

  // ── Change Password ──
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
        <CardContent className="space-y-4">
          {sshPubKey ? (
            <div>
              <Label className="mb-1.5 block text-xs text-text-secondary">{t("dash.settings.public_key")}</Label>
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
            <p className="text-sm text-text-muted">No SSH keypair generated yet</p>
          )}
          <Button variant="outline" size="sm" onClick={handleGenerateSsh} disabled={generatingSsh}>
            {generatingSsh ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
            {sshPubKey ? t("dash.settings.regen_keypair") : t("dash.settings.gen_keypair")}
          </Button>
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
      <Card>
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
