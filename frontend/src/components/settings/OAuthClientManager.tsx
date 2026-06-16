"use client";

import { useCallback, useEffect, useState } from "react";

import {
  KeyRound, Plus, Copy, CheckCircle, AlertTriangle, Trash2,
  Loader2, RotateCcw, Pencil, Power, PowerOff, Clock,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Dialog } from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { StaggerList, StaggerItem } from "@/components/ui/motion";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import type { OAuthClientInfo } from "@/lib/api";
import type { TeamContext } from "@/lib/team-context";
import { ScopeChipRow, WorkspaceScopeBadge } from "@/components/settings/credential-scope-panel";
import { OAuthSecretRevealModal } from "@/components/settings/oauth-secret-reveal-modal";
import { SettingsSection } from "@/components/settings/settings-layout";

// ── Constants ────────────────────────────────────────────────────────

/** Scope value definitions (labels/descs resolved via i18n at render time) */
const SCOPE_GROUPS_DEF = [
  {
    resourceKey: "scope_group_instances",
    scopes: [
      { value: "instances:read", labelKey: "scope_read", descKey: "scope_desc_instances_read" },
      { value: "instances:write", labelKey: "scope_write", descKey: "scope_desc_instances_write" },
    ],
  },
  {
    resourceKey: "scope_group_billing",
    scopes: [
      { value: "billing:read", labelKey: "scope_read", descKey: "scope_desc_billing_read" },
      { value: "billing:write", labelKey: "scope_write", descKey: "scope_desc_billing_write" },
    ],
  },
  {
    resourceKey: "scope_group_hosts",
    scopes: [
      { value: "hosts:read", labelKey: "scope_read", descKey: "scope_desc_hosts_read" },
      { value: "hosts:write", labelKey: "scope_write", descKey: "scope_desc_hosts_write" },
    ],
  },
  {
    resourceKey: "scope_group_identity",
    scopes: [
      { value: "profile", labelKey: "scope_profile", descKey: "scope_desc_profile" },
      { value: "email", labelKey: "scope_email", descKey: "scope_desc_email" },
      { value: "offline_access", labelKey: "scope_offline", descKey: "scope_desc_offline" },
    ],
  },
  {
    resourceKey: "scope_group_gpu",
    scopes: [
      { value: "gpu:read", labelKey: "scope_read", descKey: "scope_desc_gpu_read" },
    ],
  },
  {
    resourceKey: "scope_group_marketplace",
    scopes: [
      { value: "marketplace:read", labelKey: "scope_read", descKey: "scope_desc_marketplace_read" },
    ],
  },
  {
    resourceKey: "scope_group_events",
    scopes: [
      { value: "events:read", labelKey: "scope_read", descKey: "scope_desc_events_read" },
    ],
  },
  {
    resourceKey: "scope_group_general",
    scopes: [
      { value: "api", labelKey: "scope_full_api", descKey: "scope_desc_api" },
    ],
  },
];

const ALL_SCOPE_VALUES = SCOPE_GROUPS_DEF.flatMap((g) => g.scopes.map((s) => s.value));

// ── Helpers ──────────────────────────────────────────────────────────

function useTimeAgo() {
  const { t } = useLocale();
  return (epoch: number | null | undefined): string => {
    if (!epoch) return t("dash.settings.oauth.time_never");
    const diff = Date.now() / 1000 - epoch;
    if (diff < 60) return t("dash.settings.oauth.time_just_now");
    if (diff < 3600) return t("dash.settings.oauth.time_minutes", { n: Math.floor(diff / 60) });
    if (diff < 86400) return t("dash.settings.oauth.time_hours", { n: Math.floor(diff / 3600) });
    if (diff < 2592000) return t("dash.settings.oauth.time_days", { n: Math.floor(diff / 86400) });
    return new Date(epoch * 1000).toLocaleDateString();
  };
}

function formatDate(epoch: number | null | undefined): string {
  if (!epoch) return "\u2014";
  return new Date(epoch * 1000).toLocaleString(undefined, {
    month: "short", day: "numeric", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

// ── Scope Matrix ─────────────────────────────────────────────────────

function ScopeMatrix({
  selected,
  onChange,
  disabled,
}: {
  selected: Set<string>;
  onChange: (scopes: Set<string>) => void;
  disabled?: boolean;
}) {
  const { t } = useLocale();
  const toggle = (scope: string) => {
    const next = new Set(selected);
    if (next.has(scope)) next.delete(scope);
    else next.add(scope);
    if (scope === "api" && next.has("api")) {
      onChange(new Set(["api"]));
      return;
    }
    if (scope !== "api") next.delete("api");
    onChange(next);
  };

  const isApi = selected.has("api");

  return (
    <div className="space-y-1">
      {SCOPE_GROUPS_DEF.map((group) => (
        <div key={group.resourceKey} className="rounded-lg border border-border/60 overflow-hidden">
          <div className="bg-surface-hover/50 px-3 py-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">
              {t(`dash.settings.oauth.${group.resourceKey}`)}
            </span>
          </div>
          <div className="divide-y divide-border/40">
            {group.scopes.map((scope) => {
              const checked = isApi || selected.has(scope.value);
              const isFullApi = scope.value === "api";
              return (
                <label
                  key={scope.value}
                  className={cn(
                    "flex items-center gap-3 px-3 py-2 transition-colors cursor-pointer",
                    disabled ? "opacity-50 pointer-events-none" : "hover:bg-surface-hover/30",
                    checked && !isFullApi && "bg-accent-cyan/[0.03]",
                    checked && isFullApi && "bg-accent-gold/[0.03]",
                  )}
                >
                  <input
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggle(scope.value)}
                    disabled={disabled || (isApi && scope.value !== "api")}
                    className="h-3.5 w-3.5 rounded border-border accent-accent-cyan"
                  />
                  <div className="flex-1 min-w-0">
                    <span className={cn(
                      "text-sm font-medium",
                      isFullApi ? "text-accent-gold" : "text-text-primary",
                    )}>
                      {t(`dash.settings.oauth.${scope.labelKey}`)}
                    </span>
                    <code className="ml-2 text-[10px] text-text-muted font-mono">
                      {scope.value}
                    </code>
                  </div>
                </label>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Create Client Dialog ─────────────────────────────────────────────

function CreateClientDialog({
  open,
  onClose,
  onCreated,
  team,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (client: OAuthClientInfo, secret: string) => void;
  team: TeamContext;
}) {
  const { t } = useLocale();
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<Set<string>>(new Set(["instances:read"]));
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    if (!open) return;
    setName("");
    setScopes(new Set(["instances:read"]));
  }, [open]);

  const handleCreate = async () => {
    const trimmed = name.trim();
    if (!trimmed) { toast.error(t("dash.settings.oauth.err_name_required")); return; }
    if (scopes.size === 0) { toast.error(t("dash.settings.oauth.err_scopes_required")); return; }
    setCreating(true);
    try {
      const res = await api.createOAuthClient(trimmed, Array.from(scopes));
      const secret = res.client_secret || res.client.client_secret || "";
      onCreated({ ...res.client, status: "active" }, secret);
      onClose();
      if (!secret) {
        toast.error(t("dash.settings.oauth.secret_missing"));
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.settings.oauth.create_failed"));
    } finally {
      setCreating(false);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={onClose}
      title={t("dash.settings.oauth.create_btn")}
      description={t("dash.settings.oauth.create_hint", { grantType: "client_credentials" })}
      maxWidth="max-w-xl"
    >
      <div className="space-y-4 pt-4">
        <div className="flex flex-wrap items-center gap-2">
          <WorkspaceScopeBadge
            workspaceLabel={team.isTeamMember ? "team" : "personal"}
            teamName={team.teamName}
          />
        </div>
        <div>
          <Label className="text-xs text-text-muted mb-1.5 block">{t("dash.settings.oauth.create_name_label")}</Label>
          <Input
            placeholder={t("dash.settings.oauth.create_name_placeholder")}
            value={name}
            onChange={(e) => setName(e.target.value)}
            autoFocus
          />
        </div>
        <div>
          <Label className="text-xs text-text-muted mb-1.5 block">{t("dash.settings.oauth.create_scopes_label")}</Label>
          <ScopeMatrix selected={scopes} onChange={setScopes} />
        </div>
        <div className="flex justify-end gap-3 pt-2">
          <Button variant="outline" size="sm" onClick={onClose} disabled={creating}>
            {t("dash.settings.oauth.edit_cancel")}
          </Button>
          <Button onClick={handleCreate} disabled={creating} size="sm">
            {creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
            {t("dash.settings.oauth.create")}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

// ── Edit Client Dialog ───────────────────────────────────────────────

function EditClientDialog({
  client,
  open,
  onClose,
  onSaved,
}: {
  client: OAuthClientInfo;
  open: boolean;
  onClose: () => void;
  onSaved: (updated: Partial<OAuthClientInfo>) => void;
}) {
  const { t } = useLocale();
  const [name, setName] = useState(client.client_name);
  const [scopes, setScopes] = useState<Set<string>>(new Set(client.scopes));
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(client.client_name);
    setScopes(new Set(client.scopes));
  }, [client]);

  const handleSave = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) { toast.error(t("dash.settings.oauth.edit_name_required")); return; }
    if (scopes.size === 0) { toast.error(t("dash.settings.oauth.err_scopes_required")); return; }
    setSaving(true);
    try {
      const scopeArr = Array.from(scopes);
      await api.updateOAuthClient(client.client_id, {
        client_name: trimmedName,
        scopes: scopeArr,
      });
      onSaved({ client_name: trimmedName, scopes: scopeArr });
      toast.success(t("dash.settings.oauth.updated"));
      onClose();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.settings.oauth.update_failed"));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} title={t("dash.settings.oauth.edit_title")} description={client.client_id} maxWidth="max-w-xl">
      <div className="space-y-4 pt-4">
        <div>
          <Label className="text-xs text-text-muted mb-1.5 block">{t("dash.settings.oauth.edit_name_label")}</Label>
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <Label className="text-xs text-text-muted mb-1.5 block">{t("dash.settings.oauth.edit_scopes_label")}</Label>
          <ScopeMatrix selected={scopes} onChange={setScopes} />
        </div>
        <div className="flex justify-end gap-3 pt-2">
          <Button variant="outline" size="sm" onClick={onClose} disabled={saving}>{t("dash.settings.oauth.edit_cancel")}</Button>
          <Button size="sm" onClick={handleSave} disabled={saving}>
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle className="h-3.5 w-3.5" />}
            {t("dash.settings.oauth.edit_save")}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

// ── Rotate Secret Dialog ─────────────────────────────────────────────

function RotateSecretDialog({
  client,
  open,
  onClose,
  onRevealed,
}: {
  client: OAuthClientInfo;
  open: boolean;
  onClose: () => void;
  onRevealed: (secret: string) => void;
}) {
  const { t } = useLocale();
  const [rotating, setRotating] = useState(false);

  const handleRotate = async () => {
    setRotating(true);
    try {
      const res = await api.rotateOAuthClientSecret(client.client_id);
      if (!res.client_secret) {
        toast.error(t("dash.settings.oauth.secret_missing"));
        return;
      }
      onRevealed(res.client_secret);
      onClose();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.settings.oauth.rotate_failed"));
    } finally {
      setRotating(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} title={t("dash.settings.oauth.rotate_title")} description={client.client_name}>
      <div className="space-y-4 pt-4">
        <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 text-accent-gold mt-0.5 shrink-0" />
            <div className="text-xs text-text-secondary">
              <p className="font-medium text-accent-gold mb-1">{t("dash.settings.oauth.rotate_warning_title")}</p>
              <p>{t("dash.settings.oauth.rotate_warning_body")}</p>
            </div>
          </div>
        </div>
        <div className="rounded-lg border border-border/60 bg-surface-hover/30 p-3">
          <p className="text-xs text-text-muted mb-1">{t("dash.settings.oauth.rotate_client_label")}</p>
          <p className="text-sm font-medium">{client.client_name}</p>
          <code className="text-xs text-text-muted font-mono">{client.client_id}</code>
        </div>
        <div className="flex justify-end gap-3">
          <Button variant="outline" size="sm" onClick={onClose}>{t("dash.settings.oauth.rotate_cancel")}</Button>
          <Button
            size="sm"
            onClick={handleRotate}
            disabled={rotating}
            className="bg-accent-gold hover:bg-accent-gold/90 text-navy"
          >
            {rotating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
            {t("dash.settings.oauth.rotate_confirm")}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

// ── Client Row ───────────────────────────────────────────────────────

function ClientRow({
  client,
  team,
  readOnly,
  onEdit,
  onRotate,
  onToggleStatus,
  onDelete,
}: {
  client: OAuthClientInfo;
  team: TeamContext;
  readOnly?: boolean;
  onEdit: () => void;
  onRotate: () => void;
  onToggleStatus: () => void;
  onDelete: () => void;
}) {
  const { t } = useLocale();
  const timeAgo = useTimeAgo();
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const isDisabled = client.status === "disabled";

  const copy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success(t("dash.settings.oauth.copied"));
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div
      className={cn(
        "rounded-xl border p-4 transition-colors",
        isDisabled
          ? "border-border/40 bg-navy-light/15 opacity-80"
          : "border-border/60 bg-navy-light/25 hover:border-border",
      )}
    >
      <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="min-w-0 flex-1 space-y-3">
          <div className="flex items-start gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-accent-cyan/10 ring-1 ring-accent-cyan/20">
              <KeyRound className="h-4 w-4 text-accent-cyan" />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <p className={cn("text-sm font-semibold truncate", isDisabled && "line-through opacity-70")}>
                  {client.client_name}
                </p>
                <Badge variant={isDisabled ? "dead" : "active"}>
                  {isDisabled ? t("dash.settings.oauth.status_disabled") : t("dash.settings.oauth.status_active")}
                </Badge>
                {client.is_first_party && (
                  <span className="rounded-full bg-accent-gold/10 px-2 py-0.5 text-[10px] font-medium text-accent-gold ring-1 ring-accent-gold/20">
                    {t("dash.settings.oauth.first_party")}
                  </span>
                )}
                <WorkspaceScopeBadge
                  workspaceLabel={client.workspace_label}
                  teamName={client.workspace_label === "team" ? team.teamName : undefined}
                  compact
                />
              </div>
              <code className="mt-1 block text-xs text-text-muted font-mono truncate">
                {client.client_id}
              </code>
            </div>
          </div>
          <ScopeChipRow scopes={client.scopes || []} maxVisible={6} />
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-text-muted">
            {client.last_used != null && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                {t("dash.settings.oauth.used_ago", { time: timeAgo(client.last_used) })}
              </span>
            )}
            <span>{t("dash.settings.oauth.detail_created")}: {formatDate(client.created_at)}</span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={() => copy(client.client_id, client.client_id)}
          >
            {copiedId === client.client_id ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            {t("dash.settings.oauth.tip_copy_id")}
          </Button>
          {!client.is_first_party && !readOnly && (
            <>
              <Button variant="outline" size="sm" onClick={onEdit}>
                <Pencil className="h-3.5 w-3.5" />
                {t("dash.settings.oauth.tip_edit")}
              </Button>
              <Button variant="outline" size="sm" onClick={onRotate}>
                <RotateCcw className="h-3.5 w-3.5" />
                {t("dash.settings.oauth.tip_rotate")}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={onToggleStatus}
              >
                {isDisabled ? <Power className="h-3.5 w-3.5" /> : <PowerOff className="h-3.5 w-3.5" />}
                {isDisabled ? t("dash.settings.oauth.tip_enable") : t("dash.settings.oauth.tip_disable")}
              </Button>
              <Button variant="outline" size="sm" className="border-accent-red/30 text-accent-red hover:bg-accent-red/10" onClick={onDelete}>
                <Trash2 className="h-3.5 w-3.5" />
                {t("dash.settings.oauth.tip_delete")}
              </Button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ── Main Manager Component ───────────────────────────────────────────

interface OAuthClientManagerProps {
  clients: OAuthClientInfo[];
  onClientsChange: (clients: OAuthClientInfo[]) => void;
  team: TeamContext;
}

export function OAuthClientManager({ clients, onClientsChange, team }: OAuthClientManagerProps) {
  const { t } = useLocale();
  const [showCreate, setShowCreate] = useState(false);
  const [editingClient, setEditingClient] = useState<OAuthClientInfo | null>(null);
  const [rotatingClient, setRotatingClient] = useState<OAuthClientInfo | null>(null);
  const [deletingClient, setDeletingClient] = useState<OAuthClientInfo | null>(null);
  const [togglingClient, setTogglingClient] = useState<OAuthClientInfo | null>(null);
  const [secretReveal, setSecretReveal] = useState<{
    clientId: string;
    secret: string;
    scopes: string[];
  } | null>(null);

  const refreshClients = useCallback(async () => {
    try {
      const res = await api.fetchOAuthClients();
      onClientsChange(res.clients || []);
    } catch { /* silent */ }
  }, [onClientsChange]);

  useEffect(() => {
    const onTeamChanged = () => { void refreshClients(); };
    window.addEventListener("xcelsior-team-changed", onTeamChanged);
    return () => window.removeEventListener("xcelsior-team-changed", onTeamChanged);
  }, [refreshClients]);

  const writeBlocked = team.isTeamMember && !team.canWriteInstances;

  const openSecretReveal = (clientId: string, secret: string, scopes: string[]) => {
    setSecretReveal({ clientId, secret, scopes });
  };

  const handleCreated = (client: OAuthClientInfo, secret: string) => {
    if (secret) {
      openSecretReveal(client.client_id, secret, client.scopes || []);
    } else {
      toast.error(t("dash.settings.oauth.secret_missing"));
      void refreshClients();
    }
  };

  const handleSecretRevealClose = () => {
    setSecretReveal(null);
    void refreshClients();
  };

  const handleEditSaved = (clientId: string, updates: Partial<OAuthClientInfo>) => {
    onClientsChange(clients.map((c) => c.client_id === clientId ? { ...c, ...updates } : c));
  };

  const handleToggleStatus = async () => {
    if (!togglingClient) return;
    const newStatus = togglingClient.status === "disabled" ? "active" : "disabled";
    try {
      await api.updateOAuthClient(togglingClient.client_id, { status: newStatus });
      onClientsChange(
        clients.map((c) => c.client_id === togglingClient.client_id ? { ...c, status: newStatus } : c),
      );
      toast.success(newStatus === "active" ? t("dash.settings.oauth.enabled") : t("dash.settings.oauth.disabled"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.settings.oauth.status_failed"));
    } finally {
      setTogglingClient(null);
    }
  };

  const handleDelete = async () => {
    if (!deletingClient) return;
    try {
      await api.deleteOAuthClient(deletingClient.client_id);
      onClientsChange(clients.filter((c) => c.client_id !== deletingClient.client_id));
      toast.success(t("dash.settings.oauth.deleted"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t("dash.settings.oauth.delete_failed"));
    } finally {
      setDeletingClient(null);
    }
  };

  const activeClients = clients.filter((c) => c.status !== "disabled");
  const disabledClients = clients.filter((c) => c.status === "disabled");

  return (
    <>
      <SettingsSection
        icon={KeyRound}
        title={t("dash.settings.oauth.title")}
        description={t("dash.settings.oauth.subtitle")}
        accent="cyan"
        highlight
        badge={
          clients.length > 0 ? (
            <div className="flex flex-wrap items-center gap-1.5">
              {disabledClients.length > 0 && (
                <span className="rounded-full bg-accent-red/10 px-2 py-0.5 text-[10px] font-medium text-accent-red ring-1 ring-accent-red/20">
                  {t("dash.settings.oauth.disabled_count", { count: disabledClients.length })}
                </span>
              )}
              <span className="rounded-full bg-accent-cyan/10 px-2 py-0.5 text-[10px] font-medium text-accent-cyan ring-1 ring-accent-cyan/20">
                {t("dash.settings.oauth.active_count", { count: activeClients.length })}
              </span>
            </div>
          ) : undefined
        }
      >
        <div className="space-y-5">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-2">
              <WorkspaceScopeBadge
                workspaceLabel={team.isTeamMember ? "team" : "personal"}
                teamName={team.teamName}
              />
              <p className="text-xs text-text-muted">{t("dash.settings.oauth.info", { grantType: "client_credentials" })}</p>
            </div>
            <Button
              size="sm"
              onClick={() => setShowCreate(true)}
              disabled={writeBlocked}
              className="shrink-0"
            >
              <Plus className="h-3.5 w-3.5" />
              {t("dash.settings.oauth.create_btn")}
            </Button>
          </div>

          {writeBlocked && (
            <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/10 px-3 py-2 text-xs text-accent-gold">
              {t("dash.settings.credentials.viewer_create_blocked")}
            </div>
          )}

          {clients.length === 0 ? (
            <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border/70 py-12 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface-hover mb-3">
                <KeyRound className="h-5 w-5 text-text-muted" />
              </div>
              <p className="text-sm font-medium text-text-secondary">{t("dash.settings.oauth.no_clients")}</p>
              <p className="text-xs text-text-muted mt-1 mb-4">{t("dash.settings.oauth.no_clients_hint")}</p>
              {!writeBlocked && (
                <Button size="sm" onClick={() => setShowCreate(true)}>
                  <Plus className="h-3.5 w-3.5" />
                  {t("dash.settings.oauth.create_btn")}
                </Button>
              )}
            </div>
          ) : (
            <StaggerList className="space-y-3">
              {clients.map((client) => (
                <StaggerItem key={client.client_id}>
                  <ClientRow
                    client={client}
                    team={team}
                    readOnly={writeBlocked}
                    onEdit={() => !writeBlocked && setEditingClient(client)}
                    onRotate={() => !writeBlocked && setRotatingClient(client)}
                    onToggleStatus={() => !writeBlocked && setTogglingClient(client)}
                    onDelete={() => !writeBlocked && setDeletingClient(client)}
                  />
                </StaggerItem>
              ))}
            </StaggerList>
          )}
        </div>
      </SettingsSection>

      <CreateClientDialog
        open={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={handleCreated}
        team={team}
      />

      {/* Edit dialog */}
      {editingClient && (
        <EditClientDialog
          client={editingClient}
          open={!!editingClient}
          onClose={() => setEditingClient(null)}
          onSaved={(updates) => handleEditSaved(editingClient.client_id, updates)}
        />
      )}

      {/* Rotate secret dialog */}
      {rotatingClient && (
        <RotateSecretDialog
          client={rotatingClient}
          open={!!rotatingClient}
          onClose={() => setRotatingClient(null)}
          onRevealed={(secret) => {
            openSecretReveal(rotatingClient.client_id, secret, rotatingClient.scopes || []);
          }}
        />
      )}

      <OAuthSecretRevealModal
        open={!!secretReveal}
        clientId={secretReveal?.clientId ?? ""}
        clientSecret={secretReveal?.secret ?? ""}
        scopes={secretReveal?.scopes ?? []}
        onClose={handleSecretRevealClose}
      />

      {/* Toggle status confirm */}
      <ConfirmDialog
        open={!!togglingClient}
        title={togglingClient?.status === "disabled" ? t("dash.settings.oauth.enable_title") : t("dash.settings.oauth.disable_title")}
        description={
          togglingClient?.status === "disabled"
            ? t("dash.settings.oauth.enable_desc", { name: togglingClient?.client_name ?? "" })
            : t("dash.settings.oauth.disable_desc", { name: togglingClient?.client_name ?? "" })
        }
        confirmLabel={togglingClient?.status === "disabled" ? t("dash.settings.oauth.enable_confirm") : t("dash.settings.oauth.disable_confirm")}
        variant={togglingClient?.status === "disabled" ? "default" : "danger"}
        onConfirm={handleToggleStatus}
        onCancel={() => setTogglingClient(null)}
      />

      {/* Delete confirm */}
      <ConfirmDialog
        open={!!deletingClient}
        title={t("dash.settings.oauth.delete_title")}
        description={t("dash.settings.oauth.delete_desc", { name: deletingClient?.client_name ?? "" })}
        confirmLabel={t("dash.settings.oauth.delete_confirm")}
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setDeletingClient(null)}
      />
    </>
  );
}
