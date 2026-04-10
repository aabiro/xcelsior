"use client";

import { useCallback, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  KeyRound, Plus, Copy, CheckCircle, AlertTriangle, Trash2,
  Loader2, RotateCcw, Pencil, Power, PowerOff, Shield, Clock,
  ChevronDown, ChevronRight, Eye, EyeOff, X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input, Label } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Dialog } from "@/components/ui/dialog";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { StaggerList, StaggerItem } from "@/components/ui/motion";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import * as api from "@/lib/api";
import type { OAuthClientInfo } from "@/lib/api";

// ── Constants ────────────────────────────────────────────────────────

/** All scopes supported by the platform, grouped by resource */
const SCOPE_GROUPS = [
  {
    resource: "Instances",
    scopes: [
      { value: "instances:read", label: "Read", desc: "List and inspect instances" },
      { value: "instances:write", label: "Write", desc: "Create, stop, restart, delete instances" },
    ],
  },
  {
    resource: "Billing",
    scopes: [
      { value: "billing:read", label: "Read", desc: "View wallet, invoices, usage" },
      { value: "billing:write", label: "Write", desc: "Deposit, withdraw, manage payment methods" },
    ],
  },
  {
    resource: "Hosts",
    scopes: [
      { value: "hosts:read", label: "Read", desc: "List and inspect hosts" },
      { value: "hosts:write", label: "Write", desc: "Register, update, remove hosts" },
    ],
  },
  {
    resource: "Identity",
    scopes: [
      { value: "profile", label: "Profile", desc: "Read user profile info" },
      { value: "email", label: "Email", desc: "Read user email" },
      { value: "offline_access", label: "Offline", desc: "Refresh tokens for long-lived access" },
    ],
  },
  {
    resource: "General",
    scopes: [
      { value: "api", label: "Full API", desc: "Unrestricted API access" },
    ],
  },
];

const ALL_SCOPE_VALUES = SCOPE_GROUPS.flatMap((g) => g.scopes.map((s) => s.value));

// ── Helpers ──────────────────────────────────────────────────────────

function timeAgo(epoch: number | null | undefined): string {
  if (!epoch) return "Never";
  const diff = Date.now() / 1000 - epoch;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  if (diff < 2592000) return `${Math.floor(diff / 86400)}d ago`;
  return new Date(epoch * 1000).toLocaleDateString();
}

function formatDate(epoch: number | null | undefined): string {
  if (!epoch) return "—";
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
  const toggle = (scope: string) => {
    const next = new Set(selected);
    if (next.has(scope)) next.delete(scope);
    else next.add(scope);
    // If "api" is selected, it implies everything
    if (scope === "api" && next.has("api")) {
      onChange(new Set(["api"]));
      return;
    }
    // If selecting a specific scope, remove "api"
    if (scope !== "api") next.delete("api");
    onChange(next);
  };

  const isApi = selected.has("api");

  return (
    <div className="space-y-1">
      {SCOPE_GROUPS.map((group) => (
        <div key={group.resource} className="rounded-lg border border-border/60 overflow-hidden">
          <div className="bg-surface-hover/50 px-3 py-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-text-muted">
              {group.resource}
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
                      {scope.label}
                    </span>
                    <span className="ml-2 text-xs text-text-muted">{scope.desc}</span>
                  </div>
                  <code className="text-[10px] text-text-muted font-mono shrink-0 hidden sm:block">
                    {scope.value}
                  </code>
                </label>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── Create Client Form ───────────────────────────────────────────────

function CreateClientForm({
  onCreated,
}: {
  onCreated: (client: OAuthClientInfo, secret: string) => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [name, setName] = useState("");
  const [scopes, setScopes] = useState<Set<string>>(new Set(["instances:read"]));
  const [creating, setCreating] = useState(false);

  const handleCreate = async () => {
    const trimmed = name.trim();
    if (!trimmed) { toast.error("Enter a client name"); return; }
    if (scopes.size === 0) { toast.error("Select at least one scope"); return; }
    setCreating(true);
    try {
      const res = await api.createOAuthClient(trimmed, Array.from(scopes));
      onCreated(
        { ...res.client, status: "active" },
        res.client.client_secret || "",
      );
      setName("");
      setScopes(new Set(["instances:read"]));
      setExpanded(false);
      toast.success("OAuth client created");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to create client");
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="rounded-lg border border-dashed border-border/80 bg-navy-light/20 overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-4 py-3 text-left hover:bg-surface-hover/30 transition-colors"
      >
        {expanded
          ? <ChevronDown className="h-3.5 w-3.5 text-text-muted" />
          : <ChevronRight className="h-3.5 w-3.5 text-text-muted" />
        }
        <Plus className="h-3.5 w-3.5 text-accent-cyan" />
        <span className="text-xs font-medium text-text-secondary">Create Machine Client</span>
      </button>
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="px-4 pb-4 space-y-3 border-t border-border/40">
              <div className="pt-3">
                <Label className="text-xs text-text-muted mb-1.5 block">Client Name</Label>
                <Input
                  placeholder="e.g. CI deploy runner, monitoring service"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
              <div>
                <Label className="text-xs text-text-muted mb-1.5 block">Scopes</Label>
                <ScopeMatrix selected={scopes} onChange={setScopes} />
              </div>
              <div className="flex items-center justify-between gap-3 pt-1">
                <p className="text-xs text-text-muted">
                  Creates a confidential <code className="text-[10px]">client_credentials</code> client.
                </p>
                <Button onClick={handleCreate} disabled={creating} size="sm">
                  {creating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Plus className="h-3.5 w-3.5" />}
                  Create
                </Button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Secret Reveal Banner ─────────────────────────────────────────────

function SecretBanner({
  clientId,
  clientSecret,
  scopes,
  onDismiss,
}: {
  clientId: string;
  clientSecret: string;
  scopes: string[];
  onDismiss: () => void;
}) {
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const copy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success("Copied");
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.97, y: -4 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.97, y: -4 }}
      className="rounded-xl border border-accent-cyan/40 bg-gradient-to-b from-accent-cyan/8 to-accent-cyan/3 p-4 space-y-3"
    >
      <div className="flex items-center gap-2">
        <div className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-cyan/20">
          <AlertTriangle className="h-3.5 w-3.5 text-accent-cyan" />
        </div>
        <span className="text-sm font-semibold text-accent-cyan">
          Copy this secret now — it will not be shown again
        </span>
      </div>
      <div className="rounded-lg border border-border/60 bg-background/80 p-3 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Client ID</span>
          <button
            onClick={() => copy(clientId, "new-id")}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
              copiedId === "new-id" ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
            )}
          >
            {copiedId === "new-id" ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
        <code className="block break-all rounded-md bg-surface/70 px-3 py-2 font-mono text-xs">{clientId}</code>
        <div className="flex items-center justify-between pt-1">
          <span className="text-[11px] font-medium uppercase tracking-wider text-text-muted">Client Secret</span>
          <button
            onClick={() => copy(clientSecret, "new-secret")}
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
              copiedId === "new-secret" ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
            )}
          >
            {copiedId === "new-secret" ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
        <code className="block break-all rounded-md bg-surface/70 px-3 py-2 font-mono text-xs">{clientSecret}</code>
      </div>
      <p className="text-xs text-text-muted">Scopes: {scopes.join(", ")}</p>
      <button onClick={onDismiss} className="text-xs text-text-muted hover:text-text-secondary transition-colors">
        Dismiss
      </button>
    </motion.div>
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
  const [name, setName] = useState(client.client_name);
  const [scopes, setScopes] = useState<Set<string>>(new Set(client.scopes));
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    setName(client.client_name);
    setScopes(new Set(client.scopes));
  }, [client]);

  const handleSave = async () => {
    const trimmedName = name.trim();
    if (!trimmedName) { toast.error("Name is required"); return; }
    if (scopes.size === 0) { toast.error("Select at least one scope"); return; }
    setSaving(true);
    try {
      const scopeArr = Array.from(scopes);
      await api.updateOAuthClient(client.client_id, {
        client_name: trimmedName,
        scopes: scopeArr,
      });
      onSaved({ client_name: trimmedName, scopes: scopeArr });
      toast.success("Client updated");
      onClose();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Update failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Dialog open={open} onClose={onClose} title="Edit OAuth Client" description={client.client_id} maxWidth="max-w-xl">
      <div className="space-y-4 pt-4">
        <div>
          <Label className="text-xs text-text-muted mb-1.5 block">Client Name</Label>
          <Input value={name} onChange={(e) => setName(e.target.value)} />
        </div>
        <div>
          <Label className="text-xs text-text-muted mb-1.5 block">Scopes</Label>
          <ScopeMatrix selected={scopes} onChange={setScopes} />
        </div>
        <div className="flex justify-end gap-3 pt-2">
          <Button variant="outline" size="sm" onClick={onClose} disabled={saving}>Cancel</Button>
          <Button size="sm" onClick={handleSave} disabled={saving}>
            {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <CheckCircle className="h-3.5 w-3.5" />}
            Save Changes
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
}: {
  client: OAuthClientInfo;
  open: boolean;
  onClose: () => void;
}) {
  const [rotating, setRotating] = useState(false);
  const [newSecret, setNewSecret] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const handleRotate = async () => {
    setRotating(true);
    try {
      const res = await api.rotateOAuthClientSecret(client.client_id);
      setNewSecret(res.client_secret);
      toast.success("Secret rotated");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Rotation failed");
    } finally {
      setRotating(false);
    }
  };

  const copy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success("Copied");
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleClose = () => {
    setNewSecret(null);
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} title="Rotate Client Secret" description={client.client_name}>
      <div className="space-y-4 pt-4">
        {!newSecret ? (
          <>
            <div className="rounded-lg border border-accent-gold/30 bg-accent-gold/5 p-3">
              <div className="flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 text-accent-gold mt-0.5 shrink-0" />
                <div className="text-xs text-text-secondary">
                  <p className="font-medium text-accent-gold mb-1">This will invalidate the current secret</p>
                  <p>Any service using the old secret will stop working immediately. Make sure you can update all consumers before rotating.</p>
                </div>
              </div>
            </div>
            <div className="rounded-lg border border-border/60 bg-surface-hover/30 p-3">
              <p className="text-xs text-text-muted mb-1">Client</p>
              <p className="text-sm font-medium">{client.client_name}</p>
              <code className="text-xs text-text-muted font-mono">{client.client_id}</code>
            </div>
            <div className="flex justify-end gap-3">
              <Button variant="outline" size="sm" onClick={handleClose}>Cancel</Button>
              <Button
                size="sm"
                onClick={handleRotate}
                disabled={rotating}
                className="bg-accent-gold hover:bg-accent-gold/90 text-navy"
              >
                {rotating ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RotateCcw className="h-3.5 w-3.5" />}
                Rotate Secret
              </Button>
            </div>
          </>
        ) : (
          <>
            <div className="rounded-xl border border-accent-cyan/40 bg-gradient-to-b from-accent-cyan/8 to-accent-cyan/3 p-4 space-y-3">
              <div className="flex items-center gap-2">
                <div className="flex h-6 w-6 items-center justify-center rounded-full bg-accent-cyan/20">
                  <AlertTriangle className="h-3.5 w-3.5 text-accent-cyan" />
                </div>
                <span className="text-sm font-semibold text-accent-cyan">
                  Copy the new secret now
                </span>
              </div>
              <div className="rounded-lg border border-border/60 bg-background/80 p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[11px] font-medium uppercase tracking-wider text-text-muted">New Secret</span>
                  <button
                    onClick={() => copy(newSecret, "rotated")}
                    className={cn(
                      "flex h-7 w-7 items-center justify-center rounded-lg transition-colors",
                      copiedId === "rotated" ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
                    )}
                  >
                    {copiedId === "rotated" ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                  </button>
                </div>
                <code className="block break-all rounded-md bg-surface/70 px-3 py-2 font-mono text-xs">{newSecret}</code>
              </div>
            </div>
            <div className="flex justify-end">
              <Button size="sm" onClick={handleClose}>Done</Button>
            </div>
          </>
        )}
      </div>
    </Dialog>
  );
}

// ── Client Row ───────────────────────────────────────────────────────

function ClientRow({
  client,
  onEdit,
  onRotate,
  onToggleStatus,
  onDelete,
}: {
  client: OAuthClientInfo;
  onEdit: () => void;
  onRotate: () => void;
  onToggleStatus: () => void;
  onDelete: () => void;
}) {
  const [expanded, setExpanded] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const isDisabled = client.status === "disabled";

  const copy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    toast.success("Copied");
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div
      className={cn(
        "group rounded-lg border overflow-hidden transition-all",
        isDisabled
          ? "border-border/40 bg-navy-light/15 opacity-75"
          : "border-border/60 bg-navy-light/30 hover:border-border hover:bg-surface-hover",
      )}
    >
      {/* Main row */}
      <div className="flex items-center gap-3 p-3">
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-accent-cyan/8 ring-1 ring-accent-cyan/15 transition-colors hover:bg-accent-cyan/15"
        >
          {expanded
            ? <ChevronDown className="h-4 w-4 text-accent-cyan" />
            : <KeyRound className="h-4 w-4 text-accent-cyan" />
          }
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <p className={cn("text-sm font-medium truncate", isDisabled && "line-through opacity-70")}>
              {client.client_name}
            </p>
            <Badge variant={isDisabled ? "dead" : "active"}>
              {isDisabled ? "Disabled" : "Active"}
            </Badge>
            <span className="shrink-0 rounded-full bg-accent-cyan/10 px-2 py-0.5 text-[10px] font-medium text-accent-cyan ring-1 ring-accent-cyan/20">
              {client.client_type}
            </span>
            {client.is_first_party && (
              <span className="shrink-0 rounded-full bg-accent-gold/10 px-2 py-0.5 text-[10px] font-medium text-accent-gold ring-1 ring-accent-gold/20">
                1st party
              </span>
            )}
          </div>
          <code className="mt-0.5 block text-xs text-text-muted font-mono truncate">
            {client.client_id}
          </code>
          <div className="mt-1 flex items-center gap-3 text-[11px] text-text-muted flex-wrap">
            <span className="flex items-center gap-1">
              <Shield className="h-3 w-3" />
              {(client.scopes || []).join(", ")}
            </span>
            {client.last_used != null && (
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                Used {timeAgo(client.last_used)}
              </span>
            )}
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex shrink-0 items-center gap-0.5 opacity-60 group-hover:opacity-100 transition-opacity">
          <button
            onClick={() => copy(client.client_id, client.client_id)}
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
              copiedId === client.client_id ? "text-emerald bg-emerald/10" : "text-text-muted hover:text-text-primary hover:bg-surface-hover",
            )}
            title="Copy client ID"
          >
            {copiedId === client.client_id ? <CheckCircle className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
          {!client.is_first_party && (
            <>
              <button
                onClick={onEdit}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-accent-cyan hover:bg-accent-cyan/10 transition-colors"
                title="Edit client"
              >
                <Pencil className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={onRotate}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-accent-gold hover:bg-accent-gold/10 transition-colors"
                title="Rotate secret"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
              <button
                onClick={onToggleStatus}
                className={cn(
                  "flex h-8 w-8 items-center justify-center rounded-lg transition-colors",
                  isDisabled
                    ? "text-text-muted hover:text-emerald hover:bg-emerald/10"
                    : "text-text-muted hover:text-accent-gold hover:bg-accent-gold/10",
                )}
                title={isDisabled ? "Enable client" : "Disable client"}
              >
                {isDisabled ? <Power className="h-3.5 w-3.5" /> : <PowerOff className="h-3.5 w-3.5" />}
              </button>
              <button
                onClick={onDelete}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-text-muted hover:text-accent-red hover:bg-accent-red/10 transition-colors"
                title="Delete client"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Expanded detail panel */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="border-t border-border/40 px-3 py-3 space-y-3">
              {/* Timestamps */}
              <div className="grid grid-cols-3 gap-3">
                <div className="rounded-lg bg-surface-hover/40 p-2.5">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-0.5">Created</p>
                  <p className="text-xs text-text-secondary">{formatDate(client.created_at)}</p>
                </div>
                <div className="rounded-lg bg-surface-hover/40 p-2.5">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-0.5">Updated</p>
                  <p className="text-xs text-text-secondary">{formatDate(client.updated_at)}</p>
                </div>
                <div className="rounded-lg bg-surface-hover/40 p-2.5">
                  <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-0.5">Last Used</p>
                  <p className="text-xs text-text-secondary">{client.last_used ? formatDate(client.last_used) : "Never"}</p>
                </div>
              </div>

              {/* Grant types */}
              <div>
                <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-1.5">Grant Types</p>
                <div className="flex flex-wrap gap-1.5">
                  {(client.grant_types || []).map((g) => (
                    <code key={g} className="rounded-md bg-surface-hover px-2 py-0.5 text-[11px] font-mono text-text-secondary">
                      {g}
                    </code>
                  ))}
                </div>
              </div>

              {/* Scopes as chips */}
              <div>
                <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-1.5">Scopes</p>
                <div className="flex flex-wrap gap-1.5">
                  {(client.scopes || []).map((s) => (
                    <span
                      key={s}
                      className={cn(
                        "rounded-md px-2 py-0.5 text-[11px] font-mono",
                        s === "api"
                          ? "bg-accent-gold/10 text-accent-gold ring-1 ring-accent-gold/20"
                          : s.includes(":write")
                            ? "bg-accent-red/10 text-accent-red ring-1 ring-accent-red/20"
                            : "bg-accent-cyan/10 text-accent-cyan ring-1 ring-accent-cyan/20",
                      )}
                    >
                      {s}
                    </span>
                  ))}
                </div>
              </div>

              {/* Redirect URIs */}
              {client.redirect_uris && client.redirect_uris.length > 0 && (
                <div>
                  <p className="text-[10px] font-medium uppercase tracking-wider text-text-muted mb-1.5">Redirect URIs</p>
                  <div className="space-y-1">
                    {client.redirect_uris.map((uri) => (
                      <code key={uri} className="block rounded-md bg-surface-hover px-2 py-1 text-[11px] font-mono text-text-secondary break-all">
                        {uri}
                      </code>
                    ))}
                  </div>
                </div>
              )}

              {client.created_by_email && (
                <p className="text-[10px] text-text-muted">
                  Created by {client.created_by_email}
                </p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Main Manager Component ───────────────────────────────────────────

interface OAuthClientManagerProps {
  clients: OAuthClientInfo[];
  onClientsChange: (clients: OAuthClientInfo[]) => void;
}

export function OAuthClientManager({ clients, onClientsChange }: OAuthClientManagerProps) {
  // Modal state
  const [editingClient, setEditingClient] = useState<OAuthClientInfo | null>(null);
  const [rotatingClient, setRotatingClient] = useState<OAuthClientInfo | null>(null);
  const [deletingClient, setDeletingClient] = useState<OAuthClientInfo | null>(null);
  const [togglingClient, setTogglingClient] = useState<OAuthClientInfo | null>(null);
  const [newSecret, setNewSecret] = useState<{ clientId: string; secret: string; scopes: string[] } | null>(null);

  const refreshClients = useCallback(async () => {
    try {
      const res = await api.fetchOAuthClients();
      onClientsChange(res.clients || []);
    } catch { /* silent */ }
  }, [onClientsChange]);

  const handleCreated = (client: OAuthClientInfo, secret: string) => {
    if (secret) {
      setNewSecret({ clientId: client.client_id, secret, scopes: client.scopes });
    }
    refreshClients();
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
      toast.success(newStatus === "active" ? "Client enabled" : "Client disabled");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to update status");
    } finally {
      setTogglingClient(null);
    }
  };

  const handleDelete = async () => {
    if (!deletingClient) return;
    try {
      await api.deleteOAuthClient(deletingClient.client_id);
      onClientsChange(clients.filter((c) => c.client_id !== deletingClient.client_id));
      toast.success("Client deleted");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Failed to delete client");
    } finally {
      setDeletingClient(null);
    }
  };

  const activeClients = clients.filter((c) => c.status !== "disabled");
  const disabledClients = clients.filter((c) => c.status === "disabled");

  return (
    <>
      <div className="glow-card rounded-xl border border-border bg-surface brand-top-accent">
        {/* Header */}
        <div className="border-b border-border/60 px-5 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-cyan/10">
                <KeyRound className="h-4 w-4 text-accent-cyan" />
              </div>
              <div>
                <h3 className="text-sm font-semibold">OAuth Clients</h3>
                <p className="text-xs text-text-muted">Machine-to-machine credentials for service integrations</p>
              </div>
            </div>
            {clients.length > 0 && (
              <div className="flex items-center gap-2">
                {disabledClients.length > 0 && (
                  <span className="rounded-full bg-accent-red/10 px-2.5 py-0.5 text-[11px] font-medium text-accent-red ring-1 ring-accent-red/20">
                    {disabledClients.length} disabled
                  </span>
                )}
                <span className="rounded-full bg-accent-cyan/10 px-2.5 py-0.5 text-[11px] font-medium text-accent-cyan ring-1 ring-accent-cyan/20">
                  {activeClients.length} active
                </span>
              </div>
            )}
          </div>
        </div>

        <div className="p-5 space-y-4">
          {/* Info callout */}
          <div className="rounded-lg border border-accent-cyan/20 bg-accent-cyan/5 p-3 text-xs text-text-secondary">
            OAuth 2.0 <code className="text-[10px]">client_credentials</code> flow for automated services.
            Each client gets a unique ID/secret pair. Rotate secrets regularly; disable clients you're not using.
          </div>

          {/* Create form */}
          <CreateClientForm onCreated={handleCreated} />

          {/* Secret reveal */}
          <AnimatePresence>
            {newSecret && (
              <SecretBanner
                clientId={newSecret.clientId}
                clientSecret={newSecret.secret}
                scopes={newSecret.scopes}
                onDismiss={() => setNewSecret(null)}
              />
            )}
          </AnimatePresence>

          {/* Client list */}
          {clients.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-surface-hover mb-3">
                <KeyRound className="h-5 w-5 text-text-muted" />
              </div>
              <p className="text-sm text-text-muted">No OAuth clients yet</p>
              <p className="text-xs text-text-muted mt-1">Create your first machine client above.</p>
            </div>
          ) : (
            <div className="space-y-2">
              {clients.map((client) => (
                <ClientRow
                  key={client.client_id}
                  client={client}
                  onEdit={() => setEditingClient(client)}
                  onRotate={() => setRotatingClient(client)}
                  onToggleStatus={() => setTogglingClient(client)}
                  onDelete={() => setDeletingClient(client)}
                />
              ))}
            </div>
          )}
        </div>
      </div>

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
        />
      )}

      {/* Toggle status confirm */}
      <ConfirmDialog
        open={!!togglingClient}
        title={togglingClient?.status === "disabled" ? "Enable Client" : "Disable Client"}
        description={
          togglingClient?.status === "disabled"
            ? `Re-enable "${togglingClient?.client_name}"? It will be able to authenticate again immediately.`
            : `Disable "${togglingClient?.client_name}"? All active tokens will continue to work until they expire, but no new tokens can be issued.`
        }
        confirmLabel={togglingClient?.status === "disabled" ? "Enable" : "Disable"}
        variant={togglingClient?.status === "disabled" ? "default" : "danger"}
        onConfirm={handleToggleStatus}
        onCancel={() => setTogglingClient(null)}
      />

      {/* Delete confirm */}
      <ConfirmDialog
        open={!!deletingClient}
        title="Delete OAuth Client"
        description={`Permanently delete "${deletingClient?.client_name}"? This revokes all tokens and cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        onConfirm={handleDelete}
        onCancel={() => setDeletingClient(null)}
      />
    </>
  );
}
