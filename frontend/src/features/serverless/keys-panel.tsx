"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Key, Plus, Trash2, Loader2, AlertTriangle } from "lucide-react";
import type { ServerlessApiKey } from "@/lib/api";
import * as api from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { toast } from "sonner";
import { CopyableText } from "./copyable-text";
import { ServerlessEmptyState } from "./serverless-ui";

interface KeysPanelProps {
  endpointId: string;
  keys: ServerlessApiKey[];
  canWrite: boolean;
  onRefresh: () => void;
}

export function KeysPanel({ endpointId, keys, canWrite, onRefresh }: KeysPanelProps) {
  const { t } = useLocale();
  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState("default");
  const [revealedKey, setRevealedKey] = useState<string | null>(null);
  const [revoking, setRevoking] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    setCreating(true);
    try {
      const res = await api.createServerlessKey(endpointId, { name: newName.trim() || "default" });
      setRevealedKey(res.key.api_key);
      toast.success(t("dash.serverless.key_created"));
      onRefresh();
    } catch (e: unknown) {
      toast.error(e instanceof Error ? e.message : t("dash.serverless.key_failed"));
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (keyId: string) => {
    if (!canWrite) return toast.error(t("dash.serverless.viewer_blocked"));
    setRevoking(keyId);
    try {
      await api.revokeServerlessKey(endpointId, keyId);
      toast.success(t("dash.serverless.key_revoked"));
      onRefresh();
    } catch {
      toast.error(t("dash.serverless.key_revoke_failed"));
    } finally {
      setRevoking(null);
    }
  };

  return (
    <div className="space-y-4">
      {revealedKey && (
        <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-4">
          <div className="flex items-center gap-2 text-amber-500 text-sm font-medium mb-2">
            <AlertTriangle className="h-4 w-4" />
            {t("dash.serverless.key_reveal_warning")}
          </div>
          <CopyableText text={revealedKey} className="text-sm text-text-primary" />
          <Button variant="outline" size="sm" className="mt-3" onClick={() => setRevealedKey(null)}>
            {t("dash.serverless.key_dismiss")}
          </Button>
        </div>
      )}

      {canWrite && (
        <div className="flex flex-wrap gap-2 items-end">
          <div className="flex-1 min-w-[160px]">
            <label className="block text-xs font-medium mb-1">{t("dash.serverless.key_name")}</label>
            <Input value={newName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewName(e.target.value)} />
          </div>
          <Button onClick={handleCreate} disabled={creating}>
            {creating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
            {t("dash.serverless.key_create")}
          </Button>
        </div>
      )}

      {keys.length === 0 ? (
        <ServerlessEmptyState icon={Key} title={t("dash.serverless.keys_empty")} />
      ) : (
        <div className="space-y-2">
          {keys.map((k) => (
            <div
              key={k.key_id}
              className="glow-card flex items-center justify-between gap-3 rounded-xl border border-border bg-surface px-4 py-3 hover:border-accent-violet/20 transition-colors"
            >
              <div className="min-w-0">
                <p className="font-medium text-sm">{k.name || "default"}</p>
                <p className="font-mono text-xs text-text-muted">{k.key_prefix}••••••••</p>
                {k.last_used_at && (
                  <p className="text-xs text-text-muted mt-0.5">
                    {t("dash.serverless.key_last_used")}: {new Date(k.last_used_at * 1000).toLocaleString()}
                  </p>
                )}
              </div>
              <div className="flex items-center gap-2 shrink-0">
                {k.scopes && <Badge variant="default" className="text-[10px]">{k.scopes}</Badge>}
                {canWrite && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-red-500"
                    disabled={revoking === k.key_id}
                    onClick={() => handleRevoke(k.key_id)}
                  >
                    {revoking === k.key_id ? <Loader2 className="h-4 w-4 animate-spin" /> : <Trash2 className="h-4 w-4" />}
                  </Button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}