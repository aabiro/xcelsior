"use client";

import { useCallback, useEffect, useState } from "react";
import { KeyRound, Loader2, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";
import { toast } from "sonner";

function formatWhen(seconds: number): string | null {
  if (!seconds) return null;
  return new Date(seconds * 1000).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/**
 * Management view for long-lived agent keys.
 *
 * The MCP page is the fast path that mints a key and reveals it once; this is
 * where keys are audited and retired. Only the masked prefix is ever shown —
 * the plaintext exists solely in the reveal at creation time.
 */
export function AgentKeysPanel() {
  const { t } = useLocale();
  const [keys, setKeys] = useState<api.AgentKey[] | null>(null);
  const [revoking, setRevoking] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await api.fetchAgentKeys();
      setKeys(res.keys);
    } catch {
      toast.error(t("dash.settings.keys.load_failed"));
      setKeys([]);
    }
  }, [t]);

  useEffect(() => {
    void load();
  }, [load]);

  const handleRevoke = async (key: api.AgentKey) => {
    // A key in use is, by definition, wired into something that will break.
    if (!window.confirm(t("dash.settings.keys.revoke_confirm"))) return;
    setRevoking(key.key_id);
    try {
      await api.revokeAgentKey(key.key_id);
      toast.success(t("dash.settings.keys.revoked"));
      await load();
    } catch {
      toast.error(t("dash.settings.keys.revoke_failed"));
    } finally {
      setRevoking(null);
    }
  };

  return (
    <div className="rounded-xl border border-border bg-surface/40 p-4">
      <div className="mb-1 flex items-center gap-2">
        <KeyRound className="h-4 w-4 text-text-muted" />
        <h3 className="text-sm font-semibold text-text-primary">
          {t("dash.settings.keys.title")}
        </h3>
      </div>
      <p className="mb-4 text-xs text-text-secondary">{t("dash.settings.keys.subtitle")}</p>

      {keys === null ? (
        <div className="space-y-2">
          <div className="skeleton h-10 w-full rounded" />
          <div className="skeleton h-10 w-2/3 rounded" />
        </div>
      ) : keys.length === 0 ? (
        <p className="text-xs text-text-muted">{t("dash.settings.keys.empty")}</p>
      ) : (
        <ul className="divide-y divide-border/50">
          {keys.map((key) => {
            const lastUsed = formatWhen(key.last_used_at);
            return (
              <li key={key.key_id} className="flex items-center justify-between gap-3 py-2.5">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm text-text-primary">{key.name}</span>
                    {key.in_use && (
                      <span className="rounded-full bg-accent-cyan/10 px-2 py-0.5 text-[10px] font-medium text-accent-cyan">
                        {t("dash.settings.keys.in_use")}
                      </span>
                    )}
                  </div>
                  <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-text-muted">
                    <span className="font-mono">{key.key_prefix}</span>
                    <span>
                      {t("dash.settings.keys.created")} {formatWhen(key.created_at)}
                    </span>
                    <span>
                      {lastUsed
                        ? `${t("dash.settings.keys.last_used")} ${lastUsed}`
                        : t("dash.settings.keys.never_used")}
                    </span>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => void handleRevoke(key)}
                  disabled={revoking === key.key_id}
                  aria-label={t("dash.settings.keys.revoke")}
                >
                  {revoking === key.key_id ? (
                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                  ) : (
                    <Trash2 className="mr-1 h-3 w-3" />
                  )}
                  {t("dash.settings.keys.revoke")}
                </Button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
