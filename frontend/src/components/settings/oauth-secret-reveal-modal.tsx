"use client";

import { useState } from "react";
import { AlertTriangle, CheckCircle, Copy, KeyRound } from "lucide-react";
import { Dialog } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { useLocale } from "@/lib/locale";
import { toast } from "sonner";
import { ScopeChipRow } from "@/components/settings/credential-scope-panel";

export function OAuthSecretRevealModal({
  open,
  clientId,
  clientSecret,
  scopes,
  onClose,
}: {
  open: boolean;
  clientId: string;
  clientSecret: string;
  scopes: string[];
  onClose: () => void;
}) {
  const { t } = useLocale();
  const [copiedField, setCopiedField] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);

  const copy = (text: string, field: string) => {
    void navigator.clipboard.writeText(text).then(() => {
      setCopiedField(field);
      toast.success(t("dash.settings.oauth.copied"));
      setTimeout(() => setCopiedField(null), 2000);
    });
  };

  const handleClose = () => {
    setAcknowledged(false);
    setCopiedField(null);
    onClose();
  };

  return (
    <Dialog
      open={open}
      onClose={handleClose}
      title={t("dash.settings.oauth.secret_reveal_title")}
      description={t("dash.settings.oauth.secret_reveal_desc")}
    >
      <div className="space-y-4 pt-2">
        <div className="rounded-xl border border-accent-cyan/35 bg-gradient-to-b from-accent-cyan/10 to-accent-cyan/5 p-4">
          <div className="flex items-start gap-3">
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent-cyan/20 ring-1 ring-accent-cyan/30">
              <AlertTriangle className="h-4 w-4 text-accent-cyan" />
            </div>
            <p className="text-sm text-text-secondary leading-relaxed">
              {t("dash.settings.oauth.secret_warning")}
            </p>
          </div>
        </div>

        <div className="space-y-3 rounded-xl border border-border/60 bg-background/70 p-4">
          <SecretField
            label={t("dash.settings.oauth.client_id_label")}
            value={clientId}
            fieldId="client-id"
            copiedField={copiedField}
            onCopy={copy}
            mono
          />
          <SecretField
            label={t("dash.settings.oauth.client_secret_label")}
            value={clientSecret}
            fieldId="client-secret"
            copiedField={copiedField}
            onCopy={copy}
            mono
            highlight
          />
        </div>

        {scopes.length > 0 && (
          <div className="rounded-xl border border-border/50 bg-surface/50 px-4 py-3">
            <p className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
              {t("dash.settings.oauth.detail_scopes")}
            </p>
            <ScopeChipRow scopes={scopes} size="md" />
          </div>
        )}

        <label className="flex cursor-pointer items-start gap-3 rounded-xl border border-border/50 bg-navy-light/25 px-4 py-3">
          <input
            type="checkbox"
            checked={acknowledged}
            onChange={(e) => setAcknowledged(e.target.checked)}
            className="mt-0.5 h-4 w-4 rounded border-border accent-accent-cyan"
          />
          <span className="text-sm text-text-secondary">{t("dash.settings.oauth.secret_ack")}</span>
        </label>

        <div className="flex justify-end gap-2 pt-1">
          <Button
            size="sm"
            onClick={handleClose}
            disabled={!acknowledged || !clientSecret}
            className="min-w-[120px]"
          >
            <KeyRound className="h-3.5 w-3.5" />
            {t("dash.settings.oauth.done")}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

function SecretField({
  label,
  value,
  fieldId,
  copiedField,
  onCopy,
  mono,
  highlight,
}: {
  label: string;
  value: string;
  fieldId: string;
  copiedField: string | null;
  onCopy: (text: string, field: string) => void;
  mono?: boolean;
  highlight?: boolean;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-text-muted">{label}</span>
        <button
          type="button"
          onClick={() => onCopy(value, fieldId)}
          className={cn(
            "flex h-7 items-center gap-1 rounded-lg px-2 text-xs transition-colors",
            copiedField === fieldId
              ? "bg-emerald/10 text-emerald"
              : "text-text-muted hover:bg-surface-hover hover:text-text-primary",
          )}
        >
          {copiedField === fieldId ? (
            <CheckCircle className="h-3.5 w-3.5" />
          ) : (
            <Copy className="h-3.5 w-3.5" />
          )}
          Copy
        </button>
      </div>
      <button
        type="button"
        onClick={() => onCopy(value, fieldId)}
        className={cn(
          "block w-full rounded-lg border px-3 py-2.5 text-left text-xs transition-colors hover:bg-surface-hover",
          mono && "font-mono break-all",
          highlight
            ? "border-accent-cyan/30 bg-accent-cyan/5 text-text-primary"
            : "border-border/50 bg-surface/60 text-text-secondary",
        )}
      >
        {value || "—"}
      </button>
    </div>
  );
}