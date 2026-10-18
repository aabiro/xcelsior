"use client";

import { useEffect, useRef, useState } from "react";
import { AlertTriangle, Check, Copy, KeyRound, ShieldCheck } from "lucide-react";
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
  const [nudge, setNudge] = useState(0);
  const copiedTimer = useRef<ReturnType<typeof setTimeout>>(undefined);

  useEffect(() => {
    setAcknowledged(false);
    setCopiedField(null);
  }, [clientId, clientSecret, open]);
  useEffect(() => () => clearTimeout(copiedTimer.current), []);

  const copy = (text: string, field: string) => {
    void navigator.clipboard
      .writeText(text)
      .then(() => {
        setCopiedField(field);
        toast.success(t("dash.settings.oauth.copied"));
        clearTimeout(copiedTimer.current);
        copiedTimer.current = setTimeout(() => setCopiedField(null), 2000);
      })
      .catch(() => toast.error(t("dash.settings.oauth.copy_failed")));
  };

  const finish = () => {
    setAcknowledged(false);
    setCopiedField(null);
    onClose();
  };

  // The secret is unrecoverable once this closes. An overlay click, Escape or
  // the × used to discard it silently; now every exit needs the same
  // confirmation the Done button always did.
  const requestClose = () => {
    if (acknowledged || !clientSecret) {
      finish();
      return;
    }
    setNudge((n) => n + 1);
    toast.warning(t("dash.settings.oauth.secret_close_blocked"));
  };

  return (
    <Dialog
      open={open}
      onClose={requestClose}
      title={t("dash.settings.oauth.secret_reveal_title")}
      description={t("dash.settings.oauth.secret_reveal_desc")}
      maxWidth="max-w-3xl"
      className="!rounded-3xl"
    >
      <div className="space-y-5 pt-5">
        <div className="flex items-start gap-3 rounded-2xl border border-accent-gold/30 bg-accent-gold/[0.07] px-4 py-3.5">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-accent-gold/15 ring-1 ring-accent-gold/30">
            <AlertTriangle className="h-4 w-4 text-accent-gold" />
          </div>
          <p className="pt-1 text-sm leading-relaxed text-text-secondary">{t("dash.settings.oauth.secret_warning")}</p>
        </div>

        <div>
          <FieldLabel>{t("dash.settings.oauth.client_id_label")}</FieldLabel>
          <div className="flex min-w-0 items-center gap-2 rounded-xl border border-border/60 bg-surface/60 py-1.5 pl-4 pr-1.5">
            <code className="oauth-secret-scroll min-w-0 flex-1 select-all overflow-x-auto whitespace-nowrap py-1.5 font-mono text-[13px] text-text-secondary">
              {clientId || "-"}
            </code>
            <CopyButton
              copied={copiedField === "client-id"}
              onClick={() => copy(clientId, "client-id")}
              label={t("dash.settings.oauth.copy")}
              copiedLabel={t("dash.settings.oauth.copied_short")}
            />
          </div>
        </div>

        <div>
          <FieldLabel
            action={
              <Button
                size="sm"
                onClick={() => copy(clientSecret, "client-secret")}
                disabled={!clientSecret}
                className={cn(
                  "h-8 gap-1.5",
                  copiedField === "client-secret" && "!bg-none !bg-emerald !text-navy",
                )}
              >
                {copiedField === "client-secret" ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
                {copiedField === "client-secret" ? t("dash.settings.oauth.copied_short") : t("dash.settings.oauth.copy_secret")}
              </Button>
            }
          >
            {t("dash.settings.oauth.client_secret_label")}
            <span className="ml-2 rounded-full border border-accent-gold/30 bg-accent-gold/10 px-2 py-px text-[10px] font-semibold normal-case tracking-normal text-accent-gold">
              {t("dash.settings.oauth.secret_shown_once")}
            </span>
          </FieldLabel>
          {/* Gradient frame + glow: this is the one value on the page that
              cannot be recovered, so it is the one that has to be seen — whole,
              on one line, with nothing else competing for its width. */}
          <div className="oauth-secret-frame rounded-2xl p-[1.5px]">
            <div className="flex min-w-0 items-center gap-3 rounded-[15px] bg-[var(--popover-solid)] py-3 pl-4 pr-4">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-accent-cyan/12 ring-1 ring-accent-cyan/30">
                <KeyRound className="h-4 w-4 text-accent-cyan" />
              </div>
              <code
                className="oauth-secret-scroll oauth-secret-value min-w-0 flex-1 select-all overflow-x-auto whitespace-nowrap py-1.5 font-mono text-[15px] font-semibold tracking-wide"
                aria-label={t("dash.settings.oauth.client_secret_label")}
              >
                {clientSecret || "-"}
              </code>
            </div>
          </div>
        </div>

        {scopes.length > 0 && (
          <div>
            <FieldLabel>{t("dash.settings.oauth.detail_scopes")}</FieldLabel>
            <ScopeChipRow scopes={scopes} size="md" />
          </div>
        )}

        <div className="flex flex-col gap-3 border-t border-border/50 pt-5 sm:flex-row sm:items-center sm:justify-between">
          <label
            key={nudge}
            className={cn(
              "flex cursor-pointer items-start gap-3 rounded-xl px-1 py-1 text-sm text-text-secondary",
              nudge > 0 && !acknowledged && "oauth-secret-nudge",
            )}
          >
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
              className="mt-0.5 h-4 w-4 shrink-0 rounded border-border accent-accent-cyan"
            />
            <span>{t("dash.settings.oauth.secret_ack")}</span>
          </label>
          <Button size="sm" onClick={finish} disabled={!acknowledged || !clientSecret} className="min-w-[120px] shrink-0">
            <ShieldCheck className="h-3.5 w-3.5" />
            {t("dash.settings.oauth.done")}
          </Button>
        </div>
      </div>
    </Dialog>
  );
}

function FieldLabel({ children, action }: { children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="mb-2 flex min-h-8 items-center justify-between gap-3">
      <div className="flex items-center text-[11px] font-semibold uppercase tracking-wider text-text-muted">{children}</div>
      {action}
    </div>
  );
}

function CopyButton({
  copied,
  onClick,
  label,
  copiedLabel,
}: {
  copied: boolean;
  onClick: () => void;
  label: string;
  copiedLabel: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex h-8 shrink-0 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium transition-colors",
        copied ? "bg-emerald/10 text-emerald" : "text-text-muted hover:bg-surface-hover hover:text-text-primary",
      )}
    >
      {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? copiedLabel : label}
    </button>
  );
}
