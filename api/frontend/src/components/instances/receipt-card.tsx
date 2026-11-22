"use client";

/**
 * The receipt for an instance's finished runs, read from the ledger that billed
 * them (`GET /api/instances/{id}/receipt`). Every figure is the billed one:
 * exact seconds from the closed meter, the rate the run was locked at, and the
 * wallet debits the total reconciles to. The whole-hours line is a comparison,
 * labelled as one; nothing is billed that way.
 */
import type { ReactNode } from "react";
import { CheckCircle2, AlertTriangle, Receipt, ShieldCheck } from "lucide-react";
import { Card } from "@/components/ui/card";
import type { RunReceipt } from "@/lib/api";
import { formatDuration, formatMicros } from "@/lib/format-units";
import { useLocale } from "@/lib/locale";

const STATE_KEYS: Record<string, string> = {
  verified: "dash.receipt.state_verified",
  overdue: "dash.receipt.state_overdue",
  deverified: "dash.receipt.state_deverified",
  unverified: "dash.receipt.state_unverified",
  verifying: "dash.receipt.state_verifying",
};

function verificationKey(receipt: RunReceipt): string | null {
  const v = receipt.host_verification_at_start;
  if (!v) return null;
  // `freshness` is what placement saw (verified, or verified but overdue);
  // older runs only have the stored state.
  return STATE_KEYS[v.freshness || v.state] ?? STATE_KEYS.unverified;
}

export function ReceiptCard({ receipt, footer }: { receipt: RunReceipt; footer?: ReactNode }) {
  const { t } = useLocale();
  const stateKey = verificationKey(receipt);
  const verified = receipt.host_verification_at_start?.freshness === "verified"
    || (receipt.host_verification_at_start?.source === "history"
      && receipt.host_verification_at_start?.state === "verified");
  const runs = receipt.attempts.filter((a) => a.finished && a.started_at !== null).length;
  return (
    <Card data-testid="receipt-card">
      <div className="flex items-center gap-2 mb-3">
        <Receipt className="h-4 w-4 text-accent-gold" />
        <h2 className="text-sm font-semibold text-text-secondary">{t("dash.receipt.title")}</h2>
        {runs > 1 && (
          <span className="text-xs text-text-muted">{t("dash.receipt.runs", { count: runs })}</span>
        )}
        <span className="ml-auto text-xs font-mono text-text-muted">{receipt.job_id.slice(0, 12)}</span>
      </div>

      <dl className="grid gap-3 sm:grid-cols-4 text-sm">
        <div>
          <dt className="text-xs text-text-muted">{t("dash.receipt.billed_time")}</dt>
          <dd className="font-mono font-medium mt-0.5">{formatDuration(receipt.billed_seconds)}</dd>
          <dd className="text-xs text-text-muted font-mono">
            {t("dash.receipt.exact_seconds", { seconds: receipt.billed_seconds.toFixed(2) })}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-text-muted">{t("dash.receipt.rate")}</dt>
          <dd className="font-mono font-medium mt-0.5">
            {receipt.rate_micros_per_hour !== null ? `${formatMicros(receipt.rate_micros_per_hour)}/hr` : "—"}
          </dd>
          {receipt.pricing_mode && (
            <dd className="text-xs text-text-muted">{receipt.pricing_mode.replace("_", "-")}</dd>
          )}
        </div>
        <div>
          <dt className="text-xs text-text-muted">{t("dash.receipt.total")}</dt>
          <dd className="font-mono font-medium mt-0.5" data-testid="receipt-total">
            {formatMicros(receipt.total_micros, { exact: true })}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-text-muted">{t("dash.receipt.paid")}</dt>
          <dd className="font-mono font-medium mt-0.5">
            {formatMicros(receipt.wallet_debit_micros, { exact: true })}
          </dd>
          <dd className="text-xs text-text-muted">
            {receipt.wallet_debit_count === 1
              ? t("dash.receipt.debit_one")
              : t("dash.receipt.debits", { count: receipt.wallet_debit_count })}
          </dd>
        </div>
      </dl>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs">
        {receipt.reconciled ? (
          <span className="inline-flex items-center gap-1 text-emerald" data-testid="receipt-reconciled">
            <CheckCircle2 className="h-3.5 w-3.5" />
            {t("dash.receipt.reconciled")}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1 text-amber-400" data-testid="receipt-unreconciled">
            <AlertTriangle className="h-3.5 w-3.5" />
            {t("dash.receipt.outstanding", {
              amount: formatMicros(receipt.outstanding_micros, { exact: true }),
            })}
          </span>
        )}
        {receipt.credits_micros > 0 && (
          <span className="text-text-muted">
            {t("dash.receipt.credits", { amount: formatMicros(receipt.credits_micros, { exact: true }) })}
          </span>
        )}
        {stateKey && (
          <span className={`inline-flex items-center gap-1 ${verified ? "text-ice-blue" : "text-text-muted"}`}>
            <ShieldCheck className="h-3.5 w-3.5" />
            {t("dash.receipt.verification")}: {t(stateKey)}
          </span>
        )}
      </div>

      {receipt.hourly_rounded_micros !== null && (
        <p className="mt-3 text-xs text-text-muted" data-testid="receipt-hourly">
          {t("dash.receipt.hourly")}:{" "}
          <span className="font-mono">{formatMicros(receipt.hourly_rounded_micros, { exact: true })}</span>
        </p>
      )}
      {receipt.rounding === "legacy_4dp" && (
        <p className="mt-1 text-xs text-text-muted">{t("dash.receipt.legacy_rounding")}</p>
      )}
      {!receipt.final && <p className="mt-1 text-xs text-text-muted">{t("dash.receipt.not_final")}</p>}
      {footer}
    </Card>
  );
}
