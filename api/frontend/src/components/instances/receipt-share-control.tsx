"use client";

/**
 * The owner's switch for counting a finished run on the public workload page
 * for its template and GPU ("vLLM on RTX 3060: what real runs cost").
 *
 * Shown only for runs that can count, or will once billing settles. A run that
 * never can (a custom image, a failure, under a minute) shows nothing: there is
 * no choice to make. The page shows counts, medians and ranges only, and only
 * once three runs are shared.
 */
import { useEffect, useState } from "react";
import Link from "next/link";
import { Globe2 } from "lucide-react";
import { toast } from "sonner";
import { fetchReceiptShare, setReceiptShare, type ReceiptShareState } from "@/lib/api";
import { useLocale } from "@/lib/locale";
import { cn } from "@/lib/utils";

const SETTLING = new Set(["not_finished", "not_reconciled"]);

export function ReceiptShareControl({ jobId }: { jobId: string }) {
  const { t } = useLocale();
  const [state, setState] = useState<ReceiptShareState | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchReceiptShare(jobId)
      .then((r) => { if (!cancelled) setState(r); })
      .catch(() => { if (!cancelled) setState(null); });
    return () => { cancelled = true; };
  }, [jobId]);

  if (!state || !state.template_label || !state.gpu_model) return null;
  const settling = !state.eligible && state.blocker !== null && SETTLING.has(state.blocker);
  if (!state.eligible && !settling && !state.shared) return null;

  const toggle = async () => {
    setSaving(true);
    try {
      setState(await setReceiptShare(jobId, !state.shared));
    } catch {
      toast.error(t("dash.receipt_share.failed"));
    } finally {
      setSaving(false);
    }
  };

  const page = state.page;
  const workload = t("dash.receipt_share.workload", { template: state.template_label, gpu: state.gpu_model });
  return (
    <div className="mt-4 border-t border-border/60 pt-3" data-testid="receipt-share">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="flex items-center gap-1.5 text-sm font-medium text-text-primary">
            <Globe2 className="h-3.5 w-3.5 text-text-muted" />
            {t("dash.receipt_share.title", { workload })}
          </p>
          <p className="mt-0.5 text-xs text-text-secondary">
            {settling
              ? t("dash.receipt_share.settling")
              : t("dash.receipt_share.desc", { runs: page?.min_runs ?? 3 })}
          </p>
          {state.shared && page && (
            <p className="mt-1 text-xs text-text-muted" data-testid="receipt-share-page">
              {page.live ? (
                <Link href={`/receipts/${page.slug}`} className="text-accent-cyan hover:underline">
                  {t("dash.receipt_share.view_page", { runs: page.runs ?? page.min_runs })}
                </Link>
              ) : page.gpu_has_supply ? (
                t("dash.receipt_share.waiting_runs", { shared: page.runs ?? 1, runs: page.min_runs })
              ) : (
                t("dash.receipt_share.waiting_supply")
              )}
            </p>
          )}
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={state.shared}
          aria-label={t("dash.receipt_share.title", { workload })}
          disabled={saving || (!state.shared && !state.eligible)}
          onClick={toggle}
          className={cn(
            "relative mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full transition-colors duration-200 disabled:opacity-50",
            state.shared ? "bg-emerald" : "bg-border",
          )}
        >
          <span
            className={cn(
              "inline-block h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200",
              state.shared ? "translate-x-6" : "translate-x-1",
            )}
          />
        </button>
      </div>
    </div>
  );
}
