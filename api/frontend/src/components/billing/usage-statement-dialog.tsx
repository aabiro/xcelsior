"use client";

import { useEffect, useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Dialog } from "@/components/ui/dialog";
import { useLocale } from "@/lib/locale";
import * as api from "@/lib/api";

export function UsageStatementDialog({ customerId, invoice, onClose, onDownload }: {
  customerId: string;
  invoice: api.Invoice;
  onClose: () => void;
  onDownload: () => void;
}) {
  const { t, locale } = useLocale();
  const [attempt, setAttempt] = useState(0);
  const requestKey = `${customerId}:${invoice.period_start}:${invoice.period_end}:${attempt}`;
  const [result, setResult] = useState<{ key: string; statement?: api.UsageStatement; failed?: boolean } | null>(null);
  const statement = result?.key === requestKey ? result.statement : undefined;
  const failed = result?.key === requestKey && result.failed;
  const month = new Date(invoice.period_start * 1000).toLocaleDateString(locale === "fr" ? "fr-CA" : "en-CA", {
    month: "long", year: "numeric", timeZone: "UTC",
  });
  const money = (amount: number, maximumFractionDigits = 2) => amount.toLocaleString(locale === "fr" ? "fr-CA" : "en-CA", {
    style: "currency", currency: "CAD", minimumFractionDigits: 2, maximumFractionDigits,
  });

  useEffect(() => {
    let current = true;
    api.fetchUsageStatement(customerId, invoice.period_start, invoice.period_end)
      .then((response) => { if (current) setResult({ key: requestKey, statement: response.invoice }); })
      .catch(() => { if (current) setResult({ key: requestKey, failed: true }); });
    return () => { current = false; };
  }, [customerId, invoice.period_start, invoice.period_end, requestKey]);

  return (
    <Dialog open onClose={onClose} title={t("dash.billing.statement_title")} description={`${month} · CAD`} maxWidth="max-w-3xl">
      {failed ? (
        <div role="alert" className="flex items-center justify-between gap-3 py-4">
          <p className="text-sm text-text-muted">{t("dash.billing.statement_error")}</p>
          <Button variant="outline" size="sm" onClick={() => setAttempt((n) => n + 1)}>{t("dash.billing.statement_retry")}</Button>
        </div>
      ) : !statement ? (
        <div role="status" aria-label={t("dash.billing.statement_loading")} className="flex justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-text-muted" />
        </div>
      ) : (
        <div className="space-y-5 pt-4">
          <p className="break-all font-mono text-xs text-text-muted">{statement.invoice_id}</p>
          {statement.line_items.length === 0 ? (
            <p className="text-sm text-text-muted">{t("dash.billing.no_invoices")}</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[480px] text-left text-sm">
                <thead className="border-b border-border text-xs text-text-muted">
                  <tr>
                    <th scope="col" className="pb-3 pr-4">{t("dash.billing.statement_description")}</th>
                    <th scope="col" className="pb-3 pr-4 text-right">{t("dash.billing.statement_quantity")}</th>
                    <th scope="col" className="pb-3 pr-4 text-right">{t("dash.billing.statement_rate")}</th>
                    <th scope="col" className="pb-3 text-right">{t("dash.billing.statement_amount")}</th>
                  </tr>
                </thead>
                <tbody>
                  {statement.line_items.map((item, i) => (
                    <tr key={`${item.job_id}:${item.category}:${i}`} className="border-b border-border/60 align-top">
                      <td className="py-3 pr-4">
                        <p>{item.description}</p>
                        {item.gpu_model && <p className="mt-1 text-xs text-text-muted">{item.gpu_model}</p>}
                        {item.job_id && <p className="mt-1 break-all font-mono text-xs text-text-muted">{item.job_id}</p>}
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums">
                        {item.quantity.toLocaleString(locale, { maximumFractionDigits: 4 })}
                        <span className="mt-1 block text-xs text-text-muted">{item.unit}</span>
                      </td>
                      <td className="py-3 pr-4 text-right tabular-nums">{money(item.unit_price_cad, 6)}</td>
                      <td className="py-3 text-right font-mono">{money(item.subtotal_cad, 6)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <div className="flex items-center justify-between gap-4 border-t border-border pt-4">
            <span className="text-sm font-medium">{t("dash.billing.statement_total")}</span>
            <span className="font-mono text-xl font-semibold">{money(statement.total_cad)}</span>
          </div>
          <p className="text-xs leading-relaxed text-text-muted">{t("dash.billing.invoices_desc")}</p>
          <div className="flex justify-end">
            <Button variant="outline" size="sm" onClick={onDownload}><Download className="h-3.5 w-3.5" /> CSV</Button>
          </div>
        </div>
      )}
    </Dialog>
  );
}
