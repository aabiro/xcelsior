"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowDownRight, ArrowUpRight, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import * as api from "@/lib/api";
import { useLocale } from "@/lib/locale";

export const HISTORY_PAGE_SIZE = 20;

/**
 * The wallet's transactions, a page at a time, newest first.
 *
 * Paged by the API's keyset cursor rather than by offset: the ledger grows at
 * the head while someone reads it, and offsets would repeat rows across pages
 * as charges land. Newer pages are reached through the cursors already used,
 * so going back is exact.
 *
 * `refreshKey` changes when the page reloads its own data (a deposit, a claim,
 * a team switch), and returns this list to the newest page, which is where
 * the change appears.
 */
export function TransactionHistory({
  customerId,
  refreshKey,
  pageSize = HISTORY_PAGE_SIZE,
}: {
  customerId: string;
  refreshKey?: unknown;
  pageSize?: number;
}) {
  const { t } = useLocale();
  // cursors[i] is the `before` that fetches page i; page 0 needs none.
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const [pageIndex, setPageIndex] = useState(0);
  const [page, setPage] = useState<api.WalletHistoryPage | null>(null);
  const [loading, setLoading] = useState(true);
  const [failed, setFailed] = useState(false);
  // Only the latest request may land: a refresh can start a page-0 fetch while
  // an older-page fetch is still in flight, and the slower one must not win.
  const latest = useRef(0);

  useEffect(() => {
    setCursors([null]);
    setPageIndex(0);
  }, [customerId, refreshKey]);

  const before = cursors[pageIndex] ?? null;
  const fetchPage = useCallback(async () => {
    if (!customerId) return;
    const id = ++latest.current;
    setLoading(true);
    setFailed(false);
    try {
      const result = await api.fetchWalletHistoryPage(customerId, { limit: pageSize, before });
      if (id === latest.current) setPage(result);
    } catch {
      if (id === latest.current) setFailed(true);
    } finally {
      if (id === latest.current) setLoading(false);
    }
  }, [customerId, pageSize, before]);

  useEffect(() => {
    void fetchPage();
    // `refreshKey` refetches page 0 even when the cursor did not change.
  }, [fetchPage, refreshKey]);

  const older = () => {
    if (!page?.next_cursor) return;
    setCursors((c) => [...c.slice(0, pageIndex + 1), page.next_cursor]);
    setPageIndex((i) => i + 1);
  };
  const newer = () => setPageIndex((i) => Math.max(0, i - 1));

  const rows = page?.transactions ?? [];
  const total = page?.total ?? 0;
  const first = rows.length ? pageIndex * pageSize + 1 : 0;
  const last = pageIndex * pageSize + rows.length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{t("dash.billing.transactions")}</CardTitle>
        <CardDescription>{t("dash.billing.transactions_desc")}</CardDescription>
      </CardHeader>
      <CardContent>
        {failed ? (
          <div className="flex items-center justify-between gap-3 rounded-lg border border-border p-3">
            <p className="text-sm text-text-muted">Transactions could not be loaded.</p>
            <Button variant="outline" size="sm" onClick={() => void fetchPage()}>
              Retry
            </Button>
          </div>
        ) : !page && loading ? (
          <div className="flex justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-text-muted" />
          </div>
        ) : rows.length === 0 ? (
          <p className="text-sm text-text-muted">{t("dash.billing.no_transactions")}</p>
        ) : (
          <>
            <div className={`space-y-2 transition-opacity ${loading ? "opacity-60" : ""}`} data-testid="transaction-rows">
              {rows.map((tx) => {
                const isCredit = tx.amount_cad > 0;
                return (
                  <div
                    key={tx.tx_id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-border p-3"
                  >
                    <div className="flex min-w-0 items-center gap-3">
                      {isCredit ? (
                        <ArrowDownRight className="h-4 w-4 shrink-0 text-emerald" />
                      ) : (
                        <ArrowUpRight className="h-4 w-4 shrink-0 text-accent-red" />
                      )}
                      <div className="min-w-0">
                        <p className="truncate text-sm font-medium">
                          {tx.description || tx.tx_type || "Transaction"}
                        </p>
                        <p className="text-xs text-text-muted">
                          {api.walletTxTime(tx)}
                          {tx.job_id && <span className="ml-2">· Job {tx.job_id.slice(0, 8)}</span>}
                        </p>
                      </div>
                    </div>
                    <span className={`shrink-0 font-mono text-sm font-medium ${isCredit ? "text-emerald" : "text-accent-red"}`}>
                      {isCredit ? "+" : ""}${tx.amount_cad.toFixed(2)}
                    </span>
                  </div>
                );
              })}
            </div>
            {(pageIndex > 0 || page?.next_cursor) && (
              <div className="mt-4 flex items-center justify-between gap-3">
                <p className="text-xs text-text-muted" data-testid="history-range">
                  {first}–{last} of {total}
                </p>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={newer} disabled={pageIndex === 0 || loading}>
                    <ChevronLeft className="h-3.5 w-3.5" /> Newer
                  </Button>
                  <Button variant="outline" size="sm" onClick={older} disabled={!page?.next_cursor || loading}>
                    Older <ChevronRight className="h-3.5 w-3.5" />
                  </Button>
                </div>
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
