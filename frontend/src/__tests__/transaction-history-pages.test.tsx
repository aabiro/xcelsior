import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

/**
 * The billing page's transaction list pages through the ledger with the API's
 * cursor: newest first, Older follows `next_cursor`, Newer goes back through
 * the cursors already used, and the range says where the page sits.
 */

const ROWS = Array.from({ length: 45 }, (_, i) => ({
  tx_id: `tx-${String(45 - i).padStart(2, "0")}`,
  tx_type: "deposit",
  amount_cad: 1 + i,
  description: `Deposit ${45 - i}`,
  created_at: 1_790_000_000 - i * 60,
}));

// A keyset server over ROWS: `before` is the tx_id of the previous page's last row.
const fetchWalletHistoryPage = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { walletTxTime: actual.walletTxTime, fetchWalletHistoryPage };
});
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (k: string) => k, locale: "en" }) }));

import { TransactionHistory } from "@/components/billing/transaction-history";

beforeEach(() => {
  fetchWalletHistoryPage.mockReset();
  fetchWalletHistoryPage.mockImplementation(async (_cid: string, { limit, before }: { limit: number; before?: string | null }) => {
    const start = before ? ROWS.findIndex((r) => r.tx_id === before) + 1 : 0;
    const page = ROWS.slice(start, start + limit);
    const more = start + limit < ROWS.length;
    return { ok: true, customer_id: "c", transactions: page, next_cursor: more ? page[page.length - 1].tx_id : null, total: ROWS.length };
  });
});

const range = () => screen.getByTestId("history-range").textContent;
const button = (name: RegExp) => screen.getByRole("button", { name });

describe("transaction history pages", () => {
  it("shows the newest page first, with where it sits", async () => {
    render(<TransactionHistory customerId="c" />);
    await waitFor(() => expect(range()).toBe("1–20 of 45"));
    expect(screen.getByText("Deposit 45")).toBeInTheDocument();
    expect(screen.queryByText("Deposit 25")).not.toBeInTheDocument();
    expect(button(/newer/i)).toBeDisabled();
  });

  it("walks older to the end and back, by the cursors it was given", async () => {
    render(<TransactionHistory customerId="c" />);
    await waitFor(() => expect(range()).toBe("1–20 of 45"));

    fireEvent.click(button(/older/i));
    await waitFor(() => expect(range()).toBe("21–40 of 45"));
    expect(fetchWalletHistoryPage).toHaveBeenLastCalledWith("c", { limit: 20, before: "tx-26" });

    fireEvent.click(button(/older/i));
    await waitFor(() => expect(range()).toBe("41–45 of 45"));
    expect(screen.getByText("Deposit 1")).toBeInTheDocument();
    expect(button(/older/i)).toBeDisabled();

    fireEvent.click(button(/newer/i));
    await waitFor(() => expect(range()).toBe("21–40 of 45"));
    expect(fetchWalletHistoryPage).toHaveBeenLastCalledWith("c", { limit: 20, before: "tx-26" });
  });

  it("a short ledger has no paging controls", async () => {
    fetchWalletHistoryPage.mockResolvedValue({
      ok: true, customer_id: "c", transactions: ROWS.slice(0, 3), next_cursor: null, total: 3,
    });
    render(<TransactionHistory customerId="c" />);
    await screen.findByText("Deposit 45");
    expect(screen.queryByTestId("history-range")).not.toBeInTheDocument();
  });

  it("returns to the newest page when the billing page reloads", async () => {
    const { rerender } = render(<TransactionHistory customerId="c" refreshKey={1} />);
    await waitFor(() => expect(range()).toBe("1–20 of 45"));
    fireEvent.click(button(/older/i));
    await waitFor(() => expect(range()).toBe("21–40 of 45"));
    rerender(<TransactionHistory customerId="c" refreshKey={2} />);
    await waitFor(() => expect(range()).toBe("1–20 of 45"));
  });

  it("offers a retry rather than an empty ledger when a page fails", async () => {
    fetchWalletHistoryPage.mockRejectedValueOnce(new Error("502"));
    render(<TransactionHistory customerId="c" />);
    expect(await screen.findByText("Transactions could not be loaded.")).toBeInTheDocument();
    expect(screen.queryByText("dash.billing.no_transactions")).not.toBeInTheDocument();
    fireEvent.click(button(/retry/i));
    await waitFor(() => expect(range()).toBe("1–20 of 45"));
  });
});
