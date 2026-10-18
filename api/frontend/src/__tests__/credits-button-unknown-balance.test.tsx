import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, act } from "@testing-library/react";

/**
 * The header wallet must not state a balance it did not read.
 *
 * Production's mobile sweep showed "$0.00" for an account holding $100.00 once
 * requests began failing: on any error except 401 the button fell back to 0
 * "for new users". New users never needed it, because the API creates the
 * wallet on first read and returns its zero balance itself.
 */

const fetchWallet = vi.hoisted(() => vi.fn());

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ApiError: actual.ApiError, fetchWallet };
});
vi.mock("@/lib/auth", () => ({
  useAuth: () => ({ user: { user_id: "u-1", customer_id: "cust-1", email: "a@b.c" } }),
}));
vi.mock("@/lib/team-context", () => ({
  getBillingCustomerId: () => "cust-1",
  getTeamContext: () => ({ canManageBilling: true }),
}));
vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (k: string) => k, locale: "en" }) }));

import { CreditsButton } from "@/components/CreditsButton";

beforeEach(() => {
  vi.useRealTimers();
  fetchWallet.mockReset();
});

describe("header wallet balance", () => {
  it("says nothing it does not know when the first read fails", async () => {
    fetchWallet.mockRejectedValue(new Error("429 Too Many Requests"));
    render(<CreditsButton />);
    await waitFor(() => expect(fetchWallet).toHaveBeenCalled());
    await waitFor(() => expect(screen.getByRole("button").textContent).toContain("-"));
    expect(screen.getByRole("button").textContent).not.toContain("$0.00");
  });

  it("keeps the last balance it read through a later failure", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    fetchWallet.mockResolvedValueOnce({ ok: true, wallet: { balance_cad: 100 } });
    render(<CreditsButton />);
    await waitFor(() => expect(screen.getByRole("button").textContent).toContain("$100.00"));

    fetchWallet.mockRejectedValue(new Error("502 Bad Gateway"));
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    await waitFor(() => expect(fetchWallet.mock.calls.length).toBeGreaterThanOrEqual(2));
    expect(screen.getByRole("button").textContent).toContain("$100.00");
  });

  it("shows a real zero balance as zero", async () => {
    fetchWallet.mockResolvedValue({ ok: true, wallet: { balance_cad: 0 } });
    render(<CreditsButton />);
    await waitFor(() => expect(screen.getByRole("button").textContent).toContain("$0.00"));
  });
});
