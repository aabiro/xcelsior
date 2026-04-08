import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";

const apiMocks = vi.hoisted(() => ({
  fetchWallet: vi.fn(),
  fetchWalletHistory: vi.fn(),
  fetchInvoices: vi.fn(),
  fetchUsageSummary: vi.fn(),
  fetchReservedPlans: vi.fn(),
  checkCryptoEnabled: vi.fn(),
  checkLightningEnabled: vi.fn(),
  resetWalletTestingState: vi.fn(),
  checkFreeCreditsStatus: vi.fn(),
  claimFreeCredits: vi.fn(),
}));

const authMocks = vi.hoisted(() => ({
  useAuth: vi.fn(),
}));

const toastMocks = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);

vi.mock("@/lib/auth", () => ({
  useAuth: authMocks.useAuth,
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string) => key,
    locale: "en",
  }),
}));

vi.mock("@/components/billing/deposit-modal", () => ({
  DepositModal: () => null,
}));

vi.mock("@/components/billing/crypto-deposit-modal", () => ({
  CryptoDepositModal: () => null,
}));

vi.mock("sonner", () => ({
  toast: toastMocks,
}));

vi.mock("framer-motion", async () => {
  const ReactModule = await import("react");

  function createMotionTag(tag: keyof React.JSX.IntrinsicElements) {
    return ReactModule.forwardRef<HTMLElement, React.HTMLAttributes<HTMLElement>>(
      ({ children, ...props }, ref) => ReactModule.createElement(tag, { ...props, ref }, children),
    );
  }

  return {
    AnimatePresence: ({ children }: { children: React.ReactNode }) => children,
    animate: (
      _from: number,
      to: number,
      options?: { onUpdate?: (value: number) => void; onComplete?: () => void },
    ) => {
      options?.onUpdate?.(to);
      options?.onComplete?.();
      return { stop: vi.fn() };
    },
    motion: {
      div: createMotionTag("div"),
      span: createMotionTag("span"),
    },
  };
});

import BillingPage from "@/app/(dashboard)/dashboard/billing/page";

describe("BillingPage free credits flow", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    authMocks.useAuth.mockReturnValue({
      user: { user_id: "user-1", customer_id: "cust-1", is_admin: false },
    });

    apiMocks.fetchWallet.mockResolvedValue({
      ok: true,
      wallet: { customer_id: "cust-1", balance_cad: 5, currency: "CAD" },
    });
    apiMocks.fetchWalletHistory.mockResolvedValue({ ok: true, transactions: [] });
    apiMocks.fetchInvoices.mockResolvedValue({ ok: true, invoices: [] });
    apiMocks.fetchUsageSummary.mockResolvedValue({
      ok: true,
      job_count: 0,
      total_gpu_hours: 0,
      total_cost_cad: 0,
      canadian_compute_cad: 0,
      hosts_used: 0,
    });
    apiMocks.fetchReservedPlans.mockResolvedValue({});
    apiMocks.checkCryptoEnabled.mockResolvedValue({ ok: true, enabled: false });
    apiMocks.checkLightningEnabled.mockResolvedValue({ ok: true, enabled: false });
    apiMocks.checkFreeCreditsStatus.mockResolvedValue({ ok: true, claimed: false });
    apiMocks.resetWalletTestingState.mockResolvedValue({
      ok: true,
      wallet: { customer_id: "cust-1", balance_cad: 0, currency: "CAD" },
      cleared_transactions: 2,
      promo_available: true,
    });
    apiMocks.claimFreeCredits.mockResolvedValue({
      ok: true,
      amount_cad: 10,
      balance_cad: 15,
      already_claimed: false,
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it("shows transfer state, then check state, then removes the promo banner", async () => {
    render(<BillingPage />);

    await screen.findByText("dash.billing.free_credits_title");
    await waitFor(() => {
      expect(screen.getAllByText("$5.00").length).toBeGreaterThan(0);
    });

    fireEvent.click(screen.getByRole("button", { name: /dash\.billing\.claim_credits/i }));

    await screen.findByText("dash.billing.credits_transferring_title");
    await waitFor(() => {
      expect(screen.getAllByText("$15.00").length).toBeGreaterThan(0);
    });

    await act(async () => {
      vi.advanceTimersByTime(2200);
    });

    await screen.findByText("dash.billing.credits_added_badge");

    await act(async () => {
      vi.advanceTimersByTime(1400);
    });

    await waitFor(() => {
      expect(screen.queryByText("dash.billing.free_credits_title")).not.toBeInTheDocument();
    });
  });

  it("hides the reset control for non-admin users", async () => {
    render(<BillingPage />);

    await screen.findByText("dash.billing.wallet_credits");
    expect(screen.queryByRole("button", { name: /dash\.billing\.admin_reset_action/i })).not.toBeInTheDocument();
  });

  it("lets admins reset wallet testing state and restores the promo banner", async () => {
    authMocks.useAuth.mockReturnValue({
      user: { user_id: "user-1", customer_id: "cust-1", is_admin: true },
    });
    apiMocks.fetchWallet.mockResolvedValue({
      ok: true,
      wallet: { customer_id: "cust-1", balance_cad: 15, currency: "CAD" },
    });
    apiMocks.checkFreeCreditsStatus.mockResolvedValue({ ok: true, claimed: true });

    render(<BillingPage />);

    await screen.findByText("dash.billing.wallet_credits");
    expect(screen.queryByText("dash.billing.free_credits_title")).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /dash\.billing\.admin_reset_action/i }));
    await screen.findByText("dash.billing.admin_reset_confirm_title");

    fireEvent.click(screen.getByRole("button", { name: /dash\.billing\.admin_reset_confirm_action/i }));

    await waitFor(() => {
      expect(apiMocks.resetWalletTestingState).toHaveBeenCalledWith("cust-1");
    });
    await screen.findByText("dash.billing.free_credits_title");
    await waitFor(() => {
      expect(screen.getAllByText("$0.00").length).toBeGreaterThan(0);
    });
  });

  it("disables bitcoin deposits when the backend reports the service unavailable", async () => {
    apiMocks.checkCryptoEnabled.mockResolvedValue({
      ok: true,
      enabled: true,
      available: false,
      reason: "Bitcoin RPC error: {'code': -4, 'message': 'Database already exists.'}",
    });

    render(<BillingPage />);

    await screen.findByText("Bitcoin Deposits");
    expect(screen.getByText("Bitcoin deposits are temporarily unavailable.")).toBeInTheDocument();
    expect(screen.queryByText(/database already exists/i)).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /unavailable/i })).toBeDisabled();
  });
});
