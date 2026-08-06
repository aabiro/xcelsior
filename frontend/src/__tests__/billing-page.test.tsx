import React from "react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor, act } from "@testing-library/react";

const apiMocks = vi.hoisted(() => ({
  fetchWallet: vi.fn(),
  fetchWalletHistory: vi.fn(),
  fetchInvoices: vi.fn(),
  fetchUsageSummary: vi.fn(),
  fetchReservedPlans: vi.fn(),
  fetchPaymentMethods: vi.fn(),
  fetchAutoTopup: vi.fn(),
  configureAutoTopup: vi.fn(),
  createSetupIntent: vi.fn(),
  deletePaymentMethod: vi.fn(),
  checkCryptoEnabled: vi.fn(),
  checkLightningEnabled: vi.fn(),
  resetWalletTestingState: vi.fn(),
  checkFreeCreditsStatus: vi.fn(),
  claimFreeCredits: vi.fn(),
  // Charges the bank stopped pending verification. The page loads these on
  // mount; unmocked it is `undefined` and the effect throws.
  fetchPendingVerification: vi.fn(),
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
  // The page clears `?resume=` after an SCA challenge is finished, so it needs
  // a router. Without this export every test in the file fails at render with
  // "No useRouter export is defined on the next/navigation mock".
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
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

vi.mock("@/components/billing/lightning-deposit-modal", () => ({
  LightningDepositModal: () => null,
}));

vi.mock("@/components/billing/payment-method-modal", () => ({
  PaymentMethodModal: () => null,
}));

vi.mock("@/components/team/team-context-banner", () => ({
  TeamContextBanner: () => null,
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
  vi.setConfig({ testTimeout: 15_000 });
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    authMocks.useAuth.mockReturnValue({
      user: { user_id: "user-1", customer_id: "cust-1", is_admin: false },
    });

    // No challenge outstanding: the default for every test here, so the SCA
    // panel stays hidden and these assertions are about the wallet itself.
    apiMocks.fetchPendingVerification.mockResolvedValue({
      ok: true,
      customer_id: "cust-1",
      pending: [],
      count: 0,
      message: "No charges are waiting on cardholder verification.",
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
    apiMocks.fetchPaymentMethods.mockResolvedValue({ ok: true, payment_methods: [] });
    apiMocks.fetchAutoTopup.mockResolvedValue({
      ok: true,
      auto_topup: {
        enabled: false,
        amount_cad: 25,
        threshold_cad: 5,
        payment_method_id: "",
        has_payment_method: false,
      },
    });
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

  it("renders low-balance banner when wallet.low_balance is set", async () => {
    apiMocks.fetchWallet.mockResolvedValue({
      ok: true,
      wallet: {
        customer_id: "cust-1",
        balance_cad: 3,
        currency: "CAD",
        low_balance: true,
        hard_stop: false,
        low_balance_threshold_cad: 5,
      },
    });
    apiMocks.checkFreeCreditsStatus.mockResolvedValue({ ok: true, claimed: true });
    render(<BillingPage />);
    expect(await screen.findByTestId("wallet-low-balance-banner")).toBeInTheDocument();
    expect(screen.getByText(/Low balance warning/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Add credits/i })).toBeInTheDocument();
  });

  it("renders hard-stop banner when wallet.hard_stop is set", async () => {
    apiMocks.fetchWallet.mockResolvedValue({
      ok: true,
      wallet: {
        customer_id: "cust-1",
        balance_cad: 0,
        currency: "CAD",
        low_balance: true,
        hard_stop: true,
        low_balance_threshold_cad: 5,
      },
    });
    apiMocks.checkFreeCreditsStatus.mockResolvedValue({ ok: true, claimed: true });
    render(<BillingPage />);
    expect(await screen.findByTestId("wallet-hard-stop-banner")).toBeInTheDocument();
    expect(screen.getByText(/Hard stop/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Top up now/i })).toBeInTheDocument();
    // Hard-stop supersedes low-balance banner
    expect(screen.queryByTestId("wallet-low-balance-banner")).not.toBeInTheDocument();
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

  it("shows lightning alongside bitcoin in the grouped crypto deposits section", async () => {
    apiMocks.checkCryptoEnabled.mockResolvedValue({
      ok: true,
      enabled: true,
      available: true,
    });
    apiMocks.checkLightningEnabled.mockResolvedValue({
      ok: true,
      enabled: true,
      available: true,
      node_alias: "xcelsior-lnd",
      num_active_channels: 4,
    });

    render(<BillingPage />);

    await screen.findByText("Crypto Deposits");
    expect(screen.getByText("Bitcoin Deposits")).toBeInTheDocument();
    expect(screen.getByText("Lightning Network")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /deposit btc/i })).toBeEnabled();
    expect(screen.getByRole("button", { name: /deposit via lightning/i })).toBeEnabled();
  });
});
