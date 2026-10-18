import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

/**
 * The payout-state card belongs to a provider account.
 *
 * Production's signed-in run showed a customer who had never registered as a
 * provider a warning that Stripe "could not be reached for this account". There
 * was no account. The page probed `/api/providers/{customer_id}`, took the 404,
 * and rendered `PayoutRequirements` with nothing, which correctly reads as
 * "unknown". The question it answered was never the one being asked.
 */

const apiMocks = vi.hoisted(() => ({
  fetchProviderEarnings: vi.fn(),
  fetchProvider: vi.fn(),
  fetchGstThreshold: vi.fn(),
  checkPayPalEnabled: vi.fn(),
  abandonOnboarding: vi.fn(),
  disconnectStripeProvider: vi.fn(),
  registerProvider: vi.fn(),
}));
const authState = vi.hoisted(() => ({ user: {} as Record<string, string> }));

vi.mock("@/lib/api", () => apiMocks);
vi.mock("@/lib/auth", () => ({
  useAuth: () => ({ user: authState.user, refreshUser: vi.fn() }),
}));
vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key, locale: "en" }),
}));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
}));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn(), info: vi.fn() } }));
// Neither is under test, and each would reach for Stripe or PayPal on mount.
vi.mock("@/components/providers/stripe-connect-embedded", () => ({
  StripeConnectEmbedded: () => null,
}));
vi.mock("@/components/providers/paypal-connect-card", () => ({
  PayPalConnectCard: () => null,
}));
vi.mock("@/lib/recharts", () => {
  const Box = ({ children }: { children?: React.ReactNode }) => <div>{children}</div>;
  return {
    BarChart: Box, Bar: Box, XAxis: Box, YAxis: Box, Tooltip: Box,
    ResponsiveContainer: Box, CartesianGrid: Box,
  };
});

import EarningsPage from "@/app/(dashboard)/dashboard/earnings/page";

const ZERO_EARNINGS = {
  earnings: { total_jobs: 0, total_earned_cad: 0, total_platform_cad: 0, total_tax_cad: 0 },
  recent_payouts: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  apiMocks.fetchProviderEarnings.mockResolvedValue(ZERO_EARNINGS);
  apiMocks.fetchGstThreshold.mockResolvedValue({
    total_revenue_cad: 0, threshold_cad: 30000, must_register: false,
  });
  apiMocks.checkPayPalEnabled.mockResolvedValue({ enabled: false, platform_mode: false });
  // What the server answers for an id with no provider row.
  apiMocks.fetchProvider.mockRejectedValue(new Error("Provider not found"));
});

describe("earnings payout state", () => {
  it("does not describe a Stripe account a customer never opened", async () => {
    authState.user = { customer_id: "cust-f84ea9d6", user_id: "u-1" };
    render(<EarningsPage />);

    expect(await screen.findByText("Become a GPU Provider")).toBeInTheDocument();
    expect(screen.queryByTestId("payout-state-unknown")).not.toBeInTheDocument();
    expect(document.body.textContent).not.toMatch(/could not be checked/);
    // No provider id means no account; asking by customer id is a guaranteed 404.
    expect(apiMocks.fetchProvider).not.toHaveBeenCalled();
  });

  it("shows a provider why they are or are not being paid", async () => {
    authState.user = { customer_id: "cust-1", provider_id: "cust-1" };
    apiMocks.fetchProvider.mockResolvedValue({
      ok: true,
      provider: {
        provider_id: "cust-1",
        provider_type: "individual",
        status: "restricted",
        email: "p@example.com",
        created_at: "2026-01-01",
        payouts: {
          charges_enabled: false,
          payouts_enabled: false,
          currently_due: ["external_account"],
          past_due: [],
          disabled_reason: "requirements.past_due",
          checked_live: true,
        },
      },
    });
    render(<EarningsPage />);

    await waitFor(() => expect(screen.getByTestId("payout-state-blocked")).toBeInTheDocument());
    expect(apiMocks.fetchProvider).toHaveBeenCalledWith("cust-1");
    expect(screen.getByText("A bank account to receive payouts")).toBeInTheDocument();
  });
});
