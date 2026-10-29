import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const confirmPayment = vi.hoisted(() => vi.fn());
const authState = vi.hoisted(() => ({ user: null as { is_admin?: boolean } | null }));

const apiMocks = vi.hoisted(() => ({
  createPaymentIntent: vi.fn(),
  fetchWallet: vi.fn(),
  depositWallet: vi.fn(),
  checkPayPalEnabled: vi.fn(),
  createPayPalOrder: vi.fn(),
  capturePayPalOrder: vi.fn(),
  adminCreditWallet: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);

vi.mock("@/lib/auth", () => ({
  useAuth: () => authState,
}));

vi.mock("sonner", () => ({
  toast: {
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
  },
}));

vi.mock("@stripe/stripe-js", () => ({
  loadStripe: vi.fn(() => Promise.resolve({})),
}));

vi.mock("@stripe/react-stripe-js", () => ({
  Elements: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PaymentElement: ({ onChange }: { onChange?: (event: { complete: boolean }) => void }) => (
    <button type="button" onClick={() => onChange?.({ complete: true })}>
      Mock payment element
    </button>
  ),
  useStripe: () => ({ confirmPayment }),
  useElements: () => ({}),
}));

vi.mock("@/lib/stripe-client", () => ({
  getStripePublishableKey: () =>
    process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY ||
    process.env.NEXT_PUBLIC_STRIPE_LIVE_PUBLISHABLE_KEY ||
    process.env.NEXT_PUBLIC_STRIPE_SANDBOX_PUBLISHABLE_KEY ||
    undefined,
  getStripePromise: () => {
    const key =
      process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY ||
      process.env.NEXT_PUBLIC_STRIPE_LIVE_PUBLISHABLE_KEY ||
      process.env.NEXT_PUBLIC_STRIPE_SANDBOX_PUBLISHABLE_KEY;
    if (!key) return null;
    return Promise.resolve({} as import("@stripe/stripe-js").Stripe);
  },
}));

import { DepositModal } from "@/components/billing/deposit-modal";

describe("DepositModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    authState.user = null;
    vi.stubEnv("NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY", "pk_test_123");
    vi.stubEnv("NEXT_PUBLIC_STRIPE_LIVE_PUBLISHABLE_KEY", "");
    vi.stubEnv("NEXT_PUBLIC_STRIPE_SANDBOX_PUBLISHABLE_KEY", "");
    apiMocks.checkPayPalEnabled.mockResolvedValue({ enabled: true });
    apiMocks.fetchWallet.mockResolvedValue({
      ok: true,
      wallet: { balance_cad: 0 },
    });
    apiMocks.createPaymentIntent.mockResolvedValue({
      ok: true,
      intent: { client_secret: "pi_test_secret" },
    });
    confirmPayment.mockResolvedValue({
      error: null,
      paymentIntent: { status: "succeeded" },
    });
  });

  it(
    "shows embedded payment element and exposes PayPal as a secondary option",
    async () => {
      render(
        <DepositModal
          customerId="cust-1"
          onClose={() => {}}
          onSuccess={() => {}}
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: "$25" }));
      fireEvent.click(screen.getByRole("button", { name: /continue/i }));

      await waitFor(
        () => {
          expect(screen.getByText("Secure payment")).toBeInTheDocument();
          expect(apiMocks.createPaymentIntent).toHaveBeenCalled();
          expect(apiMocks.checkPayPalEnabled).toHaveBeenCalled();
        },
        { timeout: 10000 },
      );

      expect(screen.getByText("Or use PayPal")).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /pay with paypal/i })).toBeInTheDocument();
    },
    15000,
  );

  it("polls wallet after Stripe success instead of calling depositWallet", async () => {
    apiMocks.fetchWallet
      .mockResolvedValueOnce({ ok: true, wallet: { balance_cad: 10 } })
      .mockResolvedValueOnce({ ok: true, wallet: { balance_cad: 35 } });

    const onSuccess = vi.fn();
    render(
      <DepositModal
        customerId="cust-1"
        onClose={() => {}}
        onSuccess={onSuccess}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "$25" }));
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(screen.getByText("Mock payment element")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /mock payment element/i }));
    fireEvent.click(screen.getByRole("button", { name: /pay \$25\.00 cad/i }));

    await waitFor(() => {
      expect(confirmPayment).toHaveBeenCalled();
      expect(apiMocks.depositWallet).not.toHaveBeenCalled();
      expect(apiMocks.fetchWallet.mock.calls.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("shows unavailable state when Stripe is missing and PayPal is disabled", async () => {
    vi.stubEnv("NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY", "");
    vi.stubEnv("NEXT_PUBLIC_STRIPE_LIVE_PUBLISHABLE_KEY", "");
    vi.stubEnv("NEXT_PUBLIC_STRIPE_SANDBOX_PUBLISHABLE_KEY", "");
    apiMocks.checkPayPalEnabled.mockResolvedValue({ enabled: false });

    render(
      <DepositModal
        customerId="cust-1"
        onClose={() => {}}
        onSuccess={() => {}}
      />,
    );

    await waitFor(() => {
      expect(screen.getByText(/payments unavailable/i)).toBeInTheDocument();
      expect(screen.getByText(/contact support/i)).toBeInTheDocument();
    });
    expect(apiMocks.depositWallet).not.toHaveBeenCalled();
  });

  it("starts a platform admin on the admin credit, not a card payment", async () => {
    authState.user = { is_admin: true };
    apiMocks.adminCreditWallet.mockResolvedValue({ ok: true, tx_id: "tx-1", balance_cad: 50, amount_cad: 50 });
    const onSuccess = vi.fn();

    render(<DepositModal customerId="cust-1" onClose={() => {}} onSuccess={onSuccess} />);

    expect(screen.getByRole("tab", { name: "Admin credit" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Credit without payment")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText(/reason/i), { target: { value: "launch test" } });
    fireEvent.click(screen.getByRole("button", { name: /credit \$50\.00/i }));

    await waitFor(() => {
      expect(apiMocks.adminCreditWallet).toHaveBeenCalledWith("cust-1", 50, "launch test", expect.any(String));
      expect(onSuccess).toHaveBeenCalledWith(50);
    });
    expect(apiMocks.createPaymentIntent).not.toHaveBeenCalled();
    expect(confirmPayment).not.toHaveBeenCalled();
  });

  it("lets a platform admin credit with the reason left blank", async () => {
    // The reason was required with no sign that it was, so the Credit button
    // stayed disabled and admins were locked out of topping up.
    authState.user = { is_admin: true };
    apiMocks.adminCreditWallet.mockResolvedValue({ ok: true, tx_id: "tx-2", balance_cad: 25, amount_cad: 25 });
    render(<DepositModal customerId="cust-1" onClose={() => {}} onSuccess={() => {}} />);

    fireEvent.click(screen.getByRole("button", { name: "$25" }));
    const credit = screen.getByRole("button", { name: /credit \$25\.00/i });
    expect(credit).toBeEnabled();
    fireEvent.click(credit);

    await waitFor(() => {
      expect(apiMocks.adminCreditWallet).toHaveBeenCalledWith("cust-1", 25, "Platform admin top-up", expect.any(String));
    });
  });

  it("explains a reason that is too short instead of silently disabling", () => {
    authState.user = { is_admin: true };
    render(<DepositModal customerId="cust-1" onClose={() => {}} onSuccess={() => {}} />);
    fireEvent.change(screen.getByLabelText(/reason/i), { target: { value: "ok" } });
    expect(screen.getByText(/at least 3 characters, or leave it blank/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /credit \$50\.00/i })).toBeDisabled();
  });

  it("warns a platform admin that Real payment charges a card", () => {
    authState.user = { is_admin: true };

    render(<DepositModal customerId="cust-1" onClose={() => {}} onSuccess={() => {}} />);
    fireEvent.click(screen.getByRole("tab", { name: "Real payment" }));

    expect(screen.getByText(/this charges a real card/i)).toBeInTheDocument();
    expect(screen.getByText("Choose deposit amount")).toBeInTheDocument();
  });

  it("gives a regular user the card payment with no admin switch", () => {
    authState.user = { is_admin: false };

    render(<DepositModal customerId="cust-1" onClose={() => {}} onSuccess={() => {}} />);

    expect(screen.queryByRole("tab", { name: "Admin credit" })).not.toBeInTheDocument();
    expect(screen.getByText("Choose deposit amount")).toBeInTheDocument();
  });
});
