import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const confirmPayment = vi.hoisted(() => vi.fn());

const apiMocks = vi.hoisted(() => ({
  createPaymentIntent: vi.fn(),
  fetchWallet: vi.fn(),
  depositWallet: vi.fn(),
  checkPayPalEnabled: vi.fn(),
  createPayPalOrder: vi.fn(),
  capturePayPalOrder: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);

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

import { DepositModal } from "@/components/billing/deposit-modal";

describe("DepositModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv("NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY", "pk_test_123");
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
});