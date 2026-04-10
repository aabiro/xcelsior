import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

const apiMocks = vi.hoisted(() => ({
  createPaymentIntent: vi.fn(),
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
  loadStripe: vi.fn(() => ({})),
}));

vi.mock("@stripe/react-stripe-js", () => ({
  Elements: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  CardElement: ({ onChange }: { onChange?: (event: { complete: boolean }) => void }) => (
    <button type="button" onClick={() => onChange?.({ complete: true })}>
      Mock card element
    </button>
  ),
  useStripe: () => ({ confirmCardPayment: vi.fn() }),
  useElements: () => ({ getElement: vi.fn(() => ({})) }),
}));

import { DepositModal } from "@/components/billing/deposit-modal";

describe("DepositModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv("NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY", "pk_test_123");
    apiMocks.checkPayPalEnabled.mockResolvedValue({ enabled: true });
  });

  it("shows card checkout first and exposes PayPal as a secondary option", async () => {
    render(
      <DepositModal
        customerId="cust-1"
        onClose={() => {}}
        onSuccess={() => {}}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "$25" }));
    fireEvent.click(screen.getByRole("button", { name: /continue/i }));

    await waitFor(() => {
      expect(apiMocks.checkPayPalEnabled).toHaveBeenCalled();
    });

    expect(screen.getByText("Card details")).toBeInTheDocument();
    expect(screen.getByText("Or use PayPal")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /pay with paypal/i })).toBeInTheDocument();
    expect(screen.queryByText(/stripe/i)).not.toBeInTheDocument();
  });
});
