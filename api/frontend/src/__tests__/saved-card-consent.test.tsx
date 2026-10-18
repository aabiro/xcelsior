import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";

/**
 * What the Add card modal says about when a saved card is charged.
 *
 * Stripe's line for an off-session SetupIntent ("you allow Xcelsior to charge
 * your card for future payments") read as a standing authorisation even with
 * auto-reload off. With it off, a saved card is charged only when the customer
 * adds credits; the auto-reload sweep selects only wallets that turned it on.
 */

const elementProps = vi.hoisted(() => ({ last: null as null | Record<string, unknown> }));

vi.mock("@stripe/react-stripe-js", () => ({
  Elements: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  PaymentElement: (props: Record<string, unknown>) => {
    elementProps.last = props;
    return <div data-testid="payment-element" />;
  },
  useStripe: () => null,
  useElements: () => null,
}));
vi.mock("@/lib/stripe-client", () => ({ getStripePromise: () => Promise.resolve({}) }));
vi.mock("@/lib/api", () => ({
  createSetupIntent: vi.fn().mockResolvedValue({ client_secret: "seti_test_secret" }),
}));
vi.mock("posthog-js", () => ({ default: { capture: vi.fn(), captureException: vi.fn() } }));
vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

import { PaymentMethodModal, savedCardConsent } from "@/components/billing/payment-method-modal";

beforeEach(() => {
  elementProps.last = null;
});

describe("saved card consent", () => {
  it("promises no charge beyond the customer's own top-ups when auto-reload is off", () => {
    const text = savedCardConsent({ enabled: false, amount_cad: 25, threshold_cad: 5 });
    expect(text).toBe("It is charged only when you choose to add credits.");
    expect(savedCardConsent(undefined)).toBe(text);
  });

  it("states the auto-reload amount and threshold when it is on", () => {
    expect(savedCardConsent({ enabled: true, amount_cad: 25, threshold_cad: 5 })).toBe(
      "It is charged when you add credits, and by auto-reload: $25.00 whenever your balance falls below $5.00.",
    );
  });

  it("the modal shows that sentence, says it is secure, and turns off Stripe's generic line", async () => {
    render(
      <PaymentMethodModal
        onClose={() => {}}
        onSuccess={() => {}}
        autoReload={{ enabled: false, amount_cad: 25, threshold_cad: 5 }}
      />,
    );
    const consent = await screen.findByTestId("saved-card-consent");
    expect(consent.textContent).toContain("saved securely");
    expect(consent.textContent).toContain("only when you choose to add credits");
    expect(document.body.textContent).not.toMatch(/future payments|Embedded checkout|automatic top-ups/i);
    expect((elementProps.last?.options as { terms?: { card?: string } }).terms?.card).toBe("never");
  });
});
