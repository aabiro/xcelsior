import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";

/**
 * The browser half of P1's SCA recovery.
 *
 * An off-session charge refused with `authentication_required` is not failed
 * and not complete: the issuer wants the cardholder. This panel is the only
 * browser detour the plan keeps, and the properties worth testing are about
 * what it says and when it acts, not how it looks.
 *
 * * It must not charge on load. Opening a link is consent to look.
 * * It must say the money has **not** been taken, or someone reads a pending
 *   challenge as a completed payment.
 * * Success must not claim the wallet moved — the webhook decides that.
 * * It must end by pointing back at the terminal, because the plan's whole
 *   claim is that the browser is a detour rather than a destination.
 */

const handleNextAction = vi.hoisted(() => vi.fn());

const apiMocks = vi.hoisted(() => ({
  resumePendingVerification: vi.fn(),
}));

vi.mock("@/lib/api", () => apiMocks);

vi.mock("@/lib/stripe-client", () => ({
  getStripePromise: () => Promise.resolve({ handleNextAction }),
  getStripePublishableKey: () => "pk_test_probe",
}));

import { ScaResumePanel } from "@/components/billing/sca-resume-panel";

const INTENT = "pi_probe_requires_action";

beforeEach(() => {
  handleNextAction.mockReset();
  apiMocks.resumePendingVerification.mockReset();
  apiMocks.resumePendingVerification.mockResolvedValue({
    ok: true,
    resumable: true,
    client_secret: "pi_probe_secret_abc123",
    amount_cad: 25,
    description: "Wallet top-up",
    message: "Confirm this payment with your bank to complete it.",
  });
  handleNextAction.mockResolvedValue({});
});

describe("ScaResumePanel", () => {
  it("does not confirm anything until the person asks", async () => {
    render(<ScaResumePanel intentId={INTENT} />);
    // Give any stray effect a chance to fire before asserting absence.
    await waitFor(() => expect(screen.getByRole("button")).toBeTruthy());
    expect(apiMocks.resumePendingVerification).not.toHaveBeenCalled();
    expect(handleNextAction).not.toHaveBeenCalled();
  });

  it("says plainly that the card has not been charged", () => {
    render(
      <ScaResumePanel
        intentId={INTENT}
        known={{
          stripe_intent_id: INTENT,
          amount_cad: 25,
          description: "Wallet top-up",
          created_at: 0,
          resume_url: "",
        }}
      />,
    );
    expect(screen.getByText(/not charged/i)).toBeTruthy();
    expect(screen.getByText(/\$25\.00 CAD/)).toBeTruthy();
  });

  it("fetches the secret only on the explicit action, then confirms with the bank", async () => {
    render(<ScaResumePanel intentId={INTENT} />);
    fireEvent.click(screen.getByRole("button", { name: /verify with my bank/i }));

    await waitFor(() =>
      expect(apiMocks.resumePendingVerification).toHaveBeenCalledWith(INTENT),
    );
    await waitFor(() =>
      expect(handleNextAction).toHaveBeenCalledWith({
        clientSecret: "pi_probe_secret_abc123",
      }),
    );
  });

  it("ends by sending the person back to the terminal", async () => {
    render(<ScaResumePanel intentId={INTENT} />);
    fireEvent.click(screen.getByRole("button", { name: /verify with my bank/i }));

    await waitFor(() => expect(screen.getByText(/verification complete/i)).toBeTruthy());
    expect(screen.getByText(/return to your terminal/i)).toBeTruthy();
  });

  it("does not claim the balance moved, because the webhook decides that", async () => {
    render(<ScaResumePanel intentId={INTENT} />);
    fireEvent.click(screen.getByRole("button", { name: /verify with my bank/i }));

    await waitFor(() => expect(screen.getByText(/verification complete/i)).toBeTruthy());
    // "credited" / "added to your balance" would be a claim we cannot make here.
    expect(screen.getByText(/once the payment\s+processor confirms/i)).toBeTruthy();
  });

  it("surfaces a refusal instead of implying success", async () => {
    apiMocks.resumePendingVerification.mockResolvedValue({
      ok: true,
      resumable: false,
      client_secret: "",
      amount_cad: 25,
      description: "Wallet top-up",
      message: "This payment is not waiting on verification.",
    });

    render(<ScaResumePanel intentId={INTENT} />);
    fireEvent.click(screen.getByRole("button", { name: /verify with my bank/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    expect(screen.queryByText(/verification complete/i)).toBeNull();
    expect(handleNextAction).not.toHaveBeenCalled();
  });

  it("reports a declined challenge rather than swallowing it", async () => {
    handleNextAction.mockResolvedValue({
      error: { message: "Your bank declined the verification." },
    });

    render(<ScaResumePanel intentId={INTENT} />);
    fireEvent.click(screen.getByRole("button", { name: /verify with my bank/i }));

    await waitFor(() => expect(screen.getByRole("alert")).toBeTruthy());
    expect(screen.getByText(/bank declined/i)).toBeTruthy();
    expect(screen.queryByText(/verification complete/i)).toBeNull();
  });
});
