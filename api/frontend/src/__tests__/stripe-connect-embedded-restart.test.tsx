import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

/**
 * When Stripe no longer recognizes the connected account, the account-session
 * route answers 409 on purpose. The panel must offer a way out (restart
 * onboarding) and must not hand connect-js a rejected promise, which error
 * tracking then records as an unhandled exception.
 */

const mocks = vi.hoisted(() => ({
  createProviderAccountSession: vi.fn(),
  loadConnectAndInitialize: vi.fn(),
  capture: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ApiError: actual.ApiError, createProviderAccountSession: mocks.createProviderAccountSession };
});
vi.mock("@stripe/connect-js/pure", () => ({
  loadConnectAndInitialize: mocks.loadConnectAndInitialize,
}));
vi.mock("@/lib/stripe-client", () => ({ getStripePublishableKey: () => "pk_test_123" }));
vi.mock("posthog-js", () => ({ default: { capture: mocks.capture } }));

import { ApiError } from "@/lib/api";
import { StripeConnectEmbedded } from "@/components/providers/stripe-connect-embedded";

const STALE_MESSAGE =
  "Stripe no longer recognizes your connected account. Restart Stripe onboarding to reconnect payouts.";

beforeEach(() => {
  vi.clearAllMocks();
  mocks.loadConnectAndInitialize.mockImplementation(() => ({
    create: () => document.createElement("div"),
  }));
});

describe("StripeConnectEmbedded", () => {
  it("offers to restart onboarding on a 409 and does not start connect-js", async () => {
    mocks.createProviderAccountSession.mockRejectedValue(new ApiError(409, STALE_MESSAGE));
    const onRestart = vi.fn();
    render(
      <StripeConnectEmbedded providerId="prov-1" mode="manage" onRestartOnboarding={onRestart} />,
    );

    expect(await screen.findByText(STALE_MESSAGE)).toBeInTheDocument();
    expect(mocks.loadConnectAndInitialize).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: /restart stripe onboarding/i }));
    expect(onRestart).toHaveBeenCalledTimes(1);
    expect(mocks.capture).toHaveBeenCalledWith("stripe_onboarding_restarted", {
      source: "payouts_panel",
      mode: "manage",
    });
  });

  it("shows no restart action for other failures", async () => {
    mocks.createProviderAccountSession.mockRejectedValue(
      new ApiError(502, "Stripe could not open the payouts session. Try again in a few minutes."),
    );
    render(
      <StripeConnectEmbedded providerId="prov-1" mode="manage" onRestartOnboarding={vi.fn()} />,
    );

    expect(await screen.findByText(/could not open the payouts session/)).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /restart stripe onboarding/i })).not.toBeInTheDocument();
  });

  it("hands connect-js the prefetched secret once, then fetches fresh ones", async () => {
    mocks.createProviderAccountSession
      .mockResolvedValueOnce({ client_secret: "secret-1" })
      .mockResolvedValueOnce({ client_secret: "secret-2" });
    render(<StripeConnectEmbedded providerId="prov-1" mode="manage" />);

    await waitFor(() => expect(screen.getByText("Connect components ready")).toBeInTheDocument());
    const { fetchClientSecret } = mocks.loadConnectAndInitialize.mock.calls[0][0];
    await expect(fetchClientSecret()).resolves.toBe("secret-1");
    await expect(fetchClientSecret()).resolves.toBe("secret-2");
    expect(mocks.createProviderAccountSession).toHaveBeenCalledTimes(2);
  });

  it("shows the error box when connect.js fails to load instead of a blank panel", async () => {
    mocks.createProviderAccountSession.mockResolvedValue({ client_secret: "secret-1" });
    const create = vi.fn(() => document.createElement("div"));
    mocks.loadConnectAndInitialize.mockImplementation(() => ({
      create,
      debugInstance: () => Promise.reject(new Error("Failed to load Connect.js")),
    }));
    render(<StripeConnectEmbedded providerId="prov-1" mode="manage" />);

    expect(await screen.findByText("Failed to load Connect.js")).toBeInTheDocument();
    expect(create).not.toHaveBeenCalled();
    expect(screen.queryByText("Connect components ready")).not.toBeInTheDocument();
  });
});
