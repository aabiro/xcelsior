import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import { PayPalConnectCard } from "@/components/providers/paypal-connect-card";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("@/lib/api", () => ({
  startPayPalProviderOnboard: vi.fn(),
  refreshPayPalProvider: vi.fn(),
}));

describe("PayPalConnectCard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders connect CTA and benefits when not started", () => {
    render(
      <PayPalConnectCard
        providerId="prov-1"
        paypal={{ enabled: true, status: "not_started" }}
        platformPayPalEnabled
      />,
    );
    expect(screen.getByRole("button", { name: "dash.earnings.paypal_connect" })).toBeInTheDocument();
    expect(screen.getByText("dash.earnings.paypal_wallet_clarity")).toBeInTheDocument();
    expect(screen.getByText("dash.earnings.paypal_benefit_instant")).toBeInTheDocument();
    expect(screen.getByText("dash.earnings.paypal_benefit_dual")).toBeInTheDocument();
  });

  it("renders connected state with badge", () => {
    render(
      <PayPalConnectCard
        providerId="prov-1"
        paypal={{ enabled: true, status: "active", onboarded_at: 1700000000 }}
        platformPayPalEnabled
      />,
    );
    expect(screen.getByText("dash.earnings.paypal_ready")).toBeInTheDocument();
    expect(screen.getByText("dash.earnings.paypal_connected")).toBeInTheDocument();
  });

  it("renders onboarding resume and check status", () => {
    render(
      <PayPalConnectCard
        providerId="prov-1"
        paypal={{ enabled: true, status: "onboarding" }}
        platformPayPalEnabled
      />,
    );
    expect(screen.getAllByText("dash.earnings.paypal_resume").length).toBeGreaterThan(0);
    expect(screen.getByText("dash.earnings.paypal_check_status")).toBeInTheDocument();
  });

  it("renders unavailable when platform PayPal disabled", () => {
    render(
      <PayPalConnectCard
        providerId="prov-1"
        paypal={{ enabled: false, status: "not_started" }}
        platformPayPalEnabled={false}
      />,
    );
    expect(screen.getByText("dash.earnings.paypal_unavailable")).toBeInTheDocument();
  });
});