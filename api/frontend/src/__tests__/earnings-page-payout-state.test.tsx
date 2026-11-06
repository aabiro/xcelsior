import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";

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
  apiFetch: vi.fn(),
  saveSellerProfile: vi.fn(),
  updateProfile: vi.fn(),
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

import { AccountProfileForm, AddressLookup, LocationFields } from "@/components/ui/province-selector";
import EarningsPage from "@/app/(dashboard)/dashboard/earnings/page";

const ZERO_EARNINGS = {
  earnings: { total_jobs: 0, total_earned_cad: 0, total_platform_cad: 0, total_tax_cad: 0 },
  recent_payouts: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  apiMocks.apiFetch.mockImplementation(async (path: string) => ({
    countries: [{code: "CA", name: "Canada"}, {code: "US", name: "United States"}],
    subdivisions: path.includes("country=CA") ? [{code: "ON", name: "Ontario", region: "ca-on"}] : [],
    requires_subdivision: path.includes("country=CA"),
    required_address_fields: ["address_line1", "city", "postal_code"],
  }));
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
    expect(screen.queryByLabelText("Search address")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Country")).not.toBeInTheDocument();
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


describe("required account location", () => {
  it("recovers an invalid old country and displays names rather than codes", async () => {
    const onCountryChange=vi.fn(); const onProvinceChange=vi.fn();
    render(<LocationFields country="old-region" province="" onCountryChange={onCountryChange} onProvinceChange={onProvinceChange} />);
    const country=await screen.findByRole("option", {name:"Canada"});
    expect(country).toHaveValue("CA");
    const select=screen.getByLabelText("Country");
    expect(select).toBeRequired();
    expect(select).not.toBeDisabled();
    expect(select).toHaveValue("");
    expect((select as HTMLSelectElement).checkValidity()).toBe(false);
    fireEvent.change(select,{target:{value:"CA"}});
    expect(onCountryChange).toHaveBeenCalledWith("CA");
    expect(onProvinceChange).toHaveBeenCalledWith("");
  });

  it("requires the province while data loads and when the saved value is invalid", async () => {
    render(<LocationFields country="CA" province="wrong" onCountryChange={vi.fn()} onProvinceChange={vi.fn()} />);
    const province=screen.getByLabelText("State, province, or territory");
    expect(province).toBeRequired();
    expect(province).not.toBeDisabled();
    await screen.findByRole("option", {name:"Ontario"});
    expect(province).toHaveValue("");
    expect((province as HTMLSelectElement).checkValidity()).toBe(false);
  });
});


it("saving profile details preserves a separately edited name and clears hidden business data", async () => {
  authState.user = { name:"Original Name", country:"CA", province:"ON", address_line1:"1 Test Street", address_line2:"", city:"London", postal_code:"N6A 1A1", account_type:"individual", onboarding_intent:"rent", business_name:"Old Company", gst_hst_number:"123456789RT0001" };
  apiMocks.updateProfile.mockResolvedValue({ok:true});
  const view=render(<AccountProfileForm settings />);
  await waitFor(() => expect(screen.getByRole("button", {name:"Save profile"})).toBeEnabled());
  authState.user = {...authState.user, name:"Updated Name"};
  view.rerender(<AccountProfileForm settings />);
  fireEvent.click(screen.getByRole("button", {name:"Save profile"}));
  await screen.findByText("Profile saved.");
  const saved=apiMocks.updateProfile.mock.calls[0][0];
  expect(saved).not.toHaveProperty("name");
  expect(saved).toMatchObject({country:"CA",province:"ON", business_name:"", gst_hst_number:""});
});


it.each([false,true])("address retrieval survives tabbing but respects manual edits: manual=%s", async (manualEdit) => {
  vi.useFakeTimers();
  try {
    let deliver!: (value: unknown) => void;
    const selected = {address_line1:"1 Test Street", address_line2:"", city:"London", postal_code:"N6A 1A1", country:"CA", province:"ON"};
    apiMocks.apiFetch.mockImplementation((path: string) => path.endsWith("search")
      ? Promise.resolve({suggestions:[{id:"address-one", text:"1 Test Street", description:"London", is_address:true}]})
      : new Promise((resolve) => { deliver=resolve; }));
    const onSelect=vi.fn();
    const view=render(<AddressLookup country="CA" onSelect={onSelect} />);
    const input=screen.getByRole("combobox",{name:"Search address"});
    fireEvent.change(input,{target:{value:"Test Street"}});
    await act(async () => { await vi.advanceTimersByTimeAsync(310); });
    fireEvent.click(screen.getByRole("option",{name:/1 Test Street/}));
    fireEvent.blur(input);
    if(manualEdit) view.rerender(<AddressLookup country="CA" onSelect={onSelect} editRevision={1} />);
    await act(async () => { deliver({address:selected}); });
    if(manualEdit) expect(onSelect).not.toHaveBeenCalled();
    else expect(onSelect).toHaveBeenCalledWith(selected);
  } finally { vi.useRealTimers(); }
});
