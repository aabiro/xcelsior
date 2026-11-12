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


it("puts address search ahead of country and lets a personal account skip business details", async () => {
  authState.user = { name: "Original Name", country: "", province: "", account_type: "", onboarding_intent: "" };
  const { container } = render(<AccountProfileForm step="address" />);
  const search = await screen.findByRole("combobox", { name: "Search address" });
  const country = screen.getByLabelText("Country");
  expect(search.compareDocumentPosition(country) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  expect(search).toBeEnabled();
  expect(container).toHaveTextContent("Search for your address, or enter it below.");
  expect(container).not.toHaveTextContent("Select your country first");
});


it("searches for an address before a country is chosen", async () => {
  vi.useFakeTimers();
  try {
    apiMocks.apiFetch.mockImplementation((path: string) => {
      if (String(path).includes("search")) return Promise.resolve({ suggestions: [] });
      return Promise.resolve({ countries: [], subdivisions: [], requires_subdivision: false, required_address_fields: [] });
    });
    render(<AddressLookup country="" onSelect={vi.fn()} />);
    const input = screen.getByRole("combobox", { name: "Search address" });
    expect(input).toBeEnabled();
    fireEvent.change(input, { target: { value: "123 Main" } });
    await act(async () => { await vi.advanceTimersByTimeAsync(310); });
    const searchCall = apiMocks.apiFetch.mock.calls.find((call) => String(call[0]).includes("search"));
    expect(searchCall).toBeTruthy();
    expect(JSON.parse(searchCall![1].body).country).toBe("");
  } finally { vi.useRealTimers(); }
});


it("account setup saves the account type without a renter or provider choice", async () => {
  authState.user = { name: "Original Name", account_type: "", onboarding_intent: "provide" };
  apiMocks.updateProfile.mockResolvedValue({ ok: true });
  render(<AccountProfileForm step="account" />);
  const continueButton = screen.getByRole("button", { name: "Continue" });
  expect(continueButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Account type"), { target: { value: "individual" } });
  expect(continueButton).toBeEnabled();
  fireEvent.click(continueButton);
  await waitFor(() => expect(apiMocks.updateProfile).toHaveBeenCalled());
  expect(apiMocks.updateProfile.mock.calls[0][0]).not.toHaveProperty("onboarding_intent");
  expect(apiMocks.updateProfile.mock.calls[0][0]).toMatchObject({ account_type: "individual", business_name: "", gst_hst_number: "" });
  expect(continueButton).toBeDisabled();
});


it("says a renter or provider choice can change in settings and asks for business details only for a business account", async () => {
  authState.user = { name: "Original Name", account_type: "", onboarding_intent: "" };
  const { container } = render(<AccountProfileForm step="account" />);
  const continueButton = screen.getByRole("button", { name: "Continue" });
  expect(container.querySelector("form")).toHaveAttribute("novalidate");
  expect(continueButton).toBeDisabled();
  expect(screen.queryByText("How will you use Xcelsior?")).not.toBeInTheDocument();
  expect(screen.queryByRole("radio")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Legal business name")).not.toBeInTheDocument();
  // Bare focus/blur without editing does not mark Account type invalid or enable Continue.
  fireEvent.blur(screen.getByLabelText("Account type"));
  expect(screen.getByLabelText("Account type")).toHaveAttribute("aria-invalid", "false");
  expect(screen.queryByText("Select an account type.")).not.toBeInTheDocument();
  expect(continueButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Account type"), { target: { value: "individual" } });
  expect(continueButton).toBeEnabled();
  expect(screen.getByText("A personal account does not need a business name, even if you provide GPUs.")).toBeInTheDocument();
  expect(screen.queryByLabelText("Legal business name")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Account type"), { target: { value: "business" } });
  expect(continueButton).toBeDisabled();
  const businessInput = screen.getByLabelText("Legal business name");
  expect(businessInput).toBeInTheDocument();
  expect(businessInput).toHaveAttribute("aria-invalid", "false");
  // Bare blur without editing does not flag Legal business name invalid.
  fireEvent.blur(businessInput);
  expect(businessInput).toHaveAttribute("aria-invalid", "false");
  expect(screen.queryByText("Enter your legal business name.")).not.toBeInTheDocument();
  // Editing and clearing it back to empty, then blurring, shows the inline error.
  fireEvent.change(businessInput, { target: { value: "P" } });
  fireEvent.change(businessInput, { target: { value: "" } });
  fireEvent.blur(businessInput);
  expect(businessInput).toHaveAttribute("aria-invalid", "true");
  expect(screen.getByText("Enter your legal business name.")).toBeInTheDocument();
  fireEvent.change(businessInput, { target: { value: "Port Dover Marine Inc." } });
  expect(businessInput).toHaveAttribute("aria-invalid", "false");
  expect(screen.queryByText("Enter your legal business name.")).not.toBeInTheDocument();
  expect(continueButton).toBeEnabled();
  expect(screen.getByLabelText("GST/HST registration number (optional)")).toBeInTheDocument();
  expect(screen.queryByText("A personal account does not need a business name, even if you provide GPUs.")).not.toBeInTheDocument();
});


it("keeps Continue disabled on pre-filled onboarding steps until a field is actually edited", async () => {
  authState.user = { name: "Original Name", account_type: "individual", country: "CA", province: "ON", address_line1: "123 Harbour St", address_line2: "", city: "Port Dover", postal_code: "N0A 1N0" };
  const accountView = render(<AccountProfileForm step="account" />);
  const accountContinue = screen.getByRole("button", { name: "Continue" });
  expect(accountContinue).toBeDisabled();
  fireEvent.focus(screen.getByLabelText("Account type"));
  fireEvent.blur(screen.getByLabelText("Account type"));
  expect(accountContinue).toBeDisabled();
  accountView.unmount();

  render(<AccountProfileForm step="address" />);
  await screen.findByRole("option", { name: "Ontario" });
  const addressContinue = screen.getByRole("button", { name: "Continue" });
  expect(addressContinue).toBeDisabled();
  fireEvent.change(screen.getByLabelText("City or locality"), { target: { value: "Simcoe" } });
  expect(addressContinue).toBeEnabled();
  fireEvent.change(screen.getByLabelText("City or locality"), { target: { value: "Port Dover" } });
  expect(addressContinue).toBeDisabled();
});


it("disables Continue on the address step until all required address fields are filled", async () => {
  authState.user = { name: "Original Name", country: "CA", province: "", address_line1: "", city: "", postal_code: "", account_type: "individual" };
  render(<AccountProfileForm step="address" />);
  const continueButton = screen.getByRole("button", { name: "Continue" });
  await screen.findByRole("option", { name: "Ontario" });
  expect(continueButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("State, province, or territory"), { target: { value: "ON" } });
  expect(continueButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Street address"), { target: { value: "123 Harbour St" } });
  expect(continueButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("City or locality"), { target: { value: "Port Dover" } });
  expect(continueButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Postal code"), { target: { value: "N0A 1N0" } });
  expect(continueButton).toBeEnabled();
});


it("tells someone already in settings that they can switch here", async () => {
  authState.user = { name: "Original Name", account_type: "individual", onboarding_intent: "provide", country: "CA", province: "ON" };
  render(<AccountProfileForm settings />);
  expect(screen.getByText("You can switch between renting and providing here anytime. Payout setup stays on Earnings.")).toBeInTheDocument();
  expect(screen.getByText("A personal account does not need a business name, even if you provide GPUs.")).toBeInTheDocument();
  expect(await screen.findByRole("option", { name: "Canada" })).toBeInTheDocument();
});


it("saving profile details preserves a separately edited name and clears hidden business data", async () => {
  authState.user = { name:"Original Name", country:"CA", province:"ON", address_line1:"1 Test Street", address_line2:"", city:"London", postal_code:"N6A 1A1", account_type:"individual", onboarding_intent:"rent", business_name:"Old Company", gst_hst_number:"123456789RT0001" };
  apiMocks.updateProfile.mockResolvedValue({ok:true});
  const view=render(<AccountProfileForm settings />);
  await screen.findByRole("option", { name: "Ontario" });
  const saveButton = screen.getByRole("button", {name:"Save profile"});
  expect(saveButton).toBeDisabled();
  fireEvent.change(screen.getByLabelText("Apartment, suite, or unit (optional)"), { target: { value: "Suite 200" } });
  await waitFor(() => expect(saveButton).toBeEnabled());
  authState.user = {...authState.user, name:"Updated Name"};
  view.rerender(<AccountProfileForm settings />);
  fireEvent.click(saveButton);
  await screen.findByText("Profile saved.");
  expect(saveButton).toBeDisabled();
  const saved=apiMocks.updateProfile.mock.calls[0][0];
  expect(saved).not.toHaveProperty("name");
  expect(saved).toMatchObject({country:"CA",province:"ON", address_line2:"Suite 200", business_name:"", gst_hst_number:""});
});


it("fills the address from the picked suggestion without a second request, with no watermark", async () => {
  vi.useFakeTimers();
  try {
    const address = {address_line1:"300 Dufferin Avenue", address_line2:"", city:"London", postal_code:"N6B 2P8", country:"CA", province:"ON"};
    apiMocks.apiFetch.mockResolvedValue({suggestions:[{id:"photon-0", text:"300 Dufferin Avenue", description:"London, ON, N6B 2P8, Canada", address}]});
    const onSelect=vi.fn();
    const { container } = render(<AddressLookup country="CA" onSelect={onSelect} />);
    fireEvent.change(screen.getByRole("combobox",{name:"Search address"}),{target:{value:"300 Duff"}});
    await act(async () => { await vi.advanceTimersByTimeAsync(260); });
    const listbox = screen.getByRole("listbox", { name: "Address suggestions" });
    expect(listbox.className).toContain("bg-[var(--popover-solid)]");
    expect(container.textContent).not.toMatch(/OpenStreetMap|©/);
    // The typed words are bold, as in Google Maps.
    expect(Array.from(listbox.querySelectorAll("strong"), (node) => node.textContent)).toEqual(["300", "Duff"]);
    const sent = JSON.parse(String(apiMocks.apiFetch.mock.calls[0][1].body));
    expect(sent).toEqual({country:"CA", text:"300 Duff", lang:"en"});
    fireEvent.click(screen.getByRole("option",{name:/300 Dufferin Avenue/}));
    expect(onSelect).toHaveBeenCalledWith(address);
    expect(apiMocks.apiFetch).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
  } finally { vi.useRealTimers(); }
});


it("keeps the last suggestions on screen while the next ones load", async () => {
  vi.useFakeTimers();
  try {
    let deliver!: (value: unknown) => void;
    const first = {suggestions:[{id:"a", text:"300 Dufferin Avenue", description:"London", address:{}}]};
    apiMocks.apiFetch.mockResolvedValueOnce(first).mockImplementationOnce(() => new Promise((resolve) => { deliver = resolve; }));
    render(<AddressLookup country="CA" onSelect={vi.fn()} />);
    const input = screen.getByRole("combobox",{name:"Search address"});
    fireEvent.change(input,{target:{value:"300 Duf"}});
    await act(async () => { await vi.advanceTimersByTimeAsync(260); });
    fireEvent.change(input,{target:{value:"300 Dufferin Av"}});
    await act(async () => { await vi.advanceTimersByTimeAsync(260); });
    expect(screen.getByRole("option",{name:/300 Dufferin Avenue/})).toBeInTheDocument();  // still showing
    await act(async () => { deliver({suggestions:[{id:"b", text:"300 Dufferin Avenue West", description:"London", address:{}}]}); });
    expect(screen.getByRole("option",{name:/Avenue West/})).toBeInTheDocument();
  } finally { vi.useRealTimers(); }
});


it("treats a search with no match as fine and points to the fields below", async () => {
  vi.useFakeTimers();
  try {
    apiMocks.apiFetch.mockResolvedValue({suggestions:[]});
    render(<AddressLookup country="CA" onSelect={vi.fn()} />);
    fireEvent.change(screen.getByRole("combobox",{name:"Search address"}),{target:{value:"1234 Fairview Rd"}});
    await act(async () => { await vi.advanceTimersByTimeAsync(310); });
    expect(screen.getByText(/No matching address found/)).toBeInTheDocument();
  } finally { vi.useRealTimers(); }
});


function withAddressCheck(review: Record<string, unknown>) {
  const catalog = apiMocks.apiFetch.getMockImplementation()!;
  apiMocks.apiFetch.mockImplementation(async (path: string, init?: RequestInit) =>
    path === "/api/locations/check" ? review : catalog(path, init));
}

async function typeAddress(line1: string, city: string, postal: string) {
  authState.user = { name:"Original Name", country:"CA", province:"ON", address_line1:"", city:"", postal_code:"", account_type:"individual" };
  render(<AccountProfileForm step="address" />);
  await screen.findByRole("option", { name: "Ontario" });
  fireEvent.change(screen.getByLabelText("Street address"), { target: { value: line1 } });
  fireEvent.change(screen.getByLabelText("City or locality"), { target: { value: city } });
  fireEvent.change(screen.getByLabelText("Postal code"), { target: { value: postal } });
  fireEvent.click(screen.getByRole("button", { name: "Continue" }));
}

it("shows the address back in postal format, with any warning, and saves only once it is confirmed", async () => {
  const cleaned = {country:"CA", province:"ON", address_line1:"4915 50 St", address_line2:"", city:"Lloydminster", postal_code:"S9V 0A1"};
  withAddressCheck({
    address: cleaned, errors: [],
    warnings: [{field:"postal_code", code:"postal_code_region", message:"S9V 0A1 is usually a postal code in Saskatchewan, not Ontario. Check it, or keep it if your address is right."}],
    lines: ["4915 50 St", "Lloydminster ON S9V 0A1", "Canada"],
  });
  apiMocks.updateProfile.mockResolvedValue({ok:true});
  await typeAddress("4915 50 St,", "Lloydminster", "s9v0a1");
  expect(await screen.findByText("Is this address right?")).toBeInTheDocument();
  expect(screen.getByText("Lloydminster ON S9V 0A1")).toBeInTheDocument();
  expect(screen.getByText(/usually a postal code in Saskatchewan/)).toBeInTheDocument();
  expect(apiMocks.updateProfile).not.toHaveBeenCalled();
  const sent = apiMocks.apiFetch.mock.calls.find((call) => call[0] === "/api/locations/check");
  expect(JSON.parse(String(sent![1].body))).toMatchObject({address_line1:"4915 50 St,", postal_code:"s9v0a1"});
  fireEvent.click(screen.getByRole("button", { name: "Yes, continue" }));
  await waitFor(() => expect(apiMocks.updateProfile).toHaveBeenCalled());
  expect(apiMocks.updateProfile.mock.calls[0][0]).toMatchObject(cleaned);  // stored as shown, not as typed
});


it("lets someone go back and edit instead of saving", async () => {
  withAddressCheck({
    address: {country:"CA", province:"ON", address_line1:"1 Test St", address_line2:"", city:"London", postal_code:"N6A 1A1"},
    errors: [], warnings: [], lines: ["1 Test St", "London ON N6A 1A1", "Canada"],
  });
  await typeAddress("1 Test St", "London", "n6a1a1");
  fireEvent.click(await screen.findByRole("button", { name: "Edit address" }));
  expect(screen.queryByText("Is this address right?")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Continue" })).toBeInTheDocument();
  expect(apiMocks.updateProfile).not.toHaveBeenCalled();
});


it("stops an address that cannot be valid and says why", async () => {
  withAddressCheck({
    address: {country:"CA", province:"ON", address_line1:"1 Test St", address_line2:"", city:"London", postal_code:"12345"},
    errors: [{field:"postal_code", message:"12345 isn't a valid postal code in Canada."}], warnings: [], lines: [],
  });
  await typeAddress("1 Test St", "London", "12345");
  expect(await screen.findByRole("alert")).toHaveTextContent("12345 isn't a valid postal code in Canada.");
  expect(screen.queryByText("Is this address right?")).not.toBeInTheDocument();
  expect(apiMocks.updateProfile).not.toHaveBeenCalled();
});


it("offers every country's regions, requires one only where the post does, and disables only an empty list", async () => {
  authState.user = { name:"Original Name", country:"GB", province:"", account_type:"individual" };
  apiMocks.apiFetch.mockImplementation(async (path: string) => {
    const country = new URLSearchParams(path.split("?")[1] ?? "").get("country");
    const catalogs: Record<string, object> = {
      GB: {subdivisions:[{code:"ENG", name:"England", region:"gb-eng"}, {code:"SCT", name:"Scotland", region:"gb-sct"}], requires_subdivision:false,
        subdivision_mode:"optional", labels:{subdivision:"Region", city:"Town or city", postal_code:"Postal code"}},
      US: {subdivisions:[{code:"NY", name:"New York", region:"us-ny"}], requires_subdivision:true,
        subdivision_mode:"required", labels:{subdivision:"State", city:"City", postal_code:"ZIP code"}},
      HK: {subdivisions:[], requires_subdivision:false, subdivision_mode:"none", labels:{subdivision:"Area", city:"District", postal_code:"Postal code"}},
    };
    return {
      countries: [{code:"GB", name:"United Kingdom"}, {code:"US", name:"United States"}, {code:"HK", name:"Hong Kong"}],
      ...(country ? catalogs[country] : {subdivisions: [], requires_subdivision: false}),
      required_address_fields: ["address_line1", "city", "postal_code"],
    };
  });
  render(<AccountProfileForm step="address" />);
  const region = await screen.findByLabelText("Region (optional)");
  expect(region).not.toBeRequired();
  expect(region).toBeEnabled();
  expect(await screen.findByRole("option", { name: "Scotland" })).toBeInTheDocument();
  expect(screen.getByLabelText("Town or city")).toBeInTheDocument();
  expect(apiMocks.apiFetch.mock.calls.some((call) => String(call[0]).includes("country=GB&purpose=address"))).toBe(true);
  fireEvent.change(screen.getByLabelText("Country"), { target: { value: "US" } });
  expect(await screen.findByLabelText("State")).toBeRequired();
  expect(screen.getByLabelText("ZIP code")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Country"), { target: { value: "HK" } });
  await waitFor(() => expect(screen.getByLabelText("Area")).toBeDisabled());
  expect(screen.getByRole("option", { name: "Not applicable" })).toBeInTheDocument();
});
