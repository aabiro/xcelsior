import type { ReactNode } from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ProviderProfileForm, SellerProfileCard } from "@/components/providers/seller-profile-card";
import en from "@/lib/i18n/en";

const apiMocks = vi.hoisted(() => ({
  saveSellerProfile: vi.fn(),
  apiFetch: vi.fn(),
}));

const authState = vi.hoisted(() => ({
  user: {
    name: "Ada Lovelace",
    address_line1: "1 Test Street",
    address_line2: "Unit 2",
    city: "Toronto",
    province: "ON",
    postal_code: "M5V 2T6",
    country: "CA",
  } as Record<string, string> | null,
}));

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => <a href={href}>{children}</a>,
}));

vi.mock("sonner", () => ({ toast: { success: vi.fn() } }));

vi.mock("@/lib/locale", () => ({
  useLocale: () => ({ t: (key: string) => key }),
}));

vi.mock("@/lib/auth", () => ({
  useAuth: () => ({ user: authState.user }),
}));

vi.mock("@/lib/api", () => apiMocks);

const provider = { legal_name: "Ada Lovelace", tax_id_kind: "", gst_hst_number: "" };

describe("SellerProfileCard", () => {
  beforeEach(() => {
    apiMocks.saveSellerProfile.mockReset();
    apiMocks.saveSellerProfile.mockResolvedValue({ ok: true });
    authState.user = {
      name: "Ada Lovelace",
      address_line1: "1 Test Street",
      address_line2: "Unit 2",
      city: "Toronto",
      province: "ON",
      postal_code: "M5V 2T6",
      country: "CA",
    };
  });

  it("keeps the account address, drops the tax checkbox, and continues when the form is filled", async () => {
    const onSaved = vi.fn();
    const onEditAddress = vi.fn();
    const user = userEvent.setup();
    render(
      <SellerProfileCard mode="onboarding" provider={provider} onSaved={onSaved} onEditAddress={onEditAddress} />,
    );

    expect(screen.queryByRole("checkbox")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("dash.earnings.seller_address")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Search address")).not.toBeInTheDocument();
    expect(screen.getByText("1 Test Street, Unit 2 · Toronto ON M5V 2T6 · CA")).toBeInTheDocument();
    const continueButton = screen.getByRole("button", { name: "dash.earnings.seller_continue" });
    expect(continueButton).toBeEnabled();
    expect(screen.getByRole("link", { name: "dash.earnings.seller_terms" })).toHaveAttribute("href", "/terms#provider-terms");

    await user.click(continueButton);
    expect(apiMocks.saveSellerProfile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("dash.earnings.seller_kind_error");

    await user.click(screen.getByText("dash.earnings.seller_tax_sin"));
    await user.type(screen.getByLabelText("dash.earnings.seller_tax_id"), "130692544");
    await user.click(screen.getByRole("button", { name: "dash.earnings.seller_continue" }));

    expect(apiMocks.saveSellerProfile).toHaveBeenCalledWith({
      legal_name: "Ada Lovelace",
      address_line1: "1 Test Street",
      address_line2: "Unit 2",
      city: "Toronto",
      province: "ON",
      postal_code: "M5V 2T6",
      country: "CA",
      tax_id_kind: "sin",
      tax_id: "130692544",
      gst_hst_number: "",
      accepted_terms: true,
    });
    expect(onSaved).toHaveBeenCalledOnce();

    await user.click(screen.getByRole("button", { name: "dash.earnings.seller_address_edit" }));
    expect(onEditAddress).toHaveBeenCalledOnce();
  });

  it("says what is missing when the account has no address", async () => {
    authState.user = { name: "Ada Lovelace" };
    const user = userEvent.setup();
    render(<SellerProfileCard mode="onboarding" provider={provider} onSaved={vi.fn()} />);
    await user.click(screen.getByText("dash.earnings.seller_tax_bn"));
    await user.type(screen.getByLabelText("dash.earnings.seller_tax_id"), "123456789");
    await user.click(screen.getByRole("button", { name: "dash.earnings.seller_continue" }));
    expect(apiMocks.saveSellerProfile).not.toHaveBeenCalled();
    expect(screen.getByRole("alert")).toHaveTextContent("dash.earnings.seller_address_missing");
  });

  it("does not lecture about the CRA or personal tax responsibility", () => {
    expect(en["dash.earnings.seller_desc"]).not.toMatch(/CRA|ARC/i);
    expect(en["dash.earnings.seller_gst_hint"]).not.toMatch(/own taxes/i);
    expect(en["dash.earnings.seller_terms"]).not.toMatch(/checkbox|I am an independent/i);
    expect(en["dash.earnings.seller_desc"]).toBe("Legal name and tax identifier for payouts.");
    expect(en["dash.earnings.seller_tax_gst"]).toBe("GST/HST number");
    expect(en["dash.earnings.seller_tax_accepted"]).toMatch(/GST\/HST/);
  });

  it("accepts a GST/HST number as the tax identifier and hides the extra field", async () => {
    const user = userEvent.setup();
    render(<SellerProfileCard mode="onboarding" provider={provider} onSaved={vi.fn()} />);
    await user.click(screen.getByText("dash.earnings.seller_tax_gst"));
    expect(screen.queryByText("dash.earnings.seller_gst_hint")).not.toBeInTheDocument();
    await user.type(screen.getByLabelText("dash.earnings.seller_tax_id"), "123456789RT0001");
    await user.click(screen.getByRole("button", { name: "dash.earnings.seller_continue" }));
    expect(apiMocks.saveSellerProfile).toHaveBeenCalledWith(expect.objectContaining({
      tax_id_kind: "gst",
      tax_id: "123456789RT0001",
      gst_hst_number: "123456789RT0001",
    }));
  });

  it("leaves legal name empty when the account name is only a display name", async () => {
    authState.user = {
      name: "Probe Placeholder",
      business_name: "Probe Co",
      address_line1: "1 Test Street",
      city: "Toronto",
      province: "ON",
      postal_code: "M5V 2T6",
      country: "CA",
    };
    apiMocks.apiFetch.mockResolvedValue({ profile: { legal_name: "", tax_id_kind: "" } });
    render(<ProviderProfileForm mode="onboarding" onSaved={vi.fn()} />);
    const input = await screen.findByLabelText("dash.earnings.seller_legal_name");
    expect(input).toHaveValue("");
    expect(input).toHaveAttribute("autocomplete", "off");
  });

  it("keeps the settings save button disabled until a field changes and disables it again after saving", async () => {
    const user = userEvent.setup();
    render(
      <SellerProfileCard
        mode="settings"
        provider={{ legal_name: "Ada Lovelace", tax_id_kind: "sin", tax_id_last4: "2544", gst_hst_number: "" }}
        onSaved={vi.fn()}
      />,
    );
    const saveBtn = screen.getByRole("button", { name: "dash.earnings.seller_save" });
    expect(saveBtn).toBeDisabled();

    await user.type(screen.getByLabelText("dash.earnings.seller_gst"), "123456789RT0001");
    expect(saveBtn).toBeEnabled();

    await user.click(saveBtn);
    expect(apiMocks.saveSellerProfile).toHaveBeenCalledOnce();
    expect(saveBtn).toBeDisabled();
  });
});
