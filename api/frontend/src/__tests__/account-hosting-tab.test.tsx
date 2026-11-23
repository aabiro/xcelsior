import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { User } from "@/lib/auth";

// Account type, the business profile, GPU hosting and payout details were
// blended into the Profile tab, with a "Change account type" button that only
// scrolled back up to the same select (owner review, 2026-11-22). They now
// share one tab, with one account-type switch on it.

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("@/components/ui/province-selector", () => ({
  AccountProfileForm: ({ settings }: { settings?: boolean }) => (
    <select data-testid="account-type" data-settings={settings ? "yes" : "no"} />
  ),
}));
vi.mock("@/components/settings/business-profile-section", () => ({
  BusinessProfileSection: () => <section data-testid="business-profile" />,
}));
vi.mock("@/components/providers/seller-profile-card", () => ({
  ProviderProfileForm: () => <form data-testid="payout-details" />,
}));
vi.mock("@/components/hosting/start-hosting-button", () => ({
  StartHostingButton: () => <button type="button">Start hosting</button>,
}));

import { AccountHostingTab } from "@/components/settings/account-hosting-tab";

function user(overrides: Partial<User> = {}): User {
  return { user_id: "u-1", email: "renter@example.test", role: "user", account_type: "individual", ...overrides } as User;
}

const onUpdated = vi.fn(async () => {});

describe("AccountHostingTab", () => {
  it("holds the single account-type switch", () => {
    render(<AccountHostingTab user={user()} onUpdated={onUpdated} />);
    expect(screen.getAllByTestId("account-type")).toHaveLength(1);
    expect(screen.getByTestId("account-type").dataset.settings).toBe("yes");
    expect(screen.queryByText("dash.business.locked_action")).toBeNull();
  });

  it("shows the business profile only to a business account", () => {
    const personal = render(<AccountHostingTab user={user()} onUpdated={onUpdated} />);
    expect(screen.queryByTestId("business-profile")).toBeNull();
    personal.unmount();
    render(<AccountHostingTab user={user({ account_type: "business" })} onUpdated={onUpdated} />);
    expect(screen.getByTestId("business-profile")).toBeInTheDocument();
  });

  it("offers Start hosting to a renter, with no payout form yet", () => {
    render(<AccountHostingTab user={user()} onUpdated={onUpdated} />);
    expect(screen.getByRole("button", { name: "Start hosting" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /dash.settings.hosting_manage/ })).toBeNull();
    expect(screen.queryByTestId("payout-details")).toBeNull();
  });

  it("links a host to the hosting page and shows payout details", () => {
    render(<AccountHostingTab user={user({ role: "provider", provider_id: "p-1" })} onUpdated={onUpdated} />);
    expect(screen.getByRole("link", { name: /dash.settings.hosting_manage/ })).toHaveAttribute("href", "/dashboard/hosting");
    expect(screen.getByTestId("payout-details")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start hosting" })).toBeNull();
  });

  it("gives the old deep links somewhere to land", () => {
    render(<AccountHostingTab user={user({ account_type: "business", role: "provider" })} onUpdated={onUpdated} />);
    expect(document.getElementById("account")).not.toBeNull();
    expect(document.getElementById("hosting")).not.toBeNull();
  });
});
