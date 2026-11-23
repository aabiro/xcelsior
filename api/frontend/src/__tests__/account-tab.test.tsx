import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { User } from "@/lib/auth";

// Account type and the business profile were blended into the Profile tab,
// with a "Change account type" button that only scrolled back up to the same
// select (owner review, 2026-11-22). They have their own Account tab. GPU
// hosting and payout details are on /dashboard/hosting, not in Settings: the
// owner wants account and hosting kept separate (2026-11-23).

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("@/components/ui/province-selector", () => ({
  AccountProfileForm: ({ settings }: { settings?: boolean }) => (
    <select data-testid="account-type" data-settings={settings ? "yes" : "no"} />
  ),
}));
vi.mock("@/components/settings/business-profile-section", () => ({
  BusinessProfileSection: () => <section data-testid="business-profile" />,
}));

import { AccountTab } from "@/components/settings/account-tab";

function user(overrides: Partial<User> = {}): User {
  return { user_id: "u-1", email: "renter@example.test", role: "user", account_type: "individual", ...overrides } as User;
}

const onUpdated = vi.fn(async () => {});

describe("AccountTab", () => {
  it("holds the single account-type switch", () => {
    render(<AccountTab user={user()} onUpdated={onUpdated} />);
    expect(screen.getAllByTestId("account-type")).toHaveLength(1);
    expect(screen.getByTestId("account-type").dataset.settings).toBe("yes");
  });

  it("shows the business profile only to a business account", () => {
    const personal = render(<AccountTab user={user()} onUpdated={onUpdated} />);
    expect(screen.queryByTestId("business-profile")).toBeNull();
    personal.unmount();
    render(<AccountTab user={user({ account_type: "business" })} onUpdated={onUpdated} />);
    expect(screen.getByTestId("business-profile")).toBeInTheDocument();
  });

  it("has no hosting or payout section, even for a host", () => {
    render(<AccountTab user={user({ role: "provider", provider_id: "p-1", account_type: "business" })} onUpdated={onUpdated} />);
    expect(document.getElementById("account")).not.toBeNull();
    expect(document.getElementById("hosting")).toBeNull();
    expect(screen.queryByText(/hosting/i)).toBeNull();
  });
});
