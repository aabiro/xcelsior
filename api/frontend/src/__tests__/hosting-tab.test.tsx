import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { User } from "@/lib/auth";

// GPU hosting and payout details have their own Settings tab, separate from
// Account (owner, 2026-11-23).

vi.mock("@/lib/locale", () => ({ useLocale: () => ({ t: (key: string) => key }) }));
vi.mock("@/components/providers/seller-profile-card", () => ({
  ProviderProfileForm: () => <form data-testid="payout-details" />,
}));
vi.mock("@/components/hosting/start-hosting-button", () => ({
  StartHostingButton: () => <button type="button">Start hosting</button>,
}));

import { HostingTab } from "@/components/settings/hosting-tab";

function user(overrides: Partial<User> = {}): User {
  return { user_id: "u-1", email: "renter@example.test", role: "user", account_type: "individual", ...overrides } as User;
}

const onUpdated = vi.fn(async () => {});

describe("HostingTab", () => {
  it("offers Start hosting to a renter, with no payout form yet", () => {
    render(<HostingTab user={user()} onUpdated={onUpdated} />);
    expect(screen.getByRole("button", { name: "Start hosting" })).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: /dash.settings.hosting_manage/ })).toBeNull();
    expect(screen.queryByTestId("payout-details")).toBeNull();
  });

  it("links a host to their machines and shows payout details", () => {
    render(<HostingTab user={user({ role: "provider", provider_id: "p-1" })} onUpdated={onUpdated} />);
    expect(screen.getByRole("link", { name: /dash.settings.hosting_manage/ })).toHaveAttribute("href", "/dashboard/hosting");
    expect(screen.getByTestId("payout-details")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start hosting" })).toBeNull();
  });

  it("keeps the #hosting and #payout deep links", () => {
    render(<HostingTab user={user({ role: "provider" })} onUpdated={onUpdated} />);
    expect(document.getElementById("hosting")).not.toBeNull();
    expect(document.getElementById("payout")).not.toBeNull();
    expect(document.getElementById("account")).toBeNull();
  });
});
