import { describe, it, expect, vi, beforeEach } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { DashboardNav } from "@/components/nav/dashboard-nav";
import { NAV_GROUPS_KEY, resetSidebarState } from "@/lib/sidebar-state";

const navigation = vi.hoisted(() => ({ pathname: "/dashboard" }));

vi.mock("next/navigation", () => ({
  usePathname: () => navigation.pathname,
}));

const nav = () => (
  <DashboardNav collapsed={false} mobile={false} showServerless canAccessRole={() => false} t={(key) => key} mode="rent" />
);
const header = (key: string) => screen.getByRole("button", { name: key });

describe("dashboard nav groups", () => {
  beforeEach(() => {
    localStorage.clear();
    navigation.pathname = "/dashboard";
  });

  it("starts with every group closed and none highlighted", () => {
    render(nav());
    for (const button of screen.getAllByRole("button")) {
      expect(button).toHaveAttribute("aria-expanded", "false");
      expect(button.className).not.toContain("text-accent-cyan");
    }
  });

  it("highlights a header exactly when its group is open", () => {
    render(nav());
    fireEvent.click(header("dash.nav.storage"));
    expect(header("dash.nav.storage")).toHaveAttribute("aria-expanded", "true");
    expect(header("dash.nav.storage").className).toContain("text-accent-cyan");
    fireEvent.click(header("dash.nav.storage"));
    expect(header("dash.nav.storage")).toHaveAttribute("aria-expanded", "false");
    expect(header("dash.nav.storage").className).not.toContain("text-accent-cyan");
  });

  it("opens the current page's group on arrival and lets its header close it", async () => {
    navigation.pathname = "/dashboard/billing";
    render(nav());
    expect(header("dash.nav.business")).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(header("dash.nav.business"));
    expect(header("dash.nav.business")).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(screen.queryByRole("link", { name: "dash.billing" })).toBeNull());
  });

  it("keeps the arrival group and the groups you opened open as you move between pages", () => {
    navigation.pathname = "/dashboard/billing";
    const { rerender } = render(nav());
    fireEvent.click(header("dash.nav.storage"));
    navigation.pathname = "/dashboard/templates";
    rerender(nav());
    for (const key of ["dash.nav.compute", "dash.nav.storage", "dash.nav.business"]) {
      expect(header(key)).toHaveAttribute("aria-expanded", "true");
    }
    expect(header("dash.nav.insights")).toHaveAttribute("aria-expanded", "false");
  });

  it("forgets open groups and every account's Rent/Host choice on sign-out", () => {
    localStorage.setItem(NAV_GROUPS_KEY, JSON.stringify({ storage: true }));
    localStorage.setItem("xcelsior-workspace-mode:u1", "host");
    localStorage.setItem("xcelsior-workspace-mode:u2", "host");
    localStorage.setItem("xcelsior-sidebar-collapsed", "true");
    resetSidebarState();
    expect(localStorage.getItem(NAV_GROUPS_KEY)).toBeNull();
    expect(localStorage.getItem("xcelsior-workspace-mode:u1")).toBeNull();
    expect(localStorage.getItem("xcelsior-workspace-mode:u2")).toBeNull();
    expect(localStorage.getItem("xcelsior-sidebar-collapsed")).toBe("true");
    render(nav());
    expect(header("dash.nav.storage")).toHaveAttribute("aria-expanded", "false");
  });
});
