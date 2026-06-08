import { describe, expect, it, vi } from "vitest";

const switchActiveTeam = vi.hoisted(() =>
  vi.fn().mockResolvedValue({ active_team_id: "team-xyz" }),
);

vi.mock("@/lib/api", () => ({
  switchActiveTeam,
}));

import {
  applyActiveTeamSwitch,
  getBillingCustomerId,
  getTeamContext,
  formatTeamRoleLabel,
} from "@/lib/team-context";

describe("team-context", () => {
  it("uses billing_customer_id when present", () => {
    expect(
      getBillingCustomerId({
        billing_customer_id: "team-wallet",
        customer_id: "personal",
        user_id: "user-1",
      }),
    ).toBe("team-wallet");
  });

  it("builds team context from auth user", () => {
    const ctx = getTeamContext({
      user_id: "u1",
      email: "m@example.com",
      role: "submitter",
      team_id: "team-abc",
      team_name: "Acme GPU",
      team_role: "viewer",
      team_plan: "pro",
      billing_customer_id: "owner-wallet",
      team_can_manage_billing: false,
      team_can_write_instances: false,
    });
    expect(ctx.isTeamMember).toBe(true);
    expect(ctx.teamName).toBe("Acme GPU");
    expect(ctx.billingCustomerId).toBe("owner-wallet");
    expect(ctx.canManageBilling).toBe(false);
    expect(ctx.canWriteInstances).toBe(false);
  });

  it("formats role labels", () => {
    expect(formatTeamRoleLabel("admin")).toBe("Admin");
    expect(formatTeamRoleLabel("viewer")).toBe("Viewer");
  });

  it("personal workspace when team_id is unset", () => {
    const ctx = getTeamContext({
      user_id: "u1",
      email: "solo@example.com",
      role: "submitter",
      customer_id: "personal-wallet",
      billing_customer_id: "personal-wallet",
      team_can_manage_billing: true,
      team_can_write_instances: true,
    });
    expect(ctx.isTeamMember).toBe(false);
    expect(ctx.billingCustomerId).toBe("personal-wallet");
  });

  it("falls back to customer_id then user_id for billing scope", () => {
    expect(getBillingCustomerId({ customer_id: "cust-1", user_id: "u-1" })).toBe("cust-1");
    expect(getBillingCustomerId({ user_id: "u-1" })).toBe("u-1");
    expect(getBillingCustomerId(null)).toBe("");
  });

  it("admin team member has full write and billing flags", () => {
    const ctx = getTeamContext({
      user_id: "u1",
      email: "admin@example.com",
      role: "submitter",
      team_id: "team-1",
      team_name: "Ops",
      team_role: "admin",
      team_plan: "pro",
      billing_customer_id: "team-wallet",
      team_can_manage_billing: true,
      team_can_write_instances: true,
    });
    expect(ctx.isTeamMember).toBe(true);
    expect(ctx.teamRole).toBe("admin");
    expect(ctx.canManageBilling).toBe(true);
    expect(ctx.canWriteInstances).toBe(true);
  });

  it("dispatches xcelsior-team-changed after workspace switch", async () => {
    const refreshUser = vi.fn().mockResolvedValue(undefined);
    const dispatched: CustomEvent[] = [];
    const handler = (e: Event) => dispatched.push(e as CustomEvent);
    window.addEventListener("xcelsior-team-changed", handler);

    await applyActiveTeamSwitch("team-xyz", refreshUser);

    expect(switchActiveTeam).toHaveBeenCalledWith("team-xyz");
    expect(refreshUser).toHaveBeenCalledOnce();
    expect(dispatched).toHaveLength(1);
    expect(dispatched[0].detail).toEqual({ teamId: "team-xyz" });

    window.removeEventListener("xcelsior-team-changed", handler);
  });
});