import { describe, expect, it } from "vitest";
import { getBillingCustomerId, getTeamContext, formatTeamRoleLabel } from "@/lib/team-context";

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
});