import { describe, it, expect } from "vitest";
import {
  TOOL_SCOPES,
  satisfiesScope,
  scopeUnion,
  describeScopeRequirement,
} from "../../src/auth/scopes.js";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";

/**
 * Regression tests for the authorization defect.
 *
 * The previous check was:
 *
 *   if (userScopes.includes("api")) return true;
 *   return required.some((s) => userScopes.includes(s));
 *
 * Two independent defects: `api` short-circuited every check, and `.some()`
 * accepted any ONE of a tool's scopes. Every one of the 41 contracts also
 * listed `"api"`. Each test below was confirmed to fail against that version.
 */
describe("scope enforcement", () => {
  it("requires every scope in allOf, not just one of them", () => {
    // `schedule_under_budget` needs instances:write + gpu:read + marketplace:read.
    // Under `.some()`, a read-only token satisfied a spending write.
    expect(satisfiesScope(["gpu:read"], TOOL_SCOPES.schedule_under_budget)).toBe(false);
    expect(satisfiesScope(["marketplace:read"], TOOL_SCOPES.schedule_under_budget)).toBe(false);
    expect(
      satisfiesScope(["instances:write", "gpu:read", "marketplace:read"],
        TOOL_SCOPES.schedule_under_budget),
    ).toBe(true);
  });

  it("does not let a read scope authorize a spending write", () => {
    expect(satisfiesScope(["billing:read"], TOOL_SCOPES.run_training_job)).toBe(false);
    expect(satisfiesScope(["instances:write", "billing:read"], TOOL_SCOPES.run_training_job))
      .toBe(true);
  });

  it("has no wildcard grant at all", () => {
    // `api` short-circuited every check. It is removed from the scope
    // vocabulary rather than narrowed, so no value means "everything".
    for (const tool of Object.keys(TOOL_SCOPES)) {
      expect(satisfiesScope(["api"], TOOL_SCOPES[tool]), tool).toBe(false);
    }
    expect(satisfiesScope(["hosts:evict"], TOOL_SCOPES.evict_host_workloads)).toBe(true);
    expect(satisfiesScope(["instances:read"], TOOL_SCOPES.list_instances)).toBe(true);
  });

  it("treats anyOf as genuine alternatives", () => {
    // An action plan may concern an instance, an endpoint, or a host.
    expect(satisfiesScope(["instances:read"], TOOL_SCOPES.get_mcp_action_status)).toBe(true);
    expect(satisfiesScope(["inference:read"], TOOL_SCOPES.get_mcp_action_status)).toBe(true);
    expect(satisfiesScope(["billing:read"], TOOL_SCOPES.get_mcp_action_status)).toBe(false);
  });

  it("fails closed on an unknown tool or empty requirement", () => {
    // Call sites used `TOOL_SCOPES[tool] || ["api"]`, so an unregistered tool
    // required only the broad grant — fail-open exactly where it must not be.
    expect(satisfiesScope(["api"], TOOL_SCOPES["no_such_tool"])).toBe(false);
    expect(satisfiesScope(["instances:write"], {})).toBe(false);
    expect(satisfiesScope([], TOOL_SCOPES.list_instances)).toBe(false);
    expect(satisfiesScope(undefined, TOOL_SCOPES.list_instances)).toBe(false);
  });

  it("no contract lists api as a requirement", () => {
    for (const [name, requirement] of Object.entries(TOOL_SCOPES)) {
      expect(scopeUnion(requirement), name).not.toContain("api");
      expect(scopeUnion(requirement).length, `${name} has an empty requirement`)
        .toBeGreaterThan(0);
    }
  });

  it("exposes the requirement on contracts, with the flat union as metadata only", () => {
    const contract = TOOL_CONTRACTS.run_training_job;
    expect(contract.scopeRequirement.allOf).toEqual(["instances:write", "billing:read"]);
    expect([...contract.requiredScopes].sort()).toEqual(["billing:read", "instances:write"]);
  });

  it("describes requirements accurately enough to act on", () => {
    expect(describeScopeRequirement(TOOL_SCOPES.run_training_job))
      .toBe("all of: instances:write, billing:read");
    expect(describeScopeRequirement(TOOL_SCOPES.get_mcp_action_status))
      .toBe("one of: instances:read, inference:read, hosts:read");
    expect(describeScopeRequirement(undefined)).toMatch(/no contract/);
  });

  /**
   * Tools a Quick Connect token deliberately cannot reach.
   *
   * The default connector credential carries `billing:read` and not
   * `billing:write`, so it can read the funding configuration and not change
   * it. `top_up_wallet` charges a real card, and the plan of record is explicit
   * that it "is `billing:write` from the start — it charges a real card, so it
   * is never reachable by a read-scoped credential"
   * (docs/mcp-agent-native-implementation-plan.md, P0).
   *
   * Listed by name rather than skipped by scope, so adding a second
   * money-moving tool is a decision someone records here instead of an
   * exemption that widens quietly.
   */
  const NOT_REACHABLE_BY_QUICK_CONNECT = new Set([
    "top_up_wallet",
    // Changes what gets charged *unattended*. Same reasoning: the default
    // connector token holds `billing:read`, and widening automatic spending is
    // not something it should be able to do.
    "configure_auto_topup",
  ]);

  it("still admits the Quick Connect scope set for every customer tool", () => {
    // Regression guard on the fix itself: tightening allOf must not lock the
    // default connector credential out of the surface it is issued for.
    const quickConnect = [
      "instances:read", "instances:write", "instances:operate", "billing:read",
      "gpu:read", "marketplace:read", "inference:read", "inference:write", "events:read",
    ];
    for (const [name, contract] of Object.entries(TOOL_CONTRACTS)) {
      if (contract.tenantClass === "operator") continue;
      if (name === "search" || name === "fetch") continue;
      if (NOT_REACHABLE_BY_QUICK_CONNECT.has(name)) continue;
      expect(satisfiesScope(quickConnect, TOOL_SCOPES[name]), name).toBe(true);
    }
  });

  it("refuses the money-moving tools to the Quick Connect scope set", () => {
    // The other half, so the exemption above cannot become a blanket skip. A
    // tool named here must actually be out of reach — if `billing:write` were
    // added to the connector default, this fails and the decision surfaces.
    const quickConnect = [
      "instances:read", "instances:write", "instances:operate", "billing:read",
      "gpu:read", "marketplace:read", "inference:read", "inference:write", "events:read",
    ];
    for (const name of NOT_REACHABLE_BY_QUICK_CONNECT) {
      expect(satisfiesScope(quickConnect, TOOL_SCOPES[name]), name).toBe(false);
    }
  });
});
