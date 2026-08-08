import { describe, expect, it } from "vitest";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import {
  isOperatorTool,
  principalIsOperator,
  scopesForProfile,
  toolIsVisible,
  toolsInProfile,
} from "../../src/tools/profiles.js";

const CUSTOMER_SCOPES = ["instances:read", "instances:write", "billing:read", "gpu:read"];
const OPERATOR_SCOPES = [...CUSTOMER_SCOPES, "hosts:operate", "control_plane:read"];

describe("trust-surface profiles", () => {
  it("keeps every platform-global tool out of the customer profile", () => {
    const customer = toolsInProfile("customer");
    for (const name of [
      "drain_host", "undrain_host", "evict_host_workloads", "retry_agent_command",
      "get_scheduler_health", "get_host_capacity", "list_reconciliation_findings",
    ]) {
      expect(customer, `${name} must not appear in a public directory listing`)
        .not.toContain(name);
    }
  });

  it("keeps the customer's own action-plan status tool in the customer profile", () => {
    // It reads a plan owned by the calling principal. It was previously
    // classified operator because `hosts:read` appears among its alternative
    // scopes, which would have removed "did my launch get approved?" from the
    // public surface.
    expect(toolsInProfile("customer")).toContain("get_mcp_action_status");
    expect(isOperatorTool("get_mcp_action_status")).toBe(false);
  });

  it("gives the operator profile everything the customer profile has", () => {
    const operator = new Set(toolsInProfile("operator"));
    for (const name of toolsInProfile("customer")) expect(operator).toContain(name);
    expect(operator.size).toBeGreaterThan(toolsInProfile("customer").length);
    // Everything except the optional company-knowledge tools, which are only
    // registered when the deployment opts in.
    expect(operator.size).toBe(Object.keys(TOOL_CONTRACTS).length - 2);
  });

  it("excludes the optional company-knowledge tools unless enabled", () => {
    // The profile has to describe what the server actually registers, or the
    // snapshot, the advertised scopes, and the E2E all describe a surface
    // nobody serves.
    expect(toolsInProfile("customer")).not.toContain("search");
    expect(toolsInProfile("customer")).not.toContain("fetch");
    const withKnowledge = toolsInProfile("customer", { companyKnowledge: true });
    expect(withKnowledge).toContain("search");
    expect(withKnowledge).toContain("fetch");
  });

  it("hides operator tools from a token without operator scopes, even on the operator host", () => {
    expect(toolIsVisible("drain_host", "operator", OPERATOR_SCOPES)).toBe(true);
    expect(toolIsVisible("drain_host", "operator", CUSTOMER_SCOPES)).toBe(false);
    expect(toolIsVisible("drain_host", "customer", OPERATOR_SCOPES)).toBe(false);
  });

  it("never reads unresolved scopes as operator authority", () => {
    expect(principalIsOperator(undefined)).toBe(false);
    expect(principalIsOperator([])).toBe(false);
    expect(toolIsVisible("evict_host_workloads", "operator", undefined)).toBe(false);
  });

  it("advertises only the profile's own scopes, never the blanket api scope", () => {
    const customer = scopesForProfile("customer");
    expect(customer).not.toContain("api");
    expect(customer).not.toContain("hosts:evict");
    expect(customer).not.toContain("control_plane:operate");
    expect(customer).toContain("instances:read");
    expect(scopesForProfile("operator")).toContain("hosts:evict");
  });

  it("refuses to expose a tool with no contract", () => {
    expect(toolIsVisible("not_a_real_tool", "operator", OPERATOR_SCOPES)).toBe(false);
  });
});

describe("annotation accuracy", () => {
  it("marks exactly the tools that read state we do not control as open-world", () => {
    const openWorld = Object.entries(TOOL_CONTRACTS)
      .filter(([, contract]) => contract.annotations.openWorldHint)
      .map(([name]) => name)
      .sort();
    expect(openWorld).toEqual([
      // Live third-party marketplace inventory and pricing.
      "get_spot_prices", "list_available_gpus", "search_marketplace",
      // Company knowledge indexes the published docs site, which changes
      // without a deploy on our side.
      "fetch", "search",
    ].sort());
  });

  it("marks exactly the non-undoable tools as destructive", () => {
    const destructive = Object.entries(TOOL_CONTRACTS)
      .filter(([, contract]) => contract.annotations.destructiveHint)
      .map(([name]) => name)
      .sort();
    // drain_host is absent deliberately: the versioned endpoint it calls stops
    // new placements and leaves running workloads alone. See contracts.ts.
    // `delete_volume` joins them: a deleted volume's contents cannot be
    // recovered. `detach_volume` deliberately does not — it is confirm-gated
    // because it disrupts a running job, but re-attaching restores the mount,
    // and `destructiveHint` is a claim about undoability rather than about
    // needing care. See the reasoning in contracts.ts.
    // The serverless exits join them. Deleting an endpoint stops its id
    // resolving — a replacement is a new deployment, not the same one back.
    // Cancelling an inference job ends it; there is no resume, only a new job,
    // which is the same shape as cancel_instance.
    expect(destructive).toEqual([
      "cancel_instance", "cancel_serverless_job", "delete_serverless_endpoint",
      "delete_volume", "evict_host_workloads", "terminate_instance",
    ]);
  });

  it("never marks a tool both read-only and destructive", () => {
    for (const [name, contract] of Object.entries(TOOL_CONTRACTS)) {
      if (contract.annotations.readOnlyHint) {
        expect(contract.annotations.destructiveHint, name).toBe(false);
        expect(contract.idempotency, name).toBe("read");
      }
    }
  });

  it("never marks a destructive tool as safely retryable", () => {
    for (const [name, contract] of Object.entries(TOOL_CONTRACTS)) {
      if (contract.annotations.destructiveHint) expect(contract.retry, name).not.toBe("safe");
    }
  });
});
