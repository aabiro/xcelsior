import { describe, expect, it } from "vitest";
import { IRREVERSIBLE_TOOLS, TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import {
  CONNECTOR_WITHHELD_SCOPES,
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
      "list_hpc_clusters", "list_hpc_jobs", "get_hpc_job", "get_hpc_job_output",
      "submit_hpc_job", "cancel_hpc_job",
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

  it("never advertises a host or control-plane scope in the customer profile", () => {
    // `get_mcp_action_status` lists `hosts:read` as one alternative; that must
    // not reach the public metadata, which every connector's consent reads.
    const leaked = scopesForProfile("customer").filter((s) => s.startsWith("hosts:") || s.startsWith("control_plane:"));
    expect(leaked).toEqual([]);
    expect(scopesForProfile("operator")).toContain("hosts:read");
  });

  it("never advertises a scope a connector cannot be granted in the customer profile", () => {
    // A connector requests everything in `scopes_supported`; these three are
    // dropped from every connector grant, so advertising them only asks the
    // user to approve something that never arrives. Their tools still exist.
    const customer = scopesForProfile("customer");
    for (const scope of CONNECTOR_WITHHELD_SCOPES) expect(customer).not.toContain(scope);
    expect(customer).toContain("ssh:read");
    expect(toolsInProfile("customer")).toContain("top_up_wallet");
    // Topping up is `billing:topup`, which a connector may hold, so it is
    // advertised; `billing:write`, which it may not, is not.
    expect(customer).toContain("billing:topup");
    expect(customer).not.toContain("billing:write");
  });

  it("advertises optional account identity without changing tool visibility", () => {
    expect(scopesForProfile("customer")).toContain("profile");
    expect(scopesForProfile("customer")).toContain("email");
    expect(toolIsVisible("list_available_gpus", "customer", [])).toBe(true);
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
      // Reads `/api/v2/gpu/available` and `/api/v2/marketplace/spot-prices` —
      // the two feeds behind the first two above — and then **spends money
      // against the answer**. Added 2026-08-14; it had been missed because the
      // rationale in contracts.ts said "everything else reads our own
      // records", and a summary of a list is not a check of it.
      "schedule_under_budget",
      // Company knowledge indexes the published docs site, which changes
      // without a deploy on our side.
      "fetch", "search",
    ].sort());
  });

  it("flags every customer write destructive, and operator writes only when irreversible", () => {
    // The Connectors Directory requires destructiveHint on any tool that
    // modifies or deletes data, so on the public surface every write carries
    // it. Operator tools are never published there and keep the narrower
    // reading: drain_host stops new placements and leaves running work alone,
    // so it is not flagged. See the note above TOOL_POLICY in contracts.ts.
    const wrong: string[] = [];
    for (const [name, c] of Object.entries(TOOL_CONTRACTS)) {
      if (c.annotations.readOnlyHint) continue;
      const want = name === "request_permission" ? false : c.tenantClass === "tenant" ? true : IRREVERSIBLE_TOOLS.has(name as never);
      if (c.annotations.destructiveHint !== want) wrong.push(`${name}: ${c.annotations.destructiveHint}`);
    }
    expect(wrong).toEqual([]);
  });

  it("keeps every irreversible tool flagged destructive", () => {
    for (const name of IRREVERSIBLE_TOOLS) {
      expect(TOOL_CONTRACTS[name].annotations.destructiveHint, name).toBe(true);
    }
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
