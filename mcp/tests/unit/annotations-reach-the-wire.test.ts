import { describe, it, expect } from "vitest";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../../src/client/api.js";
import { installToolAudit } from "../../src/audit/context.js";
import { registerAllTools } from "../../src/tools/index.js";
import { registerDiscoveryTools } from "../../src/tools/discovery.js";
import { registerBillingTools } from "../../src/tools/billing.js";
import { registerComputeTools } from "../../src/tools/compute.js";
import { registerGuardrailTools } from "../../src/tools/guardrails.js";
import { registerWorkflowTools } from "../../src/tools/workflows.js";
import { registerServerlessTools } from "../../src/tools/serverless.js";
import { registerMonitoringTools } from "../../src/tools/monitoring.js";
import { registerDiagnosticTools } from "../../src/tools/diagnostics.js";
import { registerOperatorTools } from "../../src/tools/operator.js";
import { registerVolumeTools } from "../../src/tools/volumes.js";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";

/**
 * What a client actually receives must be what the policy table decided.
 *
 * ## Measure the composition, not a part of it
 *
 * This file first asserted against `registerAllTools` called with a bare
 * recorder, and reported that **38 of 60 tools shipped with no annotations**.
 * That was wrong, and wrong in an instructive way: `createMcpServer` calls
 * `installToolAudit` *before* `registerAllTools`, and the audit wrapper sets
 * `annotations: { ...contract.annotations }` on every registration. The
 * contract has always won on every server that is actually built. Registering
 * without the audit wrapper is not a production path — it is only reachable
 * from a test, and the test was the only thing reaching it.
 *
 * So the composed order is what is exercised here. It is a guard that did not
 * exist: nothing asserted that a server built the way production builds one
 * carries the contract's annotations for every tool, and the property is worth
 * pinning precisely because it depends on two components being wired in one
 * order.
 */

const HINTS = ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] as const;

/** Every scope any tool requires, so profile filtering hides nothing. */
const ALL_SCOPES = [
  ...new Set(Object.values(TOOL_CONTRACTS).flatMap((c) => [...c.requiredScopes])),
];

/** The production wiring: audit installed first, then tools registered. */
function registerAsProductionDoes(): Map<string, Record<string, unknown>> {
  const recorded = new Map<string, Record<string, unknown>>();
  const recorder = {
    registerTool(name: string, config: Record<string, unknown>) {
      recorded.set(name, config);
      return undefined;
    },
  } as unknown as McpServer;
  const user = { scopes: ALL_SCOPES } as never;
  installToolAudit(recorder, {} as XcelsiorApiClient, user, "streamable_http", "operator");
  registerAllTools(recorder, {} as XcelsiorApiClient, user, {
    companyKnowledge: {} as never,
  });
  return recorded;
}

describe("tool annotations reach the wire", () => {
  it("registers every tool in the contract registry", () => {
    // Calibration. An empty recording agrees with everything, and the first
    // version of this file drew a conclusion from a recording that was real
    // but taken from the wrong seam.
    const recorded = registerAsProductionDoes();
    expect(recorded.size).toBe(Object.keys(TOOL_CONTRACTS).length);
    expect(recorded.size).toBeGreaterThan(40);
  });

  it("gives every registered tool an annotations object", () => {
    const missing = [...registerAsProductionDoes().entries()]
      .filter(([, config]) => !config.annotations)
      .map(([name]) => name)
      .sort();
    expect(
      missing,
      "these tools would reach the client with no annotations, so MCP's " +
        "defaults apply: not read-only, destructive, open world",
    ).toEqual([]);
  });

  it("publishes exactly the annotations the contract declares", () => {
    const drift: string[] = [];
    for (const [name, config] of registerAsProductionDoes()) {
      const want = TOOL_CONTRACTS[name].annotations as Record<string, boolean>;
      const got = (config.annotations ?? {}) as Record<string, boolean>;
      for (const hint of HINTS) {
        if (got[hint] !== want[hint]) {
          drift.push(`${name}.${hint}: wire=${got[hint]} contract=${want[hint]}`);
        }
      }
    }
    expect(
      drift,
      "the server would advertise something other than the policy table",
    ).toEqual([]);
  });

  it("marks every destructive tool destructive on the wire", () => {
    // Named separately because this is the one a model acts on: it decides
    // whether a call needs confirmation.
    const recorded = registerAsProductionDoes();
    const wrong = Object.entries(TOOL_CONTRACTS)
      .filter(([, c]) => c.annotations.destructiveHint)
      .filter(([name]) => {
        const a = (recorded.get(name)?.annotations ?? {}) as Record<string, boolean>;
        return a.destructiveHint !== true;
      })
      .map(([name]) => name)
      .sort();
    expect(wrong, "irreversible tools not advertised as destructive").toEqual([]);
  });

  it("marks every read-only tool read-only on the wire", () => {
    const recorded = registerAsProductionDoes();
    const wrong = Object.entries(TOOL_CONTRACTS)
      .filter(([, c]) => c.annotations.readOnlyHint)
      .filter(([name]) => {
        const a = (recorded.get(name)?.annotations ?? {}) as Record<string, boolean>;
        return a.readOnlyHint !== true;
      })
      .map(([name]) => name)
      .sort();
    expect(
      wrong,
      "read-only tools that would default to destructive for a spec-compliant client",
    ).toEqual([]);
  });

  it("has no call-site literal that contradicts the contract", () => {
    // `assertAnnotationsMatchContract` throws on this — but only after
    // `toolIsVisible` has returned early, so a contradicting literal on an
    // **operator-only** tool is never validated while running the customer
    // profile. It would throw on the operator deployment and nowhere else.
    //
    // Registrars are called directly here, with no audit wrapper, precisely to
    // see the literals as typed rather than as overridden.
    const raw = new Map<string, Record<string, unknown>>();
    const recorder = {
      registerTool(name: string, config: Record<string, unknown>) {
        raw.set(name, config);
        return undefined;
      },
    } as unknown as McpServer;
    const client = {} as XcelsiorApiClient;
    registerDiscoveryTools(recorder, client, undefined);
    registerBillingTools(recorder, client, undefined);
    registerComputeTools(recorder, client, undefined);
    registerGuardrailTools(recorder, client, undefined);
    registerWorkflowTools(recorder, client, undefined);
    registerServerlessTools(recorder, client, undefined);
    registerMonitoringTools(recorder, client, undefined);
    registerDiagnosticTools(recorder, client, undefined);
    registerOperatorTools(recorder, client, undefined);
    registerVolumeTools(recorder, client, undefined);

    expect(raw.size, "no registrar recorded anything; this check is vacuous")
      .toBeGreaterThan(40);

    const contradictions: string[] = [];
    for (const [name, config] of raw) {
      const literal = config.annotations as Record<string, boolean> | undefined;
      if (!literal) continue;
      const want = TOOL_CONTRACTS[name]?.annotations as Record<string, boolean>;
      if (!want) continue;
      for (const hint of HINTS) {
        if (hint in literal && literal[hint] !== want[hint]) {
          contradictions.push(
            `${name}.${hint}: literal=${literal[hint]} contract=${want[hint]}`,
          );
        }
      }
    }
    expect(
      contradictions,
      "a registration hardcodes an annotation that disagrees with TOOL_POLICY",
    ).toEqual([]);
  });
});
