import { describe, it, expect } from "vitest";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../../src/client/api.js";
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
 * The annotations a client actually receives must be the ones we decided on.
 *
 * `tool-surface.json` is built in `describeToolSurface`, which reads
 * `contract.annotations` and **discards whatever `registerTool` was passed**.
 * So the manifest was correct by construction while the live server published
 * whatever each call site happened to type — and nothing compared the two.
 *
 * When this test was written, **38 of 60 tools reached the wire with no
 * annotations object at all**, including five of the six destructive ones.
 * Under the MCP defaults (`readOnlyHint` false, `destructiveHint` **true**,
 * `openWorldHint` **true**) that advertises every pure read — `get_instance`,
 * `get_wallet_balance`, `list_available_gpus` — as a destructive, open-world
 * call, which is the opposite of what the manifest promised a reviewer.
 *
 * The registry inversion made the policy table authoritative for the manifest.
 * This is what makes it authoritative for the runtime.
 */

const HINTS = ["readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"] as const;

function recordRegistrations(): Map<string, Record<string, unknown>> {
  const recorded = new Map<string, Record<string, unknown>>();
  const recorder = {
    registerTool(name: string, config: Record<string, unknown>) {
      recorded.set(name, config);
      return undefined;
    },
  };
  registerAllTools(
    recorder as unknown as McpServer,
    {} as XcelsiorApiClient,
    undefined,
    { companyKnowledge: {} as never },
  );
  return recorded;
}

describe("tool annotations reach the wire", () => {
  it("registers every tool in the contract registry", () => {
    // Calibration. An empty recording agrees with everything.
    const recorded = recordRegistrations();
    expect(recorded.size).toBe(Object.keys(TOOL_CONTRACTS).length);
    expect(recorded.size).toBeGreaterThan(40);
  });

  it("gives every registered tool an annotations object", () => {
    const missing = [...recordRegistrations().entries()]
      .filter(([, config]) => !config.annotations)
      .map(([name]) => name)
      .sort();
    expect(
      missing,
      "these tools reach the client with no annotations, so MCP's defaults " +
        "apply: not read-only, destructive, open world",
    ).toEqual([]);
  });

  it("publishes exactly the annotations the contract declares", () => {
    const drift: string[] = [];
    for (const [name, config] of recordRegistrations()) {
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
      "the running server advertises something other than the policy table. " +
        "tool-surface.json is generated from the contract and would not show this.",
    ).toEqual([]);
  });

  it("marks every destructive tool destructive on the wire", () => {
    // Named separately because this is the one a model acts on. It would be
    // covered by the check above, but a failure there reads as a drift list;
    // this one names the actual danger.
    const recorded = recordRegistrations();
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
    const recorded = recordRegistrations();
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
      "read-only tools that default to destructive for any spec-compliant client",
    ).toEqual([]);
  });

  it("has no call-site literal that contradicts the contract", () => {
    // The wrapper makes a wrong literal harmless, which is not the same as
    // making it fine: it records an author's belief about a tool that
    // disagrees with the policy table, and the next reader believes the
    // literal. Registrars are called **directly** here, bypassing the
    // wrapper, so this sees what was actually typed.
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
      if (!literal) continue; // absent is the wrapper's job, asserted above
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
      "a registration hardcodes an annotation that disagrees with TOOL_POLICY. " +
        "The wrapper overrides it, so nothing breaks — but one of the two is " +
        "wrong and the literal is what the next reader will believe.",
    ).toEqual([]);
  });
});
