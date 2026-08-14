import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import type { AuthUser } from "../auth/bearer.js";
import { registerDiscoveryTools } from "./discovery.js";
import { registerBillingTools } from "./billing.js";
import { registerComputeTools } from "./compute.js";
import { registerGuardrailTools } from "./guardrails.js";
import { registerWorkflowTools } from "./workflows.js";
import { registerServerlessTools } from "./serverless.js";
import { registerMonitoringTools } from "./monitoring.js";
import { registerDiagnosticTools } from "./diagnostics.js";
import { registerOperatorTools } from "./operator.js";
import { registerVolumeTools } from "./volumes.js";
import { registerKnowledgeTools, type KnowledgeSources } from "./knowledge.js";
import { TOOL_CONTRACTS } from "./contracts.js";

export interface ToolRegistrationOptions {
  /**
   * Company-knowledge `search`/`fetch`. Optional and off by default so the
   * base plugin submission is never waiting on it (adoption plan X1.11).
   */
  companyKnowledge?: KnowledgeSources | false;
}

/**
 * Wraps a server so every `registerTool` call carries the contract's
 * annotations, whatever the call site passed.
 *
 * ## Why this is not "just tidy the call sites"
 *
 * Annotations were passed by hand at each registration, and **38 of 60 tools
 * passed none at all** — including five of the six destructive ones. That was
 * invisible because `describeToolSurface` builds `tool-surface.json` from
 * `contract.annotations` and *discards* what registration passed, so the
 * published manifest was correct by construction while the live server
 * advertised something else. Every test asserted the manifest.
 *
 * Under the MCP defaults a missing block is not "unspecified" — it is
 * `readOnlyHint` false, `destructiveHint` **true**, `openWorldHint` **true**.
 * So every pure read we shipped was advertised to a spec-compliant client as a
 * destructive, open-world call.
 *
 * Fixing the fourteen call sites would fix today and not tomorrow: the next
 * tool added anywhere in `src/tools/` would still be one forgotten object away
 * from the same bug. Injecting here means a registration **cannot** publish
 * annotations that disagree with the policy table, which is what the S1
 * inversion was supposed to buy and only bought for the manifest.
 *
 * The contract is spread **last** deliberately. A literal at a call site is not
 * a local override to be respected — it is a second copy of a decision that
 * lives in `TOOL_POLICY`, and the copy loses. `tests/unit/annotations-reach-
 * the-wire.test.ts` asserts the result; a disagreeing literal is reported by
 * `test_tool_annotations_agree_with_scopes.py` rather than silently winning.
 */
function withContractAnnotations(server: McpServer): McpServer {
  return new Proxy(server, {
    get(target, prop, receiver) {
      const value = Reflect.get(target, prop, receiver);
      if (prop !== "registerTool" || typeof value !== "function") return value;
      return (name: string, config: Record<string, unknown>, ...rest: unknown[]) => {
        const contract = TOOL_CONTRACTS[name];
        // A tool with no contract is a definition error caught by
        // `test_every_registered_tool_declares_its_scope_requirement`. Pass it
        // through untouched rather than inventing annotations for it here.
        const merged = contract
          ? {
              ...config,
              annotations: {
                ...(config?.annotations as Record<string, unknown> | undefined),
                ...contract.annotations,
              },
            }
          : config;
        return (value as (...args: unknown[]) => unknown).call(target, name, merged, ...rest);
      };
    },
  });
}

export function registerAllTools(
  rawServer: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
  options: ToolRegistrationOptions = {},
): void {
  const server = withContractAnnotations(rawServer);
  registerDiscoveryTools(server, client, user);
  registerBillingTools(server, client, user);
  registerComputeTools(server, client, user);
  registerGuardrailTools(server, client, user);
  registerWorkflowTools(server, client, user);
  registerServerlessTools(server, client, user);
  registerMonitoringTools(server, client, user);
  registerDiagnosticTools(server, client, user);
  registerOperatorTools(server, client, user);
  registerVolumeTools(server, client, user);
  if (options.companyKnowledge) {
    registerKnowledgeTools(server, client, options.companyKnowledge, user);
  }
}
