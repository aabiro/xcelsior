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

export interface ToolRegistrationOptions {
  /**
   * Company-knowledge `search`/`fetch`. Optional and off by default so the
   * base plugin submission is never waiting on it (adoption plan X1.11).
   */
  companyKnowledge?: KnowledgeSources | false;
}


/**
 * Registers every tool on `server`.
 *
 * Annotations are **not** applied here. `installToolAudit` wraps
 * `registerTool` and sets `annotations: { ...contract.annotations }` for every
 * tool, and `createMcpServer` installs it before calling this — so the contract
 * already wins on every server that is actually built. A second mechanism here
 * was written and removed: it duplicated a decision that already had one home,
 * which is the drift this package has been removing all week.
 *
 * The consequence worth knowing: calling this function **without**
 * `installToolAudit` yields tools with whatever annotations the call sites
 * happened to pass, which for most of them is none. That is not a production
 * path — it is only reachable from a test — and
 * `tests/unit/annotations-reach-the-wire.test.ts` exercises the composed order
 * instead, because that is the one that ships.
 */
export function registerAllTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
  options: ToolRegistrationOptions = {},
): void {
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
