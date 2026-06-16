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

export function registerAllTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  registerDiscoveryTools(server, client, user);
  registerBillingTools(server, client, user);
  registerComputeTools(server, client, user);
  registerGuardrailTools(server, client, user);
  registerWorkflowTools(server, client, user);
  registerServerlessTools(server, client, user);
  registerMonitoringTools(server, client, user);
}