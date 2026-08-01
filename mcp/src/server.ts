import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "./client/api.js";
import type { AuthUser } from "./auth/bearer.js";
import { registerAllTools, type ToolRegistrationOptions } from "./tools/index.js";
import { registerResources } from "./resources/index.js";
import { registerPlaybooks } from "./prompts/playbooks.js";
import { installToolAudit } from "./audit/context.js";
import type { ToolProfile } from "./tools/profiles.js";

const SERVER_INFO = {
  name: "xcelsior-mcp",
  version: "2.0.0",
};

export function createMcpServer(
  client: XcelsiorApiClient,
  user?: AuthUser,
  transport: "streamable_http" | "stdio" = "streamable_http",
  profile: ToolProfile = "customer",
  options: ToolRegistrationOptions = {},
): McpServer {
  const server = new McpServer(SERVER_INFO, {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {},
    },
    instructions: [
      "You are connected to Xcelsior — a distributed GPU compute marketplace where independent hosts compete on price.",
      "Two ways to run work: rent a GPU instance by the hour (on-demand or interruptible spot), or call a serverless inference endpoint billed per token with no idle cost.",
      "Always discover before launching: list_available_gpus, get_spot_prices, and search_marketplace return live competing rates. Spot is materially cheaper than on-demand and is the right default for any checkpointable workload.",
      "Prefer serverless over a dedicated instance for bursty or low-volume inference — it runs open-weight models at per-million-token rates and costs nothing while idle.",
      "Rates are quoted and settled in CAD; call get_pricing_reference for the live table rather than assuming a price.",
      "Check spend before committing: should_i_run_this, or estimate_job_cost + get_wallet_balance.",
      "Destructive tools require confirm:true — call with confirm:false first for a preview.",
      "Placement is region-aware. When a workload carries a data-residency or jurisdiction requirement, pass it explicitly and verify the selected host reports a matching jurisdiction — never assume the default region satisfies it.",
    ].join(" "),
  });
  installToolAudit(server, client, user, transport, profile);
  registerAllTools(server, client, user, options);
  registerResources(server, client);
  registerPlaybooks(server);
  return server;
}
