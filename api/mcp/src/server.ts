import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "./client/api.js";
import type { AuthUser } from "./auth/bearer.js";
import { registerAllTools, type ToolRegistrationOptions } from "./tools/index.js";
import { registerResources } from "./resources/index.js";
import { registerPlaybooks } from "./prompts/playbooks.js";
import { installToolAudit } from "./audit/context.js";
import type { ToolProfile } from "./tools/profiles.js";
import type { Icon } from "@modelcontextprotocol/sdk/types.js";

// Everything a client can show about the server before a tool is called.
// VS Code builds its tool prefix from `title` (else `name`), lowercased, unsafe
// runs to "_", cut to 13 characters: "Xcelsior MCP" -> "mcp_xcelsior_mcp_…".
// The longer display name lives in server.json for the registry listing.
const SERVER_INFO = {
  name: "xcelsior-mcp",
  title: "Xcelsior MCP",
  description: "Rent GPUs, launch spot instances and run serverless inference on the Xcelsior marketplace.",
  version: "2.0.0",
  websiteUrl: "https://xcelsior.ca/mcp",
};

export function createMcpServer(
  client: XcelsiorApiClient,
  user?: AuthUser,
  transport: "streamable_http" | "stdio" = "streamable_http",
  profile: ToolProfile = "customer",
  options: ToolRegistrationOptions = {},
  icons: Icon[] = [],
): McpServer {
  const server = new McpServer({ ...SERVER_INFO, ...(icons.length ? { icons } : {}) }, {
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
      "When you first connect, call get_permissions and tell the user in plain words what this connection can and cannot do on their account, and where they change it.",
      "If a tool answers insufficient_scope, say which permission is missing and offer request_permission, which returns an approval link only the user can approve; never work around a refusal with another tool.",
    ].join(" "),
  });
  installToolAudit(server, client, user, transport, profile);
  registerAllTools(server, client, user, options);
  registerResources(server, client);
  registerPlaybooks(server);
  return server;
}
