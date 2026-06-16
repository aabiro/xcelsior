import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "./client/api.js";
import type { AuthUser } from "./auth/bearer.js";
import { registerAllTools } from "./tools/index.js";
import { registerResources } from "./resources/index.js";
import { registerPlaybooks } from "./prompts/playbooks.js";

const SERVER_INFO = {
  name: "xcelsior-mcp",
  version: "0.2.0",
};

export function createMcpServer(client: XcelsiorApiClient, user?: AuthUser): McpServer {
  const server = new McpServer(SERVER_INFO, {
    capabilities: {
      tools: {},
      resources: {},
      prompts: {},
    },
    instructions: [
      "You are connected to Xcelsior — a GPU compute marketplace.",
      "Use list_available_gpus and get_spot_prices before launching workloads.",
      "Use should_i_run_this or estimate_job_cost + get_wallet_balance before spend.",
      "Destructive tools require confirm:true — call with confirm:false first for a preview.",
      "Canadian data residency and PIPEDA compliance are supported — prefer CA regions when required.",
    ].join(" "),
  });
  registerAllTools(server, client, user);
  registerResources(server, client);
  registerPlaybooks(server);
  return server;
}