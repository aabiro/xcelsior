// Shared MCP connection helpers — used by both the /dashboard/mcp Connect page
// and the Settings → AI Agents wizard (McpAgentSetup.tsx) so the connection URL
// and per-agent config JSON have a single source of truth.

export function mcpUrl(): string {
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8770/mcp";
  }
  return "https://xcelsior.ca/mcp";
}

// MCP client config formats differ by agent:
//  - Cursor  (~/.cursor/mcp.json):  `mcpServers` + `url` (transport inferred)
//  - Claude  (.mcp.json):           `mcpServers` + explicit `type: "http"`
//  - VS Code (.vscode/mcp.json):    `servers` (not mcpServers) + `type: "http"`
export function configJson(agentId: string, tokenPlaceholder = "YOUR_OAUTH_TOKEN"): string {
  const url = mcpUrl();
  const headers = { Authorization: `Bearer ${tokenPlaceholder}` };
  if (agentId === "github") {
    return JSON.stringify(
      {
        mcpServers: {
          "xcelsior-readonly": {
            type: "http",
            url,
            headers: {
              Authorization: "Bearer ${COPILOT_MCP_XCELSIOR_ACCESS_TOKEN}",
            },
            tools: [
              "list_available_gpus",
              "get_spot_prices",
              "get_pricing_reference",
              "search_marketplace",
              "list_tiers",
            ],
          },
        },
      },
      null,
      2,
    );
  }
  if (agentId === "vscode") {
    return JSON.stringify({ servers: { xcelsior: { type: "http", url, headers } } }, null, 2);
  }
  if (agentId === "claude") {
    return JSON.stringify({ mcpServers: { xcelsior: { type: "http", url, headers } } }, null, 2);
  }
  return JSON.stringify({ mcpServers: { xcelsior: { url, headers } } }, null, 2);
}

// Where each client expects the config file to live (shown above the snippet).
export function configPath(agentId: string): string {
  if (agentId === "github") return "GitHub -> Settings -> Copilot -> MCP servers";
  if (agentId === "vscode") return ".vscode/mcp.json";
  if (agentId === "claude") return ".mcp.json (project root) or claude_desktop_config.json";
  return "~/.cursor/mcp.json";
}
