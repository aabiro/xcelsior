// Shared MCP connection helpers — used by both the /dashboard/mcp Connect page
// and the Settings → AI Agents wizard (McpAgentSetup.tsx) so the connection URL
// and per-agent config JSON have a single source of truth.

// The canonical connector URL — what a user pastes into Claude, ChatGPT, Grok,
// Cursor, or VS Code, and the exact RFC 8707 resource identifier tokens are
// bound to. Everything user-facing must show this, path included: a connector
// echoes the value back as its `resource` parameter, so the origin alone is a
// different identifier and gets rejected.
export const MCP_CONNECTOR_URL = "https://mcp.xcelsior.ca/mcp";
export const MCP_RESOURCE = MCP_CONNECTOR_URL;

export function mcpUrl(): string {
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8770/mcp";
  }
  return MCP_CONNECTOR_URL;
}

// Same-origin compatibility route, used only for the in-app health probe.
// `mcp.xcelsior.ca` serves no CORS headers — deliberately, it is an
// agent-to-server endpoint, not a browser one — so a fetch from the dashboard
// to the canonical host would be blocked before it reached us and would read
// as "MCP is down".
export function mcpHealthUrl(): string {
  if (typeof window !== "undefined" && window.location.hostname === "localhost") {
    return "http://localhost:8770/mcp/health";
  }
  return "https://xcelsior.ca/mcp/health";
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

// ── One-click installs (adoption plan X5.26) ────────────────────────────────
//
// Deep links carry no credential. They install the *server*; the client then
// performs OAuth on first use, which is the whole point of the front door —
// a link that embedded a token would be a token in a URL, in a bookmark, in
// a browser history, and in whatever chat it got pasted into.

/** The server entry a deep link installs. No `headers`: OAuth handles auth. */
function deepLinkConfig(kind: "cursor" | "vscode"): string {
  const url = MCP_CONNECTOR_URL;
  return kind === "vscode"
    ? JSON.stringify({ name: "xcelsior", type: "http", url })
    : JSON.stringify({ url });
}

export interface OneClickInstall {
  id: string;
  label: string;
  /** A deep link the browser can open, or null when the client uses a command. */
  href: string | null;
  /** A command to paste, for clients with no deep-link scheme. */
  command: string | null;
}

/** base64, in both the browser and the Node render pass. */
function toBase64(value: string): string {
  if (typeof btoa === "function") return btoa(value);
  // Node render pass; `globalThis` keeps the browser bundle free of `Buffer`.
  const nodeBuffer = (globalThis as { Buffer?: { from(s: string, e: string): { toString(e: string): string } } })
    .Buffer;
  return nodeBuffer ? nodeBuffer.from(value, "utf-8").toString("base64") : value;
}

export function oneClickInstalls(): OneClickInstall[] {
  const cursorPayload = toBase64(deepLinkConfig("cursor"));
  return [
    {
      id: "cursor",
      label: "Cursor",
      href: `cursor://anysphere.cursor-deeplink/mcp/install?name=xcelsior&config=${encodeURIComponent(cursorPayload)}`,
      command: null,
    },
    {
      id: "vscode",
      label: "VS Code",
      href: `vscode:mcp/install?${encodeURIComponent(deepLinkConfig("vscode"))}`,
      command: null,
    },
    {
      id: "claude-code",
      label: "Claude Code",
      href: null,
      command: `claude mcp add --transport http xcelsior ${MCP_CONNECTOR_URL}`,
    },
    {
      id: "copilot-cli",
      label: "Copilot CLI",
      href: null,
      command: `copilot mcp add xcelsior --transport http --url ${MCP_CONNECTOR_URL}`,
    },
    {
      id: "npx",
      label: "Local (stdio)",
      href: null,
      command: "npx @xcelsior-gpu/mcp",
    },
  ];
}
