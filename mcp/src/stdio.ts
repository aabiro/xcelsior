#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig } from "./config.js";
import { createMcpServer } from "./server.js";
import { createApiClient, validateBearer } from "./auth/bearer.js";

// stdio transport: launched as a subprocess by the MCP client (VS Code, Claude,
// Cursor, …). Unlike the hosted HTTP transport it never speaks HTTP to the
// client, so the client can never receive a 401 and can never be dragged into
// an OAuth discovery / dynamic-client-registration flow. This is the friction-
// free local path — the token is supplied out-of-band via the environment
// (XCELSIOR_ACCESS_TOKEN), exactly like every other stdio MCP server.
//
// Robustness rule: a missing or expired token must NEVER prevent the server
// from starting. If it did, the client would show the server as "failed to
// start" and — in some clients — retry through the interactive auth flow we are
// deliberately avoiding. Instead we always start, and let each tool call return
// the API's own typed 401 with an actionable "refresh your token" message. The
// bearer is still enforced end-to-end by the API on every call.
async function main(): Promise<void> {
  const config = loadConfig();
  const token = (process.env.XCELSIOR_ACCESS_TOKEN || process.env.XCELSIOR_OAUTH_TOKEN || "").trim();
  const quickConnect = `${config.apiUrl.replace(/\/$/, "")}/dashboard/mcp`;

  let user = undefined;
  if (!token) {
    console.error(
      `[xcelsior-mcp] No XCELSIOR_ACCESS_TOKEN set — starting anyway; tool calls will ` +
        `return 401 until a token is supplied. Create one at ${quickConnect} (Quick Connect).`,
    );
  } else {
    // Best-effort identity resolution for nicer context. Never fatal: an expired
    // token still starts the server so the client stays connected and shows a
    // clean per-tool error instead of an auth popup.
    try {
      user = (await validateBearer(config.apiUrl, token)) ?? undefined;
      if (!user) {
        console.error(
          `[xcelsior-mcp] Token present but invalid or expired — starting anyway; tool ` +
            `calls will return 401. Refresh it at ${quickConnect}.`,
        );
      }
    } catch (err) {
      console.error(`[xcelsior-mcp] Identity check failed (${String(err)}); starting anyway.`);
    }
  }

  const client = createApiClient(config.apiUrl, token);
  const server = createMcpServer(client, user);
  await server.connect(new StdioServerTransport());
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
