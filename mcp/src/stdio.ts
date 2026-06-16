#!/usr/bin/env node
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { loadConfig } from "./config.js";
import { createMcpServer } from "./server.js";
import { createApiClient, validateBearer } from "./auth/bearer.js";

async function main(): Promise<void> {
  const config = loadConfig();
  const token = (process.env.XCELSIOR_ACCESS_TOKEN || process.env.XCELSIOR_OAUTH_TOKEN || "").trim();
  if (!token) {
    console.error(
      "XCELSIOR_ACCESS_TOKEN is required. Create an OAuth client at https://xcelsior.ca/dashboard/settings?tab=mcp",
    );
    process.exit(1);
  }

  const user = await validateBearer(config.apiUrl, token);
  if (!user) {
    console.error("Invalid or expired XCELSIOR_ACCESS_TOKEN");
    process.exit(1);
  }

  const client = createApiClient(config.apiUrl, token);
  const server = createMcpServer(client, user);
  const transport = new StdioServerTransport();
  await server.connect(transport);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});