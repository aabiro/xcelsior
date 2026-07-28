/**
 * Authenticated deploy gate for a direct MCP replica.
 *
 * The bearer token is accepted on stdin so it never appears in argv, logs, or
 * the container environment. The official SDK performs the complete
 * initialize/session handshake before listing tools.
 */
import { readFileSync } from "node:fs";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const endpoint = process.argv[2];
const token = readFileSync(0, "utf8").trim();
if (!endpoint || !token) {
  throw new Error("usage: token-on-stdin | node dist/protocol-smoke.js <mcp-url>");
}

const client = new Client({ name: "xcelsior-deploy-smoke", version: "2.0.0" });
const transport = new StreamableHTTPClientTransport(new URL(endpoint), {
  requestInit: { headers: { Authorization: `Bearer ${token}` } },
});

try {
  await client.connect(transport);
  const { tools } = await client.listTools();
  if (!tools.some((tool) => tool.name === "create_instance")) {
    throw new Error("create_instance is absent from the authenticated tool registry");
  }
} finally {
  await transport.close().catch(() => undefined);
}
