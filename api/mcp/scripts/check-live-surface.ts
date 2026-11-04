/** Compare the actual source registry with an authenticated live tools/list.
 * Usage: npx tsx scripts/check-live-surface.ts TOKEN_FILE [MCP_URL]
 * TOKEN_FILE contains a bearer token or an agent-key JSON response. No tool
 * handlers are invoked; this checks declarations, not handler side effects.
 */
import { readFile } from "node:fs/promises";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { describeToolSurface } from "../src/tools/surface.js";

const file = process.argv[2];
if (!file) throw new Error("Usage: check-live-surface.ts TOKEN_FILE [MCP_URL]");
const endpoint = process.argv[3] || "https://mcp.xcelsior.ca/mcp";
const stored = (await readFile(file, "utf8")).trim();
const token = stored.startsWith("{") ? JSON.parse(stored).access_token : stored;
if (typeof token !== "string" || !token) throw new Error("Token file has no bearer token");
const client = new Client({ name: "xcelsior-source-surface-check", version: "1" });
try {
  await client.connect(new StreamableHTTPClientTransport(new URL(endpoint), {
    requestInit: { headers: { Authorization: `Bearer ${token}` } },
  }));
  const actual = (await client.listTools()).tools;
  const expected = describeToolSurface("customer");
  const byName = new Map(actual.map(tool => [tool.name, tool]));
  const expectedNames = new Set(expected.map(tool => tool.name));
  const findings: string[] = [];
  for (const source of expected) {
    const live = byName.get(source.name);
    if (!live) {
      findings.push(`${source.name}: missing from production`);
      continue;
    }
    for (const [key, value] of Object.entries(source.annotations)) {
      if ((live.annotations as Record<string, unknown> | undefined)?.[key] !== value) {
        findings.push(`${source.name}: annotation ${key} differs`);
      }
    }
    if (Boolean(live.outputSchema) !== source.hasOutputSchema) {
      findings.push(`${source.name}: output-schema presence differs`);
    }
  }
  for (const tool of actual) {
    if (!expectedNames.has(tool.name)) findings.push(`${tool.name}: absent from customer source registry`);
  }
  console.log(JSON.stringify({ endpoint, source_tools: expected.length, live_tools: actual.length, findings }, null, 2));
  if (findings.length) process.exitCode = 1;
} catch (error) {
  // Errors must never make a token-file credential part of a pasteable report.
  console.error(String(error).replaceAll(token, "[REDACTED]"));
  process.exitCode = 1;
} finally {
  await client.close();
}
