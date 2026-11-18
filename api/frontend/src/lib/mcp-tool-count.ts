import releaseFacts from "./release-facts.json";

// The tools the hosted MCP server publishes to a customer. Not typed here and
// not configurable: release-facts.json is written from api/mcp/tool-surface.json
// by api/scripts/generate_release_facts.py, and a test fails when it is stale.
export const MCP_TOOL_COUNT: number = releaseFacts.mcpToolCount;

export function withMcpToolCount(text: string): string {
  return text.replaceAll("{mcpToolCount}", String(MCP_TOOL_COUNT));
}
