// The advertised customer-profile total. NEXT_PUBLIC_* is inlined at build time.
// Keep the fallback aligned with api/mcp/tool-surface.json.
export const DEFAULT_MCP_TOOL_COUNT = 89;

const configuredCount = process.env.NEXT_PUBLIC_MCP_TOOL_COUNT?.trim();
const count = configuredCount ? Number(configuredCount) : DEFAULT_MCP_TOOL_COUNT;
if (!Number.isSafeInteger(count) || count <= 0) {
  throw new Error("NEXT_PUBLIC_MCP_TOOL_COUNT must be a positive integer");
}

export const MCP_TOOL_COUNT = count;

export function withMcpToolCount(text: string): string {
  return text.replaceAll("{mcpToolCount}", String(MCP_TOOL_COUNT));
}
