import releaseFacts from "./release-facts.json";

// The tools the hosted MCP server publishes to a customer. Not typed here and
// not configurable: release-facts.json is written from api/mcp/tool-surface.json
// by api/scripts/generate_release_facts.py, and a test fails when it is stale.
export const MCP_TOOL_COUNT: number = releaseFacts.mcpToolCount;

// The checks a host must pass to be verified ("7 checks"): the length of
// VERIFICATION_CHECKS in hosts/verification.py, through the same file.
export const VERIFICATION_CHECK_COUNT: number = releaseFacts.verificationCheckCount;

// Fills every release-fact placeholder copy may carry. Named for the first one
// it filled; i18n, the server translator and the blog all pass text through it.
export function withMcpToolCount(text: string): string {
  return text
    .replaceAll("{mcpToolCount}", String(MCP_TOOL_COUNT))
    .replaceAll("{verificationCheckCount}", String(VERIFICATION_CHECK_COUNT));
}
