import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";

const PRICING_CACHE_MS = 300_000;
let pricingCache: { at: number; body: string } | null = null;

export function registerResources(server: McpServer, client: XcelsiorApiClient): void {
  server.registerResource(
    "xcelsior-llms-txt",
    "xcelsior://docs/llms",
    {
      title: "Xcelsior llms.txt",
      description: "Machine-readable API documentation for AI agents",
      mimeType: "text/plain",
    },
    async () => {
      const res = await fetch(`${client.baseUrl}/llms.txt`);
      const text = res.ok ? await res.text() : "llms.txt unavailable";
      return {
        contents: [{ uri: "xcelsior://docs/llms", mimeType: "text/plain", text }],
      };
    },
  );

  server.registerResource(
    "xcelsior-pricing-reference",
    "xcelsior://pricing/reference",
    {
      title: "GPU pricing reference",
      description: "Cached on-demand and spot CAD hourly rates",
      mimeType: "application/json",
    },
    async () => {
      const now = Date.now();
      if (!pricingCache || now - pricingCache.at > PRICING_CACHE_MS) {
        const data = await client.get("/api/pricing/reference");
        pricingCache = { at: now, body: JSON.stringify(data, null, 2) };
      }
      return {
        contents: [
          {
            uri: "xcelsior://pricing/reference",
            mimeType: "application/json",
            text: pricingCache.body,
          },
        ],
      };
    },
  );
}