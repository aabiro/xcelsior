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

  const documents: Array<[string, string, string, unknown]> = [
    ["xcelsior-scope-v1", "xcelsior://policy/v1/scopes", "MCP scope vocabulary", {
      version: "1", legacy: ["api"], scopes: [
        "instances:read", "instances:write", "instances:operate", "inference:read",
        "inference:write", "billing:read", "gpu:read", "hosts:read", "hosts:operate",
        "hosts:evict", "control_plane:read", "control_plane:operate", "mcp_actions:approve",
      ],
    }],
    ["xcelsior-queue-reasons-v1", "xcelsior://control-plane/v1/queue-reasons", "Persisted queue reason catalog", {
      version: "1", authority: "control-plane persisted placement explanation",
      reasons: ["no_eligible_host", "capacity_unavailable", "policy_denied", "funding_required", "host_stale", "placement_conflict"],
    }],
    ["xcelsior-launch-policy-v2", "xcelsior://policy/v2/launch", "Launch approval and cancellation policy", {
      version: "2", preview_required: true, approval_is_server_bound: true,
      confirm_is_intent_only: true, cancellation: "Cancelling a watch never cancels compute.",
    }],
    ["xcelsior-runtime-capabilities-v1", "xcelsior://capabilities/v1/runtime", "GPU and runtime capability definitions", {
      version: "1", source: "Use list_available_gpus and get_host_capacity for live facts.",
      isolation: "Required isolation and storage capabilities fail closed.",
    }],
  ];
  for (const [name, uri, title, body] of documents) {
    server.registerResource(name, uri, {
      title, description: title, mimeType: "application/json",
    }, async () => ({ contents: [{ uri, mimeType: "application/json", text: JSON.stringify(body, null, 2) }] }));
  }
}
