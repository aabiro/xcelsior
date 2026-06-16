import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { TOOL_SCOPES, userHasScope } from "../auth/scopes.js";
import type { AuthUser } from "../auth/bearer.js";

function scopeDenied(tool: string, user: AuthUser | undefined) {
  const required = TOOL_SCOPES[tool] || ["api"];
  if (!userHasScope(user?.scopes, required)) {
    return jsonText({
      error: "insufficient_scope",
      required,
      message: `This tool requires one of: ${required.join(", ")}`,
    });
  }
  return null;
}

export function registerDiscoveryTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "list_available_gpus",
    {
      description:
        "List GPUs currently available on the Xcelsior marketplace with VRAM, region, counts, and CAD pricing.",
      inputSchema: z.object({
        region: z.string().optional().describe("Filter by region code (e.g. ca-east)"),
      }),
    },
    async ({ region }) => {
      const denied = scopeDenied("list_available_gpus", user);
      if (denied) return denied;
      try {
        const data = await client.get("/api/v2/gpu/available", region ? { region } : undefined);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_spot_prices",
    {
      description: "Get current spot/interruptible GPU prices per model in CAD.",
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("get_spot_prices", user);
      if (denied) return denied;
      try {
        const data = await client.get("/api/v2/marketplace/spot-prices");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_pricing_reference",
    {
      description: "Reference on-demand GPU hourly rates in CAD from the live pricing table.",
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("get_pricing_reference", user);
      if (denied) return denied;
      try {
        const data = await client.get("/api/pricing/reference");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "search_marketplace",
    {
      description: "Search marketplace listings by GPU model, VRAM, region, and reputation.",
      inputSchema: z.object({
        gpu_model: z.string().optional(),
        min_vram_gb: z.number().optional(),
        region: z.string().optional(),
        min_reputation: z.number().optional(),
        sort_by: z.enum(["price", "reputation", "region", "vram"]).optional(),
      }),
    },
    async (args) => {
      const denied = scopeDenied("search_marketplace", user);
      if (denied) return denied;
      try {
        const data = await client.post("/api/v2/marketplace/search", args);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "list_tiers",
    {
      description: "List compute tier catalog (VRAM bands and tier names).",
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("list_tiers", user);
      if (denied) return denied;
      try {
        const data = await client.get("/tiers");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}