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

export function registerBillingTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "get_wallet_balance",
    {
      description: "Get wallet balance and credits for a customer (defaults to authenticated user).",
      inputSchema: z.object({
        customer_id: z.string().optional().describe("Customer ID; omit to use your account"),
      }),
    },
    async ({ customer_id }) => {
      const denied = scopeDenied("get_wallet_balance", user);
      if (denied) return denied;
      const cid = customer_id || user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required — authenticate or pass customer_id" });
      try {
        const data = await client.get(`/api/billing/wallet/${encodeURIComponent(cid)}`);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "estimate_job_cost",
    {
      description:
        "Estimate job cost in CAD with optional spot pricing and Canadian AI Compute rebate preview.",
      inputSchema: z.object({
        gpu_model: z.string().default("RTX 4090"),
        duration_hours: z.number().min(0).max(8760).default(1),
        spot: z.boolean().default(false),
        sovereignty: z.boolean().default(false),
        is_canadian: z.boolean().default(true),
      }),
    },
    async (args) => {
      const denied = scopeDenied("estimate_job_cost", user);
      if (denied) return denied;
      try {
        const data = await client.post("/api/pricing/estimate", args);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "list_invoices",
    {
      description: "List billing invoices for a customer.",
      inputSchema: z.object({
        customer_id: z.string().optional(),
      }),
    },
    async ({ customer_id }) => {
      const denied = scopeDenied("list_invoices", user);
      if (denied) return denied;
      const cid = customer_id || user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required" });
      try {
        const data = await client.get(`/api/billing/invoices/${encodeURIComponent(cid)}`);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}