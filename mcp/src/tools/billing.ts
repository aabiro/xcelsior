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
        "Estimate what a GPU job will cost, in CAD, before launching it. Returns the hourly rate and " +
        "projected total so you can compare against the wallet balance. Price on-demand by default; set " +
        "spot:true for interruptible capacity when the workload can checkpoint.",
      inputSchema: z.object({
        gpu_model: z.string().default("RTX 4090"),
        duration_hours: z.number().min(0).max(8760).default(1),
        spot: z
          .boolean()
          .default(false)
          .describe(
            "Price as interruptible spot capacity instead of on-demand. Materially cheaper, but the " +
              "instance can be reclaimed — only use for workloads that checkpoint.",
          ),
        sovereignty: z
          .boolean()
          .default(false)
          .describe(
            "Price for a sovereignty-vetted host (independently incorporated, no foreign control). " +
              "Carries a pricing premium — set only when a contract or regulation actually requires it.",
          ),
      }),
    },
    async (args) => {
      const denied = scopeDenied("estimate_job_cost", user);
      if (denied) return denied;
      try {
        // The Canadian AI Compute Access Fund has ended, so no estimate should carry its rebate.
        // EstimateRequest still defaults is_canadian to true, so pin it false explicitly —
        // omitting it would apply a rebate that no longer exists and understate real cost.
        const data = await client.post("/api/pricing/estimate", {
          gpu_model: args.gpu_model,
          duration_hours: args.duration_hours,
          spot: args.spot,
          sovereignty: args.sovereignty,
          is_canadian: false,
        });
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