import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { TOOL_SCOPES, userHasScope, scopeUnion, describeScopeRequirement } from "../auth/scopes.js";
import type { AuthUser } from "../auth/bearer.js";

function scopeDenied(tool: string, user: AuthUser | undefined) {
  const required = TOOL_SCOPES[tool];
  if (!userHasScope(user?.scopes, required)) {
    return jsonText({
      error: "insufficient_scope",
      required: scopeUnion(required),
      message: `This tool requires ${describeScopeRequirement(required)}`,
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
      }),
    },
    async (args) => {
      const denied = scopeDenied("estimate_job_cost", user);
      if (denied) return denied;
      try {
        const data = await client.post("/api/pricing/estimate", {
          gpu_model: args.gpu_model,
          duration_hours: args.duration_hours,
          spot: args.spot,
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

  server.registerTool(
    "list_payment_methods",
    {
      // No arguments. The route resolves the customer from the caller's own
      // credential, so there is no `customer_id` to pass and no way to ask for
      // somebody else's cards — unlike the routes above, which accept one and
      // check ownership server-side.
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("list_payment_methods", user);
      if (denied) return denied;
      try {
        // Returns brand, last four, expiry and which is default. No PAN, no
        // client_secret, no Stripe token — the plan's "no secret in any
        // surface" gate covers this response, and the endpoint is what the
        // dashboard already renders from.
        const data = await client.get("/api/billing/payment-methods");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}