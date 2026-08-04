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
    "top_up_wallet",
    {
      inputSchema: z.object({
        amount_cad: z
          .number()
          .gt(0)
          .max(10_000)
          .describe("Amount to charge in CAD. Confirm this with the user first — it moves money."),
        // Human selectors, because nobody says `pm_1QxYz...`. Resolution is
        // server-side: two Visas on file and a request for "the Visa" is
        // refused with both listed, rather than one being picked. Charging the
        // wrong card is not undone by an apology.
        card_last4: z
          .string()
          .optional()
          .describe("Last four digits, e.g. '4242'. Use what the user said."),
        card_brand: z
          .string()
          .optional()
          .describe("Card brand, e.g. 'visa' or 'mastercard'. Use what the user said."),
        payment_method_id: z
          .string()
          .optional()
          .describe("Exact Stripe id, if you already have one from list_payment_methods."),
        idempotency_key: z
          .string()
          .optional()
          .describe(
            "Pass the SAME key when retrying a call that timed out. Without it a retry may " +
              "charge a second time.",
          ),
      }),
    },
    async ({ amount_cad, card_last4, card_brand, payment_method_id, idempotency_key }) => {
      const denied = scopeDenied("top_up_wallet", user);
      if (denied) return denied;
      try {
        const data = await client.post(
          "/api/v2/billing/top-up",
          {
            amount_cad,
            card_last4: card_last4 ?? "",
            card_brand: card_brand ?? "",
            payment_method_id: payment_method_id ?? "",
          },
          // The client already carries idempotency as a first-class option and
          // sets the header itself; passing a raw header would bypass its
          // retry policy, which is the thing that makes the key matter.
          { idempotencyKey: idempotency_key, retry: "idempotent" },
        );
        return jsonText(data);
      } catch (e) {
        // The route answers 409 for "which card?", 402 for an SCA challenge,
        // and 502 for a decline. Each body already says `charged: false` and
        // why, so it is surfaced rather than flattened into "error" — an agent
        // that cannot tell a challenge from a failure will either retry a
        // charge that succeeded or abandon one that only needed confirming.
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