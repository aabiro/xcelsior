import { randomUUID } from "node:crypto";
import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem, formatApiError } from "../client/errors.js";
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
    "get_wallet_history",
    {
      inputSchema: z.object({
        customer_id: z.string().optional().describe("Customer ID; omit to use your account"),
        limit: z.number().int().min(1).max(200).default(50),
      }),
    },
    async ({ customer_id, limit }) => {
      const denied = scopeDenied("get_wallet_history", user);
      if (denied) return denied;
      const cid = customer_id || user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required — authenticate or pass customer_id" });
      try {
        return jsonText(
          await client.get(`/api/billing/wallet/${encodeURIComponent(cid)}/history`, {
            limit: Number(limit ?? 50),
          }),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "list_funding_options",
    { inputSchema: z.object({}) },
    async () => {
      const denied = scopeDenied("list_funding_options", user);
      if (denied) return denied;
      // Four small probes rather than four tools. They are capability flags, and
      // a surface with `is_paypal_enabled` next to `is_lightning_enabled` makes
      // a model choose between things that are one answer to one question.
      //
      // Each is settled independently: a rail that errors is reported
      // unavailable rather than failing the call, because the point of this tool
      // is to find the rail that *does* work when one has just failed.
      //
      // Each path is written **at** its `client.get` rather than passed to the
      // helper as a variable. `tests/test_classification_matches_the_tools.py`
      // pairs a literal with the verb beside it, so `probe(path)` hides the
      // route from the only check that keeps `covered` honest — and the fix for
      // that is not another entry in its indirection allowlist.
      const probe = async (call: () => Promise<unknown>) => {
        try {
          return await call() as Record<string, unknown>;
        } catch (e) {
          return { available: false, enabled: false, reason: formatApiError(e) };
        }
      };
      const [crypto, lightning, paypal, rate] = await Promise.all([
        probe(() => client.get("/api/billing/crypto/enabled")),
        probe(() => client.get("/api/billing/lightning/enabled")),
        probe(() => client.get("/api/billing/paypal/enabled")),
        probe(() => client.get("/api/billing/crypto/rate")),
      ]);
      return jsonText({
        ok: true,
        card: {
          available: true,
          note: "Charges a card already on file with top_up_wallet. Adding a card is a dashboard action.",
        },
        crypto: { ...crypto, btc_cad: (rate as { btc_cad?: unknown }).btc_cad },
        lightning,
        paypal,
      });
    },
  );

  server.registerTool(
    "create_crypto_deposit",
    {
      inputSchema: z.object({
        amount_cad: z.number().gt(0).max(10_000).describe("How much to add, in CAD"),
        customer_id: z.string().optional().describe("Customer ID; omit to use your account"),
        idempotency_key: z
          .string()
          .optional()
          .describe("Reuse the same key to retry safely; a new amount under an old key is refused"),
      }),
    },
    async ({ amount_cad, customer_id, idempotency_key }) => {
      const denied = scopeDenied("create_crypto_deposit", user);
      if (denied) return denied;
      const cid = customer_id || user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required — authenticate or pass customer_id" });
      try {
        return jsonText(
          await client.post(
            "/api/billing/crypto/deposit",
            { customer_id: cid, amount_cad },
            // The route reads an Idempotency-Key header and answers 409 when a
            // reused key carries a different amount. Without one, a retried call
            // is a second deposit at a second rate.
            { idempotencyKey: idempotency_key ?? `btc-deposit-${randomUUID()}` },
          ),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_crypto_deposit",
    {
      inputSchema: z.object({
        deposit_id: z.string().min(1).describe("From create_crypto_deposit"),
      }),
    },
    async ({ deposit_id }) => {
      const denied = scopeDenied("get_crypto_deposit", user);
      if (denied) return denied;
      try {
        return jsonText(
          await client.get(`/api/billing/crypto/deposit/${encodeURIComponent(String(deposit_id))}`),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "create_lightning_deposit",
    {
      inputSchema: z.object({
        amount_cad: z.number().gt(0).max(10_000).describe("How much to add, in CAD"),
        customer_id: z.string().optional().describe("Customer ID; omit to use your account"),
        idempotency_key: z
          .string()
          .optional()
          .describe("Reuse the same key to retry safely; a new amount under an old key is refused"),
      }),
    },
    async ({ amount_cad, customer_id, idempotency_key }) => {
      const denied = scopeDenied("create_lightning_deposit", user);
      if (denied) return denied;
      const cid = customer_id || user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required — authenticate or pass customer_id" });
      try {
        return jsonText(
          await client.post(
            "/api/billing/lightning/deposit",
            { customer_id: cid, amount_cad },
            { idempotencyKey: idempotency_key ?? `ln-deposit-${randomUUID()}` },
          ),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_lightning_deposit",
    {
      inputSchema: z.object({
        deposit_id: z.string().min(1).describe("From create_lightning_deposit"),
      }),
    },
    async ({ deposit_id }) => {
      const denied = scopeDenied("get_lightning_deposit", user);
      if (denied) return denied;
      try {
        return jsonText(
          await client.get(`/api/billing/lightning/deposit/${encodeURIComponent(String(deposit_id))}`),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_spend_envelope",
    {
      inputSchema: z.object({
        customer_id: z.string().optional().describe("Customer ID; omit to use your account"),
      }),
    },
    async ({ customer_id }) => {
      const denied = scopeDenied("get_spend_envelope", user);
      if (denied) return denied;
      const cid = customer_id || user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required — authenticate or pass customer_id" });
      try {
        const data = await client.get(
          `/api/billing/wallet/${encodeURIComponent(cid)}/depletion`,
        );
        // `seconds_to_zero` is null when nothing is running, which is not the
        // same as "no time left" and reads that way if passed through bare.
        const seconds = (data as Record<string, unknown>)?.seconds_to_zero;
        return jsonText({
          ...(data as Record<string, unknown>),
          runway:
            seconds == null
              ? "nothing is running, so the balance is not being consumed"
              : `${(Number(seconds) / 3600).toFixed(1)} hours at the current burn rate`,
          at_zero:
            "running instances are stopped automatically with reason 'low_balance'; " +
            "auto-top-up, if configured, charges the saved card before that happens",
        });
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
    "configure_auto_topup",
    {
      inputSchema: z.object({
        enabled: z.boolean().describe("false turns auto top-up off entirely"),
        amount_cad: z
          .number()
          .gt(0)
          .max(10_000)
          .describe("How much to charge each time the threshold is crossed"),
        threshold_cad: z
          .number()
          .gte(0)
          .max(10_000)
          .describe("Charge when the balance falls below this"),
        payment_method_id: z
          .string()
          .optional()
          .describe("Card to charge; omit to keep the one already configured"),
        plan_id: z
          .string()
          .optional()
          .describe("An approved plan from a previous widening call; omit on the first call"),
      }),
    },
    async ({ enabled, amount_cad, threshold_cad, payment_method_id, plan_id }) => {
      const denied = scopeDenied("configure_auto_topup", user);
      if (denied) return denied;
      try {
        // The response carries `previous`, so the model can tell the user what
        // actually changed rather than echoing back what it just sent — the
        // difference between "auto top-up is $50" and "I raised it from $20 to
        // $50", which is the sentence that lets someone catch a mistake.
        // Second phase: an approval already exists. The settings come from the
        // **plan**, not from this call, so an approval cannot be spent on
        // different values — which is the whole point of approving one.
        if (plan_id) {
          return jsonText(
            await client.post(
              `/api/v2/billing/auto-topup-plans/${encodeURIComponent(plan_id)}/execute`,
              {},
            ),
          );
        }
        const body = {
          enabled,
          amount_cad,
          threshold_cad,
          stripe_payment_method_id: payment_method_id ?? "",
        };
        try {
          return jsonText(await client.post("/api/v2/billing/auto-topup", body));
        } catch (inner) {
          // Gate P1 clause 6: raising a spend cap needs approval, lowering does
          // not. The API answers a widening with 409 naming the plan route. A
          // bare 409 is a dead end for an agent — the caller is told what it
          // may not do and left holding nothing — so prepare the plan and hand
          // back something approvable. Narrowing never reaches here.
          const problem = apiProblem(inner) as { status?: number };
          if (problem?.status !== 409) throw inner;
          const plan = await client.post("/api/v2/billing/auto-topup-plans", body);
          return jsonText({
            ...(plan as Record<string, unknown>),
            next_step:
              "This widens unattended spending, so it needs approval. Send the " +
              "user to approval_url, then call configure_auto_topup again with " +
              "plan_id and nothing else changed.",
          });
        }
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
            "Leave this out unless you are retrying a call that timed out, in which case " +
              "pass the SAME key you sent the first time. Omitting it is safe: one is " +
              "generated per call, so asking twice tops up twice and a retry does not.",
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
          //
          // One is generated when the caller omits it, and that is load-bearing
          // twice over. Without a key the client disables retries entirely
          // (`api.ts`), so a timeout became a single blind attempt; and the
          // route fell back to bucketing by customer, amount and card in a
          // five-minute window, which quietly merged a *deliberate* second
          // top-up into the first and reported success. One key per invocation
          // means one intent per invocation: ask twice and you are charged
          // twice, retry once and you are charged once.
          {
            idempotencyKey: idempotency_key ?? `topup-${randomUUID()}`,
            retry: "idempotent",
          },
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

  server.registerTool(
    "get_auto_topup",
    {
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("get_auto_topup", user);
      if (denied) return denied;
      try {
        // The read half. Until this existed the only way to learn the current
        // settings was to POST a change and read `previous` out of the
        // response — you had to write in order to read, on the one surface
        // that authorises unattended charges.
        const data = await client.get("/api/v2/billing/auto-topup");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "list_pending_verifications",
    {
      // No arguments, same as list_payment_methods: the route resolves the
      // customer from the caller's credential.
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("list_pending_verifications", user);
      if (denied) return denied;
      try {
        // Deliberately the **list** route and not
        // `pending-verification/{id}/resume`. That one returns a
        // `client_secret` — a bearer credential that can complete a charge —
        // and the route's own docstring is explicit that it is `billing:write`
        // for that reason. A tool response goes into a model's context and into
        // audit records; a payment-completing credential does not belong in
        // either. The link the user follows in a browser is the resume path,
        // and this tool exists to tell them a link is waiting.
        const data = await client.get("/api/v2/billing/pending-verification");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}