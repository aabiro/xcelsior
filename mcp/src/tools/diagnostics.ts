import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import { TOOL_SCOPES, userHasScope, scopeUnion, describeScopeRequirement } from "../auth/scopes.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem } from "../client/errors.js";
import { structuredResult } from "../lib/format.js";

const output = z.object({ ok: z.boolean().optional() }).passthrough();
const id = z.string().min(1).max(160);

function registerRead(
  server: McpServer,
  client: XcelsiorApiClient,
  user: AuthUser | undefined,
  name: keyof typeof TOOL_SCOPES,
  inputSchema: z.ZodObject<Record<string, z.ZodTypeAny>>,
  request: (args: Record<string, unknown>) => Promise<unknown>,
): void {
  server.registerTool(name, {
    inputSchema,
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, async (args) => {
    const required = TOOL_SCOPES[name];
    if (!userHasScope(user?.scopes, required)) {
      return structuredResult({ ok: false, code: "insufficient_scope", required }, `Access denied: requires ${describeScopeRequirement(required)}.`);
    }
    try {
      const value = await request(args) as Record<string, unknown>;
      return structuredResult(value, `${name} completed.`);
    } catch (error) {
      return structuredResult(apiProblem(error), `${name} failed.`);
    }
  });
}

/** The write counterpart of `registerRead`. Same scope refusal and error
 *  shaping; POSTs, and does not claim to be read-only. Annotations are set
 *  from the contract by `installToolAudit` regardless, so these are the
 *  honest values rather than the authoritative ones. */
function registerWrite(
  server: McpServer,
  client: XcelsiorApiClient,
  user: AuthUser | undefined,
  name: keyof typeof TOOL_SCOPES,
  inputSchema: z.ZodObject<Record<string, z.ZodTypeAny>>,
  request: (args: Record<string, unknown>) => Promise<unknown>,
): void {
  server.registerTool(name, {
    inputSchema,
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, async (args) => {
    const required = TOOL_SCOPES[name];
    if (!userHasScope(user?.scopes, required)) {
      return structuredResult({ ok: false, code: "insufficient_scope", required }, `Access denied: requires ${describeScopeRequirement(required)}.`);
    }
    try {
      const value = await request(args) as Record<string, unknown>;
      return structuredResult(value, `${name} completed.`);
    } catch (error) {
      return structuredResult(apiProblem(error), `${name} failed.`);
    }
  });
}

export function registerDiagnosticTools(server: McpServer, client: XcelsiorApiClient, user?: AuthUser): void {
  registerRead(server, client, user, "explain_instance_placement", z.object({ job_id: id }), a => client.get(`/api/v1/instances/${encodeURIComponent(String(a.job_id))}/placement-explanation`));
  registerRead(server, client, user, "simulate_instance_placement", z.object({ spec: z.record(z.unknown()) }), a => client.post("/api/v1/placements/simulate", a.spec));
  registerRead(server, client, user, "evaluate_placement_preference", z.object({
    spec: z.record(z.unknown()),
    preference: z.object({
      min_uptime_pct: z.number().min(0).max(100).optional(),
      min_tier: z.string().max(32).optional(),
      require_verified: z.boolean().default(false),
      max_premium_pct: z.number().min(0).max(10000).optional(),
    }),
  }), a => client.post("/api/v1/placements/evaluate", { spec: a.spec, preference: a.preference }));
  registerRead(server, client, user, "get_instance_timeline", z.object({ job_id: id }), a => client.get(`/api/v1/instances/${encodeURIComponent(String(a.job_id))}/timeline`));
  registerRead(server, client, user, "get_event_history",
    z.object({
      entity_type: z.enum(["job", "host"]).describe("Which kind of thing to read events for"),
      entity_id: id.describe("A job_id from list_instances, or a host_id"),
      limit: z.number().int().min(1).max(200).default(50),
    }),
    a => client.get(`/api/events/${encodeURIComponent(String(a.entity_type))}/${encodeURIComponent(String(a.entity_id))}`, { limit: Number(a.limit ?? 50) }));
  registerRead(server, client, user, "list_providers",
    z.object({ status: z.string().max(40).default("") }),
    a => client.get("/api/providers", { status: String(a.status ?? "") }));
  registerRead(server, client, user, "get_provider_account",
    z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}`));
  registerWrite(server, client, user, "claim_reputation_milestones",
    z.object({}),
    () => client.post("/api/reputation/me/claim", {}));
  registerRead(server, client, user, "get_reputation_leaderboard",
    z.object({ entity_type: z.enum(["host", "user"]).default("host"), limit: z.number().int().min(1).max(100).default(20) }),
    a => client.get("/api/reputation/leaderboard", { entity_type: String(a.entity_type ?? "host"), limit: Number(a.limit ?? 20) }));
  registerRead(server, client, user, "get_reputation_history",
    z.object({ entity_id: id.describe("The caller's own provider_id from list_providers, or a host_id from get_host_capacity"), limit: z.number().int().min(1).max(200).default(50) }),
    a => client.get(`/api/reputation/${encodeURIComponent(String(a.entity_id))}/history`, { limit: Number(a.limit ?? 50) }));
  registerWrite(server, client, user, "request_provider_payout",
    z.object({
      provider_id: id.describe("From list_providers, or the caller's own"),
      job_id: id.describe("A completed instance from list_instances"),
      payment_rail: z.enum(["stripe", "paypal"]).default("stripe"),
    }),
    a => client.post(
      `/api/providers/${encodeURIComponent(String(a.provider_id))}/payout`
      + `?job_id=${encodeURIComponent(String(a.job_id))}`
      + `&payment_rail=${encodeURIComponent(String(a.payment_rail ?? "stripe"))}`,
      {}));
  registerWrite(server, client, user, "register_provider",
    z.object({
      provider_type: z.enum(["individual", "company"]).default("individual"),
      legal_name: z.string().max(200).default(""),
      province: z.string().max(40).default("").describe("ON, QC, BC … or a region code"),
      country: z.string().max(2).default("CA").describe("ISO-3166 alpha-2"),
      corporation_name: z.string().max(200).default("").describe("Required for provider_type=company"),
      business_number: z.string().max(40).default(""),
      gst_hst_number: z.string().max(40).default(""),
    }),
    async a => {
      const data = await client.post("/api/providers/register", {
        // The API derives the provider identity from the credential and ignores
        // any id sent, so none is sent. `email` must match the caller's own or
        // the route refuses — it is not a parameter the model gets to choose.
        provider_id: "",
        email: user?.email ?? "",
        provider_type: String(a.provider_type ?? "individual"),
        legal_name: String(a.legal_name ?? ""),
        province: String(a.province ?? ""),
        country: String(a.country ?? "CA"),
        corporation_name: String(a.corporation_name ?? ""),
        business_number: String(a.business_number ?? ""),
        gst_hst_number: String(a.gst_hst_number ?? ""),
      }) as Record<string, unknown>;
      // The route serves the browser too, so it returns an AccountLink and the
      // Connect account id. Neither leaves here: an AccountLink can set the
      // external bank account, which makes the URL a payout-destination-change
      // capability, and Stripe's own guidance is not to distribute it. Same
      // call as `list_pending_verifications` declining the `client_secret`.
      delete data.onboarding_url;
      delete data.stripe_account_id;
      return { ...data, finish_onboarding_at: "the earnings page in the dashboard" };
    });
  registerRead(server, client, user, "get_paypal_status",
    z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}/paypal`));
  registerRead(server, client, user, "get_my_reputation",
    z.object({}),
    () => client.get("/api/reputation/me"));
  registerRead(server, client, user, "get_reputation_journey",
    z.object({}),
    () => client.get("/api/reputation/me/journey"));
  registerRead(server, client, user, "get_trust_tiers",
    z.object({}),
    () => client.get("/api/trust-tiers"));
  registerRead(server, client, user, "get_reputation_breakdown",
    z.object({ entity_id: id.describe("The caller's own provider_id from list_providers, or a host_id from get_host_capacity") }),
    a => client.get(`/api/reputation/${encodeURIComponent(String(a.entity_id))}/breakdown`));
  registerRead(server, client, user, "get_provider_earnings",
    z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}/earnings`));
  registerRead(server, client, user, "get_active_lease", z.object({ job_id: id }), a => client.get(`/api/v1/instances/${encodeURIComponent(String(a.job_id))}/active-lease`));
  registerRead(server, client, user, "get_scheduler_health", z.object({}), () => client.get("/api/v1/control-plane/health"));
  registerRead(server, client, user, "get_host_capacity", z.object({ host_id: id }), a => client.get(`/api/v1/hosts/${encodeURIComponent(String(a.host_id))}/capacity`));
  registerRead(server, client, user, "list_reconciliation_findings", z.object({
    status: z.enum(["open", "resolved", "all"]).default("open"),
    cursor: z.string().max(512).optional(),
    limit: z.number().int().min(1).max(200).default(100),
  }), a => client.get("/api/v1/control-plane/reconciliation-findings", {
    status: String(a.status), cursor: a.cursor ? String(a.cursor) : undefined, limit: Number(a.limit),
  }));
  registerRead(server, client, user, "get_mcp_action_status", z.object({ plan_id: id }), a => client.get(`/api/v1/launch-plans/${encodeURIComponent(String(a.plan_id))}`));
}
