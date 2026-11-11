import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import { TOOL_SCOPES, userHasScope, describeScopeRequirement } from "../auth/scopes.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem } from "../client/errors.js";
import { structuredResult } from "../lib/format.js";

const output = z.object({ ok: z.boolean().optional() }).passthrough();
const id = z.string().min(1).max(160);

/** Shared scope check and error shaping for diagnostic and provider tools.
 *  Each tool calls `server.registerTool` directly with inline `inputSchema`
 *  and literal boolean `annotations` so static analyzers see every schema and
 *  hint without evaluating runtime helpers; `installToolAudit` still enforces
 *  contract parity at registration time. */
function diagnosticHandler(
  user: AuthUser | undefined,
  name: keyof typeof TOOL_SCOPES,
  request: (args: Record<string, unknown>) => Promise<unknown>,
) {
  return async (args: Record<string, unknown>) => {
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
  };
}

export function registerDiagnosticTools(server: McpServer, client: XcelsiorApiClient, user?: AuthUser): void {
  server.registerTool("explain_instance_placement", {
    inputSchema: z.object({ job_id: id }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "explain_instance_placement", a => client.get(`/api/v1/instances/${encodeURIComponent(String(a.job_id))}/placement-explanation`)));

  server.registerTool("simulate_instance_placement", {
    inputSchema: z.object({ spec: z.record(z.string(), z.unknown()) }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "simulate_instance_placement", a => client.post("/api/v1/placements/simulate", a.spec)));

  server.registerTool("evaluate_placement_preference", {
    inputSchema: z.object({
      spec: z.record(z.string(), z.unknown()),
      preference: z.object({
        min_uptime_pct: z.number().min(0).max(100).optional(),
        min_tier: z.string().max(32).optional(),
        require_verified: z.boolean().default(false),
        max_premium_pct: z.number().min(0).max(10000).optional(),
      }),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "evaluate_placement_preference", a => client.post("/api/v1/placements/evaluate", { spec: a.spec, preference: a.preference })));

  server.registerTool("get_instance_timeline", {
    inputSchema: z.object({ job_id: id }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_instance_timeline", a => client.get(`/api/v1/instances/${encodeURIComponent(String(a.job_id))}/timeline`)));

  server.registerTool("get_event_history", {
    inputSchema: z.object({
      entity_type: z.enum(["job", "host"]).describe("Which kind of thing to read events for"),
      entity_id: id.describe("A job_id from list_instances, or a host_id"),
      limit: z.number().int().min(1).max(200).default(50),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_event_history", a => client.get(`/api/events/${encodeURIComponent(String(a.entity_type))}/${encodeURIComponent(String(a.entity_id))}`, { limit: Number(a.limit ?? 50) })));

  server.registerTool("list_providers", {
    inputSchema: z.object({ status: z.string().max(40).default("") }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "list_providers", a => client.get("/api/providers", { status: String(a.status ?? "") })));

  server.registerTool("get_provider_account", {
    inputSchema: z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_provider_account", a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}`)));

  server.registerTool("claim_reputation_milestones", {
    inputSchema: z.object({}),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "claim_reputation_milestones", () => client.post("/api/reputation/me/claim", {})));

  server.registerTool("get_reputation_leaderboard", {
    inputSchema: z.object({ entity_type: z.enum(["host", "user"]).default("host"), limit: z.number().int().min(1).max(100).default(20) }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_reputation_leaderboard", a => client.get("/api/reputation/leaderboard", { entity_type: String(a.entity_type ?? "host"), limit: Number(a.limit ?? 20) })));

  server.registerTool("get_reputation_history", {
    inputSchema: z.object({ entity_id: id.describe("The caller's own provider_id from list_providers, or a host_id from get_host_capacity"), limit: z.number().int().min(1).max(200).default(50) }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_reputation_history", a => client.get(`/api/reputation/${encodeURIComponent(String(a.entity_id))}/history`, { limit: Number(a.limit ?? 50) })));

  server.registerTool("request_provider_payout", {
    inputSchema: z.object({
      provider_id: id.describe("From list_providers, or the caller's own"),
      job_id: id.describe("A completed instance from list_instances"),
      payment_rail: z.enum(["stripe", "paypal", "wise"]).default("stripe"),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "request_provider_payout", a => client.post(
    `/api/providers/${encodeURIComponent(String(a.provider_id))}/payout`
    + `?job_id=${encodeURIComponent(String(a.job_id))}`
    + `&payment_rail=${encodeURIComponent(String(a.payment_rail ?? "stripe"))}`,
    {})));

  server.registerTool("register_provider", {
    inputSchema: z.object({
      provider_type: z.enum(["individual", "company"]).default("individual"),
      legal_name: z.string().max(200).default(""),
      province: z.string().max(40).default("").describe("ON, QC, BC … or a region code"),
      country: z.string().max(2).default("").describe("ISO-3166 alpha-2; omit to use the saved seller profile"),
      corporation_name: z.string().max(200).default("").describe("Required for provider_type=company"),
      business_number: z.string().max(40).default(""),
      gst_hst_number: z.string().max(40).default(""),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "register_provider", async a => {
    const data = await client.post("/api/providers/register", {
      // The API derives the provider identity from the credential and ignores
      // any id sent, so none is sent. `email` must match the caller's own or
      // the route refuses — it is not a parameter the model gets to choose.
      provider_id: "",
      email: user?.email ?? "",
      provider_type: String(a.provider_type ?? "individual"),
      legal_name: String(a.legal_name ?? ""),
      province: String(a.province ?? ""),
      country: String(a.country ?? ""),
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
    return { ...data, finish_onboarding_at: "https://xcelsior.ca/onboarding/provider" };
  }));

  server.registerTool("get_paypal_status", {
    inputSchema: z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_paypal_status", a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}/paypal`)));

  server.registerTool("get_wise_status", {
    inputSchema: z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_wise_status", a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}/wise`)));

  server.registerTool("get_my_reputation", {
    inputSchema: z.object({}),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_my_reputation", () => client.get("/api/reputation/me")));

  server.registerTool("get_reputation_journey", {
    inputSchema: z.object({}),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_reputation_journey", () => client.get("/api/reputation/me/journey")));

  server.registerTool("get_trust_tiers", {
    inputSchema: z.object({}),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_trust_tiers", () => client.get("/api/trust-tiers")));

  server.registerTool("get_reputation_breakdown", {
    inputSchema: z.object({ entity_id: id.describe("The caller's own provider_id from list_providers, or a host_id from get_host_capacity") }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_reputation_breakdown", a => client.get(`/api/reputation/${encodeURIComponent(String(a.entity_id))}/breakdown`)));

  server.registerTool("get_provider_earnings", {
    inputSchema: z.object({ provider_id: id.describe("From list_providers, or the caller's own") }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_provider_earnings", a => client.get(`/api/providers/${encodeURIComponent(String(a.provider_id))}/earnings`)));

  server.registerTool("get_active_lease", {
    inputSchema: z.object({ job_id: id }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_active_lease", a => client.get(`/api/v1/instances/${encodeURIComponent(String(a.job_id))}/active-lease`)));

  server.registerTool("get_scheduler_health", {
    inputSchema: z.object({}),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_scheduler_health", () => client.get("/api/v1/control-plane/health")));

  server.registerTool("get_host_capacity", {
    inputSchema: z.object({ host_id: id }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_host_capacity", a => client.get(`/api/v1/hosts/${encodeURIComponent(String(a.host_id))}/capacity`)));

  server.registerTool("list_reconciliation_findings", {
    inputSchema: z.object({
      status: z.enum(["open", "resolved", "all"]).default("open"),
      cursor: z.string().max(512).optional(),
      limit: z.number().int().min(1).max(200).default(100),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "list_reconciliation_findings", a => client.get("/api/v1/control-plane/reconciliation-findings", {
    status: String(a.status), cursor: a.cursor ? String(a.cursor) : undefined, limit: Number(a.limit),
  })));

  // Slurm: read the reconciler's copy. These never reach a cluster, which is
  // the point: agents can watch without adding load (Alliance's automation
  // nodes are "not suitable for usage of AI agents").
  server.registerTool("list_hpc_clusters", {
    inputSchema: z.object({}),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "list_hpc_clusters", () => client.get("/api/v1/hpc/clusters")));

  server.registerTool("list_hpc_jobs", {
    inputSchema: z.object({
      cluster_id: id.optional(),
      state: z.enum(["open", "settled", "all"]).default("open"),
      cursor: z.string().max(512).optional(),
      limit: z.number().int().min(1).max(200).default(100),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "list_hpc_jobs", a => client.get("/api/v1/hpc/jobs", {
    cluster_id: a.cluster_id ? String(a.cluster_id) : undefined,
    state: String(a.state), cursor: a.cursor ? String(a.cursor) : undefined, limit: Number(a.limit),
  })));

  server.registerTool("get_hpc_job", {
    inputSchema: z.object({ xcelsior_job_id: id }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_hpc_job", a => client.get(`/api/v1/hpc/jobs/${encodeURIComponent(String(a.xcelsior_job_id))}`)));

  server.registerTool("get_hpc_job_output", {
    inputSchema: z.object({
      xcelsior_job_id: id,
      stream: z.enum(["out", "err"]).default("out"),
    }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_hpc_job_output", a => client.get(`/api/v1/hpc/jobs/${encodeURIComponent(String(a.xcelsior_job_id))}/output`, {
    stream: String(a.stream),
  })));

  server.registerTool("get_mcp_action_status", {
    inputSchema: z.object({ plan_id: id }),
    outputSchema: output,
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, diagnosticHandler(user, "get_mcp_action_status", a => client.get(`/api/v1/launch-plans/${encodeURIComponent(String(a.plan_id))}`)));
}
