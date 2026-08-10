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
