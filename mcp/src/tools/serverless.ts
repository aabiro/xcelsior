import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem, formatApiError } from "../client/errors.js";
import { jsonText, structuredResult } from "../lib/format.js";
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

export function registerServerlessTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "list_serverless_endpoints",
    {
      inputSchema: z.object({}),
    },
    async () => {
      const denied = scopeDenied("list_serverless_endpoints", user);
      if (denied) return denied;
      try {
        const data = await client.get("/api/v2/serverless/endpoints");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "create_serverless_endpoint",
    {
      inputSchema: z.object({
        name: z.string().min(1).max(128),
        model_ref: z.string().describe("HuggingFace model id or image ref"),
        gpu_tier: z.string().default("RTX 4090"),
        gpu_count: z.number().int().min(1).max(8).default(1),
        region: z.string().default("ca-east"),
        min_workers: z.number().int().min(0).max(32).default(0),
        max_workers: z.number().int().min(1).max(32).default(2),
        confirm: z.boolean().default(false),
        plan_id: z.string().min(1).max(160).optional(),
        idempotency_key: z.string().uuid().optional(),
      }),
      outputSchema: z.object({ preview: z.boolean().optional(), plan_id: z.string().optional() }).passthrough(),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async (args) => {
      const denied = scopeDenied("create_serverless_endpoint", user);
      if (denied) return denied;
      const { confirm, plan_id, idempotency_key, ...payload } = args;
      if (!confirm || !plan_id) {
        try {
          const plan = await client.post<Record<string, unknown>>(
            "/api/v1/serverless/endpoint-plans", payload,
            { idempotencyKey: idempotency_key, retry: idempotency_key ? "idempotent" : "none" },
          );
          return structuredResult({
            ...plan,
            approval_required: Boolean(confirm && !plan_id),
          }, "Review and approve the serverless endpoint plan before execution.");
        } catch (e) {
          return structuredResult({ preview: true, ...apiProblem(e) });
        }
      }
      try {
        const data = await client.post<Record<string, unknown>>(
          `/api/v1/serverless/endpoint-plans/${encodeURIComponent(plan_id)}/execute`,
          { confirm: true },
          { idempotencyKey: idempotency_key, retry: idempotency_key ? "idempotent" : "none" },
        );
        return structuredResult(data, `Serverless endpoint plan ${plan_id} executed.`);
      } catch (e) {
        return structuredResult({ ...apiProblem(e), plan_id });
      }
    },
  );

  server.registerTool(
    "should_i_run_pel_job",
    {
      inputSchema: z.object({
        endpoint_id: z.string().optional(),
        model_ref: z.string().optional(),
        estimated_input_tokens: z.number().int().min(0).default(1000),
        estimated_output_tokens: z.number().int().min(0).default(500),
        duration_hours: z.number().min(0).max(168).default(0.1),
        gpu_tier: z.string().default("RTX 4090"),
      }),
    },
    async (args) => {
      const denied = scopeDenied("should_i_run_pel_job", user);
      if (denied) return denied;
      try {
        const data = await client.post("/api/v2/serverless/should-i-run-this", args);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "run_serverless_job",
    {
      inputSchema: z.object({
        endpoint_id: z.string(),
        input: z.record(z.unknown()).default({}),
        webhook: z.string().url().optional(),
      }),
    },
    async ({ endpoint_id, input, webhook }) => {
      const denied = scopeDenied("run_serverless_job", user);
      if (denied) return denied;
      try {
        const body: Record<string, unknown> = { input };
        if (webhook) body.webhook = webhook;
        const data = await client.post(`/v1/serverless/${encodeURIComponent(endpoint_id)}/run`, body);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_serverless_job_status",
    {
      inputSchema: z.object({
        endpoint_id: z.string(),
        job_id: z.string(),
      }),
    },
    async ({ endpoint_id, job_id }) => {
      const denied = scopeDenied("get_serverless_job_status", user);
      if (denied) return denied;
      try {
        const data = await client.get(
          `/v1/serverless/${encodeURIComponent(endpoint_id)}/status/${encodeURIComponent(job_id)}`,
        );
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  /**
   * The exits.
   *
   * GT0 surfaced a pattern across the whole surface: entrances exist where
   * exits do not. An agent could create a serverless endpoint and run jobs on
   * it, and had no tool to stop either — the money-moving direction was
   * reachable and the money-stopping direction was not.
   *
   * Both are confirm-gated for the same reason `cancel_instance` is: they end
   * work in flight, and a preview costs one call while an unwanted cancellation
   * costs the run.
   *
   * A note on scope. `TOOL_SCOPES` requires `inference:write` for both, and the
   * MCP layer enforces it before the request is made — but the routes
   * themselves check ownership only and read no scope at all (34 of 35 on that
   * surface; see `tests/test_serverless_writes_honour_scope.py`). So the
   * enforcement here is real for anything arriving through a tool and is *not*
   * a substitute for scoping the routes. That work is tracked, not done.
   */
  server.registerTool(
    "cancel_serverless_job",
    {
      inputSchema: z.object({
        endpoint_id: z.string().min(1).max(160),
        job_id: z.string().min(1).max(160),
        confirm: z.boolean().default(false),
      }),
    },
    async ({ endpoint_id, job_id, confirm }) => {
      const denied = scopeDenied("cancel_serverless_job", user);
      if (denied) return denied;
      if (!confirm) {
        return jsonText({
          preview: true,
          endpoint_id,
          job_id,
          message:
            "Set confirm:true to cancel this inference job. It stops the work " +
            "and its spend; anything the job had not returned is lost.",
        });
      }
      try {
        return jsonText(
          // One template literal, not two concatenated. Split across a `+` the
          // path becomes unreadable to `tests/test_tools_reach_the_routes_they_call.py`,
          // which resolves every tool call site against the live route table —
          // and it read this as `POST /api/v2/serverless/endpoints/{}`, a route
          // that does not exist. A path a guard cannot parse is a path nobody
          // verifies.
          await client.post(
            `/api/v2/serverless/endpoints/${encodeURIComponent(endpoint_id)}/jobs/${encodeURIComponent(job_id)}/cancel`,
          ),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "delete_serverless_endpoint",
    {
      inputSchema: z.object({
        endpoint_id: z.string().min(1).max(160),
        confirm: z.boolean().default(false),
      }),
    },
    async ({ endpoint_id, confirm }) => {
      const denied = scopeDenied("delete_serverless_endpoint", user);
      if (denied) return denied;
      if (!confirm) {
        return jsonText({
          preview: true,
          endpoint_id,
          message:
            "Set confirm:true to delete this endpoint. It stops the endpoint " +
            "serving and ends its idle cost. Jobs still in flight on it are " +
            "cancelled, and the endpoint id stops resolving.",
        });
      }
      try {
        return jsonText(
          await client.delete(
            `/api/v2/serverless/endpoints/${encodeURIComponent(endpoint_id)}`,
          ),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}
