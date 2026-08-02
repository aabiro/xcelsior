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
}
