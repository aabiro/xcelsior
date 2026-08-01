import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem, formatApiError } from "../client/errors.js";
import { jsonText, structuredResult } from "../lib/format.js";
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

export function registerComputeTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "list_instances",
    {
      inputSchema: z.object({
        status: z
          .string()
          .optional()
          .describe("Filter: queued, assigned, starting, running, completed, failed, cancelled"),
        cursor: z.string().max(128).optional(),
        limit: z.number().int().min(1).max(200).default(100),
      }),
    },
    async ({ status, cursor, limit }) => {
      const denied = scopeDenied("list_instances", user);
      if (denied) return denied;
      try {
        const data = await client.get<Record<string, unknown>>("/instances", status ? { status } : undefined);
        const rows = Array.isArray(data.instances) ? data.instances : [];
        let offset = 0;
        if (cursor) {
          try { offset = Number(Buffer.from(cursor, "base64url").toString("utf8")); } catch { offset = -1; }
          if (!Number.isSafeInteger(offset) || offset < 0) return jsonText({ ok: false, code: "invalid_cursor" });
        }
        const page = rows.slice(offset, offset + limit);
        return jsonText({
          ...data, instances: page,
          next_cursor: offset + limit < rows.length
            ? Buffer.from(String(offset + limit)).toString("base64url") : null,
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_instance",
    {
      inputSchema: z.object({
        job_id: z.string().describe("Instance job ID"),
      }),
    },
    async ({ job_id }) => {
      const denied = scopeDenied("get_instance", user);
      if (denied) return denied;
      try {
        const data = await client.get(`/api/v1/instances/${encodeURIComponent(job_id)}`);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_instance_logs",
    {
      inputSchema: z.object({
        job_id: z.string(),
        limit: z.number().int().min(1).max(500).default(100),
      }),
    },
    async ({ job_id, limit }) => {
      const denied = scopeDenied("get_instance_logs", user);
      if (denied) return denied;
      try {
        const data = await client.get(`/instances/${encodeURIComponent(job_id)}/logs`, { limit });
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "create_instance",
    {
      inputSchema: z.object({
        name: z.string().min(1).max(128),
        vram_needed_gb: z.number().min(0).default(0),
        num_gpus: z.number().int().min(1).max(64).default(1),
        gpu_model: z.string().optional(),
        host_id: z.string().optional(),
        image: z.string().optional(),
        git_repo: z.string().max(512).optional(),
        init_script: z.string().max(4096).optional(),
        pricing_mode: z.enum(["on_demand", "spot"]).default("on_demand"),
        interactive: z.boolean().default(true),
        confirm: z.boolean().default(false),
        plan_id: z.string().min(1).max(160).optional(),
        idempotency_key: z.string().uuid().optional(),
      }),
      outputSchema: z.object({
        preview: z.boolean().optional(),
        plan_id: z.string().optional(),
        approval_state: z.string().optional(),
        canonical_spec: z.record(z.unknown()).optional(),
        estimate: z.record(z.unknown()).optional(),
        availability: z.record(z.unknown()).optional(),
        approval_url: z.string().optional(),
        expires_at: z.string().optional(),
      }).passthrough(),
      annotations: {
        readOnlyHint: false,
        destructiveHint: false,
        idempotentHint: true,
        openWorldHint: false,
      },
    },
    async (args) => {
      const denied = scopeDenied("create_instance", user);
      if (denied) return denied;

      const { confirm, plan_id, idempotency_key, ...payload } = args;
      if (!confirm || !plan_id) {
        try {
          const plan = await client.post<Record<string, unknown>>("/api/v1/launch-plans", payload, {
            idempotencyKey: idempotency_key,
            retry: idempotency_key ? "idempotent" : "none",
          });
          const result = {
            ...plan,
            preview: true,
            approval_required: Boolean(confirm && !plan_id),
            message: confirm && !plan_id
              ? "Approval is required. Approve this prepared plan, then call create_instance with confirm:true and plan_id."
              : "Review and approve this launch plan before execution.",
          };
          return structuredResult(result, String(result.message));
        } catch (e) {
          return structuredResult({ preview: true, ...apiProblem(e) });
        }
      }

      try {
        const data = await client.post<Record<string, unknown>>(
          `/api/v1/launch-plans/${encodeURIComponent(plan_id)}/execute`,
          { confirm: true },
          { idempotencyKey: idempotency_key, retry: idempotency_key ? "idempotent" : "none" },
        );
        return structuredResult(data, `Launch plan ${plan_id} executed.`);
      } catch (e) {
        return structuredResult({ ...apiProblem(e), plan_id });
      }
    },
  );

  server.registerTool(
    "cancel_instance",
    {
      inputSchema: z.object({
        job_id: z.string(),
        confirm: z.boolean().default(false),
      }),
    },
    async ({ job_id, confirm }) => {
      const denied = scopeDenied("cancel_instance", user);
      if (denied) return denied;
      if (!confirm) {
        return jsonText({
          preview: true,
          message: "Set confirm:true to cancel this instance.",
          job_id,
        });
      }
      try {
        const data = await client.post(`/instances/${encodeURIComponent(job_id)}/cancel`);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "terminate_instance",
    {
      inputSchema: z.object({
        job_id: z.string(),
        confirm: z.boolean().default(false),
      }),
    },
    async ({ job_id, confirm }) => {
      const denied = scopeDenied("terminate_instance", user);
      if (denied) return denied;
      if (!confirm) {
        return jsonText({
          preview: true,
          message: "Set confirm:true to permanently terminate this instance.",
          job_id,
        });
      }
      try {
        const data = await client.post(`/instances/${encodeURIComponent(job_id)}/terminate`);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}
