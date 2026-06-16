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

export function registerComputeTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "list_instances",
    {
      description: "List your GPU instances (jobs). Optionally filter by status.",
      inputSchema: z.object({
        status: z
          .string()
          .optional()
          .describe("Filter: queued, assigned, starting, running, completed, failed, cancelled"),
      }),
    },
    async ({ status }) => {
      const denied = scopeDenied("list_instances", user);
      if (denied) return denied;
      try {
        const data = await client.get("/instances", status ? { status } : undefined);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_instance",
    {
      description: "Get details for a single instance by job_id.",
      inputSchema: z.object({
        job_id: z.string().describe("Instance job ID"),
      }),
    },
    async ({ job_id }) => {
      const denied = scopeDenied("get_instance", user);
      if (denied) return denied;
      try {
        const data = await client.get(`/instance/${encodeURIComponent(job_id)}`);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_instance_logs",
    {
      description: "Get buffered log lines for an instance (non-streaming).",
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
      description:
        "Create a GPU instance. Set confirm:true to launch; confirm:false returns cost preview only.",
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
      }),
    },
    async (args) => {
      const denied = scopeDenied("create_instance", user);
      if (denied) return denied;

      const { confirm, ...payload } = args;
      if (!confirm) {
        try {
          const estimate = await client.post("/api/pricing/estimate", {
            gpu_model: args.gpu_model || "RTX 4090",
            duration_hours: 1,
            spot: args.pricing_mode === "spot",
          });
          return jsonText({
            preview: true,
            message: "Set confirm:true to create this instance.",
            config: payload,
            estimate,
          });
        } catch (e) {
          return jsonText({ error: formatApiError(e) });
        }
      }

      try {
        const data = await client.post("/instance", payload);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "cancel_instance",
    {
      description: "Cancel a queued or running instance. Requires confirm:true.",
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
      description: "Permanently terminate an instance (irreversible). Requires confirm:true.",
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