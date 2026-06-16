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

export function registerServerlessTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "list_serverless_endpoints",
    {
      description: "List your serverless inference endpoints.",
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
      description:
        "Create a serverless inference endpoint. Set confirm:true to deploy; confirm:false returns a preview.",
      inputSchema: z.object({
        name: z.string().min(1).max(128),
        model_ref: z.string().describe("HuggingFace model id or image ref"),
        gpu_tier: z.string().default("RTX 4090"),
        gpu_count: z.number().int().min(1).max(8).default(1),
        region: z.string().default("ca-east"),
        min_workers: z.number().int().min(0).max(32).default(0),
        max_workers: z.number().int().min(1).max(32).default(2),
        confirm: z.boolean().default(false),
      }),
    },
    async (args) => {
      const denied = scopeDenied("create_serverless_endpoint", user);
      if (denied) return denied;
      const { confirm, ...payload } = args;
      if (!confirm) {
        return jsonText({
          preview: true,
          message: "Set confirm:true to create this serverless endpoint.",
          config: payload,
        });
      }
      try {
        const data = await client.post("/api/v2/serverless/endpoints", payload);
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "run_serverless_job",
    {
      description: "Enqueue an async inference job on a serverless endpoint.",
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
      description: "Poll status for a serverless async job.",
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