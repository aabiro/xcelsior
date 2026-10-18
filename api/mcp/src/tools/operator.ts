import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import { TOOL_SCOPES, userHasScope } from "../auth/scopes.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem } from "../client/errors.js";
import { structuredResult } from "../lib/format.js";

const output = z.object({ ok: z.boolean().optional() }).passthrough();
const base = { expected_version: z.number().int().nonnegative(), idempotency_key: z.string().uuid() };

export function registerOperatorTools(server: McpServer, client: XcelsiorApiClient, user?: AuthUser): void {
  const tools = [
    ["retry_instance", "instances:operate", z.object({ job_id: z.string().min(1).max(160), ...base }), (a: any) => `/api/v1/instances/${encodeURIComponent(a.job_id)}/retry`],
    ["reconcile_instance", "instances:operate", z.object({ job_id: z.string().min(1).max(160), ...base }), (a: any) => `/api/v1/instances/${encodeURIComponent(a.job_id)}/reconcile`],
    ["drain_host", "hosts:operate", z.object({ host_id: z.string().min(1).max(160), reason: z.string().min(1).max(500), ...base }), (a: any) => `/api/v1/hosts/${encodeURIComponent(a.host_id)}/drain`],
    ["undrain_host", "hosts:operate", z.object({ host_id: z.string().min(1).max(160), ...base }), (a: any) => `/api/v1/hosts/${encodeURIComponent(a.host_id)}/undrain`],
  ] as const;
  for (const [name, , inputSchema, path] of tools) {
    server.registerTool(name, {
      inputSchema, outputSchema: output,
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    }, async (args: Record<string, unknown>) => mutate(name, path(args), args));
  }

  server.registerTool("retry_agent_command", {
    inputSchema: z.object({ command_id: z.string().uuid(), ...base }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, async (args) => mutate(
    "retry_agent_command",
    `/api/v1/control-plane/commands/${encodeURIComponent(args.command_id)}/retry`,
    args,
  ));

  server.registerTool("evict_host_workloads", {
    inputSchema: z.object({ host_id: z.string().min(1).max(160), reason: z.string().min(1).max(500), confirm: z.boolean().default(false), plan_id: z.string().max(160).optional(), ...base }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  }, async (args) => {
    const required = TOOL_SCOPES.evict_host_workloads;
    if (!userHasScope(user?.scopes, required)) return structuredResult({ ok: false, code: "insufficient_scope", required });
    try {
      if (!args.confirm || !args.plan_id) {
        const plan = await client.post<Record<string, unknown>>(
          `/api/v1/hosts/${encodeURIComponent(args.host_id)}/eviction-plans`,
          { expected_version: args.expected_version, reason: args.reason },
          { idempotencyKey: args.idempotency_key, retry: "idempotent" },
        );
        return structuredResult({ ...plan, approval_required: true });
      }
      const result = await client.post<Record<string, unknown>>(
        `/api/v1/hosts/${encodeURIComponent(args.host_id)}/eviction-plans/${encodeURIComponent(args.plan_id)}/execute`,
        {}, { idempotencyKey: args.idempotency_key, retry: "idempotent" },
      );
      return structuredResult(result, `Eviction plan ${args.plan_id} executed.`);
    } catch (error) {
      return structuredResult(apiProblem(error));
    }
  });

  async function mutate(name: keyof typeof TOOL_SCOPES, path: string, args: Record<string, unknown>) {
    const required = TOOL_SCOPES[name];
    if (!userHasScope(user?.scopes, required)) return structuredResult({ ok: false, code: "insufficient_scope", required });
    try {
      const body = { ...args }; delete body.job_id; delete body.host_id; delete body.idempotency_key;
      const data = await client.post<Record<string, unknown>>(path, body, { idempotencyKey: String(args.idempotency_key), retry: "idempotent" });
      return structuredResult(data, `${name} accepted.`);
    } catch (error) {
      return structuredResult(apiProblem(error), `${name} failed.`);
    }
  }
}
