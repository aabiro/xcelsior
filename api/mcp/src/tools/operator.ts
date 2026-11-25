import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import { scopeDenial } from "../auth/denial.js";
import { TOOL_SCOPES, userHasScope } from "../auth/scopes.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem } from "../client/errors.js";
import { structuredResult } from "../lib/format.js";
import { readPreviewTarget } from "../lib/preview-target.js";

const output = z.object({ ok: z.boolean().optional() }).passthrough();
const base = { expected_version: z.number().int().nonnegative(), idempotency_key: z.string().uuid() };

export function registerOperatorTools(server: McpServer, client: XcelsiorApiClient, user?: AuthUser): void {
  server.registerTool("retry_instance", {
    inputSchema: z.object({ job_id: z.string().min(1).max(160), ...base, confirm: z.boolean().default(false).describe("false (the default) previews and changes nothing; true acts"), }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  }, async (args) => args.confirm
    ? mutate("retry_instance", `/api/v1/instances/${encodeURIComponent(args.job_id)}/retry`, args)
    : previewInstance("retry_instance", args.job_id,
      "Would queue it again; it bills again from when it starts running."));

  server.registerTool("reconcile_instance", {
    inputSchema: z.object({ job_id: z.string().min(1).max(160), ...base, confirm: z.boolean().default(false).describe("false (the default) previews and changes nothing; true acts"), }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
  }, async (args) => args.confirm
    ? mutate("reconcile_instance", `/api/v1/instances/${encodeURIComponent(args.job_id)}/reconcile`, args)
    : previewInstance("reconcile_instance", args.job_id,
      "Would correct its recorded state to match what the host reports; it does not start, stop or move the workload, or change its billing."));

  server.registerTool("drain_host", {
    inputSchema: z.object({ host_id: z.string().min(1).max(160), reason: z.string().min(1).max(500), ...base }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, async (args) => mutate("drain_host", `/api/v1/hosts/${encodeURIComponent(args.host_id)}/drain`, args));

  server.registerTool("undrain_host", {
    inputSchema: z.object({ host_id: z.string().min(1).max(160), reason: z.string().min(1).max(500), ...base }),
    outputSchema: output,
    annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
  }, async (args) => mutate("undrain_host", `/api/v1/hosts/${encodeURIComponent(args.host_id)}/undrain`, args));

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
    if (!userHasScope(user?.scopes, required)) return structuredResult({ ok: false, code: "insufficient_scope", ...scopeDenial("evict_host_workloads", user) });
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
      const summary = result.pending === true
        ? `Eviction plan ${args.plan_id} is pending fenced worker settlement.`
        : result.status === "succeeded"
          ? `Eviction plan ${args.plan_id} settled.`
          : `Eviction plan ${args.plan_id} request accepted.`;
      return structuredResult(result, summary);
    } catch (error) {
      return structuredResult(apiProblem(error));
    }
  });

  /** Read the instance an act would touch, and describe the act; change nothing. */
  async function previewInstance(name: keyof typeof TOOL_SCOPES, jobId: string, effect: string) {
    const required = TOOL_SCOPES[name];
    if (!userHasScope(user?.scopes, required)) return structuredResult({ ok: false, code: "insufficient_scope", ...scopeDenial(name, user) });
    const target = await readPreviewTarget(
      () => client.get(`/api/v1/instances/${encodeURIComponent(jobId)}`),
      `instance ${jobId}`,
      "list_instances",
    );
    if (!target.found) return target.result;
    const inst = (target.value.instance ?? target.value.job ?? target.value) as Record<string, unknown>;
    return structuredResult(
      { ok: true, preview: true, job_id: jobId, status: inst.status ?? null, version: inst.version ?? null },
      `${inst.name ?? jobId} is ${inst.status ?? "in an unknown state"}. ${effect} Nothing has changed. Call again with confirm:true.`,
    );
  }

  async function mutate(name: keyof typeof TOOL_SCOPES, path: string, args: Record<string, unknown>) {
    const required = TOOL_SCOPES[name];
    if (!userHasScope(user?.scopes, required)) return structuredResult({ ok: false, code: "insufficient_scope", ...scopeDenial(name, user) });
    try {
      const body = { ...args }; delete body.job_id; delete body.host_id; delete body.idempotency_key; delete body.confirm;
      const data = await client.post<Record<string, unknown>>(path, body, { idempotencyKey: String(args.idempotency_key), retry: "idempotent" });
      return structuredResult(data, `${name} accepted.`);
    } catch (error) {
      return structuredResult(apiProblem(error), `${name} failed.`);
    }
  }
}
