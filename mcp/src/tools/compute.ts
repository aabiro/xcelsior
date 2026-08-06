import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { ApiError, apiProblem, formatApiError } from "../client/errors.js";
import { jsonText, structuredResult } from "../lib/format.js";
import { inspectSshKeyInput } from "../lib/ssh-key.js";
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

  server.registerTool(
    "register_ssh_key",
    {
      inputSchema: z.object({
        public_key: z
          .string()
          .min(1)
          .describe(
            "Your SSH *public* key — the contents of a .pub file, e.g. " +
              "'ssh-ed25519 AAAAC3... you@laptop'. Never send a private key.",
          ),
        name: z
          .string()
          .optional()
          .describe("A label for this key, e.g. 'laptop'. Defaults to 'default'."),
      }),
    },
    async ({ public_key, name }) => {
      const denied = scopeDenied("register_ssh_key", user);
      if (denied) return denied;

      // Classified before anything is sent: the one tool whose wrong argument
      // is itself a secret. See lib/ssh-key.ts.
      const inspection = inspectSshKeyInput(public_key);
      if (inspection.verdict !== "public") {
        return jsonText({
          ok: false,
          error:
            inspection.verdict === "private"
              ? "private_key_supplied"
              : "not_an_ssh_public_key",
          message: inspection.message,
        });
      }

      try {
        const data = await client.post("/api/ssh/keys", {
          public_key: inspection.key,
          name: name ?? "default",
        });
        return jsonText({
          ...(data as Record<string, unknown>),
          note:
            "Registered. Running interactive instances you own pick this up " +
            "without relaunching. Anyone holding the matching private key can " +
            "open a shell on them.",
        });
      } catch (e) {
        // The contract publishes `idempotentHint: true` for this tool, and the
        // endpoint is not: a second POST of the same key returns 409 "This key
        // is already added". A model that retried a call whose response was
        // lost would read that 409 as a failure and tell the user their key is
        // not registered — when it is. The desired end state holds, so report
        // it as reached, and say it was already there rather than claiming to
        // have done something.
        if (e instanceof ApiError && e.status === 409) {
          return jsonText({
            ok: true,
            already_registered: true,
            message:
              "That key was already registered on this account; nothing " +
              "changed. Your running interactive instances already accept it.",
          });
        }
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}
