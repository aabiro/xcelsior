import { createHash } from "node:crypto";
import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { ApiError, apiProblem, formatApiError } from "../client/errors.js";
import { jsonText, structuredResult } from "../lib/format.js";
import { notOnThisAccount, readPreviewTarget } from "../lib/preview-target.js";
import { inspectSshKeyInput } from "../lib/ssh-key.js";
import { loadConfig } from "../config.js";
import { scopeDenial } from "../auth/denial.js";
import type { AuthUser } from "../auth/bearer.js";

/**
 * Where a user's SSH lands, and where the terminal websocket lives.
 *
 * Read once at module load from the same env var the rest of the platform
 * uses. The job record's `host_ip` is the tailnet address — the dashboard shows
 * it under "Direct SSH (requires mesh network)" — so it is not what a user off
 * the mesh connects to, and `open_instance_access` must not hand it out as if
 * it were.
 */
const { sshHost, apiUrl } = loadConfig();
const wsBase = apiUrl.replace(/^http/, "ws");

function scopeDenied(tool: string, user: AuthUser | undefined) {
  const denial = scopeDenial(tool, user);
  return denial ? jsonText(denial) : null;
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
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
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
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
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
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
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
        tier: z.enum(["standard", "premium", "dedicated"]).optional().describe("Paid service tier. Dedicated reserves the entire physical host and requires its full GPU count."),
        service_tier_price_id: z.string().max(128).optional(),
        exclusive_gpu: z.boolean().optional().describe("Optional paid GPU-only exclusivity for Standard or Premium. Included in Dedicated."),
        gpu_exclusive_price_id: z.string().max(128).optional(),
        interactive: z.boolean().default(true),
        confirm: z.boolean().default(false),
        plan_id: z.string().min(1).max(160).optional(),
        idempotency_key: z.string().uuid().optional(),
      }),
      outputSchema: z.object({
        preview: z.boolean().optional(),
        plan_id: z.string().optional(),
        approval_state: z.string().optional(),
        canonical_spec: z.record(z.string(), z.unknown()).optional(),
        estimate: z.record(z.string(), z.unknown()).optional(),
        availability: z.record(z.string(), z.unknown()).optional(),
        approval_url: z.string().optional(),
        expires_at: z.string().optional(),
      }).passthrough(),
      annotations: {
        readOnlyHint: false,
        destructiveHint: true,
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
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    },
    async ({ job_id, confirm }) => {
      const denied = scopeDenied("cancel_instance", user);
      if (denied) return denied;
      if (!confirm) {
        const target = await readPreviewTarget(
          () => client.get(`/api/v1/instances/${encodeURIComponent(job_id)}`),
          `instance ${job_id}`,
          "list_instances",
        );
        if (!target.found) return target.result;
        const instance = (target.value.instance ?? target.value.job ?? target.value) as Record<string, unknown>;
        return jsonText({
          ok: true,
          preview: true,
          message: `Set confirm:true to cancel ${instance.name ?? job_id} (${instance.status ?? "status unknown"}).`,
          job_id,
          name: instance.name ?? null,
          status: instance.status ?? null,
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
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    },
    async ({ job_id, confirm }) => {
      const denied = scopeDenied("terminate_instance", user);
      if (denied) return denied;
      if (!confirm) {
        const target = await readPreviewTarget(
          () => client.get(`/api/v1/instances/${encodeURIComponent(job_id)}`),
          `instance ${job_id}`,
          "list_instances",
        );
        if (!target.found) return target.result;
        const instance = (target.value.instance ?? target.value.job ?? target.value) as Record<string, unknown>;
        return jsonText({
          ok: true,
          preview: true,
          message: `Set confirm:true to permanently terminate ${instance.name ?? job_id} (${instance.status ?? "status unknown"}).`,
          job_id,
          name: instance.name ?? null,
          status: instance.status ?? null,
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
        confirm: z.boolean().default(false).describe("false (the default) previews and changes nothing; true acts"),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    },
    async ({ public_key, name, confirm }) => {
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

      if (!confirm) {
        // Standing access: anyone with the matching private key gets a shell
        // on the user's instances. Say which key, by the fingerprint the user
        // can compare with `ssh-keygen -lf`, before adding it.
        const [type, blob] = inspection.key.split(/\s+/);
        const fingerprint = `SHA256:${createHash("sha256")
          .update(Buffer.from(blob ?? "", "base64"))
          .digest("base64")
          .replace(/=+$/, "")}`;
        return jsonText({
          ok: true,
          preview: true,
          key_type: type,
          fingerprint,
          name: name ?? "default",
          message:
            `Would register ${type} key ${fingerprint} as '${name ?? "default"}'. Anyone holding ` +
            "its private key could open a shell on this account's running instances. Nothing " +
            "has changed. Check the fingerprint with the user, then call again with confirm:true.",
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

  server.registerTool(
    "list_ssh_keys",
    {
      // No arguments: the route resolves the account from the caller's own
      // credential, so there is no way to ask for somebody else's keys.
      inputSchema: z.object({}),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async () => {
      const denied = scopeDenied("list_ssh_keys", user);
      if (denied) return denied;
      try {
        // Fingerprints and names only — the response carries public keys, which
        // are public by construction, and no private material can exist here.
        const data = await client.get("/api/ssh/keys");
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "delete_ssh_key",
    {
      inputSchema: z.object({
        key_id: z.string().min(1).max(160),
        confirm: z
          .boolean()
          .default(false)
          .describe("false returns a preview of the key that would be removed"),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    },
    async ({ key_id, confirm }) => {
      const denied = scopeDenied("delete_ssh_key", user);
      if (denied) return denied;
      // Preview first, for the same reason `delete_volume` does: revocation
      // reaches running instances, so someone working through that key is
      // disconnected. A model that guessed an id gets one chance to notice.
      if (!confirm) {
        try {
          const listing = (await client.get("/api/ssh/keys")) as Record<string, unknown>;
          const keys = (listing.keys as Array<Record<string, unknown>>) ?? [];
          const match = keys.find((k) => String(k.key_id ?? k.id) === key_id);
          if (!match) return notOnThisAccount(`key ${key_id}`, "list_ssh_keys");
          return jsonText({
            ok: true,
            preview: true,
            confirm_required: true,
            key: match,
            message:
              `Would remove ${match.name ?? key_id} (${match.fingerprint ?? "no fingerprint"}). ` +
              "Shell access is revoked on running instances too, disconnecting anyone " +
              "using it. Call again with confirm:true.",
          });
        } catch (e) {
          return jsonText({ error: formatApiError(e) });
        }
      }
      try {
        const data = await client.delete(`/api/ssh/keys/${encodeURIComponent(key_id)}`);
        return jsonText({
          ...(data as Record<string, unknown>),
          note:
            "Revoked, including on running instances. The user can restore access " +
            "by registering the same public key again; nobody else can do that for them.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "open_instance_access",
    {
      inputSchema: z.object({
        job_id: z.string(),
        method: z
          .enum(["auto", "ssh", "terminal"])
          .default("auto")
          .describe(
            "'ssh' returns the endpoint to connect to and spends nothing. " +
              "'terminal' mints a single-use browser/websocket ticket. 'auto' " +
              "prefers ssh when the instance publishes a port.",
          ),
        confirm: z.boolean().default(false).describe("false (the default) previews and changes nothing; true acts"),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
    },
    async ({ job_id, method, confirm }) => {
      const denied = scopeDenied("open_instance_access", user);
      if (denied) return denied;

      let instance: Record<string, unknown>;
      try {
        const data = await client.get(`/instance/${encodeURIComponent(job_id)}`);
        instance = ((data as Record<string, unknown>)?.instance ?? data) as Record<string, unknown>;
      } catch (e) {
        return jsonText({ ok: false, error: formatApiError(e) });
      }

      const status = String(instance?.status ?? "unknown");
      if (status !== "running") {
        // Checked before minting anything. A terminal ticket is a credential
        // with a clock on it, and issuing one for an instance that cannot
        // accept a connection burns it for nothing and reads to the model as
        // "access is ready".
        return jsonText({
          ok: false,
          error: "instance_not_running",
          status,
          message:
            `This instance is '${status}', not running, so there is nothing to ` +
            "connect to yet. Use watch_instance to wait for it to reach running.",
        });
      }

      const sshPort = Number(instance?.ssh_port ?? 0) || null;
      // Only ever the API's value — never derived, never defaulted. A
      // fingerprint this tool invented would be worse than none: null makes a
      // model say "this cannot be verified", a wrong value makes it say
      // "verified".
      const rawFingerprint = instance?.host_key_fingerprint;
      const hostKeyFingerprint =
        typeof rawFingerprint === "string" && /^SHA256:[A-Za-z0-9+/]{43}$/.test(rawFingerprint)
          ? rawFingerprint
          : null;
      const useSsh = method === "ssh" || (method === "auto" && sshPort !== null);

      if (!confirm) {
        return jsonText({
          ok: true,
          preview: true,
          job_id,
          method: useSsh ? "ssh" : "terminal",
          message: useSsh
            ? `Would return the SSH endpoint for ${instance?.name ?? job_id}; nothing is minted. ` +
              "Call again with confirm:true."
            : `Would mint a single-use browser terminal ticket for ${instance?.name ?? job_id}: a ` +
              "short-lived credential that opens a shell. Nothing has been issued. Call again with confirm:true.",
        });
      }

      if (useSsh) {
        if (sshPort === null) {
          return jsonText({
            ok: false,
            error: "no_ssh_endpoint",
            message:
              "This instance publishes no SSH port — it was most likely " +
              "launched non-interactive. Call again with method:'terminal'.",
          });
        }
        return jsonText({
          ok: true,
          method: "ssh",
          host: sshHost,
          port: sshPort,
          user: "root",
          command: `ssh root@${sshHost} -p ${sshPort}`,
          // A4 of docs/host-key-fingerprint-plan.md. The value when the
          // platform has observed one, and **the null path is not deleted** —
          // older instances, non-interactive launches, proxy-terminated hosts
          // where the container legitimately holds no keys, and agents that
          // have not yet reported all produce it, and today's wording stays
          // correct for them forever.
          //
          // It arrives paired with the port: the API returns it only when a
          // port resolved, because a fingerprint beside the wrong port is a
          // failed verification that reads to the user as an attack.
          host_key_fingerprint: hostKeyFingerprint,
          host_key_note: hostKeyFingerprint
            ? "Verify before trusting the connection. Run: ssh-keyscan -p " +
              `${sshPort} ${sshHost} 2>/dev/null | ssh-keygen -lf -` +
              " and check the SHA256 value matches host_key_fingerprint above. " +
              "If it does not match, do not connect and report it."
            : "Xcelsior does not yet publish a host-key fingerprint for this " +
              "endpoint because it has not observed one, so the first " +
              "connection cannot be verified against one. Tell the user " +
              "that, rather than telling them to accept the key.",
          requires:
            "A registered public key. If ssh refuses the connection, call " +
            "register_ssh_key with the contents of a .pub file.",
        });
      }

      try {
        const data = (await client.post("/api/terminal/ticket", {
          instance_id: job_id,
        })) as Record<string, unknown>;
        return jsonText({
          ok: true,
          method: "terminal",
          ticket: data?.ticket,
          expires_in_seconds: data?.expires_in,
          websocket_url: `${wsBase}/ws/terminal/${encodeURIComponent(job_id)}`,
          note:
            "Single-use and short-lived: this ticket is consumed by the first " +
            "connection and cannot be replayed. Ask for another if it expires.",
        });
      } catch (e) {
        return jsonText({ ok: false, error: formatApiError(e) });
      }
    },
  );
}
