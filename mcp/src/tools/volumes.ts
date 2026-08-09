import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
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

/**
 * P3 — durable state.
 *
 * An instance's disk dies with the instance. A volume is the only place work
 * survives a relaunch, and until now none of it was reachable without a
 * browser: GT0 classified all twenty volume and artifact operations as `gap`.
 *
 * Two properties shape this file.
 *
 * **Destruction previews before it acts.** `detach_volume` and `delete_volume`
 * both require `confirm:true` and return a preview otherwise. The plan asks for
 * detach specifically to sit behind approval "since it can disrupt a running
 * workload" — a detach pulls the filesystem out from under a job that is
 * writing to it. Delete is worse and is treated the same way.
 *
 * **The retention clock is surfaced, because its absence is the failure.** The
 * plan's words: artifacts expire and "the retention clock should be visible to
 * the human too — it is currently invisible, which is how work gets lost".
 * `get_artifact_expiry` is that clock, and it is deliberately a read an agent
 * can perform unprompted before a user loses something.
 */
export function registerVolumeTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "list_volumes",
    { inputSchema: z.object({}) },
    async () => {
      const denied = scopeDenied("list_volumes", user);
      if (denied) return denied;
      try {
        return jsonText(await client.get("/api/v2/volumes"));
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_volume",
    { inputSchema: z.object({ volume_id: z.string().min(1).max(160) }) },
    async ({ volume_id }) => {
      const denied = scopeDenied("get_volume", user);
      if (denied) return denied;
      try {
        return jsonText(await client.get(`/api/v2/volumes/${encodeURIComponent(volume_id)}`));
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "create_volume",
    {
      inputSchema: z.object({
        name: z.string().min(1).max(128),
        size_gb: z.number().int().min(1).max(2000).default(50),
        region: z.string().max(64).default("ca-east"),
        encrypted: z.boolean().default(true),
      }),
    },
    async (args) => {
      const denied = scopeDenied("create_volume", user);
      if (denied) return denied;
      try {
        const data = (await client.post("/api/v2/volumes", args)) as Record<string, unknown>;
        return jsonText({
          ...data,
          note:
            "Billed per GB-month from the moment it exists, whether or not it " +
            "is attached. Delete it when the data is no longer needed.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "attach_volume",
    {
      inputSchema: z.object({
        volume_id: z.string().min(1).max(160),
        instance_id: z.string().min(1).max(160),
        mount_path: z
          .string()
          .regex(/^\/(workspace|mnt\/[a-zA-Z0-9._-]+|data)$/)
          .default("/workspace")
          .describe("Where it appears inside the container. The API accepts only these shapes."),
        mode: z.enum(["rw", "ro"]).default("rw"),
      }),
    },
    async (args) => {
      const denied = scopeDenied("attach_volume", user);
      if (denied) return denied;
      try {
        const data = (await client.post(
          `/api/v2/volumes/${encodeURIComponent(args.volume_id)}/attach`,
          { instance_id: args.instance_id, mount_path: args.mount_path, mode: args.mode },
        )) as Record<string, unknown>;
        return jsonText(data);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "detach_volume",
    {
      inputSchema: z.object({
        volume_id: z.string().min(1).max(160),
        confirm: z.boolean().default(false),
      }),
    },
    async ({ volume_id, confirm }) => {
      const denied = scopeDenied("detach_volume", user);
      if (denied) return denied;
      if (!confirm) {
        // Preview before acting, per the plan: a detach can disrupt a running
        // workload. The preview names the instance so the user is told *what*
        // is about to lose its filesystem, not merely that something will.
        let attachedTo: unknown = null;
        try {
          const vol = (await client.get(
            `/api/v2/volumes/${encodeURIComponent(volume_id)}`,
          )) as Record<string, unknown>;
          const record = (vol?.volume ?? vol) as Record<string, unknown>;
          attachedTo = record?.attached_instance_id ?? record?.instance_id ?? null;
        } catch {
          // A preview that cannot read the volume still refuses to act.
        }
        return jsonText({
          preview: true,
          volume_id,
          attached_to: attachedTo,
          message:
            attachedTo
              ? `This volume is attached to ${String(attachedTo)}. Detaching pulls the ` +
                "filesystem out from under anything writing to it — unwritten data is " +
                "lost. Set confirm:true to proceed."
              : "Set confirm:true to detach. Nothing has changed.",
        });
      }
      try {
        return jsonText(
          await client.post(`/api/v2/volumes/${encodeURIComponent(volume_id)}/detach`),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "delete_volume",
    {
      inputSchema: z.object({
        volume_id: z.string().min(1).max(160),
        confirm: z.boolean().default(false),
      }),
    },
    async ({ volume_id, confirm }) => {
      const denied = scopeDenied("delete_volume", user);
      if (denied) return denied;
      if (!confirm) {
        return jsonText({
          preview: true,
          volume_id,
          message:
            "Deleting a volume destroys its contents. This is the tool that " +
            "loses work permanently — snapshot it first if the data matters. " +
            "Set confirm:true to proceed.",
        });
      }
      try {
        return jsonText(
          await client.delete(`/api/v2/volumes/${encodeURIComponent(volume_id)}`),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "snapshot_volume",
    {
      inputSchema: z.object({
        volume_id: z.string().min(1).max(160),
        name: z.string().max(128).optional(),
      }),
    },
    async ({ volume_id, name }) => {
      const denied = scopeDenied("snapshot_volume", user);
      if (denied) return denied;
      try {
        return jsonText(
          await client.post(
            `/api/v2/volumes/${encodeURIComponent(volume_id)}/snapshots`,
            name ? { name } : {},
          ),
        );
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_artifact_expiry",
    { inputSchema: z.object({ job_id: z.string().min(1).max(160) }) },
    async ({ job_id }) => {
      const denied = scopeDenied("get_artifact_expiry", user);
      if (denied) return denied;
      try {
        const data = (await client.get(
          `/api/artifacts/${encodeURIComponent(job_id)}/expiry`,
        )) as Record<string, unknown>;
        return jsonText({
          ...data,
          note:
            "Artifacts are deleted when their retention window ends. Copy " +
            "anything that matters onto a volume, which has no clock.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
  server.registerTool(
    "promote_artifact_to_volume",
    {
      inputSchema: z.object({
        job_id: z.string().min(1).max(160).describe("The finished run whose outputs to keep"),
        volume_id: z.string().min(1).max(160).describe("The volume to copy them onto"),
        idempotency_key: z
          .string()
          .max(160)
          .optional()
          .describe(
            "Leave this out. Omitting it is safe: the same job and the same files " +
              "resolve to the same promotion, so a retry after a timeout joins the " +
              "copy already running instead of starting a second one.",
          ),
      }),
      annotations: { readOnlyHint: false, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async ({ job_id, volume_id, idempotency_key }) => {
      const denied = scopeDenied("promote_artifact_to_volume", user);
      if (denied) return denied;
      try {
        const data = (await client.post(
          `/api/v2/volumes/${encodeURIComponent(volume_id)}/promotions`,
          { job_id, idempotency_key: idempotency_key ?? "" },
        )) as Record<string, unknown>;
        // §3.6: the copy is still running when this returns. The wording is the
        // mechanism — a model that says "saved" when it means "started" is the
        // failure this whole phase exists to prevent, and there is nothing else
        // in the response to stop it.
        return jsonText({
          ...data,
          status: "started",
          note:
            "The copy has STARTED and is NOT finished. Tell the user it is " +
            "running, not that their files are saved. Check " +
            "get_promotion_status with this promotion_id before saying it is " +
            "done — a large checkpoint takes minutes and the artifacts are " +
            "not safe from their retention clock until it completes.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_promotion_status",
    {
      inputSchema: z.object({
        volume_id: z.string().min(1).max(160),
        promotion_id: z.string().min(1).max(160),
      }),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async ({ volume_id, promotion_id }) => {
      const denied = scopeDenied("get_promotion_status", user);
      if (denied) return denied;
      try {
        const data = (await client.get(
          `/api/v2/volumes/${encodeURIComponent(volume_id)}/promotions/${encodeURIComponent(promotion_id)}`,
        )) as Record<string, unknown>;
        const finished = data.state === "succeeded";
        return jsonText({
          ...data,
          note: finished
            ? "The copy finished. The files are on the volume, which has no retention clock."
            : "Still running. Do not tell the user their files are saved yet.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}
