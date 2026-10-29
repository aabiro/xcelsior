import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import type { AuthUser } from "../auth/bearer.js";
import { TOOL_SCOPES, userHasScope, describeScopeRequirement } from "../auth/scopes.js";
import { structuredResult } from "../lib/format.js";
import { formatApiError } from "../client/errors.js";

/**
 * P7's image sweep, as two tools.
 *
 * The sweep is the largest single spend on the platform — up to sixty-four
 * instances from one call — so it goes through the same prepare-approve-execute
 * path `create_instance` uses for one instance, not a shortcut around it. A
 * bulk action reached by an agent must be the *hardest* thing to do by
 * accident, not the easiest.
 *
 * Without `plan_id`, `create_image_sweep` quotes the sweep and returns a plan
 * id and an approval URL; nothing has launched. With `plan_id` and
 * `confirm:true` it executes that approved plan. The member count lives inside
 * the plan's approved arguments and is bound by its hash, so an approval for
 * three members cannot be spent on sixty-four.
 */
export function registerSweepTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  function denied(tool: string) {
    const required = TOOL_SCOPES[tool];
    return structuredResult(
      { ok: false, code: "insufficient_scope", required },
      `Access denied: requires ${describeScopeRequirement(required)}.`,
    );
  }

  server.registerTool(
    "create_image_sweep",
    {
      inputSchema: z.object({
        image_id: z.string().min(1).max(64).describe("A ready image_id from list_user_images"),
        count: z.number().int().min(1).max(64).describe("How many instances to launch"),
        name: z.string().min(1).max(96).default("sweep"),
        vram_needed_gb: z.number().min(0).default(0),
        num_gpus: z.number().int().min(1).max(8).default(1),
        gpu_model: z.string().max(64).optional(),
        command: z.string().max(4000).optional(),
        interactive: z.boolean().default(true),
        confirm: z.boolean().default(false),
        plan_id: z
          .string()
          .min(1)
          .max(160)
          .optional()
          .describe("An approved sweep plan. Omit to quote one."),
      }),
      outputSchema: z
        .object({
          ok: z.boolean().optional(),
          preview: z.boolean().optional(),
          plan_id: z.string().optional(),
          members: z.number().optional(),
          estimate_cad: z.number().optional(),
          approval_url: z.string().optional(),
          approval_required: z.boolean().optional(),
          message: z.string().optional(),
        })
        .passthrough(),
    },
    async (args: Record<string, unknown>) => {
      if (!userHasScope(user?.scopes, TOOL_SCOPES.create_image_sweep)) {
        return denied("create_image_sweep");
      }
      const { confirm, plan_id, ...payload } = args as {
        confirm?: boolean;
        plan_id?: string;
        [k: string]: unknown;
      };
      try {
        // No plan, or a plan the caller has not confirmed: quote only. Both
        // conditions matter — `confirm:true` with no plan is a caller trying to
        // skip approval, and it gets a plan rather than a launch.
        if (!confirm || !plan_id) {
          const plan = await client.post<Record<string, unknown>>(
            "/api/v1/image-sweeps",
            payload,
          );
          return structuredResult(
            {
              ...plan,
              approval_required: true,
              message:
                Boolean(confirm) && !plan_id
                  ? "Approval is required. Approve this prepared plan, then call " +
                    "create_image_sweep with confirm:true and plan_id."
                  : `Quoted ${plan.members ?? payload.count} members at about ` +
                    `${plan.estimate_cad ?? "?"} CAD. Nothing has launched. Approve the ` +
                    "plan, then call again with confirm:true and plan_id.",
            },
            "Sweep plan prepared — approval required before anything launches.",
          );
        }
        const data = await client.post<Record<string, unknown>>(
          `/api/v1/image-sweep-plans/${encodeURIComponent(plan_id)}/execute`,
          {},
        );
        return structuredResult(data, `Sweep launched from approved plan ${plan_id}.`);
      } catch (error) {
        return structuredResult({ ok: false, error: formatApiError(error) }, "create_image_sweep failed.");
      }
    },
  );

  server.registerTool(
    "create_instance_snapshot",
    {
      inputSchema: z.object({
        job_id: z.string().min(1).max(160),
        name: z.string().min(1).max(63).describe("Lowercase repository name"),
        tag: z.string().max(63).default("latest"),
        description: z.string().max(512).default(""),
      }),
      outputSchema: z.object({}).passthrough(),
    },
    async (args: Record<string, unknown>) => {
      if (!userHasScope(user?.scopes, TOOL_SCOPES.create_instance_snapshot)) {
        return denied("create_instance_snapshot");
      }
      const { job_id, ...body } = args;
      try {
        const data = await client.post<Record<string, unknown>>(
          `/instances/${encodeURIComponent(String(job_id))}/snapshot`,
          body,
        );
        return structuredResult(
          data,
          "Snapshot queued. It builds on the host in the background — check " +
            "list_user_images for when it reports ready.",
        );
      } catch (error) {
        return structuredResult(
          { ok: false, error: formatApiError(error) },
          "create_instance_snapshot failed.",
        );
      }
    },
  );

  server.registerTool(
    "list_user_images",
    {
      inputSchema: z.object({
        scope: z.enum(["mine", "team", "all"]).default("mine"),
        q: z.string().max(200).default("").describe("Filter by name substring"),
        limit: z.number().int().min(1).max(500).default(100),
      }),
      outputSchema: z.object({}).passthrough(),
    },
    async (args: Record<string, unknown>) => {
      if (!userHasScope(user?.scopes, TOOL_SCOPES.list_user_images)) {
        return denied("list_user_images");
      }
      try {
        const data = await client.get<Record<string, unknown>>("/user-images", {
          scope: String(args.scope ?? "mine"),
          q: String(args.q ?? ""),
          limit: Number(args.limit ?? 100),
        });
        return structuredResult(data, "Images retrieved.");
      } catch (error) {
        return structuredResult(
          { ok: false, error: formatApiError(error) },
          "list_user_images failed.",
        );
      }
    },
  );

  server.registerTool(
    "delete_user_image",
    {
      inputSchema: z.object({
        image_id: z.string().min(1).max(64),
        confirm: z
          .boolean()
          .default(false)
          .describe("false returns a preview of what would be deleted"),
      }),
      outputSchema: z.object({}).passthrough(),
    },
    async (args: Record<string, unknown>) => {
      if (!userHasScope(user?.scopes, TOOL_SCOPES.delete_user_image)) {
        return denied("delete_user_image");
      }
      const imageId = String(args.image_id);
      // Preview before destroying, the same shape `delete_volume` uses. The
      // deletion cannot be undone through the API, so a model that guessed an
      // id has one chance to notice.
      if (!args.confirm) {
        try {
          const listing = await client.get<Record<string, unknown>>("/user-images", {
            limit: 500,
          });
          const images = (listing.images as Array<Record<string, unknown>>) ?? [];
          const match = images.find((i) => String(i.image_id) === imageId);
          return structuredResult(
            {
              ok: true,
              preview: true,
              confirm_required: true,
              image: match ?? null,
              message: match
                ? `Would delete ${match.name}:${match.tag}. This cannot be undone — ` +
                  "call again with confirm:true."
                : `No image ${imageId} is visible on this account.`,
            },
            "Preview only — nothing was deleted.",
          );
        } catch (error) {
          return structuredResult(
            { ok: false, error: formatApiError(error) },
            "delete_user_image preview failed.",
          );
        }
      }
      try {
        const data = await client.delete<Record<string, unknown>>(
          `/user-images/${encodeURIComponent(imageId)}`,
        );
        return structuredResult(data, `Image ${imageId} deleted.`);
      } catch (error) {
        return structuredResult(
          { ok: false, error: formatApiError(error) },
          "delete_user_image failed.",
        );
      }
    },
  );

  server.registerTool(
    "get_image_sweep",
    {
      inputSchema: z.object({
        sweep_id: z.string().min(1).max(160),
      }),
      outputSchema: z.object({}).passthrough(),
    },
    async (args: Record<string, unknown>) => {
      if (!userHasScope(user?.scopes, TOOL_SCOPES.get_image_sweep)) {
        return denied("get_image_sweep");
      }
      try {
        const data = await client.get<Record<string, unknown>>(
          `/api/v1/image-sweeps/${encodeURIComponent(String(args.sweep_id))}`,
        );
        return structuredResult(data, "Sweep retrieved.");
      } catch (error) {
        return structuredResult({ ok: false, error: formatApiError(error) }, "get_image_sweep failed.");
      }
    },
  );
}
