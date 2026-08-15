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
        image_id: z.string().min(1).max(64).describe("A ready snapshot from list_user_images"),
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
