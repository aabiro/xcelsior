import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { apiProblem, formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { waitForInstance } from "../lib/polling.js";
import { scopeDenial } from "../auth/denial.js";
import type { AuthUser } from "../auth/bearer.js";

function scopeDenied(tool: string, user: AuthUser | undefined) {
  const denial = scopeDenial(tool, user);
  return denial ? jsonText(denial) : null;
}

/** A row of `/api/v2/gpu/available`, as that route writes it. */
type GpuRow = {
  gpu_model?: string;
  vram_gb?: number;
  price_per_hour_cad?: number;
  count_available?: number;
  region?: string;
};

/** A row of `/api/v2/marketplace/spot-prices` (`SpotQuote.to_api_dict`). */
type SpotRow = { gpu_model?: string; rate_cad?: number; spot_cents?: number };

/**
 * Prepare a launch plan: the only way these workflows spend money.
 *
 * Both workflows used to POST /instance on confirm:true, so "approval applies"
 * in their descriptions was not true: an agent could launch and bill a GPU
 * without the user ever seeing the plan. Now confirm:true without an approved
 * plan_id only prepares one, exactly as create_instance does.
 */
async function prepareLaunchPlan(
  client: XcelsiorApiClient,
  spec: Record<string, unknown>,
  opts: { tool: string; confirm: boolean; idempotencyKey?: string },
) {
  try {
    const plan = await client.post<Record<string, unknown>>("/api/v1/launch-plans", spec, {
      idempotencyKey: opts.idempotencyKey,
      retry: opts.idempotencyKey ? "idempotent" : "none",
    });
    return jsonText({
      ...plan,
      preview: true,
      approval_required: opts.confirm,
      message: opts.confirm
        ? `Approval is required. Ask the user to approve this plan, then call ${opts.tool} ` +
          "with confirm:true and this plan_id."
        : `Review and approve this launch plan, then call ${opts.tool} with confirm:true and ` +
          "this plan_id. Nothing is allocated or billed until then.",
    });
  } catch (e) {
    return jsonText({ preview: true, ...apiProblem(e) });
  }
}

type Executed =
  | { ok: true; jobId: string; job: Record<string, unknown> }
  | { ok: false; problem: Record<string, unknown> };

/** Execute an approved plan. The server refuses one that is not approved. */
async function executeLaunchPlan(
  client: XcelsiorApiClient,
  planId: string,
  idempotencyKey?: string,
): Promise<Executed> {
  try {
    const data = await client.post<Record<string, unknown>>(
      `/api/v1/launch-plans/${encodeURIComponent(planId)}/execute`,
      { confirm: true },
      { idempotencyKey, retry: idempotencyKey ? "idempotent" : "none" },
    );
    const job = (data.job ?? data.instance ?? {}) as Record<string, unknown>;
    const jobId = String(job.job_id ?? data.job_id ?? "");
    if (!jobId) {
      return {
        ok: false,
        problem: { ok: false, error: "the plan executed but returned no job_id", plan_id: planId, data },
      };
    }
    return { ok: true, jobId, job };
  } catch (e) {
    return { ok: false, problem: { ...apiProblem(e), plan_id: planId } };
  }
}

export function registerWorkflowTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "run_training_job",
    {
      inputSchema: z.object({
        name: z.string().min(1).max(128),
        gpu_model: z.string().default("RTX 4090"),
        vram_needed_gb: z.number().min(0).default(0),
        num_gpus: z.number().int().min(1).max(64).default(1),
        image: z.string().optional(),
        git_repo: z.string().max(512).optional(),
        init_script: z.string().max(4096).optional(),
        pricing_mode: z.enum(["on_demand", "spot"]).default("on_demand"),
        host_id: z.string().optional(),
        confirm: z
          .boolean()
          .default(false)
          .describe("Intent to launch. Launching also needs plan_id, approved by the user"),
        plan_id: z
          .string()
          .min(1)
          .max(160)
          .optional()
          .describe("The plan a call without it prepared. Executes only once the user has approved it"),
        idempotency_key: z.string().uuid().optional(),
        wait_timeout_seconds: z.number().int().min(30).max(1800).default(300),
        log_tail: z.number().int().min(1).max(200).default(50),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    },
    async (args) => {
      const denied = scopeDenied("run_training_job", user);
      if (denied) return denied;

      const spec = {
        name: args.name,
        gpu_model: args.gpu_model,
        vram_needed_gb: args.vram_needed_gb,
        num_gpus: args.num_gpus,
        image: args.image,
        git_repo: args.git_repo,
        init_script: args.init_script,
        pricing_mode: args.pricing_mode,
        host_id: args.host_id,
        interactive: true,
      };

      if (!args.confirm || !args.plan_id) {
        return prepareLaunchPlan(client, spec, {
          tool: "run_training_job",
          confirm: args.confirm,
          idempotencyKey: args.idempotency_key,
        });
      }

      const executed = await executeLaunchPlan(client, args.plan_id, args.idempotency_key);
      if (!executed.ok) return jsonText(executed.problem);
      const jobId = executed.jobId;

      try {
        const wait = await waitForInstance(client, jobId, {
          timeoutMs: args.wait_timeout_seconds * 1000,
        });

        let logs: unknown = null;
        try {
          logs = await client.get(`/instances/${encodeURIComponent(jobId)}/logs`, {
            limit: args.log_tail,
          });
        } catch {
          logs = { note: "logs not yet available" };
        }

        // A launch that is still starting when the wait ends has not failed.
        // Only a run that ended in failure is not ok.
        return jsonText({
          ok: wait.ok || wait.timedOut,
          launched: true,
          plan_id: args.plan_id,
          job_id: jobId,
          status: wait.instance.status,
          timed_out: wait.timedOut,
          instance: wait.instance,
          logs,
          ...(wait.timedOut
            ? {
                next_step:
                  "The instance is launched and still starting. Follow it with watch_instance " +
                  "and this job_id. Calling run_training_job again with the same plan_id " +
                  "returns this same job; it does not launch another.",
              }
            : {}),
        });
      } catch (e) {
        // The launch happened; only following it failed. Say so, with the id,
        // so nobody launches again to find out.
        return jsonText({
          ok: false,
          launched: true,
          plan_id: args.plan_id,
          job_id: jobId,
          error: formatApiError(e),
          next_step: "The instance was launched. Follow it with watch_instance and this job_id.",
        });
      }
    },
  );

  server.registerTool(
    "schedule_under_budget",
    {
      inputSchema: z.object({
        max_hourly_cad: z.number().positive().describe("The most to pay per GPU per hour"),
        gpu_model: z.string().optional(),
        vram_needed_gb: z.number().min(0).default(0),
        num_gpus: z.number().int().min(1).max(64).default(1),
        pricing_mode: z.enum(["on_demand", "spot"]).default("spot"),
        name: z.string().min(1).max(128).optional(),
        confirm: z
          .boolean()
          .default(false)
          .describe("Intent to launch. Launching also needs plan_id, approved by the user"),
        plan_id: z
          .string()
          .min(1)
          .max(160)
          .optional()
          .describe("The plan a call without it prepared for the cheapest match. Executes only once approved"),
        idempotency_key: z.string().uuid().optional(),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: true },
    },
    async (args) => {
      const denied = scopeDenied("schedule_under_budget", user);
      if (denied) return denied;

      // The approved plan is the launch: it fixed the GPU and the price the
      // user agreed to. Searching again here could pick something else.
      if (args.confirm && args.plan_id) {
        const executed = await executeLaunchPlan(client, args.plan_id, args.idempotency_key);
        if (!executed.ok) return jsonText(executed.problem);
        return jsonText({
          ok: true,
          launched: true,
          plan_id: args.plan_id,
          job_id: executed.jobId,
          job: executed.job,
          next_step: "Follow it with watch_instance and this job_id.",
        });
      }

      try {
        const [gpuRes, spotRes] = await Promise.all([
          client.get("/api/v2/gpu/available"),
          client.get("/api/v2/marketplace/spot-prices"),
        ]);
        // Read the fields these routes actually write. This read `price_cad`
        // and `spot_cad`, which neither returns, so every price was 0 and
        // every GPU was filtered out: the tool never found anything.
        const gpus = ((gpuRes as { gpus?: GpuRow[] }).gpus || []) as GpuRow[];
        const spotList = ((spotRes as { spot_prices?: SpotRow[] }).spot_prices || []) as SpotRow[];
        const spotByModel = new Map(
          spotList.map((s) => [
            String(s.gpu_model || ""),
            Number(s.rate_cad) || (Number(s.spot_cents) || 0) / 100,
          ]),
        );

        const candidates = gpus
          .filter((g) => (g.count_available || 0) > 0)
          .filter((g) => !args.gpu_model || g.gpu_model === args.gpu_model)
          .map((g) => {
            const model = String(g.gpu_model || "");
            const listed = Number(g.price_per_hour_cad) || 0;
            const hourly = args.pricing_mode === "spot" ? spotByModel.get(model) || listed : listed;
            return { ...g, hourly_cad: hourly };
          })
          .filter((g) => g.hourly_cad > 0 && g.hourly_cad <= args.max_hourly_cad)
          .sort((a, b) => a.hourly_cad - b.hourly_cad);

        if (!candidates.length) {
          // Nothing under the budget is an answer, not a failure.
          return jsonText({
            ok: true,
            found: false,
            message: `No GPUs found under $${args.max_hourly_cad.toFixed(2)}/hr`,
            max_hourly_cad: args.max_hourly_cad,
          });
        }

        const pick = candidates[0];
        const spec = {
          name: args.name || `mcp-budget-${Date.now()}`,
          gpu_model: pick.gpu_model,
          vram_needed_gb: args.vram_needed_gb,
          num_gpus: args.num_gpus,
          pricing_mode: args.pricing_mode,
          interactive: true,
        };

        let plan: Record<string, unknown>;
        try {
          plan = await client.post<Record<string, unknown>>("/api/v1/launch-plans", spec, {
            idempotencyKey: args.idempotency_key,
            retry: args.idempotency_key ? "idempotent" : "none",
          });
        } catch (e) {
          return jsonText({ preview: true, selected: pick, ...apiProblem(e) });
        }

        // The listing is a summary; the plan's quote is the price that would
        // be charged. A plan over the budget is withdrawn, not offered.
        const estimate = (plan.estimate ?? {}) as Record<string, unknown>;
        const burnMicros = Number(estimate.hourly_burn_micros);
        const perGpuCad = Number.isFinite(burnMicros) ? burnMicros / 1_000_000 / args.num_gpus : NaN;
        if (Number.isFinite(perGpuCad) && perGpuCad > args.max_hourly_cad + 1e-9) {
          if (plan.plan_id) {
            try {
              await client.post(`/api/v1/launch-plans/${encodeURIComponent(String(plan.plan_id))}/revoke`, {
                reason: "quoted above the caller's budget",
              });
            } catch {
              // An unapproved plan expires on its own; it can spend nothing.
            }
          }
          return jsonText({
            ok: true,
            found: false,
            selected: pick,
            quoted_hourly_cad_per_gpu: Number(perGpuCad.toFixed(4)),
            max_hourly_cad: args.max_hourly_cad,
            message:
              `${pick.gpu_model} is listed under the budget, but the launch quote is ` +
              `$${perGpuCad.toFixed(2)}/hr per GPU, above $${args.max_hourly_cad.toFixed(2)}. ` +
              "No plan was kept and nothing was launched.",
          });
        }

        return jsonText({
          ...plan,
          ok: true,
          found: true,
          preview: true,
          selected: pick,
          approval_required: Boolean(args.confirm),
          message: args.confirm
            ? "Approval is required. Ask the user to approve this plan, then call " +
              "schedule_under_budget with confirm:true and this plan_id."
            : "Review and approve this launch plan, then call schedule_under_budget with " +
              "confirm:true and this plan_id. Nothing is allocated or billed until then.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
  server.registerTool(
    "run_pipeline",
    {
      inputSchema: z.object({
        name: z.string().max(120).optional().describe("What this pipeline is for"),
        stages: z
          .array(
            z.object({
              name: z.string().min(1).max(64),
              action_type: z.string().min(1).max(64),
              on_failure: z.enum(["halt", "continue", "retry"]).optional(),
              max_attempts: z.number().int().min(1).max(10).optional(),
              estimate_micros: z.number().int().min(0).optional(),
              args: z.record(z.string(), z.unknown()).optional(),
            }),
          )
          .min(1)
          .max(20)
          .describe("Stages in the order they must run. Each one's on_failure is fixed at approval."),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: false, openWorldHint: false },
    },
    async ({ name, stages }) => {
      const denied = scopeDenied("run_pipeline", user);
      if (denied) return denied;
      try {
        const data = (await client.post("/api/v1/pipelines", {
          name: name ?? "pipeline",
          stages,
        })) as Record<string, unknown>;
        // The pipeline is quoted, not approved and not running. Saying so is
        // the mechanism — an agent that reports "pipeline started" when it
        // means "pipeline awaiting approval" is the failure this phase's whole
        // approval story exists to prevent.
        return jsonText({
          ...data,
          status: "awaiting_approval",
          note:
            "This pipeline is QUOTED and NOT YET APPROVED — nothing has run. " +
            "Tell the user the total above is what they are approving, and " +
            "that no stage starts until they approve it. After approval, use " +
            "get_pipeline_status to see which stage is live; do not report any " +
            "stage as finished until that says so.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "revoke_launch_plan",
    {
      inputSchema: z.object({
        plan_id: z.string().min(1).max(160).describe("From whatever tool quoted it"),
        reason: z.string().max(500).optional(),
      }),
      annotations: { readOnlyHint: false, destructiveHint: true, idempotentHint: true, openWorldHint: false },
    },
    async ({ plan_id, reason }) => {
      const denied = scopeDenied("revoke_launch_plan", user);
      if (denied) return denied;
      try {
        // No confirm gate, deliberately. Every other write on this surface has
        // one because it can spend or destroy; this only ever *removes* the
        // ability to spend, and making a user confirm the safe direction
        // teaches them to click through the prompts that matter.
        const data = (await client.post(
          `/api/v1/launch-plans/${encodeURIComponent(plan_id)}/revoke`,
          reason ? { reason } : {},
        )) as Record<string, unknown>;
        return jsonText({
          ...data,
          note:
            "Withdrawn. Nothing ran, so there is nothing to refund. Quoting again " +
            "produces a new plan; this id cannot be revived.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "get_pipeline_status",
    {
      inputSchema: z.object({ plan_id: z.string().min(1).max(160) }),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async ({ plan_id }) => {
      const denied = scopeDenied("get_pipeline_status", user);
      if (denied) return denied;
      try {
        const data = (await client.get(
          `/api/v1/pipelines/${encodeURIComponent(plan_id)}`,
        )) as Record<string, unknown>;
        const finished = data.finished === true;
        const failed = data.failed === true;
        return jsonText({
          ...data,
          note: finished
            ? failed
              ? "The pipeline stopped. Read each stage's failure_code — a 'skipped' stage did not fail, it never ran."
              : "Every stage finished successfully."
            : "Still running. Do not report any stage as done unless its state says succeeded.",
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}
