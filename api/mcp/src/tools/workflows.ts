import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { waitForInstance } from "../lib/polling.js";
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

type GpuRow = {
  gpu_model?: string;
  price_cad?: number;
  spot_cad?: number;
  count_available?: number;
  region?: string;
};

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
        git_repo: z.string().optional(),
        init_script: z.string().max(4096).optional(),
        pricing_mode: z.enum(["on_demand", "spot"]).default("on_demand"),
        host_id: z.string().optional(),
        confirm: z
          .boolean()
          .default(false)
          .describe("Must be true to create the instance; false returns a preview only"),
        wait_timeout_seconds: z.number().int().min(30).max(1800).default(300),
        log_tail: z.number().int().min(1).max(200).default(50),
      }),
    },
    async (args) => {
      const denied = scopeDenied("run_training_job", user);
      if (denied) return denied;

      const payload = {
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

      if (!args.confirm) {
        try {
          const estimate = await client.post("/api/pricing/estimate", {
            gpu_model: args.gpu_model,
            duration_hours: 1,
            spot: args.pricing_mode === "spot",
          });
          return jsonText({
            preview: true,
            message: "Set confirm:true to launch this training job.",
            config: payload,
            estimate,
          });
        } catch (e) {
          return jsonText({ error: formatApiError(e) });
        }
      }

      try {
        const created = (await client.post("/instance", payload)) as Record<string, unknown>;
        const instance = (created.instance as Record<string, unknown>) || created;
        const jobId = String(instance.job_id || "");
        if (!jobId) return jsonText({ error: "create_instance did not return job_id", created });

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

        return jsonText({
          ok: wait.ok,
          job_id: jobId,
          status: wait.instance.status,
          timed_out: wait.timedOut,
          instance: wait.instance,
          logs,
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );

  server.registerTool(
    "schedule_under_budget",
    {
      inputSchema: z.object({
        max_hourly_cad: z.number().positive(),
        gpu_model: z.string().optional(),
        vram_needed_gb: z.number().min(0).default(0),
        num_gpus: z.number().int().min(1).max(64).default(1),
        pricing_mode: z.enum(["on_demand", "spot"]).default("spot"),
        name: z.string().min(1).max(128).optional(),
        confirm: z.boolean().default(false),
      }),
    },
    async (args) => {
      const denied = scopeDenied("schedule_under_budget", user);
      if (denied) return denied;

      try {
        const [gpuRes, spotRes] = await Promise.all([
          client.get("/api/v2/gpu/available"),
          client.get("/api/v2/marketplace/spot-prices"),
        ]);
        const gpus = ((gpuRes as { gpus?: GpuRow[] }).gpus || []) as GpuRow[];
        const spotList =
          ((spotRes as { spot_prices?: Array<{ gpu_model?: string; spot_cad?: number }> })
            .spot_prices || []) as Array<{ gpu_model?: string; spot_cad?: number }>;
        const spotByModel = new Map(
          spotList.map((s) => [String(s.gpu_model || ""), Number(s.spot_cad) || 0]),
        );

        const candidates = gpus
          .filter((g) => (g.count_available || 0) > 0)
          .filter((g) => !args.gpu_model || g.gpu_model === args.gpu_model)
          .map((g) => {
            const model = String(g.gpu_model || "");
            const hourly =
              args.pricing_mode === "spot"
                ? spotByModel.get(model) || Number(g.spot_cad) || Number(g.price_cad) || 0
                : Number(g.price_cad) || 0;
            return { ...g, hourly_cad: hourly };
          })
          .filter((g) => g.hourly_cad > 0 && g.hourly_cad <= args.max_hourly_cad)
          .sort((a, b) => a.hourly_cad - b.hourly_cad);

        if (!candidates.length) {
          return jsonText({
            ok: false,
            message: `No GPUs found under $${args.max_hourly_cad.toFixed(2)} CAD/hr`,
            max_hourly_cad: args.max_hourly_cad,
          });
        }

        const pick = candidates[0];
        const instanceName = args.name || `mcp-budget-${Date.now()}`;
        const config = {
          name: instanceName,
          gpu_model: pick.gpu_model,
          vram_needed_gb: args.vram_needed_gb,
          num_gpus: args.num_gpus,
          pricing_mode: args.pricing_mode,
          interactive: true,
        };

        if (!args.confirm) {
          return jsonText({
            preview: true,
            message: "Set confirm:true to create an instance with the selected GPU.",
            selected: pick,
            config,
          });
        }

        const created = await client.post("/instance", config);
        return jsonText({ ok: true, selected: pick, created });
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
              args: z.record(z.unknown()).optional(),
            }),
          )
          .min(1)
          .max(20)
          .describe("Stages in the order they must run. Each one's on_failure is fixed at approval."),
      }),
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
