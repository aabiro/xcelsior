import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { waitForInstance } from "../lib/polling.js";
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
      description:
        "Launch a training instance (git_repo + init_script), wait until running, return connection info and log tail.",
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
      description:
        "Find available GPU capacity under a max hourly CAD rate, then optionally create an instance.",
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
}