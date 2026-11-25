import { describe, it, expect } from "vitest";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../../src/client/api.js";
import { registerWorkflowTools } from "../../src/tools/workflows.js";
import { ApiError } from "../../src/client/errors.js";
import { TOOL_SCOPES } from "../../src/auth/scopes.js";

/**
 * run_training_job and schedule_under_budget launch only through an approved plan.
 *
 * Both used to POST /instance on confirm:true. Their descriptions said launch-plan
 * approval applied, and it did not: an agent could launch and bill a GPU the user
 * never saw a plan for. Now a call without an approved plan_id only prepares a
 * plan, and a call with one executes that plan, nothing else.
 */

type Result = { content: Array<{ text: string }> };
type Handler = (args: Record<string, unknown>) => Promise<Result>;

function capture(tool: string, client: Partial<XcelsiorApiClient>): Handler {
  let handler: Handler | undefined;
  const recorder = {
    registerTool(name: string, _config: unknown, fn: Handler) {
      if (name === tool) handler = fn;
      return undefined;
    },
  } as unknown as McpServer;
  const user = { scopes: [...TOOL_SCOPES[tool].allOf!] } as never;
  registerWorkflowTools(recorder, client as XcelsiorApiClient, user);
  if (!handler) throw new Error(`${tool} was not registered`);
  return handler;
}

const parse = (r: Result) => JSON.parse(r.content[0].text) as Record<string, any>;

const PLAN = {
  ok: true,
  plan_id: "plan_123",
  status: "pending",
  approval_url: "/dashboard/launch-plans/plan_123",
  estimate: { hourly_burn_micros: 400_000, currency: "CAD" },
};

const MARKET = {
  async get(path: string) {
    if (path === "/api/v2/gpu/available") {
      return { gpus: [{ gpu_model: "RTX 3060", vram_gb: 12, price_per_hour_cad: 0.5, count_available: 1 }] };
    }
    return { spot_prices: [{ gpu_model: "RTX 3060", rate_cad: 0.4, spot_cents: 40 }] };
  },
};

function recordingClient(extra: Partial<XcelsiorApiClient> = {}) {
  const posted: Array<{ path: string; body: unknown }> = [];
  const client = {
    ...MARKET,
    async post(path: string, body?: unknown) {
      posted.push({ path, body });
      if (path === "/api/v1/launch-plans") return PLAN;
      if (path.endsWith("/execute")) return { ok: true, job: { job_id: "job_9", phase: "pending" } };
      if (path.endsWith("/revoke")) return { ok: true };
      throw new Error(`unexpected POST ${path}`);
    },
    ...extra,
  } as Partial<XcelsiorApiClient>;
  return { client, posted };
}

describe.each(["run_training_job", "schedule_under_budget"])("%s", (tool) => {
  const args =
    tool === "run_training_job"
      ? { name: "t", gpu_model: "RTX 3060", num_gpus: 1, pricing_mode: "spot", wait_timeout_seconds: 30, log_tail: 5 }
      : { name: "t", max_hourly_cad: 1, num_gpus: 1, pricing_mode: "spot" };

  it("prepares a plan on preview and never posts /instance", async () => {
    const { client, posted } = recordingClient();
    const out = parse(await capture(tool, client)({ ...args, confirm: false }));
    expect(posted.map((p) => p.path)).toEqual(["/api/v1/launch-plans"]);
    expect(out.plan_id).toBe("plan_123");
    expect(out.approval_url).toBe(PLAN.approval_url);
    expect(out.preview).toBe(true);
  });

  it("prepares a plan, not a launch, for confirm:true without a plan_id", async () => {
    const { client, posted } = recordingClient();
    const out = parse(await capture(tool, client)({ ...args, confirm: true }));
    expect(posted.map((p) => p.path)).toEqual(["/api/v1/launch-plans"]);
    expect(out.approval_required).toBe(true);
    expect(String(out.message)).toMatch(/approv/i);
  });

  it("executes the approved plan and nothing else", async () => {
    const { client, posted } = recordingClient({
      async get(path: string) {
        if (path.startsWith("/instance/")) {
          return { job_id: "job_9", status: "running" };
        }
        return MARKET.get(path);
      },
    } as Partial<XcelsiorApiClient>);
    const out = parse(await capture(tool, client)({ ...args, confirm: true, plan_id: "plan_123" }));
    expect(posted.map((p) => p.path)).toEqual(["/api/v1/launch-plans/plan_123/execute"]);
    expect(out.job_id).toBe("job_9");
    expect(out.launched).toBe(true);
  });

  it("passes on the server's refusal of an unapproved plan", async () => {
    const { client, posted } = recordingClient({
      async post(path: string, body?: unknown) {
        posted.push({ path, body });
        throw new ApiError("conflict", 409, undefined, {
          status: 409,
          code: "plan_not_approved",
          detail: "plan is pending; it must be approved before execution",
        });
      },
    } as Partial<XcelsiorApiClient>);
    const out = parse(await capture(tool, client)({ ...args, confirm: true, plan_id: "plan_123" }));
    expect(posted.map((p) => p.path)).toEqual(["/api/v1/launch-plans/plan_123/execute"]);
    expect(out.ok).toBe(false);
    expect(out.status).toBe(409);
    expect(out.launched).toBeUndefined();
  });
});

describe("schedule_under_budget checks the plan's quote, not only the listing", () => {
  it("withdraws a plan quoted above the budget", async () => {
    const { client, posted } = recordingClient({
      async post(path: string, body?: unknown) {
        posted.push({ path, body });
        if (path === "/api/v1/launch-plans") {
          return { ...PLAN, estimate: { hourly_burn_micros: 2_500_000 } };
        }
        return { ok: true };
      },
    } as Partial<XcelsiorApiClient>);
    const out = parse(
      await capture("schedule_under_budget", client)({ max_hourly_cad: 1, num_gpus: 1, pricing_mode: "spot", confirm: false }),
    );
    expect(posted.map((p) => p.path)).toEqual([
      "/api/v1/launch-plans",
      "/api/v1/launch-plans/plan_123/revoke",
    ]);
    expect(out.found).toBe(false);
    expect(out.plan_id).toBeUndefined();
    expect(out.quoted_hourly_cad_per_gpu).toBe(2.5);
  });

  it("binds the selected GPU into the plan", async () => {
    const { client, posted } = recordingClient();
    await capture("schedule_under_budget", client)({ max_hourly_cad: 1, num_gpus: 1, pricing_mode: "spot", confirm: false });
    expect(posted[0].body).toMatchObject({ gpu_model: "RTX 3060", pricing_mode: "spot" });
  });
});

describe("schedule_under_budget reads the prices the API returns", () => {
  // The route field names, not invented ones: it read price_cad and spot_cad,
  // which neither route writes, and so never found a GPU for anyone.
  it.each([
    ["on_demand", 0.5],
    ["spot", 0.4],
  ])("prices %s from the feed", async (pricing_mode, hourly) => {
    const { client } = recordingClient();
    const out = parse(
      await capture("schedule_under_budget", client)({ max_hourly_cad: 1, num_gpus: 1, pricing_mode, confirm: false }),
    );
    expect(out.found).toBe(true);
    expect(out.selected.hourly_cad).toBe(hourly);
  });

  it("finds nothing when the listed price is above the budget", async () => {
    const { client, posted } = recordingClient();
    const out = parse(
      await capture("schedule_under_budget", client)({ max_hourly_cad: 0.45, num_gpus: 1, pricing_mode: "on_demand", confirm: false }),
    );
    expect(out.found).toBe(false);
    expect(posted).toEqual([]);
  });
});
