import { describe, it, expect } from "vitest";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../../src/client/api.js";
import { registerBillingTools } from "../../src/tools/billing.js";
import { ApiError } from "../../src/client/errors.js";
import { TOOL_SCOPES } from "../../src/auth/scopes.js";

/**
 * Gate P1 clause 6: raising a spend cap requires approval, lowering one does not.
 *
 * The API enforces that — a widening from a non-human caller is refused **409**,
 * naming `/api/v2/billing/auto-topup-plans`. But `configure_auto_topup` posted
 * straight to `/api/v2/billing/auto-topup` and returned the refusal, so from an
 * agent's side the lever simply did not work: it could narrow auto-top-up and
 * not widen it, and the error told it about a route it never called.
 *
 * A 409 that names the way forward is still a dead end if nothing walks it, and
 * "every dead end has a lever inside the terminal" is the plan's fourth clause.
 *
 * ## What is asserted
 *
 * The **two-phase shape**: a refused widening comes back as an approvable plan,
 * and a second call carrying `plan_id` executes that plan rather than resending
 * the numbers. The second half is the security-relevant one — the settings come
 * from the plan, so an approval cannot be spent on different values.
 *
 * Narrowing must stay a single call. If approval were demanded for lowering a
 * cap, the safe direction would have friction the unsafe one lacks, which is
 * how a control gets routed around.
 */

type Handler = (args: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }>;

function captureConfigureAutoTopup(client: Partial<XcelsiorApiClient>): Handler {
  let handler: Handler | undefined;
  const recorder = {
    registerTool(name: string, _config: unknown, fn: Handler) {
      if (name === "configure_auto_topup") handler = fn;
      return undefined;
    },
  } as unknown as McpServer;
  const user = { scopes: [...TOOL_SCOPES.configure_auto_topup.allOf!] } as never;
  registerBillingTools(recorder, client as XcelsiorApiClient, user);
  if (!handler) throw new Error("configure_auto_topup was not registered");
  return handler;
}

function parse(result: { content: Array<{ text: string }> }): Record<string, unknown> {
  return JSON.parse(result.content[0].text);
}

const WIDENING = { enabled: true, amount_cad: 200, threshold_cad: 50 };

describe("widening auto-top-up offers a plan instead of a refusal", () => {
  it("turns the API's 409 into an approvable plan", async () => {
    const posted: Array<{ path: string; body: unknown }> = [];
    const handler = captureConfigureAutoTopup({
      async post(path: string, body?: unknown) {
        posted.push({ path, body });
        if (path === "/api/v2/billing/auto-topup") {
          throw new ApiError("conflict", 409, undefined, {
            status: 409,
            detail: "Widening auto-top-up requires an approved plan.",
          });
        }
        return { ok: true, preview: true, plan_id: "plan_abc", approval_url: "/dashboard/x" };
      },
    } as Partial<XcelsiorApiClient>);

    const out = parse(await handler(WIDENING));

    expect(posted.map((p) => p.path)).toEqual([
      "/api/v2/billing/auto-topup",
      "/api/v2/billing/auto-topup-plans",
    ]);
    expect(out.plan_id).toBe("plan_abc");
    expect(out.approval_url).toBe("/dashboard/x");
    expect(String(out.next_step)).toContain("plan_id");
    // The requested numbers must reach the plan, or the user approves a blank.
    expect(posted[1].body).toMatchObject({ amount_cad: 200, threshold_cad: 50 });
  });

  it("executes the approved plan without resending the numbers", async () => {
    const posted: Array<{ path: string; body: unknown }> = [];
    const handler = captureConfigureAutoTopup({
      async post(path: string, body?: unknown) {
        posted.push({ path, body });
        return { ok: true, applied: true };
      },
    } as Partial<XcelsiorApiClient>);

    // Deliberately different numbers from the ones "approved": they must be
    // ignored. The API takes the settings from the plan, and the tool must not
    // undermine that by posting its own.
    await handler({ ...WIDENING, amount_cad: 9999, plan_id: "plan_abc" });

    expect(posted).toHaveLength(1);
    expect(posted[0].path).toBe("/api/v2/billing/auto-topup-plans/plan_abc/execute");
    expect(posted[0].body).toEqual({});
  });

  it("leaves narrowing as a single call", async () => {
    const posted: string[] = [];
    const handler = captureConfigureAutoTopup({
      async post(path: string) {
        posted.push(path);
        return { ok: true, previous: {} };
      },
    } as Partial<XcelsiorApiClient>);

    await handler({ enabled: false, amount_cad: 10, threshold_cad: 5 });

    expect(posted).toEqual(["/api/v2/billing/auto-topup"]);
  });

  it("does not treat every failure as a widening", async () => {
    // A 500 must surface as an error, not quietly become a plan the user is
    // asked to approve. Matching on the status is what keeps this honest.
    const posted: string[] = [];
    const handler = captureConfigureAutoTopup({
      async post(path: string) {
        posted.push(path);
        throw new ApiError("boom", 500, undefined, { status: 500 });
      },
    } as Partial<XcelsiorApiClient>);

    const out = parse(await handler(WIDENING));

    expect(posted).toEqual(["/api/v2/billing/auto-topup"]);
    expect(String(out.error)).toContain("500");
  });
});
