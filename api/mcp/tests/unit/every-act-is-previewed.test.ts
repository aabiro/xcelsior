import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";

/**
 * Every tool that acts on an account previews first and acts only when told to.
 *
 * The connector listing says so: "any tool that changes data or starts billed
 * work needs an explicit confirm after a preview". On 2026-11-24 that was
 * false for 15 of 29 destructive tools: top_up_wallet charged a card in one
 * call, create_volume started a monthly bill, register_ssh_key granted shell
 * access, all without a preview. This test is what keeps the sentence true.
 *
 * A destructive tool must take `confirm` (preview without it, act with it) or
 * `plan_id` (prepare a plan the user approves, execute only that plan). The
 * exemptions are each in the safe direction or behind a stronger gate, and
 * each says why. Adding a tool to this list is a claim a reviewer will test.
 */
const EXEMPT: Record<string, string> = {
  revoke_launch_plan:
    "Only ever removes the ability to spend. Asking a user to confirm the safe " +
    "direction teaches them to click through the prompts that matter. (Its " +
    "plan_id names the plan to withdraw; it is not a gate.)",
  run_pipeline:
    "Only quotes a pipeline. Nothing runs until the user approves it in the " +
    "dashboard, which an agent credential cannot do.",
};
// configure_auto_topup is gated by plan_id, not exempt: raising or enabling a
// limit comes back as a plan the user approves. Lowering one applies at once,
// which only reduces what can be spent.

type SurfaceTool = {
  name: string;
  annotations?: { destructiveHint?: boolean };
  input?: Record<string, unknown>;
};

const surface = JSON.parse(
  readFileSync(new URL("../../tool-surface.json", import.meta.url), "utf8"),
) as { tools: SurfaceTool[] };

const destructive = surface.tools.filter((t) => t.annotations?.destructiveHint);

describe("every act is previewed", () => {
  it.each(destructive.map((t) => [t.name, t] as const))("%s previews before it acts", (name, tool) => {
    const input = tool.input ?? {};
    const gated = "confirm" in input || "plan_id" in input;
    if (EXEMPT[name]) return;
    expect(gated, `${name} acts in one call: give it a preview and confirm, or a plan`).toBe(true);
  });

  it("names only real, destructive tools as exempt", () => {
    const names = new Set(destructive.map((t) => t.name));
    for (const name of Object.keys(EXEMPT)) expect(names.has(name), name).toBe(true);
  });
});
