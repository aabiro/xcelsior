import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";
import { toolsInProfile } from "../../src/tools/profiles.js";

const here = dirname(fileURLToPath(import.meta.url));
const EVAL_PATH = join(here, "..", "..", "evals", "tool-selection.jsonl");

interface EvalCase {
  id: string;
  category: string;
  prompt: string;
  why?: string;
  expect_any_of?: string[];
  expect_none?: boolean;
  context?: Array<{ role: string; content: string }>;
}

const CASES: EvalCase[] = readFileSync(EVAL_PATH, "utf8")
  .split("\n")
  .filter((line) => line.trim())
  .map((line) => JSON.parse(line) as EvalCase);

const CUSTOMER_TOOLS = new Set(toolsInProfile("customer"));

/**
 * The eval set is data, and data rots quietly. These checks are cheap and run
 * in CI without an API key; the model-in-the-loop run lives in
 * `scripts/mcp_tool_eval.py` and needs one.
 */
describe("tool-selection eval set", () => {
  it("parses and is not empty", () => {
    expect(CASES.length).toBeGreaterThan(20);
  });

  it("has unique ids", () => {
    const ids = CASES.map((c) => c.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it("covers every category the plan names", () => {
    // X1.14: direct requests, indirect phrasings, follow-ups reusing earlier
    // ids, write actions requiring approval, and requests that should call no
    // tool at all.
    const categories = new Set(CASES.map((c) => c.category));
    expect([...categories].sort()).toEqual(
      ["approval", "direct", "followup", "indirect", "no_tool"],
    );
    for (const category of categories) {
      expect(CASES.filter((c) => c.category === category).length, category)
        .toBeGreaterThanOrEqual(4);
    }
  });

  it("only expects tools that exist in the public customer profile", () => {
    // An expectation naming a removed or operator-only tool would fail forever
    // and be indistinguishable from a real regression.
    for (const testCase of CASES) {
      for (const name of testCase.expect_any_of ?? []) {
        expect(TOOL_CONTRACTS[name], `${testCase.id} expects unknown tool ${name}`).toBeTruthy();
        expect(CUSTOMER_TOOLS.has(name), `${testCase.id} expects operator tool ${name}`).toBe(true);
      }
    }
  });

  it("gives every case exactly one kind of expectation", () => {
    for (const testCase of CASES) {
      const hasPositive = (testCase.expect_any_of ?? []).length > 0;
      expect(hasPositive !== Boolean(testCase.expect_none), testCase.id).toBe(true);
    }
  });

  it("explains why every case exists", () => {
    // A failing case with no rationale gets deleted rather than fixed.
    for (const testCase of CASES) {
      expect(testCase.why?.length ?? 0, testCase.id).toBeGreaterThan(20);
    }
  });

  it("gives every follow-up case prior context to reuse", () => {
    for (const testCase of CASES.filter((c) => c.category === "followup")) {
      expect((testCase.context ?? []).length, testCase.id).toBeGreaterThanOrEqual(2);
      // The point of a follow-up is that the id only exists in the earlier turn.
      expect(testCase.prompt, testCase.id).not.toMatch(/\bjb-|plan[_ ]id|[0-9a-f]{8}-/i);
    }
  });

  it("exercises the approval-gated tools", () => {
    const approvalTargets = new Set(
      CASES.filter((c) => c.category === "approval").flatMap((c) => c.expect_any_of ?? []),
    );
    for (const gated of ["create_instance", "create_serverless_endpoint"]) {
      expect(approvalTargets, `${gated} is never exercised`).toContain(gated);
    }
  });

  it("covers a majority of the public surface", () => {
    // Not every tool needs a case, but a surface where most tools are never
    // selected by any phrasing is a surface a model cannot navigate.
    const covered = new Set(CASES.flatMap((c) => c.expect_any_of ?? []));
    expect(covered.size).toBeGreaterThanOrEqual(Math.ceil(CUSTOMER_TOOLS.size * 0.5));
  });
});
