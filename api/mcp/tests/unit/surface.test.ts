import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  describeToolSurface,
  diffSurface,
  zodShape,
  type ToolSurfaceEntry,
} from "../../src/tools/surface.js";
import { z } from "zod";

const here = dirname(fileURLToPath(import.meta.url));
const SNAPSHOT_PATH = join(here, "..", "..", "tool-surface.json");
const snapshot = JSON.parse(readFileSync(SNAPSHOT_PATH, "utf8")) as {
  tools: ToolSurfaceEntry[];
};

const current = describeToolSurface("customer");
const SNAPSHOT_TEXT = readFileSync(SNAPSHOT_PATH, "utf8");

/**
 * The snapshot exactly as `scripts/update-tool-surface.ts` would write it.
 *
 * Duplicated from the generator on purpose, and it is the one duplication here
 * worth having: a test that imported the generator would pass by construction —
 * it would be comparing the generator to itself, which is the "generator that
 * read its own output" failure `tests/test_generated_artifacts_are_current.py`
 * names. The `$comment` and `generatedFor` values are part of the artifact, so
 * a hand edit to either has to fail too.
 */
function regenerate(): string {
  const fresh = {
    $comment:
      "Published MCP tool surface. Regenerate with `npm run surface:update` in the " +
      "same commit as the change. A breaking change requires a toolVersion bump — " +
      "see docs/mcp-tool-versioning.md.",
    generatedFor: "customer",
    tools: describeToolSurface("customer"),
  };
  return `${JSON.stringify(fresh, null, 2)}\n`;
}

/**
 * GX6: "a CI check that fails on an unversioned breaking tool change".
 *
 * This is that check. It is the only thing that gives
 * `_meta["xcelsior/toolVersion"]` any meaning — without it the field is a
 * constant that happens to look like a version.
 */
describe("published tool surface", () => {
  it("matches the committed snapshot, or bumps the version of what changed", () => {
    const changes = diffSurface(snapshot.tools, current);
    const previous = new Map(snapshot.tools.map((tool) => [tool.name, tool]));
    const now = new Map(current.map((tool) => [tool.name, tool]));

    const unversionedBreaking = changes.filter((change) => {
      if (!change.breaking) return false;
      const was = previous.get(change.tool);
      const is = now.get(change.tool);
      // A removed tool cannot bump its own version — removal always needs the
      // deprecation process, never a silent snapshot update.
      if (!is) return true;
      return was?.version === is.version;
    });

    expect(
      unversionedBreaking,
      "Breaking change to a published tool without a toolVersion bump.\n" +
        "Either make the change backward-compatible, or bump the tool's version " +
        "in src/tools/contracts.ts and follow docs/mcp-tool-versioning.md " +
        "(announce, overlap, changelog). Then run `npm run surface:update`.\n" +
        unversionedBreaking.map((c) => `  - ${c.tool}: ${c.detail}`).join("\n"),
    ).toEqual([]);

    // Additive changes are fine, but the snapshot still has to be refreshed so
    // the next diff is measured from reality.
    expect(
      changes.length === 0 || changes.every((c) => !c.breaking),
      "run `npm run surface:update` to record these changes:\n" +
        changes.map((c) => `  - ${c.tool}: ${c.detail}`).join("\n"),
    ).toBe(true);
    expect(
      current.length,
      "snapshot is stale — run `npm run surface:update`",
    ).toBe(snapshot.tools.length);
  });

  it("is byte-identical to a fresh generation", () => {
    /*
     * P0's gate, in full: *"regenerating the registry produces byte-identical
     * output; **a hand edit to a generated file fails the build**."*
     *
     * The checks above did not deliver the second half. `changes.every(c =>
     * !c.breaking)` is *true* for any non-breaking drift, so it passes while
     * printing "run `npm run surface:update` to record these changes" — a guard
     * that cannot fail for the reason its own message gives. The only other
     * currency check compared `current.length` to `snapshot.tools.length`,
     * which sees a tool appear or vanish and nothing else.
     *
     * Verified by editing a description in the committed snapshot: 13 tests,
     * all green.
     *
     * This is precisely the history `tests/test_public_openapi.py` records —
     * comparing only the operation set answered "are the right endpoints
     * published?" and never "does the document still describe them correctly?",
     * and five schemas drifted underneath it. Same shape, same fix: compare the
     * whole document.
     */
    expect(
      SNAPSHOT_TEXT,
      "tool-surface.json does not match a fresh generation. Either run " +
        "`npm run surface:update` in this commit, or undo the hand edit — this " +
        "file is generated and a hand edit to it is not a change anyone reviews.",
    ).toBe(regenerate());
  });

  it("snapshots the customer profile only", () => {
    const names = new Set(current.map((tool) => tool.name));
    for (const operatorTool of ["drain_host", "evict_host_workloads", "get_scheduler_health"]) {
      expect(names.has(operatorTool), `${operatorTool} leaked into the public snapshot`).toBe(false);
    }
    expect(current.every((tool) => tool.tenantClass === "tenant")).toBe(true);
  });

  it("gives every published tool a version and an output schema", () => {
    for (const tool of current) {
      expect(tool.version, tool.name).toMatch(/^\d+\.\d+\.\d+$/);
      expect(tool.hasOutputSchema, tool.name).toBe(true);
      expect(tool.requiredScopes.length, tool.name).toBeGreaterThan(0);
    }
  });
});

describe("breaking-change classification", () => {
  const base: ToolSurfaceEntry = {
    name: "demo_tool",
    version: "2.0.0",
    tenantClass: "tenant",
    requiredScopes: ["instances:read"],
    idempotency: "read",
    retry: "safe",
    annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    input: { job_id: { type: "string", optional: false }, limit: { type: "number", optional: true } },
    hasOutputSchema: true,
  };
  const change = (patch: Partial<ToolSurfaceEntry>) => diffSurface([base], [{ ...base, ...patch }]);

  it("flags a removed tool", () => {
    expect(diffSurface([base], [])[0]).toMatchObject({ breaking: true });
  });

  it("flags a newly-required input", () => {
    const result = change({ input: { ...base.input, region: { type: "string", optional: false } } });
    expect(result.some((c) => c.breaking && c.detail.includes("REQUIRED"))).toBe(true);
  });

  it("allows a new optional input", () => {
    const result = change({ input: { ...base.input, region: { type: "string", optional: true } } });
    expect(result.every((c) => !c.breaking)).toBe(true);
  });

  it("flags an optional input becoming required", () => {
    const result = change({ input: { ...base.input, limit: { type: "number", optional: false } } });
    expect(result.some((c) => c.breaking && c.detail.includes("became required"))).toBe(true);
  });

  it("flags a narrowed enum but allows a widened one", () => {
    const withEnum: ToolSurfaceEntry = {
      ...base,
      input: { mode: { type: "enum", optional: false, values: ["a", "b"] } },
    };
    const narrowed = diffSurface([withEnum], [{
      ...withEnum, input: { mode: { type: "enum", optional: false, values: ["a"] } },
    }]);
    expect(narrowed.some((c) => c.breaking)).toBe(true);
    const widened = diffSurface([withEnum], [{
      ...withEnum, input: { mode: { type: "enum", optional: false, values: ["a", "b", "c"] } },
    }]);
    expect(widened.every((c) => !c.breaking)).toBe(true);
  });

  it("flags a loosened annotation", () => {
    const result = change({
      annotations: { ...base.annotations, readOnlyHint: false },
    });
    expect(result.some((c) => c.breaking && c.detail.includes("readOnlyHint"))).toBe(true);
  });

  it("flags a narrowed scope set but allows a widened one", () => {
    expect(change({ requiredScopes: ["billing:read"] }).some((c) => c.breaking)).toBe(true);
    expect(
      change({ requiredScopes: ["instances:read", "gpu:read"] }).every((c) => !c.breaking),
    ).toBe(true);
  });

  it("treats a new tool as additive", () => {
    const added = diffSurface([base], [base, { ...base, name: "brand_new" }]);
    expect(added).toEqual([{ tool: "brand_new", breaking: false, detail: "new tool" }]);
  });
});

describe("zod shape extraction", () => {
  it("sees through optional, default, and nullable wrappers", () => {
    const shape = zodShape(
      z.object({
        required: z.string(),
        optional: z.string().optional(),
        defaulted: z.number().default(10),
        nullable: z.boolean().nullable(),
        choice: z.enum(["b", "a"]),
      }),
    );
    expect(shape.required).toEqual({ type: "string", optional: false });
    expect(shape.optional).toEqual({ type: "string", optional: true });
    expect(shape.defaulted).toEqual({ type: "number", optional: true });
    expect(shape.nullable).toEqual({ type: "boolean", optional: true });
    // Sorted so a reordered enum is not mistaken for a change.
    expect(shape.choice).toEqual({ type: "enum", optional: false, values: ["a", "b"] });
  });

  it("returns nothing for a non-object schema", () => {
    expect(zodShape(z.string())).toEqual({});
    expect(zodShape(undefined)).toEqual({});
  });
});
