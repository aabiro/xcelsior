import { describe, expect, it } from "vitest";
import { TOOL_SCOPES, userHasScope, scopeUnion } from "../../src/auth/scopes.js";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";

describe("production tool registry", () => {
  it("requires an explicit scope and complete contract for every tool", () => {
    for (const [name, requirement] of Object.entries(TOOL_SCOPES)) {
      expect(scopeUnion(requirement).length, name).toBeGreaterThan(0);
      expect(TOOL_CONTRACTS[name], name).toMatchObject({
        version: expect.stringMatching(/^\d+\.\d+\.\d+$/),
        redaction: "classified",
        timeoutMs: expect.any(Number),
      });
      expect(userHasScope(undefined, requirement), name).toBe(false);
      expect(userHasScope([], requirement), name).toBe(false);
      expect(userHasScope(["wrong:scope"], requirement), name).toBe(false);
    }
  });

  // Renamed and corrected. This case was called "does not grant legacy api
  // unless it is explicitly present" while asserting the exact opposite —
  // `expect(userHasScope(["api"], ["hosts:evict"])).toBe(true)`. The name
  // described the intended rule; the assertion pinned the bug in place, so the
  // suite stayed green while a tenant token could reach an operator scope.
  it("never lets the legacy api grant reach an operator scope", () => {
    expect(userHasScope(["api"], { allOf: ["hosts:evict"] })).toBe(false);
    expect(userHasScope(["instances:read"], { allOf: ["hosts:evict"] })).toBe(false);
    expect(userHasScope(["hosts:evict"], { allOf: ["hosts:evict"] })).toBe(true);
  });
});
