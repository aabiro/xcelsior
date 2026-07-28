import { describe, expect, it } from "vitest";
import { TOOL_SCOPES, userHasScope } from "../../src/auth/scopes.js";
import { TOOL_CONTRACTS } from "../../src/tools/contracts.js";

describe("production tool registry", () => {
  it("requires an explicit scope and complete contract for every tool", () => {
    for (const [name, scopes] of Object.entries(TOOL_SCOPES)) {
      expect(scopes.length, name).toBeGreaterThan(0);
      expect(TOOL_CONTRACTS[name], name).toMatchObject({
        version: expect.stringMatching(/^\d+\.\d+\.\d+$/),
        redaction: "classified",
        timeoutMs: expect.any(Number),
      });
      expect(userHasScope(undefined, scopes), name).toBe(false);
      expect(userHasScope([], scopes), name).toBe(false);
      expect(userHasScope(["wrong:scope"], scopes), name).toBe(false);
    }
  });

  it("does not grant legacy api unless it is explicitly present", () => {
    expect(userHasScope(["api"], ["hosts:evict"])).toBe(true);
    expect(userHasScope(["instances:read"], ["hosts:evict"])).toBe(false);
  });
});
