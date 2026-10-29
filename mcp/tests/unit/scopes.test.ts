import { describe, expect, it } from "vitest";
import { TOOL_SCOPES, userHasScope } from "../../src/auth/scopes.js";

/**
 * Two cases in this file previously asserted the authorization bug as if it
 * were the specification:
 *
 *   it("allows api wildcard", () =>
 *     expect(userHasScope(["api"], ["instances:write"])).toBe(true));
 *   it("allows matching scope", () =>
 *     expect(userHasScope(["billing:read"], ["billing:read", "api"])).toBe(true));
 *
 * The first locked in `api` as a universal bypass. The second locked in
 * any-one-of semantics — one scope satisfying a multi-scope requirement was the
 * whole defect, written down as a passing test. They are rewritten below to
 * state what the rules actually are; the full regression set lives in
 * `scope-enforcement.test.ts`.
 */
describe("userHasScope", () => {
  it("refuses the removed api wildcard everywhere", () => {
    expect(userHasScope(["api"], { allOf: ["instances:write"] })).toBe(false);
    expect(userHasScope(["api"], { allOf: ["hosts:evict"] })).toBe(false);
    expect(userHasScope(["api"], { allOf: ["control_plane:operate"] })).toBe(false);
  });

  it("requires all of allOf", () => {
    expect(userHasScope(["billing:read"], { allOf: ["billing:read", "instances:write"] }))
      .toBe(false);
    expect(userHasScope(["billing:read", "instances:write"],
      { allOf: ["billing:read", "instances:write"] })).toBe(true);
  });

  it("accepts any of anyOf", () => {
    expect(userHasScope(["billing:read"], { anyOf: ["billing:read", "instances:write"] }))
      .toBe(true);
  });

  it("denies missing scope", () => {
    expect(userHasScope(["instances:read"], TOOL_SCOPES.get_wallet_balance)).toBe(false);
  });

  it("denies when scopes are undefined or empty", () => {
    expect(userHasScope(undefined, TOOL_SCOPES.get_wallet_balance)).toBe(false);
    expect(userHasScope([], TOOL_SCOPES.get_wallet_balance)).toBe(false);
  });

  it("denies unknown and wrong scopes", () => {
    expect(userHasScope(["unknown"], TOOL_SCOPES.get_wallet_balance)).toBe(false);
    expect(userHasScope(["instances:read"], TOOL_SCOPES.get_wallet_balance)).toBe(false);
  });
});
