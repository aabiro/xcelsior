import { describe, expect, it } from "vitest";
import { userHasScope } from "../../src/auth/scopes.js";

describe("userHasScope", () => {
  it("allows api wildcard", () => {
    expect(userHasScope(["api"], ["instances:write"])).toBe(true);
  });

  it("allows matching scope", () => {
    expect(userHasScope(["billing:read"], ["billing:read", "api"])).toBe(true);
  });

  it("denies missing scope", () => {
    expect(userHasScope(["instances:read"], ["billing:read"])).toBe(false);
  });

  it("denies when scopes are undefined or empty", () => {
    expect(userHasScope(undefined, ["billing:read"])).toBe(false);
    expect(userHasScope([], ["billing:read"])).toBe(false);
  });

  it("denies unknown and wrong scopes", () => {
    expect(userHasScope(["unknown"], ["billing:read"])).toBe(false);
    expect(userHasScope(["instances:read"], ["billing:read"])).toBe(false);
  });
});
