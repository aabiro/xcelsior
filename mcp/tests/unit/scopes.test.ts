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

  it("allows when scopes undefined (session user)", () => {
    expect(userHasScope(undefined, ["billing:read"])).toBe(true);
  });
});