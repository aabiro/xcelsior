import { describe, expect, it } from "vitest";
import { permissionRows, presetsFrom, presetScopes, sameScopes, toggleScope } from "@/lib/mcp-permissions";

const CEILING = [
  "instances:read",
  "instances:write",
  "instances:connect",
  "ssh:write",
  "volumes:read",
  "volumes:write",
  "billing:read",
];

describe("MCP permission logic", () => {
  it("builds one row per resource with read first", () => {
    expect(permissionRows(CEILING)).toEqual([
      { resource: "instances", scopes: ["instances:read", "instances:write", "instances:connect"] },
      { resource: "ssh", scopes: ["ssh:write"] },
      { resource: "volumes", scopes: ["volumes:read", "volumes:write"] },
      { resource: "billing", scopes: ["billing:read"] },
    ]);
  });

  it("Read holds nothing that changes a resource; Write and Custom start with everything", () => {
    const presets = presetsFrom(CEILING);
    expect(presetScopes("read", presets)).toEqual(["instances:read", "volumes:read", "billing:read"]);
    expect(presetScopes("write", presets)).toEqual(CEILING);
    expect(presetScopes("custom", presets)).toEqual(CEILING);
  });

  it("ticking write ticks read on the same resource", () => {
    expect(toggleScope(["billing:read"], "volumes:write", CEILING)).toEqual([
      "volumes:read",
      "volumes:write",
      "billing:read",
    ]);
    // Any scope that changes the resource counts, not only `:write`.
    expect(toggleScope([], "instances:connect", CEILING)).toEqual(["instances:read", "instances:connect"]);
  });

  it("a write with no read scope on the key stands alone", () => {
    expect(toggleScope([], "ssh:write", CEILING)).toEqual(["ssh:write"]);
  });

  it("unticking read unticks what depends on it", () => {
    expect(toggleScope(CEILING, "instances:read", CEILING)).toEqual([
      "ssh:write",
      "volumes:read",
      "volumes:write",
      "billing:read",
    ]);
  });

  it("unticking write leaves read alone", () => {
    expect(toggleScope(["volumes:read", "volumes:write"], "volumes:write", CEILING)).toEqual(["volumes:read"]);
  });

  it("compares selections regardless of order", () => {
    expect(sameScopes(["a:read", "b:read"], ["b:read", "a:read"])).toBe(true);
    expect(sameScopes(["a:read"], ["a:read", "b:read"])).toBe(false);
  });
});
