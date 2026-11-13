/**
 * Telling the user what a connection can do, and asking them for more.
 *
 * The property that matters most is the negative one: the agent's side can
 * only *ask*. These tools read and post a request; nothing here, and nothing
 * the API offers a key, changes a permission. The API half of that is asserted
 * in tests/test_mcp_quick_connect.py against real routes.
 */

import { describe, expect, it } from "vitest";
import { scopeDenial } from "../../src/auth/denial.js";
import { TOOL_SCOPES, satisfiesScope } from "../../src/auth/scopes.js";
import { registerPermissionTools, toolsLockedBy } from "../../src/tools/permissions.js";
import type { AuthUser } from "../../src/auth/bearer.js";

type Handler = (args: Record<string, unknown>) => Promise<{ structuredContent: Record<string, unknown>; isError?: boolean }>;

function harness(user: AuthUser, responses: Record<string, unknown> = {}) {
  const calls: { method: string; path: string; body?: unknown }[] = [];
  const handlers = new Map<string, Handler>();
  const server = {
    registerTool(name: string, _config: unknown, handler: Handler) {
      handlers.set(name, handler);
    },
  };
  const client = {
    async get(path: string) {
      calls.push({ method: "GET", path });
      return responses[path];
    },
    async post(path: string, body: unknown) {
      calls.push({ method: "POST", path, body });
      return responses[path];
    },
  };
  registerPermissionTools(server as never, client as never, user);
  return { calls, call: (name: string, args: Record<string, unknown> = {}) => handlers.get(name)!(args) };
}

const READ_KEY: AuthUser = {
  auth_type: "agent_api_key",
  scopes: ["instances:read", "billing:read", "gpu:read"],
};

describe("a refused tool says what to do next", () => {
  it("names what is missing and points a key at request_permission", () => {
    const denial = scopeDenial("create_instance", READ_KEY)!;
    expect(denial.error).toBe("insufficient_scope");
    expect(denial.missing).toEqual(["instances:write"]);
    expect(String(denial.how_to_fix)).toContain("request_permission");
    expect(String(denial.how_to_fix)).toContain("same key keeps working");
  });

  it("lists only what the credential lacks, not everything the tool needs", () => {
    // open_instance_access needs connect and read; this key has read.
    expect(scopeDenial("open_instance_access", READ_KEY)!.missing).toEqual(["instances:connect"]);
  });

  it("does not send a signed-in connector to a tool that would refuse it", () => {
    const denial = scopeDenial("create_instance", { auth_type: "oauth", scopes: ["instances:read"] })!;
    expect(String(denial.how_to_fix)).not.toContain("request_permission");
    expect(String(denial.how_to_fix)).toContain("reconnect");
  });

  it("is null when the credential may use the tool", () => {
    expect(scopeDenial("list_instances", READ_KEY)).toBeNull();
  });
});

describe("the permission tools are open to any credential that holds anything", () => {
  it.each(["get_permissions", "request_permission"])("%s", (tool) => {
    expect(satisfiesScope(["reputation:read"], TOOL_SCOPES[tool])).toBe(true);
    expect(satisfiesScope(["ssh:write"], TOOL_SCOPES[tool])).toBe(true);
    // Nothing at all is still nothing: the registry denies an empty grant.
    expect(satisfiesScope([], TOOL_SCOPES[tool])).toBe(false);
  });
});

describe("toolsLockedBy", () => {
  it("groups the tools a read-only key cannot use under the permission that unlocks them", () => {
    const locked = toolsLockedBy(READ_KEY.scopes!);
    expect(locked["instances:write"]).toContain("create_instance");
    expect(locked["billing:topup"]).toContain("top_up_wallet");
    expect(locked["instances:read"]).toBeUndefined();
    // Operator tools are not something a customer's key is "missing".
    expect(Object.values(locked).flat()).not.toContain("drain_host");
  });

  it("is empty of customer permissions for a key that holds them all", () => {
    const everything = [...new Set(Object.values(TOOL_SCOPES).flatMap((r) => [...(r.allOf ?? []), ...(r.anyOf ?? [])]))];
    expect(toolsLockedBy(everything)).toEqual({});
  });
});

describe("get_permissions", () => {
  it("returns the API's plain-language answer with a summary and the locked tools", async () => {
    const h = harness(READ_KEY, {
      "/api/mcp/permissions": {
        ok: true,
        credential: "quick_connect",
        granted: [{ scope: "instances:read", description: "See your GPU instances" }],
        not_granted: [{ scope: "instances:write", description: "Launch and terminate GPU instances", how: "dashboard_or_request" }],
        manage_url: "https://xcelsior.ca/dashboard/mcp",
      },
    });
    const { structuredContent: out, isError } = await h.call("get_permissions");
    expect(isError).toBeUndefined();
    expect(h.calls).toEqual([{ method: "GET", path: "/api/mcp/permissions" }]);
    expect(out.summary).toContain("can: See your GPU instances");
    expect(out.summary).toContain("cannot: Launch and terminate GPU instances");
    expect(out.summary).toContain("https://xcelsior.ca/dashboard/mcp");
    expect((out.unavailable_tools as Record<string, string[]>)["instances:write"]).toContain("create_instance");
  });
});

describe("request_permission", () => {
  it("posts the request and tells the agent that only the user can approve", async () => {
    const h = harness(READ_KEY, {
      "/api/mcp/permission-requests": {
        ok: true,
        status: "pending",
        code: "ABCDE-FGHJK",
        approve_url: "https://xcelsior.ca/dashboard/mcp/approve?code=ABCDE-FGHJK",
      },
    });
    const { structuredContent: out } = await h.call("request_permission", {
      permissions: ["instances:write"],
      reason: "Launch the training run",
    });
    expect(h.calls).toEqual([
      {
        method: "POST",
        path: "/api/mcp/permission-requests",
        body: { scopes: ["instances:write"], reason: "Launch the training run" },
      },
    ]);
    expect(out.approve_url).toBe("https://xcelsior.ca/dashboard/mcp/approve?code=ABCDE-FGHJK");
    expect(String(out.next_step)).toContain("Nothing is granted yet");
    expect(String(out.next_step)).toContain("do not open the link or approve for them");
  });

  it("says to retry when the connection already holds the permission", async () => {
    const h = harness(READ_KEY, { "/api/mcp/permission-requests": { ok: true, status: "already_granted" } });
    const { structuredContent: out } = await h.call("request_permission", { permissions: ["instances:read"] });
    expect(String(out.next_step)).toContain("already holds");
  });

  it("makes exactly one call and never a second one that could grant", async () => {
    const h = harness(READ_KEY, { "/api/mcp/permission-requests": { ok: true, status: "pending" } });
    await h.call("request_permission", { permissions: ["billing:topup"] });
    expect(h.calls).toHaveLength(1);
    expect(h.calls[0].path).toBe("/api/mcp/permission-requests");
  });
});
