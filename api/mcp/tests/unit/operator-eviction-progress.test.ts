import { describe, expect, it } from "vitest";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../../src/client/api.js";
import { registerOperatorTools } from "../../src/tools/operator.js";

type Handler = (args: Record<string, unknown>) => Promise<{
  content: Array<{ text: string }>;
  structuredContent: Record<string, unknown>;
}>;

function capture(response: Record<string, unknown>) {
  let handler: Handler | undefined;
  const posted: Array<{ path: string; body: unknown }> = [];
  const server = {
    registerTool(name: string, _config: unknown, fn: Handler) {
      if (name === "evict_host_workloads") handler = fn;
    },
  } as unknown as McpServer;
  const client = {
    async post(path: string, body: unknown) {
      posted.push({ path, body });
      return response;
    },
  } as unknown as XcelsiorApiClient;
  registerOperatorTools(server, client, { scopes: ["hosts:evict"] } as never);
  if (!handler) throw new Error("eviction tool was not registered");
  return { handler, posted };
}

describe("operator eviction reports settlement progress", () => {
  it("keeps pending distinct from settled and sends only the approved plan", async () => {
    const { handler, posted } = capture({ ok: true, pending: true, status: "executing" });
    const out = await handler({ host_id: "host-1", plan_id: "plan-1", confirm: true,
      expected_version: 999, reason: "different unapproved reason" });
    expect(posted).toEqual([{ path: "/api/v1/hosts/host-1/eviction-plans/plan-1/execute", body: {} }]);
    expect(out.structuredContent.pending).toBe(true);
    expect(out.content[0].text).toContain("pending fenced worker settlement");
    expect(out.content[0].text).not.toContain("executed.");
    expect(out.content[0].text).not.toContain("settled.");
  });

  it("reports settled only for durable completion", async () => {
    const { handler } = capture({ ok: true, pending: false, status: "succeeded" });
    const out = await handler({ host_id: "host-1", plan_id: "plan-1", confirm: true });
    expect(out.structuredContent.status).toBe("succeeded");
    expect(out.content[0].text).toBe("Eviction plan plan-1 settled.");
  });

  it("requires the human approval flow for previews", async () => {
    const { handler, posted } = capture({ ok: true, preview: true, plan_id: "plan-1" });
    const out = await handler({ host_id: "host-1", expected_version: 4, reason: "maintenance", confirm: false });
    expect(posted).toEqual([{ path: "/api/v1/hosts/host-1/eviction-plans", body: {
      expected_version: 4, reason: "maintenance",
    } }]);
    expect(out.structuredContent.approval_required).toBe(true);
    expect(out.structuredContent.preview).toBe(true);
  });
});
