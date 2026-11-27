/**
 * stop_instance parks a running instance. It is not cancel and not terminate.
 *
 * The registered handler is what a connector calls. Confirm false reads the
 * instance and posts nothing. Confirm true posts /instances/{job_id}/stop.
 */

import { describe, expect, it } from "vitest";
import { registerComputeTools } from "../../src/tools/compute.js";
import type { AuthUser } from "../../src/auth/bearer.js";

type Handler = (args: { job_id: string; confirm: boolean }) => Promise<{
  content: { text: string }[];
  structuredContent: Record<string, unknown>;
  isError?: boolean;
}>;

function harness() {
  const handlers = new Map<string, Handler>();
  const calls: { method: "get" | "post"; path: string }[] = [];
  const server = {
    registerTool: (name: string, _config: unknown, handler: Handler) => {
      handlers.set(name, handler);
    },
  };
  const client = {
    get: async (path: string) => {
      calls.push({ method: "get", path });
      return { instance: { name: "ultra-sentinel", status: "running", job_id: "e42dc218" } };
    },
    post: async (path: string) => {
      calls.push({ method: "post", path });
      return { ok: true, instance: { status: "stopping", job_id: "e42dc218" } };
    },
  };
  registerComputeTools(
    server as unknown as Parameters<typeof registerComputeTools>[0],
    client as unknown as Parameters<typeof registerComputeTools>[1],
    { scopes: ["instances:write"] } as AuthUser,
  );
  return { handlers, calls };
}

describe("stop_instance", () => {
  it("previews and does not stop when confirm is false", async () => {
    const { handlers, calls } = harness();
    const result = await handlers.get("stop_instance")!({ job_id: "e42dc218", confirm: false });

    expect(result.isError).toBeUndefined();
    expect(result.structuredContent).toMatchObject({
      ok: true,
      preview: true,
      job_id: "e42dc218",
      name: "ultra-sentinel",
      status: "running",
    });
    expect(String(result.structuredContent.message)).toContain("confirm:true");
    expect(calls.map((call) => call.method)).toEqual(["get"]);
    expect(calls.some((call) => call.path.includes("/stop"))).toBe(false);
    expect(calls.some((call) => call.path.includes("/cancel"))).toBe(false);
    expect(calls.some((call) => call.path.includes("/terminate"))).toBe(false);
  });

  it("posts the reversible stop and neither cancel nor terminate when confirm is true", async () => {
    const { handlers, calls } = harness();
    const result = await handlers.get("stop_instance")!({ job_id: "e42dc218", confirm: true });

    expect(result.isError).toBeUndefined();
    expect(result.structuredContent).toMatchObject({
      ok: true,
      instance: { status: "stopping", job_id: "e42dc218" },
    });
    expect(calls).toEqual([{ method: "post", path: "/instances/e42dc218/stop" }]);
  });
});
