/**
 * P3 — durable state, and the two properties worth asserting.
 *
 * An instance's disk dies with the instance, so a volume is the only place work
 * survives a relaunch. GT0 classified all twenty volume and artifact operations
 * as `gap`: none of it was reachable without a browser.
 *
 *  1. **Destruction previews before it acts.** `detach_volume` and
 *     `delete_volume` require `confirm:true`. Asserted by counting requests
 *     rather than by reading the returned message — a preview that still sent
 *     the request would return reassuring text while doing the thing.
 *
 *  2. **The detach preview names what breaks.** The plan puts detach behind
 *     approval "since it can disrupt a running workload", which is only useful
 *     if the preview says *which* workload. "Are you sure?" is not approval;
 *     "this will pull the filesystem out from under job-7" is.
 */

import { describe, expect, it } from "vitest";
import { z } from "zod";
import { registerVolumeTools } from "../../src/tools/volumes.js";
import type { AuthUser } from "../../src/auth/bearer.js";

type Handler = (args: Record<string, unknown>) => Promise<{ content: { text: string }[] }>;

const ALL = ["volumes:read", "volumes:write", "artifacts:read", "instances:read"];

function harness(scopes: string[] = ALL, volume: Record<string, unknown> = {}) {
  const gets: string[] = [];
  const posts: { path: string; body?: unknown }[] = [];
  const deletes: string[] = [];
  const handlers = new Map<string, Handler>();
  const schemas = new Map<string, z.ZodObject<z.ZodRawShape>>();

  const server = {
    registerTool(name: string, config: { inputSchema: z.ZodObject<z.ZodRawShape> }, handler: Handler) {
      handlers.set(name, handler);
      schemas.set(name, config.inputSchema);
    },
  };
  const client = {
    async get(path: string) {
      gets.push(path);
      return { ok: true, volume: { volume_id: "vol-1", ...volume } };
    },
    async post(path: string, body?: unknown) {
      posts.push({ path, body });
      return { ok: true };
    },
    async delete(path: string) {
      deletes.push(path);
      return { ok: true };
    },
  };

  registerVolumeTools(
    server as unknown as Parameters<typeof registerVolumeTools>[0],
    client as unknown as Parameters<typeof registerVolumeTools>[1],
    { scopes } as AuthUser,
  );

  const call = async (name: string, args: Record<string, unknown> = {}) => {
    const handler = handlers.get(name);
    if (!handler) throw new Error(`${name} was never registered`);
    const parsed = schemas.get(name)!.parse(args);
    const result = await handler(parsed as Record<string, unknown>);
    return JSON.parse(result.content[0].text) as Record<string, unknown>;
  };

  return { call, gets, posts, deletes, handlers };
}

describe("volume tools", () => {
  it("registers the whole P3 set", () => {
    const { handlers } = harness();
    for (const name of [
      "list_volumes", "get_volume", "create_volume", "attach_volume",
      "detach_volume", "delete_volume", "snapshot_volume", "get_artifact_expiry",
    ]) {
      expect(handlers.has(name), name).toBe(true);
    }
  });

  it("detach without confirm sends no detach request", async () => {
    const { call, posts } = harness(ALL, { attached_instance_id: "job-7" });
    const result = await call("detach_volume", { volume_id: "vol-1" });
    expect(posts.filter((p) => p.path.includes("/detach")), "a preview detached the volume")
      .toEqual([]);
    expect(result.preview).toBe(true);
  });

  it("the detach preview names the instance that would lose its filesystem", async () => {
    const { call } = harness(ALL, { attached_instance_id: "job-7" });
    const result = await call("detach_volume", { volume_id: "vol-1" });
    expect(result.attached_to).toBe("job-7");
    expect(String(result.message)).toContain("job-7");
  });

  it("detach with confirm actually detaches", async () => {
    const { call, posts } = harness();
    await call("detach_volume", { volume_id: "vol-1", confirm: true });
    expect(posts.map((p) => p.path)).toEqual(["/api/v2/volumes/vol-1/detach"]);
  });

  it("delete without confirm sends no delete request", async () => {
    const { call, deletes } = harness();
    const result = await call("delete_volume", { volume_id: "vol-1" });
    expect(deletes, "a preview deleted the volume").toEqual([]);
    expect(result.preview).toBe(true);
    expect(String(result.message)).toMatch(/snapshot/i);
  });

  it("delete with confirm actually deletes", async () => {
    const { call, deletes } = harness();
    await call("delete_volume", { volume_id: "vol-1", confirm: true });
    expect(deletes).toEqual(["/api/v2/volumes/vol-1"]);
  });

  it("says what a new volume costs, since it bills whether attached or not", async () => {
    const { call } = harness();
    const result = await call("create_volume", { name: "weights", size_gb: 100 });
    expect(String(result.note)).toMatch(/per GB-month/i);
  });

  it("the retention clock tells the user to move work somewhere without one", async () => {
    const { call, gets } = harness();
    const result = await call("get_artifact_expiry", { job_id: "job-7" });
    expect(gets).toEqual(["/api/artifacts/job-7/expiry"]);
    expect(String(result.note)).toMatch(/volume/i);
  });

  it("refuses each tool without its scope, before touching the API", async () => {
    for (const [tool, args] of [
      ["list_volumes", {}],
      ["create_volume", { name: "x" }],
      ["delete_volume", { volume_id: "vol-1", confirm: true }],
      ["get_artifact_expiry", { job_id: "job-7" }],
    ] as [string, Record<string, unknown>][]) {
      const { call, gets, posts, deletes } = harness(["instances:read"]);
      const result = await call(tool, args);
      expect(gets, tool).toEqual([]);
      expect(posts, tool).toEqual([]);
      expect(deletes, tool).toEqual([]);
      expect(JSON.stringify(result), tool).toContain("insufficient_scope");
    }
  });
});
