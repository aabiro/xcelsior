/**
 * P2's other half: "launch → connected, without a browser."
 *
 * Two properties are worth asserting, and neither is about the happy path.
 *
 *  1. **A ticket is a credential with a clock on it.** Minting one for an
 *     instance that cannot accept a connection burns it for nothing and reads
 *     back to the model as "access is ready". So the status check must happen
 *     before the mint, which is asserted by counting requests rather than by
 *     reading the message.
 *
 *  2. **The host handed out must be the one a user can reach.** The job record
 *     carries `host_ip`, the tailnet address the dashboard shows under "Direct
 *     SSH (requires mesh network)". A user off the mesh cannot reach it. The
 *     fixtures below deliberately set `host_ip` to a tailnet address so that
 *     returning it would be visible here.
 */

import { describe, expect, it } from "vitest";
import { z } from "zod";
import { registerComputeTools } from "../../src/tools/compute.js";
import type { AuthUser } from "../../src/auth/bearer.js";

type Handler = (args: Record<string, unknown>) => Promise<{ content: { text: string }[] }>;

function harness(instance: Record<string, unknown>, scopes = ["instances:connect", "instances:read"]) {
  const gets: string[] = [];
  const posts: { path: string; body: unknown }[] = [];
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
      return { instance };
    },
    async post(path: string, body?: unknown) {
      posts.push({ path, body });
      return { ok: true, ticket: "tkt-abc123", expires_in: 60 };
    },
  };

  registerComputeTools(
    server as unknown as Parameters<typeof registerComputeTools>[0],
    client as unknown as Parameters<typeof registerComputeTools>[1],
    { scopes } as AuthUser,
  );

  const call = async (args: Record<string, unknown>) => {
    const handler = handlers.get("open_instance_access")!;
    const parsed = schemas.get("open_instance_access")!.parse(args);
    const result = await handler(parsed as Record<string, unknown>);
    return JSON.parse(result.content[0].text) as Record<string, unknown>;
  };

  return { call, gets, posts, handlers };
}

const RUNNING = {
  job_id: "job-1",
  status: "running",
  ssh_port: 30022,
  host_ip: "100.64.0.9", // tailnet — unreachable off the mesh, must not be returned
};

describe("open_instance_access", () => {
  it("is registered", () => {
    expect(harness(RUNNING).handlers.has("open_instance_access")).toBe(true);
  });

  it("returns the public gateway, never the tailnet address", async () => {
    const { call } = harness(RUNNING);
    const result = await call({ job_id: "job-1", method: "ssh" });
    expect(result.ok).toBe(true);
    expect(result.port).toBe(30022);
    expect(String(result.host)).not.toBe("100.64.0.9");
    expect(JSON.stringify(result)).not.toContain("100.64.0.9");
    expect(String(result.command)).toMatch(/^ssh root@\S+ -p 30022$/);
  });

  it("says plainly that no host-key fingerprint is published", async () => {
    // Gate P2 asks for "the SSH endpoint plus the fingerprint to verify". The
    // platform publishes none. Returning a field would be inventing it and
    // saying nothing would let a model claim the connection is verified, so
    // the absence is stated in the payload.
    const { call } = harness(RUNNING);
    const result = await call({ job_id: "job-1", method: "ssh" });
    expect(result).toHaveProperty("host_key_fingerprint", null);
    expect(String(result.host_key_note)).toMatch(/does not yet publish/i);
  });

  it("mints no ticket for an instance that is not running", async () => {
    const { call, posts } = harness({ ...RUNNING, status: "starting" });
    const result = await call({ job_id: "job-1", method: "terminal" });
    expect(posts, "a ticket was minted for an instance that cannot accept it").toEqual([]);
    expect(result.ok).toBe(false);
    expect(result.error).toBe("instance_not_running");
    expect(result.status).toBe("starting");
  });

  it("mints a single-use ticket when asked, and says it is single-use", async () => {
    const { call, posts } = harness(RUNNING);
    const result = await call({ job_id: "job-1", method: "terminal" });
    expect(posts).toEqual([{ path: "/api/terminal/ticket", body: { instance_id: "job-1" } }]);
    expect(result.ticket).toBe("tkt-abc123");
    expect(result.expires_in_seconds).toBe(60);
    expect(String(result.note)).toMatch(/single-use/i);
    expect(String(result.websocket_url)).toMatch(/^wss?:\/\/.*\/ws\/terminal\/job-1$/);
  });

  it("auto prefers ssh, so the default path spends no credential", async () => {
    const { call, posts } = harness(RUNNING);
    const result = await call({ job_id: "job-1" });
    expect(result.method).toBe("ssh");
    expect(posts).toEqual([]);
  });

  it("auto falls back to a ticket when the instance publishes no port", async () => {
    const { call, posts } = harness({ ...RUNNING, ssh_port: null });
    const result = await call({ job_id: "job-1" });
    expect(result.method).toBe("terminal");
    expect(posts).toHaveLength(1);
  });

  it("explains rather than mints when ssh is asked for and unavailable", async () => {
    const { call, posts } = harness({ ...RUNNING, ssh_port: null });
    const result = await call({ job_id: "job-1", method: "ssh" });
    expect(posts).toEqual([]);
    expect(result.error).toBe("no_ssh_endpoint");
    expect(String(result.message)).toMatch(/terminal/);
  });

  it("refuses without instances:connect, before reading anything", async () => {
    const { call, gets, posts } = harness(RUNNING, ["instances:read"]);
    const result = await call({ job_id: "job-1", method: "ssh" });
    expect(gets).toEqual([]);
    expect(posts).toEqual([]);
    expect(JSON.stringify(result)).toMatch(/instances:connect/);
  });
});
