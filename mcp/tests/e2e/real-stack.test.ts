import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { randomUUID } from "node:crypto";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const enabled = process.env.XCELSIOR_MCP_REAL_STACK_E2E === "1";
const apiUrl = (process.env.XCELSIOR_MCP_E2E_API_URL || "http://127.0.0.1:18980").replace(/\/$/, "");
const mcpPort = Number(process.env.XCELSIOR_MCP_E2E_PORT || "18981");
const redisUrl = process.env.XCELSIOR_MCP_E2E_REDIS_URL || "redis://127.0.0.1:6387/0";
const manageRedis = process.env.XCELSIOR_MCP_E2E_MANAGE_REDIS !== "0";
const externalRedisContainer = process.env.XCELSIOR_MCP_E2E_REDIS_CONTAINER || "";
const resource = "https://mcp.xcelsior.ca";
const marker = `mcp-sdk-e2e-${randomUUID().slice(0, 8)}`;

type Session = { browser: string; machine: string; customer: string; clientId: string };
type ToolResult = { structuredContent?: Record<string, any>; content?: unknown; isError?: boolean };

let mcp: ChildProcess;
let mcpBlue: ChildProcess;
let redis: ChildProcess;
let tenantA: Session;
let tenantB: Session;
let clientA: Client;
let transportA: StreamableHTTPClientTransport;
let clientBlue: Client;
let transportBlue: StreamableHTTPClientTransport;

async function jsonFetch(path: string, init: RequestInit = {}): Promise<any> {
  const response = await fetch(`${apiUrl}${path}`, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw Object.assign(new Error(`${response.status} ${path}: ${JSON.stringify(body)}`), {
      status: response.status,
      body,
    });
  }
  return body;
}

function bearer(token: string, extra: Record<string, string> = {}): Record<string, string> {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json", ...extra };
}

async function createSession(label: string): Promise<Session> {
  const email = `${marker}-${label}@example.test`;
  const registered = await jsonFetch("/api/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password: "McpE2e-StrongPass-2026!" }),
  });
  const browser = registered.access_token as string;
  const principal = await jsonFetch("/api/auth/introspect", {
    headers: bearer(browser),
  });
  const quick = await jsonFetch("/api/mcp/quick-connect", {
    headers: bearer(browser),
  });
  return {
    browser,
    machine: quick.access_token,
    customer: principal.customer_id,
    clientId: quick.client_id,
  };
}

async function createMachineToken(
  browser: string,
  name: string,
  scopes: string[],
): Promise<string> {
  const created = await jsonFetch("/api/oauth/clients", {
    method: "POST",
    headers: bearer(browser),
    body: JSON.stringify({
      client_name: name,
      client_type: "confidential",
      redirect_uris: [],
      grant_types: ["client_credentials"],
      scopes,
    }),
  });
  const token = await jsonFetch("/oauth/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "client_credentials",
      client_id: created.client.client_id,
      client_secret: created.client.client_secret,
      resource,
      scope: scopes.join(" "),
    }),
  });
  return token.access_token;
}

async function sdkClient(token: string, name: string, port = mcpPort): Promise<{
  client: Client;
  transport: StreamableHTTPClientTransport;
}> {
  const client = new Client({ name, version: "1.0.0" });
  const transport = new StreamableHTTPClientTransport(
    new URL(`http://127.0.0.1:${port}/mcp`),
    { requestInit: { headers: { Authorization: `Bearer ${token}` } } },
  );
  await client.connect(transport);
  return { client, transport };
}

async function call(client: Client, name: string, args: Record<string, unknown>): Promise<Record<string, any>> {
  const result = await client.callTool({ name, arguments: args }) as ToolResult;
  expect(
    result.structuredContent,
    `${name} returned no structuredContent: ${JSON.stringify(result)}`,
  ).toBeTruthy();
  return result.structuredContent!;
}

async function approve(planId: string, browser = tenantA.browser): Promise<void> {
  const current = await jsonFetch(`/api/v1/launch-plans/${encodeURIComponent(planId)}`, {
    headers: bearer(browser),
  });
  await jsonFetch(`/api/v1/launch-plans/${encodeURIComponent(planId)}/approve`, {
    method: "POST",
    headers: bearer(browser, { "Idempotency-Key": randomUUID() }),
    body: JSON.stringify({ expected_version: current.plan.version, confirmation: "approve" }),
  });
}

function fixture(...args: string[]): any {
  const python = process.env.XCELSIOR_MCP_E2E_PYTHON || ".venv/bin/python";
  const result = spawnSync(python, ["scripts/mcp_e2e_fixture.py", ...args], {
    cwd: new URL("../", `file://${process.cwd()}/`).pathname,
    env: {
      ...process.env,
      MCP_RATE_LIMIT_BACKEND: "redis",
      XCELSIOR_MCP_REDIS_URL: redisUrl,
    },
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`fixture ${args.join(" ")} failed:\n${result.stdout}\n${result.stderr}`);
  }
  return JSON.parse(result.stdout.trim().split("\n").at(-1)!);
}

beforeAll(async () => {
  if (!enabled) return;
  if (manageRedis) {
    redis = spawn("redis-server", [
      "--port", new URL(redisUrl).port || "6387",
      "--save", "", "--appendonly", "no", "--bind", "127.0.0.1",
    ], { stdio: "pipe" });
  }
  for (let attempt = 0; attempt < 100; attempt += 1) {
    const ping = spawnSync("redis-cli", ["-u", redisUrl, "ping"], { encoding: "utf8" });
    if (ping.status === 0 && ping.stdout.includes("PONG")) break;
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  tenantA = await createSession("tenant-a");
  tenantB = await createSession("tenant-b");
  await jsonFetch(`/api/billing/wallet/${encodeURIComponent(tenantA.customer)}/deposit`, {
    method: "POST",
    headers: bearer(tenantA.browser),
    body: JSON.stringify({ amount_cad: 1000, description: "MCP real-stack E2E" }),
  });
  mcp = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      XCELSIOR_MCP_API_URL: apiUrl,
      XCELSIOR_MCP_RESOURCE_AUDIENCE: resource,
      XCELSIOR_MCP_PUBLIC_URL: `http://127.0.0.1:${mcpPort}/mcp`,
      XCELSIOR_MCP_REDIS_URL: redisUrl,
      MCP_RATE_LIMIT_BACKEND: "redis",
      MCP_RATE_LIMIT_REQUIRE_REDIS: "true",
      MCP_RATE_LIMIT_PER_MIN: "2000",
      MCP_HOST: "127.0.0.1",
      MCP_PORT: String(mcpPort),
      MCP_PATH: "/mcp",
      OTEL_EXPORTER_OTLP_ENDPOINT: "",
    },
    stdio: "pipe",
  });
  for (let attempt = 0; attempt < 200; attempt += 1) {
    try {
      if ((await fetch(`http://127.0.0.1:${mcpPort}/readyz`)).ok) break;
    } catch { /* booting */ }
    if (mcp.exitCode !== null) throw new Error(`MCP exited during startup (${mcp.exitCode})`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  ({ client: clientA, transport: transportA } = await sdkClient(tenantA.machine, "real-stack-a"));
  mcpBlue = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      XCELSIOR_MCP_API_URL: apiUrl,
      XCELSIOR_MCP_RESOURCE_AUDIENCE: resource,
      XCELSIOR_MCP_PUBLIC_URL: `http://127.0.0.1:${mcpPort + 1}/mcp`,
      XCELSIOR_MCP_REDIS_URL: redisUrl,
      MCP_RATE_LIMIT_BACKEND: "redis",
      MCP_RATE_LIMIT_REQUIRE_REDIS: "true",
      MCP_RATE_LIMIT_PER_MIN: "2000",
      MCP_HOST: "127.0.0.1",
      MCP_PORT: String(mcpPort + 1),
      MCP_PATH: "/mcp",
      OTEL_EXPORTER_OTLP_ENDPOINT: "",
    },
    stdio: "pipe",
  });
  for (let attempt = 0; attempt < 200; attempt += 1) {
    try {
      if ((await fetch(`http://127.0.0.1:${mcpPort + 1}/readyz`)).ok) break;
    } catch { /* booting */ }
    if (mcpBlue.exitCode !== null) throw new Error(`second MCP exited during startup (${mcpBlue.exitCode})`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  ({ client: clientBlue, transport: transportBlue } = await sdkClient(
    tenantA.machine, "real-stack-blue", mcpPort + 1,
  ));
}, 30_000);

afterAll(async () => {
  if (!enabled) return;
  await transportA?.close().catch(() => undefined);
  await transportBlue?.close().catch(() => undefined);
  mcp?.kill("SIGTERM");
  mcpBlue?.kill("SIGTERM");
  fixture("cleanup", marker);
  if (manageRedis && redis && redis.exitCode === null) redis.kill("SIGTERM");
});

describe.runIf(enabled)("§26.4 real MCP + API + PostgreSQL + Redis", () => {
  it("completes the thirteen-step production journey", async () => {
    // 1–2. Real protocol initialization happened in beforeAll; validate every
    // registered tool carries schemas and annotations.
    const listed = await clientA.listTools();
    expect(fixture("spend-counter-check").ok).toBe(true);
    const replicaLists = await Promise.all([
      clientA.listTools(), clientBlue.listTools(), clientA.listTools(), clientBlue.listTools(),
    ]);
    expect(replicaLists.every((value) => value.tools.length === listed.tools.length)).toBe(true);
    expect(listed.tools.length).toBeGreaterThanOrEqual(35);
    for (const tool of listed.tools) {
      expect(tool.inputSchema, tool.name).toBeTruthy();
      expect(tool.outputSchema, tool.name).toBeTruthy();
      expect(tool.annotations, tool.name).toBeTruthy();
    }

    // 3–6. Two tenants, plan preview, human approval, execute, exact replay.
    const idempotency = randomUUID();
    const preview = await call(clientA, "create_instance", {
      name: `${marker}-instance`,
      gpu_model: "MCP E2E GPU",
      num_gpus: 1,
      confirm: false,
      idempotency_key: idempotency,
    });
    expect(preview.preview).toBe(true);
    await approve(preview.plan_id);
    const executed = await call(clientA, "create_instance", {
      name: `${marker}-instance`,
      gpu_model: "MCP E2E GPU",
      num_gpus: 1,
      confirm: true,
      plan_id: preview.plan_id,
      idempotency_key: idempotency,
    });
    expect(executed.ok).toBe(true);
    const replay = await call(clientA, "create_instance", {
      name: `${marker}-instance`,
      gpu_model: "MCP E2E GPU",
      num_gpus: 1,
      confirm: true,
      plan_id: preview.plan_id,
      idempotency_key: idempotency,
    });
    expect(replay.job.job_id).toBe(executed.job.job_id);
    expect(replay.idempotent).toBe(true);

    // 7. Place through the real scheduler and drive the fenced worker API.
    const worker = fixture("place", executed.job.job_id, "MCP E2E GPU");
    const workerHeaders = bearer(process.env.XCELSIOR_API_TOKEN || "test-token-not-for-production");
    const claimed = await jsonFetch("/agent/v2/commands/claim", {
      method: "POST",
      headers: workerHeaders,
      body: JSON.stringify({
        host_id: worker.host_id,
        worker_session_id: worker.worker_session_id,
      }),
    });
    expect(claimed.commands.some((item: any) => item.command_id === worker.command_id)).toBe(true);
    await jsonFetch("/agent/v2/leases/claim", {
      method: "POST",
      headers: workerHeaders,
      body: JSON.stringify({
        lease_id: worker.lease_id,
        job_id: worker.job_id,
        attempt_id: worker.attempt_id,
        host_id: worker.host_id,
        fencing_token: worker.fencing_token,
        worker_session_id: worker.worker_session_id,
      }),
    });
    await jsonFetch("/agent/v2/attempts/status", {
      method: "POST",
      headers: workerHeaders,
      body: JSON.stringify({
        job_id: worker.job_id,
        attempt_id: worker.attempt_id,
        host_id: worker.host_id,
        fencing_token: worker.fencing_token,
        status: "running",
      }),
    });

    // 8. Durable watch + persisted attempt timeline.
    const watched = await call(clientA, "watch_instance", {
      job_id: worker.job_id,
      duration_minutes: 1,
      poll_interval_seconds: 10,
      return_on_phase: ["running"],
    });
    expect(["running", "started"]).toContain(watched.final_status);
    const timeline = await call(clientA, "get_instance_timeline", { job_id: worker.job_id });
    expect(JSON.stringify(timeline)).toContain(worker.attempt_id);
    const traceChain = fixture("trace-chain", worker.job_id);
    expect(traceChain.ok, JSON.stringify(traceChain)).toBe(true);

    // 9. Endpoint creation uses approval; invocation itself stays low-friction.
    const endpointPlan = await call(clientA, "create_serverless_endpoint", {
      name: `${marker}-endpoint`,
      model_ref: "test/model",
      gpu_tier: "RTX 4090",
      min_workers: 0,
      max_workers: 1,
      confirm: false,
      idempotency_key: randomUUID(),
    });
    await approve(endpointPlan.plan_id);
    const endpoint = await call(clientA, "create_serverless_endpoint", {
      name: `${marker}-endpoint`,
      model_ref: "test/model",
      gpu_tier: "RTX 4090",
      min_workers: 0,
      max_workers: 1,
      confirm: true,
      plan_id: endpointPlan.plan_id,
      idempotency_key: randomUUID(),
    });
    const invocation = await call(clientA, "run_serverless_job", {
      endpoint_id: endpoint.endpoint_id ?? endpoint.endpoint?.endpoint_id,
      input: { prompt: "health check" },
    });
    expect(invocation.error).toBeUndefined();
    expect(JSON.stringify(invocation)).toMatch(/sjob-|IN_QUEUE|enqueued/i);
    expect(invocation.approval_required).not.toBe(true);

    // 10. Wrong scope and cross-tenant resource ids are both denied.
    const readOnlyToken = await createMachineToken(
      tenantA.browser, `${marker}-read-only`, ["instances:read"],
    );
    const readOnly = await sdkClient(readOnlyToken, "real-stack-read-only");
    const denied = await call(readOnly.client, "create_instance", {
      name: "must-not-launch", confirm: false,
    });
    expect(denied.code || denied.error).toBe("insufficient_scope");
    await readOnly.transport.close();
    const other = await sdkClient(tenantB.machine, "real-stack-b");
    const hidden = await call(other.client, "get_instance", { job_id: worker.job_id });
    expect(JSON.stringify(hidden)).toMatch(/not.?found|404/i);
    await other.transport.close();

    // 11. Expired quote produces a typed conflict and no workload.
    const expiring = await call(clientA, "create_instance", {
      name: `${marker}-expired`, gpu_model: "MCP E2E GPU", confirm: false,
    });
    fixture("expire", expiring.plan_id);
    await expect(approve(expiring.plan_id)).rejects.toMatchObject({ status: 409 });

    // 12. Real Redis enforces mutating calls fail-closed during outage.
    spawnSync("redis-cli", ["-u", redisUrl, "shutdown", "nosave"]);
    await expect(clientA.callTool({
      name: "create_instance",
      arguments: { name: `${marker}-redis-down`, confirm: false },
    })).rejects.toThrow(/unavailable|rate.limit/i);

    // Restart Redis for audit writes and the operator separation test.
    if (manageRedis) {
      redis = spawn("redis-server", [
        "--port", new URL(redisUrl).port || "6387",
        "--save", "", "--appendonly", "no", "--bind", "127.0.0.1",
      ], { stdio: "pipe" });
    } else {
      if (!externalRedisContainer) {
        throw new Error("external Redis restart requires XCELSIOR_MCP_E2E_REDIS_CONTAINER");
      }
      const restarted = spawnSync("docker", ["start", externalRedisContainer]);
      if (restarted.status !== 0) throw new Error("failed to restart external Redis container");
    }
    await new Promise((resolve) => setTimeout(resolve, 300));

    // 13. Drain leaves the running workload intact; eviction is separately
    // scoped, separately planned, human-approved, and then executed.
    const operatorToken = await createMachineToken(tenantA.browser, `${marker}-operator`, [
      "instances:read", "hosts:read", "hosts:operate", "hosts:evict", "control_plane:read",
    ]);
    const operator = await sdkClient(operatorToken, "real-stack-operator");
    const capacity = await call(operator.client, "get_host_capacity", { host_id: worker.host_id });
    const version = Number(capacity.version ?? capacity.host?.version ?? worker.host_version);
    const drained = await call(operator.client, "drain_host", {
      host_id: worker.host_id,
      reason: "MCP separation test",
      expected_version: version,
      idempotency_key: randomUUID(),
    });
    expect(drained.ok).toBe(true);
    const stillRunning = await call(clientA, "get_instance", { job_id: worker.job_id });
    expect(JSON.stringify(stillRunning)).toMatch(/running/);
    await call(operator.client, "get_host_capacity", {
      host_id: worker.host_id,
    });
    const currentHostVersion = fixture("host-version", worker.host_id).version;
    const eviction = await call(operator.client, "evict_host_workloads", {
      host_id: worker.host_id,
      reason: "MCP approved eviction test",
      confirm: false,
      expected_version: currentHostVersion,
      idempotency_key: randomUUID(),
    });
    expect(eviction.plan_id, JSON.stringify(eviction)).toBeTruthy();
    await approve(eviction.plan_id);
    const evicted = await call(operator.client, "evict_host_workloads", {
      host_id: worker.host_id,
      reason: "MCP approved eviction test",
      confirm: true,
      plan_id: eviction.plan_id,
      expected_version: currentHostVersion,
      idempotency_key: randomUUID(),
    });
    expect(evicted.ok).toBe(true);
    await operator.transport.close();

    // One replica may restart without interrupting the shared service.
    await transportA.close();
    mcp.kill("SIGTERM");
    const surviving = await call(clientBlue, "get_instance", { job_id: worker.job_id });
    expect(surviving.ok).toBe(true);
    mcp = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
      cwd: process.cwd(),
      env: {
        ...process.env,
        XCELSIOR_MCP_API_URL: apiUrl,
        XCELSIOR_MCP_RESOURCE_AUDIENCE: resource,
        XCELSIOR_MCP_PUBLIC_URL: `http://127.0.0.1:${mcpPort}/mcp`,
        XCELSIOR_MCP_REDIS_URL: redisUrl,
        MCP_RATE_LIMIT_BACKEND: "redis",
        MCP_RATE_LIMIT_REQUIRE_REDIS: "true",
        MCP_RATE_LIMIT_PER_MIN: "2000",
        MCP_HOST: "127.0.0.1",
        MCP_PORT: String(mcpPort),
        MCP_PATH: "/mcp",
        OTEL_EXPORTER_OTLP_ENDPOINT: "",
      },
      stdio: "pipe",
    });
    for (let attempt = 0; attempt < 200; attempt += 1) {
      try {
        if ((await fetch(`http://127.0.0.1:${mcpPort}/readyz`)).ok) break;
      } catch { /* restarting */ }
      if (mcp.exitCode !== null) throw new Error(`MCP restart failed (${mcp.exitCode})`);
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    const restarted = await sdkClient(tenantA.machine, "real-stack-restarted");
    const afterRestart = await call(restarted.client, "get_instance", { job_id: worker.job_id });
    expect(afterRestart.ok).toBe(true);
    await restarted.transport.close();
  }, 90_000);
});
