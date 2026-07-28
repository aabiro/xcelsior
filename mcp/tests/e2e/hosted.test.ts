import http from "node:http";
import { spawn, type ChildProcess } from "node:child_process";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";

const apiPort = 18869;
const mcpPort = 18870;
let api: http.Server;
let mcp: ChildProcess;
let client: Client;
let transport: StreamableHTTPClientTransport;
const auditRows: unknown[] = [];
let instanceStatus = "running";
let cancelCalls = 0;

beforeAll(async () => {
  api = http.createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const chunk of req) chunks.push(Buffer.from(chunk));
    const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : {};
    const send = (status: number, value: unknown) => {
      res.writeHead(status, { "content-type": "application/json" });
      res.end(JSON.stringify(value));
    };
    if (req.url === "/api/auth/introspect") return send(200, {
      ok: true, user_id: "user-e2e", customer_id: "tenant-a", workspace_id: "tenant-a",
      client_id: "client-e2e", subject: "client-e2e", audience: "https://mcp.test",
      scopes: ["instances:read", "instances:write", "instances:operate", "gpu:read", "billing:read", "api"],
      auth_type: "client_credentials", grant_type: "client_credentials",
    });
    if (req.url === "/api/v1/mcp/tool-audit") {
      auditRows.push(body); return send(202, { ok: true, audit_id: "audit-e2e" });
    }
    if (req.url === "/instance/job-watch") return send(200, { instance: { job_id: "job-watch", status: instanceStatus } });
    if (req.url?.startsWith("/api/instances/job-watch/telemetry")) return send(200, { telemetry: {} });
    if (req.url?.startsWith("/instances/job-watch/logs")) return send(200, { logs: [] });
    if (req.url?.startsWith("/api/v1/instances/job-watch/events")) return send(200, { events: [], next_cursor: null });
    if (req.url === "/instances/job-watch/cancel") { cancelCalls += 1; instanceStatus = "cancelled"; return send(200, { ok: true }); }
    if (req.url === "/api/v1/launch-plans") return send(200, {
      ok: true, preview: true, plan_id: "11111111-1111-4111-8111-111111111111",
      approval_state: "awaiting_approval", canonical_spec: body,
      estimate: { currency: "CAD", estimate_micros: 1000 },
      availability: { feasible: true }, approval_url: "https://example.test/approve",
      expires_at: "2030-01-01T00:00:00Z",
    });
    return send(404, { type: "about:blank", status: 404, code: "not_found", detail: req.url });
  });
  await new Promise<void>((resolve) => api.listen(apiPort, "127.0.0.1", resolve));

  mcp = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      XCELSIOR_MCP_API_URL: `http://127.0.0.1:${apiPort}`,
      XCELSIOR_MCP_RESOURCE_AUDIENCE: "https://mcp.test",
      XCELSIOR_MCP_PUBLIC_URL: `http://127.0.0.1:${mcpPort}/mcp`,
      MCP_HOST: "127.0.0.1", MCP_PORT: String(mcpPort), MCP_PATH: "/mcp",
      MCP_RATE_LIMIT_BACKEND: "memory", MCP_RATE_LIMIT_PER_MIN: "1000",
      OTEL_EXPORTER_OTLP_ENDPOINT: "",
    },
    stdio: "pipe",
  });
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      if ((await fetch(`http://127.0.0.1:${mcpPort}/health`)).ok) break;
    } catch { /* startup */ }
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  client = new Client({ name: "xcelsior-hosted-e2e", version: "1.0.0" });
  transport = new StreamableHTTPClientTransport(new URL(`http://127.0.0.1:${mcpPort}/mcp`), {
    requestInit: { headers: { Authorization: "Bearer e2e-token" } },
  });
  await client.connect(transport);
}, 20_000);

afterAll(async () => {
  await transport?.close();
  mcp?.kill("SIGTERM");
  await new Promise<void>((resolve) => api?.close(() => resolve()));
});

describe("hosted Streamable HTTP MCP", () => {
  it("publishes OAuth protected-resource metadata", async () => {
    const response = await fetch(`http://127.0.0.1:${mcpPort}/.well-known/oauth-protected-resource`);
    expect(response.status).toBe(200);
    expect(await response.json()).toMatchObject({ resource: "https://mcp.test" });
  });

  it("initializes and exposes complete structured tool contracts", async () => {
    const listed = await client.listTools();
    expect(listed.tools.length).toBeGreaterThan(25);
    for (const tool of listed.tools) {
      expect(tool.inputSchema, tool.name).toBeTruthy();
      expect(tool.outputSchema, tool.name).toBeTruthy();
      expect(tool.annotations, tool.name).toBeTruthy();
    }
  });

  it("previews create_instance through the API and writes a redacted audit", async () => {
    const result = await client.callTool({
      name: "create_instance",
      arguments: {
        name: "hosted-e2e", gpu_model: "RTX 4090", confirm: false,
        init_script: "secret-do-not-audit",
      },
    });
    expect(result.structuredContent).toMatchObject({
      preview: true, plan_id: "11111111-1111-4111-8111-111111111111",
    });
    await new Promise((resolve) => setTimeout(resolve, 50));
    expect(auditRows).toHaveLength(1);
    expect(JSON.stringify(auditRows[0])).not.toContain("secret-do-not-audit");
    expect(auditRows[0]).toMatchObject({
      tool_name: "create_instance",
      tool_version: "2.0.0",
      tenant_id: "tenant-a",
      api_route: "/api/v1/launch-plans",
      api_status: 200,
    });
  });

  it("cancels a watch without cancelling the GPU instance", async () => {
    const controller = new AbortController();
    const pending = client.callTool({
      name: "watch_instance",
      arguments: {
        job_id: "job-watch", duration_minutes: 1,
        poll_interval_seconds: 10, return_on_phase: ["completed"],
      },
    }, undefined, { signal: controller.signal, timeout: 15_000 });
    setTimeout(() => controller.abort(), 150);
    await expect(pending).rejects.toThrow();
    await new Promise((resolve) => setTimeout(resolve, 100));
    expect(instanceStatus).toBe("running");
    expect(cancelCalls).toBe(0);
  });
});
