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
    if (req.url === "/api/auth/introspect") {
      // Only the one token the suite issues. A mock that accepts every bearer
      // would make the invalid-token challenge untestable.
      if (req.headers.authorization !== "Bearer e2e-token") {
        return send(401, { detail: "Invalid or expired token" });
      }
      return send(200, {
      ok: true, user_id: "user-e2e", customer_id: "tenant-a", workspace_id: "tenant-a",
      client_id: "client-e2e", subject: "client-e2e", audience: "https://mcp.test",
      scopes: ["instances:read", "instances:write", "instances:operate", "gpu:read", "billing:read", "api"],
      auth_type: "client_credentials", grant_type: "client_credentials",
      });
    }
    if (req.url === "/api/v1/mcp/tool-audit") {
      auditRows.push(body); return send(202, { ok: true, audit_id: "audit-e2e" });
    }
    if (req.url === "/instance/job-watch") return send(200, { instance: { job_id: "job-watch", status: instanceStatus } });
    if (req.url?.startsWith("/api/instances/job-watch/telemetry")) return send(200, { telemetry: {} });
    if (req.url?.startsWith("/instances/job-watch/logs")) return send(200, { logs: [] });
    if (req.url?.startsWith("/api/v1/instances/job-watch/events")) return send(200, { events: [], next_cursor: null });
    if (req.url === "/instances/job-watch/cancel") { cancelCalls += 1; instanceStatus = "cancelled"; return send(200, { ok: true }); }
    // A deliberately leaky upstream: the response-hygiene filter is the thing
    // under test, so the fixture has to actually leak.
    if (req.url === "/instances/job-leaky/logs" || req.url?.startsWith("/instances/job-leaky/logs")) {
      return send(200, {
        logs: ["starting", "connecting with xoa_LeakedTokenValue0123456789"],
        registry_password: "hunter2",
        traceback: 'File "db.py", line 1',
        _internal_row: 42,
      });
    }
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
      XCELSIOR_MCP_OPENAI_APPS_CHALLENGE: "  openai-apps-challenge-token-e2e\n",
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
  it("publishes OAuth protected-resource metadata at both RFC 9728 locations", async () => {
    for (const path of [
      "/.well-known/oauth-protected-resource",
      "/.well-known/oauth-protected-resource/mcp",
    ]) {
      const response = await fetch(`http://127.0.0.1:${mcpPort}${path}`);
      expect(response.status, path).toBe(200);
      const document = await response.json();
      expect(document, path).toMatchObject({ resource: "https://mcp.test" });
      // Only the profile's own scopes; a public listing must never advertise
      // the operator surface it does not expose.
      expect(document.scopes_supported, path).not.toContain("hosts:evict");
      expect(document.scopes_supported, path).toContain("instances:read");
    }
  });

  it("serves the OpenAI domain-verification token and nothing else", async () => {
    // BLOCKER 4: the portal expects exactly the token — not JSON, not a list,
    // and not with the stray whitespace a secrets manager tends to append.
    const response = await fetch(`http://127.0.0.1:${mcpPort}/.well-known/openai-apps-challenge`);
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/plain");
    expect(await response.text()).toBe("openai-apps-challenge-token-e2e");
  });

  it("challenges an unauthenticated initialize with the resource metadata URL", async () => {
    // BLOCKER 1: this is the only breadcrumb a connector has. Without it the
    // client reports "couldn't connect" while our logs show a normal 401.
    const response = await fetch(`http://127.0.0.1:${mcpPort}/mcp`, {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json, text/event-stream" },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} }),
    });
    expect(response.status).toBe(401);
    const challenge = response.headers.get("www-authenticate") ?? "";
    expect(challenge).toMatch(/^Bearer /);
    expect(challenge).toContain('realm="xcelsior"');
    expect(challenge).toContain("/.well-known/oauth-protected-resource");
    // Following the header must actually land on the document we serve.
    const metadataUrl = /resource_metadata="([^"]+)"/.exec(challenge)?.[1] ?? "";
    expect(metadataUrl).toBeTruthy();
    const metadata = await fetch(
      metadataUrl.replace("https://mcp.test", `http://127.0.0.1:${mcpPort}`),
    );
    expect(metadata.status).toBe(200);
  });

  it("challenges an invalid token with error=invalid_token", async () => {
    const response = await fetch(`http://127.0.0.1:${mcpPort}/mcp`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        authorization: "Bearer not-a-real-token",
      },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: {} }),
    });
    expect(response.status).toBe(401);
    const challenge = response.headers.get("www-authenticate") ?? "";
    expect(challenge).toContain('error="invalid_token"');
    expect(challenge).toContain("resource_metadata=");
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

  it("enumerates no operator tools for a customer-profile credential", async () => {
    // GX1: a public directory credential must never see the platform-global
    // surface. Checked on the wire, not in the registry, because tools/list is
    // what a provider snapshots and shows to every end user.
    const names = new Set((await client.listTools()).tools.map((tool) => tool.name));
    for (const operatorTool of [
      "drain_host", "undrain_host", "evict_host_workloads", "retry_agent_command",
      "get_scheduler_health", "get_host_capacity", "list_reconciliation_findings",
    ]) {
      expect(names.has(operatorTool), `${operatorTool} leaked into the customer profile`)
        .toBe(false);
    }
    expect(names.has("get_mcp_action_status")).toBe(true);
    expect(names.has("list_instances")).toBe(true);
  });

  it("refuses to call an out-of-profile tool by name", async () => {
    // Filtering only the listing would leave the tool callable by anyone who
    // guessed its name, so registration is skipped entirely and the SDK
    // answers "not found" rather than dispatching.
    const result = await client.callTool({
      name: "drain_host",
      arguments: { host_id: "h1", reason: "x" },
    });
    expect(result.isError).toBe(true);
    expect(JSON.stringify(result.content)).toContain("not found");
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

  it("never hands auth material or debug payloads back to the model", async () => {
    // GX1's response-hygiene scanner, asserted on the wire. Both halves of the
    // result are checked: most clients render `content`, not structuredContent,
    // so scrubbing only the structure would leak anyway.
    const result = await client.callTool({
      name: "get_instance_logs",
      arguments: { job_id: "job-leaky", limit: 10 },
    });
    const wire = JSON.stringify(result);
    expect(wire).not.toContain("xoa_LeakedTokenValue");
    expect(wire).not.toContain("hunter2");
    expect(wire).not.toContain("traceback");
    expect(wire).not.toContain("_internal_row");
    // The legitimate payload survives.
    expect(wire).toContain("starting");
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
