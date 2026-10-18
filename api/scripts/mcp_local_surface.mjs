// Boot a local MCP server against a stub API so `tools/list` can be graded
// without a production credential.
//
// The eval only calls `initialize` + `tools/list` and then asks a model to pick
// tools from those schemas — it never executes one. So the surface a local
// server publishes is byte-identical to what a deployed one would publish from
// the same commit, and grading it is a real measurement of the descriptions.
//
// What it is NOT: a measurement of the *deployed* surface. Production is behind
// repo head, so this grades what the repo would publish, which is the thing that
// changed and the thing a fresh baseline is owed for.
import http from "node:http";
import { spawn } from "node:child_process";

const apiPort = 39411;
const mcpPort = 39412;

const api = http.createServer(async (req, res) => {
  const send = (status, value) => {
    res.writeHead(status, { "content-type": "application/json" });
    res.end(JSON.stringify(value));
  };
  if (req.url === "/api/auth/introspect") {
    if (req.headers.authorization !== "Bearer local-eval-token") {
      return send(401, { detail: "Invalid or expired token" });
    }
    // Quick Connect's scope set — the credential a pasted connector actually
    // holds. Registration is not scope-filtered, so this does not change the
    // published list; it is realistic rather than load-bearing.
    return send(200, {
      ok: true,
      user_id: "user-eval",
      customer_id: "tenant-eval",
      workspace_id: "tenant-eval",
      client_id: "client-eval",
      subject: "client-eval",
      audience: "https://mcp.local",
      scopes: [
        "instances:read", "instances:write", "instances:operate", "instances:connect",
        "gpu:read", "billing:read", "marketplace:read", "events:read",
        "inference:read", "inference:write", "artifacts:read",
        "volumes:read", "volumes:write", "ssh:write",
      ],
      auth_type: "oauth_access_token",
      grant_type: "authorization_code",
    });
  }
  return send(404, { status: 404, code: "not_found", detail: req.url });
});

await new Promise((r) => api.listen(apiPort, "127.0.0.1", r));

const mcp = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
  cwd: process.env.MCP_DIR || new URL("../mcp/", import.meta.url).pathname,
  env: {
    ...process.env,
    XCELSIOR_MCP_API_URL: `http://127.0.0.1:${apiPort}`,
    XCELSIOR_MCP_RESOURCE_AUDIENCE: "https://mcp.local",
    XCELSIOR_MCP_PUBLIC_URL: `http://127.0.0.1:${mcpPort}/mcp`,
    MCP_HOST: "127.0.0.1",
    MCP_PORT: String(mcpPort),
    MCP_PATH: "/mcp",
    MCP_RATE_LIMIT_BACKEND: "memory",
    MCP_RATE_LIMIT_PER_MIN: "100000",
    OTEL_EXPORTER_OTLP_ENDPOINT: "",
  },
  stdio: ["ignore", "pipe", "pipe"],
});
mcp.stderr.on("data", (d) => process.stderr.write(`[mcp] ${d}`));

// Poll until it answers, rather than sleeping a fixed interval and connecting
// regardless — that pattern is what let a hosted e2e "pass" against a server
// that had not started.
const deadline = Date.now() + 60_000;
let ready = false;
while (Date.now() < deadline) {
  try {
    const r = await fetch(`http://127.0.0.1:${mcpPort}/mcp`, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json, text/event-stream",
        authorization: "Bearer local-eval-token",
      },
      body: JSON.stringify({ jsonrpc: "2.0", id: 1, method: "initialize", params: {
        protocolVersion: "2025-06-18", capabilities: {},
        clientInfo: { name: "readiness", version: "0" },
      } }),
    });
    if (r.status === 200) { ready = true; break; }
  } catch { /* not up yet */ }
  await new Promise((r) => setTimeout(r, 500));
}

if (!ready) {
  console.error("FAILED: local MCP server never answered 200");
  mcp.kill("SIGKILL");
  process.exit(1);
}

console.log(`READY http://127.0.0.1:${mcpPort}/mcp`);
process.on("SIGTERM", () => { mcp.kill("SIGKILL"); api.close(); process.exit(0); });
process.on("SIGINT", () => { mcp.kill("SIGKILL"); api.close(); process.exit(0); });
// Stay alive so the eval can run against it.
await new Promise(() => {});
