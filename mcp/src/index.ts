import "./observability/setup.js";
import http from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { loadConfig } from "./config.js";
import { createMcpServer } from "./server.js";
import { createApiClient, extractBearer, validateBearer } from "./auth/bearer.js";
import { acquireWatchSlot, checkRateLimit, checkToolLimit, rateLimitReady, recordAuthFailure } from "./rate-limit.js";
import { createHash } from "node:crypto";
import { activeTransports, authFailures, metricsRegistry, rateFailures } from "./observability/metrics.js";
import { log } from "./observability/logging.js";
import { TOOL_CONTRACTS } from "./tools/contracts.js";
import { TOOL_SCOPES } from "./auth/scopes.js";

const config = loadConfig();

function readBody(req: http.IncomingMessage): Promise<unknown> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      if (!raw) return resolve(undefined);
      try {
        resolve(JSON.parse(raw));
      } catch {
        resolve(undefined);
      }
    });
    req.on("error", reject);
  });
}

function json(res: http.ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

async function handleMcp(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  parsedBody?: unknown,
): Promise<void> {
  const bearer = extractBearer(req);
  if (!bearer) {
    authFailures.inc();
    json(res, 401, {
      error: "unauthorized",
      message: "Authorization: Bearer <xoa_token> required. Create an MCP client at Xcelsior dashboard settings.",
    });
    return;
  }

  const user = await validateBearer(config.apiUrl, bearer, config.resourceAudience);
  if (!user) {
    authFailures.inc();
    const abuse = await recordAuthFailure(createHash("sha256").update(bearer).digest("hex").slice(0, 24), config.rateLimit);
    json(res, abuse.ok ? 401 : abuse.status, {
      error: abuse.ok ? "invalid_token" : abuse.code,
      message: abuse.ok ? "Bearer token invalid or expired." : abuse.message,
    });
    return;
  }
  const principalKey = user.subject || user.user_id || user.email || user.customer_id || "";
  const rate = await checkRateLimit(`${principalKey}:${user.client_id || "no-client"}`, config.rateLimit);
  if (!rate.ok) {
    rateFailures.inc({ code: rate.code });
    json(res, rate.status, { error: rate.code, message: rate.message });
    return;
  }
  const toolName = extractToolName(parsedBody);
  if (toolName) {
    const toolRate = await checkToolLimit(principalKey, user.client_id || "", toolName, config.rateLimit);
    if (!toolRate.ok) {
      rateFailures.inc({ code: toolRate.code });
      json(res, toolRate.status, { error: toolRate.code, message: toolRate.message });
      return;
    }
  }
  const watchSlot = toolName === "watch_instance"
    ? await acquireWatchSlot(principalKey, config.rateLimit)
    : { decision: { ok: true } as const, release: async () => undefined };
  if (!watchSlot.decision.ok) {
    json(res, watchSlot.decision.status, { error: watchSlot.decision.code, message: watchSlot.decision.message });
    return;
  }

  const client = createApiClient(config.apiUrl, bearer);
  const mcp = createMcpServer(client, user);
  // Stateless mode: each request is self-contained (no Mcp-Session-Id round-trip),
  // so the client's `initialize` POST gets an immediate JSON response instead of
  // hanging on a session-scoped stream we tear down per request.
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: undefined,
  });
  activeTransports.inc({ transport: "streamable_http" });

  res.on("close", () => {
    activeTransports.dec({ transport: "streamable_http" });
    void transport.close();
    void mcp.close();
  });

  await mcp.connect(transport);
  try {
    await transport.handleRequest(req, res, parsedBody);
  } finally {
    await watchSlot.release();
  }
}

const httpServer = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const path = url.pathname;

  if (path === "/.well-known/oauth-protected-resource" || path === "/.well-known/oauth-protected-resource/mcp") {
    json(res, 200, {
      resource: config.resourceAudience,
      authorization_servers: [config.oauthIssuer],
      jwks_uri: config.oauthJwksUrl,
      bearer_methods_supported: ["header"],
      scopes_supported: [
        "instances:read", "instances:write", "instances:operate", "inference:read",
        "inference:write", "billing:read", "gpu:read", "hosts:read", "hosts:operate",
        "hosts:evict", "control_plane:read", "control_plane:operate", "mcp_actions:approve",
      ],
      resource_documentation: `${config.apiUrl}/docs/mcp`,
    });
    return;
  }

  if (path === "/health" || path === "/mcp/health" || path === `${config.mcpPath}/health`) {
    json(res, 200, {
      status: "healthy",
      service: "xcelsior-mcp",
      version: "2.0.0",
      api_url: config.apiUrl,
      protocol: "2025-11-25",
      resource_audience: config.resourceAudience,
    });
    return;
  }
  if (path === "/startupz") {
    const complete = Object.keys(TOOL_SCOPES).every((name) => TOOL_CONTRACTS[name]);
    const configured = Boolean(config.apiUrl && config.resourceAudience && config.oauthIssuer && config.oauthJwksUrl);
    json(res, complete && configured ? 200 : 503, {
      ok: complete && configured,
      tool_registry_complete: complete,
      tool_count: Object.keys(TOOL_CONTRACTS).length,
      configured,
    });
    return;
  }
  if (path === "/readyz") {
    const checks = { redis: false, authorization_server: false, jwks: false };
    checks.redis = await rateLimitReady(config.rateLimit);
    try {
      checks.authorization_server = (
        await fetch(`${config.oauthIssuer}/.well-known/oauth-authorization-server`, { signal: AbortSignal.timeout(3_000) })
      ).ok;
      checks.jwks = (await fetch(config.oauthJwksUrl, { signal: AbortSignal.timeout(3_000) })).ok;
    } catch { /* dependency remains false */ }
    const ok = Object.values(checks).every(Boolean);
    json(res, ok ? 200 : 503, { ok, checks });
    return;
  }
  if (path === "/metrics") {
    res.writeHead(200, { "Content-Type": metricsRegistry.contentType });
    res.end(await metricsRegistry.metrics());
    return;
  }

  const mcpPaths = [config.mcpPath, "/mcp"];
  if (!mcpPaths.some((p) => path === p || path.startsWith(`${p}/`))) {
    json(res, 404, { error: "not_found" });
    return;
  }

  if (req.method !== "GET" && req.method !== "POST" && req.method !== "DELETE") {
    json(res, 405, { error: "method_not_allowed" });
    return;
  }

  try {
    const body = req.method === "POST" ? await readBody(req) : undefined;
    await handleMcp(req, res, body);
  } catch (err) {
    log.error({ err }, "MCP request error");
    if (!res.headersSent) {
      json(res, 500, { error: "internal_error", message: String(err) });
    }
  }
});

function extractToolName(body: unknown): string | null {
  if (!body || typeof body !== "object") return null;
  const record = body as Record<string, unknown>;
  if (record.method !== "tools/call" || !record.params || typeof record.params !== "object") return null;
  const name = (record.params as Record<string, unknown>).name;
  return typeof name === "string" ? name : null;
}

httpServer.listen(config.port, config.host, () => {
  log.info({ host: config.host, port: config.port, path: config.mcpPath, api_url: config.apiUrl }, "MCP listening");
});
