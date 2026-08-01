import "./observability/setup.js";
import http from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { acceptedAudiences, loadConfig } from "./config.js";
import { createMcpServer } from "./server.js";
import { createApiClient, extractBearer, validateBearer } from "./auth/bearer.js";
import { buildWwwAuthenticate } from "./auth/challenge.js";
import { acquireWatchSlot, checkRateLimit, checkToolLimit, rateLimitReady, recordAuthFailure } from "./rate-limit.js";
import { createHash } from "node:crypto";
import { activeTransports, authFailures, metricsRegistry, rateFailures } from "./observability/metrics.js";
import { log } from "./observability/logging.js";
import { TOOL_CONTRACTS } from "./tools/contracts.js";
import { TOOL_SCOPES } from "./auth/scopes.js";
import { scopesForProfile, toolsInProfile } from "./tools/profiles.js";

const config = loadConfig();
const PROFILE_OPTIONS = { companyKnowledge: config.companyKnowledge };
const PROFILE_SCOPES = scopesForProfile(config.toolProfile, PROFILE_OPTIONS);
const PROFILE_TOOLS = toolsInProfile(config.toolProfile, PROFILE_OPTIONS);

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

function json(res: http.ServerResponse, status: number, body: unknown, headers?: http.OutgoingHttpHeaders): void {
  res.writeHead(status, { "Content-Type": "application/json", ...headers });
  res.end(JSON.stringify(body));
}

/**
 * 401 with the RFC 9728 challenge. Both unauthenticated branches go through
 * here so neither can ever answer without telling the client where to
 * authenticate — the failure the adoption plan calls BLOCKER 1.
 */
function unauthorized(
  res: http.ServerResponse,
  body: { error: string; message: string },
  challenge?: { error: "invalid_token"; description: string },
): void {
  json(res, 401, body, {
    "WWW-Authenticate": buildWwwAuthenticate({
      realm: config.authRealm,
      resourceMetadataUrl: config.resourceMetadataUrl,
      error: challenge?.error,
      errorDescription: challenge?.description,
    }),
  });
}

async function handleMcp(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  parsedBody?: unknown,
): Promise<void> {
  const bearer = extractBearer(req);
  if (!bearer) {
    authFailures.inc();
    unauthorized(res, {
      error: "unauthorized",
      message:
        `Authorization required. Discover how to authenticate at ${config.resourceMetadataUrl}, ` +
        "or use an Xcelsior agent key from dashboard settings for automation.",
    });
    return;
  }

  const user = await validateBearer(config.apiUrl, bearer, acceptedAudiences(config));
  if (!user) {
    authFailures.inc();
    const abuse = await recordAuthFailure(createHash("sha256").update(bearer).digest("hex").slice(0, 24), config.rateLimit);
    if (abuse.ok) {
      unauthorized(
        res,
        { error: "invalid_token", message: "Bearer token invalid, expired, or bound to another resource." },
        { error: "invalid_token", description: "Bearer token invalid, expired, or bound to another resource." },
      );
    } else {
      // Past the abuse threshold this is a 429, not a 401 — a challenge header
      // on a rate-limit response would just invite a retry loop.
      json(res, abuse.status, { error: abuse.code, message: abuse.message });
    }
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
  const mcp = createMcpServer(client, user, "streamable_http", config.toolProfile, {
    companyKnowledge: config.companyKnowledge
      ? { siteUrl: config.siteUrl, docsUrl: config.docsUrl }
      : false,
  });
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
    // RFC 9728. `resource` is the exact identifier tokens must be bound to and
    // the value a connector echoes back as the `resource` parameter, so it has
    // to be the connector URL a user pastes — path and all — not the origin.
    json(res, 200, {
      resource: config.resourceAudience,
      authorization_servers: [config.oauthIssuer],
      jwks_uri: config.oauthJwksUrl,
      bearer_methods_supported: ["header"],
      // The profile's real scope set, derived from the tool registry, so a new
      // tool cannot advertise a scope the resource never mentions.
      scopes_supported: PROFILE_SCOPES,
      resource_name: "Xcelsior GPU Cloud",
      resource_documentation: `${config.apiUrl}/docs/mcp`,
      resource_policy_uri: `${config.apiUrl}/privacy`,
      resource_tos_uri: `${config.apiUrl}/terms`,
    });
    return;
  }

  if (path === "/.well-known/openai-apps-challenge") {
    // OpenAI's domain-verification probe. The body must be the bare token the
    // submission portal issued — no JSON envelope, no list, no trailing
    // newline. Configuration-backed on purpose: the token does not exist until
    // the portal issues it, and a hardcoded one would either be a lie now or
    // stale later. Until it is set, 404 is the honest answer.
    if (!config.openaiAppsChallenge) {
      json(res, 404, {
        error: "not_configured",
        message:
          "No OpenAI apps challenge token is configured. Set " +
          "XCELSIOR_MCP_OPENAI_APPS_CHALLENGE to the token the submission portal issued.",
      });
      return;
    }
    res.writeHead(200, {
      "Content-Type": "text/plain; charset=utf-8",
      "Cache-Control": "no-store",
    });
    res.end(config.openaiAppsChallenge);
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
      tool_profile: config.toolProfile,
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
      tool_profile: config.toolProfile,
      profile_tool_count: PROFILE_TOOLS.length,
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
