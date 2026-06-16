import http from "node:http";
import { randomUUID } from "node:crypto";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { loadConfig } from "./config.js";
import { createMcpServer } from "./server.js";
import { createApiClient, extractBearer, validateBearer } from "./auth/bearer.js";

const config = loadConfig();

/** Per-client rate limit buckets */
const rateBuckets = new Map<string, { count: number; resetAt: number }>();

function checkRateLimit(key: string): boolean {
  const now = Date.now();
  const windowMs = 60_000;
  let bucket = rateBuckets.get(key);
  if (!bucket || now >= bucket.resetAt) {
    bucket = { count: 0, resetAt: now + windowMs };
    rateBuckets.set(key, bucket);
  }
  bucket.count += 1;
  return bucket.count <= config.rateLimitPerMinute;
}

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
    json(res, 401, {
      error: "unauthorized",
      message: "Authorization: Bearer <oauth_token> required. Create an MCP client at Xcelsior dashboard settings.",
    });
    return;
  }

  const rateKey = bearer.slice(0, 16);
  if (!checkRateLimit(rateKey)) {
    json(res, 429, { error: "rate_limit_exceeded", message: "Too many MCP requests; retry in 60s." });
    return;
  }

  const user = await validateBearer(config.apiUrl, bearer);
  if (!user) {
    json(res, 401, { error: "invalid_token", message: "Bearer token invalid or expired." });
    return;
  }

  const client = createApiClient(config.apiUrl, bearer);
  const mcp = createMcpServer(client, user);
  const transport = new StreamableHTTPServerTransport({
    sessionIdGenerator: () => randomUUID(),
  });

  res.on("close", () => {
    void transport.close();
    void mcp.close();
  });

  await mcp.connect(transport);
  await transport.handleRequest(req, res, parsedBody);
}

const httpServer = http.createServer(async (req, res) => {
  const url = new URL(req.url || "/", `http://${req.headers.host || "localhost"}`);
  const path = url.pathname;

  if (path === "/health" || path === "/mcp/health" || path === `${config.mcpPath}/health`) {
    json(res, 200, {
      status: "healthy",
      service: "xcelsior-mcp",
      version: "0.2.0",
      api_url: config.apiUrl,
    });
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
    console.error("MCP request error:", err);
    if (!res.headersSent) {
      json(res, 500, { error: "internal_error", message: String(err) });
    }
  }
});

httpServer.listen(config.port, config.host, () => {
  console.log(`Xcelsior MCP listening on http://${config.host}:${config.port}${config.mcpPath}`);
  console.log(`Upstream API: ${config.apiUrl}`);
});