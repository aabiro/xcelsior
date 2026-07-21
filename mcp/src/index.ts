import http from "node:http";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { loadConfig } from "./config.js";
import { createMcpServer } from "./server.js";
import { createApiClient, extractBearer, validateBearer } from "./auth/bearer.js";
import { checkRateLimit } from "./rate-limit.js";

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
    json(res, 401, {
      error: "unauthorized",
      message: "Authorization: Bearer <xoa_token> required. Create an MCP client at Xcelsior dashboard settings.",
    });
    return;
  }

  const rateKey = bearer.slice(0, 16);
  const rate = await checkRateLimit(rateKey, config.rateLimit);
  if (!rate.ok) {
    json(res, rate.status, { error: rate.code, message: rate.message });
    return;
  }

  const user = await validateBearer(config.apiUrl, bearer);
  if (!user) {
    json(res, 401, { error: "invalid_token", message: "Bearer token invalid or expired." });
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