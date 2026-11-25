/**
 * Helpers for the live rehearsal: a real API (started by
 * scripts/rehearse-live.sh), the MCP server from source, and a connector token
 * obtained through DCR + PKCE + the consent screen.
 */
import { spawn, spawnSync, type ChildProcess } from "node:child_process";
import { createHash, randomBytes, randomUUID } from "node:crypto";

export const apiUrl = (process.env.XCELSIOR_MCP_E2E_API_URL || "http://127.0.0.1:18980").replace(/\/$/, "");
export const redisUrl = process.env.XCELSIOR_MCP_E2E_REDIS_URL || "redis://127.0.0.1:6387/0";
/** The canonical RFC 8707 identifier: the connector URL, path included. */
export const resource = "https://mcp.xcelsior.ca/mcp";

export type Session = { browser: string; cookies: string; customer: string };

export async function jsonFetch(path: string, init: RequestInit = {}): Promise<any> {
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

export function bearer(token: string, extra: Record<string, string> = {}): Record<string, string> {
  return { Authorization: `Bearer ${token}`, "Content-Type": "application/json", ...extra };
}

function cookieHeader(response: Response): string {
  return response.headers
    .getSetCookie()
    .map((c) => c.split(";")[0])
    .join("; ");
}

/** A fresh password account, signed in, as the browser would be. */
export async function createSession(marker: string, label: string): Promise<Session> {
  const email = `${marker}-${label}@example.test`;
  const password = "McpRehearsal-StrongPass-2026!";
  const response = await fetch(`${apiUrl}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name: "Rehearsal Tenant" }),
  });
  const registered = await response.json();
  if (!response.ok) throw new Error(`register ${response.status}: ${JSON.stringify(registered)}`);
  const browser = registered.access_token as string;
  const principal = await jsonFetch("/api/auth/introspect", { headers: bearer(browser) });
  return { browser, cookies: cookieHeader(response), customer: principal.customer_id };
}

export async function approve(planId: string, browser: string): Promise<void> {
  const current = await jsonFetch(`/api/v1/launch-plans/${encodeURIComponent(planId)}`, {
    headers: bearer(browser),
  });
  await jsonFetch(`/api/v1/launch-plans/${encodeURIComponent(planId)}/approve`, {
    method: "POST",
    headers: bearer(browser, { "Idempotency-Key": randomUUID() }),
    body: JSON.stringify({ expected_version: current.plan.version, confirmation: "approve" }),
  });
}

export function fixture(...args: string[]): any {
  const python = process.env.XCELSIOR_MCP_E2E_PYTHON || ".venv/bin/python";
  const result = spawnSync(python, ["scripts/mcp_e2e_fixture.py", ...args], {
    cwd: new URL("../", `file://${process.cwd()}/`).pathname,
    env: { ...process.env, MCP_RATE_LIMIT_BACKEND: "redis", XCELSIOR_MCP_REDIS_URL: redisUrl },
    encoding: "utf8",
  });
  if (result.status !== 0) {
    throw new Error(`fixture ${args.join(" ")} failed:\n${result.stdout}\n${result.stderr}`);
  }
  return JSON.parse(result.stdout.trim().split("\n").at(-1)!);
}

let mcp: ChildProcess | undefined;

export async function startMcp(port: number): Promise<void> {
  mcp = spawn(process.execPath, ["--import", "tsx", "src/index.ts"], {
    cwd: process.cwd(),
    env: {
      ...process.env,
      XCELSIOR_MCP_API_URL: apiUrl,
      XCELSIOR_MCP_RESOURCE_AUDIENCE: resource,
      XCELSIOR_MCP_PUBLIC_URL: `http://127.0.0.1:${port}/mcp`,
      XCELSIOR_MCP_REDIS_URL: redisUrl,
      MCP_RATE_LIMIT_BACKEND: "redis",
      MCP_RATE_LIMIT_REQUIRE_REDIS: "true",
      MCP_RATE_LIMIT_PER_MIN: "2000",
      MCP_HOST: "127.0.0.1",
      MCP_PORT: String(port),
      MCP_PATH: "/mcp",
      OTEL_EXPORTER_OTLP_ENDPOINT: "",
    },
    stdio: ["ignore", "inherit", "inherit"],
  });
  for (let attempt = 0; attempt < 200; attempt += 1) {
    try {
      if ((await fetch(`http://127.0.0.1:${port}/readyz`)).ok) return;
    } catch {
      /* booting */
    }
    if (mcp.exitCode !== null) throw new Error(`MCP exited during startup (${mcp.exitCode})`);
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  throw new Error("MCP did not become ready");
}

export async function stopMcp(): Promise<void> {
  const proc = mcp;
  if (!proc || proc.exitCode !== null || proc.signalCode !== null) return;
  await new Promise<void>((resolve) => {
    const escalate = setTimeout(() => proc.kill("SIGKILL"), 5_000);
    proc.once("exit", () => {
      clearTimeout(escalate);
      resolve();
    });
    proc.kill("SIGTERM");
  });
}

/** Same origin as the API, whatever issuer the metadata names. */
function local(url: string): string {
  const parsed = new URL(url);
  return `${apiUrl}${parsed.pathname}${parsed.search}`;
}

/**
 * The token a directory connector gets: register with DCR, ask for every
 * scope the authorization server advertises, approve the consent screen
 * signed in as the tenant, and exchange the code with PKCE for the MCP
 * resource. The server, not this code, decides what is granted. Pass `scopes`
 * to ask for less, as a user who unticks permissions would end up with.
 */
export async function connectorToken(
  tenant: Session,
  _mcpUrl: string,
  /** Ask for these instead of everything advertised: a narrower connection. */
  scopes?: string[],
): Promise<string> {
  const metadata = await jsonFetch("/.well-known/oauth-authorization-server");
  const redirectUri = "http://127.0.0.1:8765/cb";
  const scope = (scopes ?? (metadata.scopes_supported as string[])).join(" ");
  const registered = await jsonFetch(new URL(local(metadata.registration_endpoint)).pathname, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      client_name: "Live rehearsal connector",
      redirect_uris: [redirectUri],
      grant_types: ["authorization_code", "refresh_token"],
      response_types: ["code"],
      token_endpoint_auth_method: "none",
      scope,
    }),
  });
  const clientId = registered.client_id as string;

  const verifier = randomBytes(32).toString("base64url");
  const challenge = createHash("sha256").update(verifier).digest("base64url");
  const state = randomBytes(12).toString("base64url");
  const authorizeUrl = new URL(local(metadata.authorization_endpoint));
  for (const [key, value] of Object.entries({
    response_type: "code",
    client_id: clientId,
    redirect_uri: redirectUri,
    state,
    code_challenge: challenge,
    code_challenge_method: "S256",
    scope,
    resource,
  })) {
    authorizeUrl.searchParams.set(key, value);
  }
  const signedIn = { Cookie: tenant.cookies, Authorization: `Bearer ${tenant.browser}` };
  let authorize = await fetch(authorizeUrl, {
    headers: { ...signedIn, Accept: "text/html" },
    redirect: "manual",
  });
  if (authorize.status === 200) {
    const html = await authorize.text();
    const key = html.match(/name="consent_key"\s+value="([^"]+)"/);
    if (!key) throw new Error(`no consent form: ${html.slice(0, 400)}`);
    authorize = await fetch(local(metadata.authorization_endpoint), {
      method: "POST",
      headers: { ...signedIn, "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ consent_key: key[1], decision: "approve" }),
      redirect: "manual",
    });
  }
  const location = authorize.headers.get("location") || "";
  const callback = new URL(location, redirectUri);
  if (![302, 303].includes(authorize.status) || callback.searchParams.get("state") !== state) {
    throw new Error(`authorize answered ${authorize.status} ${location} ${await authorize.text()}`);
  }
  const code = callback.searchParams.get("code");
  if (!code) throw new Error(`no code in ${location}`);
  const token = await fetch(local(metadata.token_endpoint), {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      client_id: clientId,
      code,
      redirect_uri: redirectUri,
      code_verifier: verifier,
      resource,
    }),
  });
  const bundle = await token.json();
  if (!token.ok) throw new Error(`token ${token.status}: ${JSON.stringify(bundle)}`);
  console.log(`[rehearsal] connector granted: ${bundle.scope}`);
  return bundle.access_token as string;
}
