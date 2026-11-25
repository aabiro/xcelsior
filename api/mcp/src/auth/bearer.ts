import type { IncomingMessage } from "node:http";
import type { XcelsiorApiClient } from "../client/api.js";
import { XcelsiorApiClient as Client } from "../client/api.js";
import { ApiError } from "../client/errors.js";

/**
 * The token could not be checked: the API throttled the check, failed, or did
 * not answer. That says nothing about the token, so it must not be reported
 * as invalid — a client told "invalid_token" throws a good token away and
 * sends the user back through sign-in.
 */
export class TokenCheckUnavailable extends Error {
  constructor(
    readonly reason: string,
    readonly retryAfterSec = 5,
  ) {
    super(`token check unavailable: ${reason}`);
    this.name = "TokenCheckUnavailable";
  }
}

/**
 * X-Forwarded-For for calls this server makes on a caller's behalf: the chain
 * it received with its own peer appended, the way nginx's
 * $proxy_add_x_forwarded_for does. The API resolves it with the same trusted
 * proxy rules it applies to traffic from nginx, so entries a caller made up
 * are skipped and the address nginx saw is used.
 */
export function forwardedFor(req: IncomingMessage): string | undefined {
  const received = req.headers["x-forwarded-for"];
  const chain = (Array.isArray(received) ? received.join(",") : received ?? "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
  const peer = (req.socket?.remoteAddress ?? "").replace(/^::ffff:/, "");
  if (peer) chain.push(peer);
  return chain.length ? chain.join(", ") : undefined;
}

export interface AuthUser {
  email?: string;
  customer_id?: string;
  user_id?: string;
  scopes?: string[];
  auth_type?: string;
  workspace_id?: string;
  team_id?: string;
  client_id?: string;
  subject?: string;
  audience?: string | string[];
  jti?: string;
  grant_type?: string;
  platform_admin?: boolean;
}

export function extractBearer(req: IncomingMessage): string | null {
  const auth = req.headers.authorization;
  if (!auth?.startsWith("Bearer ")) return null;
  const token = auth.slice(7).trim();
  return token || null;
}

/**
 * The principal a token stands for, or null when the API says it stands for
 * none. Throws TokenCheckUnavailable when the API could not say.
 */
export async function validateBearer(
  apiUrl: string,
  bearer: string,
  expectedAudiences?: string[],
  forwarded?: string,
): Promise<AuthUser | null> {
  let principal: (AuthUser & { ok?: boolean }) | null;
  try {
    const client = new Client({ baseUrl: apiUrl, bearer, forwardedFor: forwarded });
    // /api/auth/introspect accepts machine (client_credentials) tokens, which the
    // MCP gateway uses; /api/auth/me rejects them with 403.
    principal = await client.get<AuthUser & { ok?: boolean }>("/api/auth/introspect");
  } catch (error) {
    if (error instanceof ApiError && error.status !== 429 && error.status < 500) return null;
    const status = error instanceof ApiError ? `HTTP ${error.status}` : String(error);
    const retryAfter =
      error instanceof ApiError && error.problem?.retry_after_ms
        ? Math.max(1, Math.ceil(error.problem.retry_after_ms / 1000))
        : 5;
    throw new TokenCheckUnavailable(status, retryAfter);
  }
  if (!principal || (principal as { ok?: boolean }).ok === false) return null;
  const tenant = principal.workspace_id || principal.customer_id;
  if (!tenant) return null;
  if (expectedAudiences?.length) {
    const audiences = Array.isArray(principal.audience) ? principal.audience : [principal.audience];
    // RFC 8707 audience binding stays a hard gate through the resource-identifier
    // migration: the accepted set is narrow and dated, never "any audience".
    if (!expectedAudiences.some((expected) => audiences.includes(expected))) return null;
  }
  return principal;
}

export function createApiClient(apiUrl: string, bearer: string, forwarded?: string): XcelsiorApiClient {
  return new Client({ baseUrl: apiUrl, bearer, forwardedFor: forwarded });
}
