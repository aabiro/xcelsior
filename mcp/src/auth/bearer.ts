import type { IncomingMessage } from "node:http";
import type { XcelsiorApiClient } from "../client/api.js";
import { XcelsiorApiClient as Client } from "../client/api.js";

export interface AuthUser {
  email?: string;
  customer_id?: string;
  user_id?: string;
  scopes?: string[];
  auth_type?: string;
}

export function extractBearer(req: IncomingMessage): string | null {
  const auth = req.headers.authorization;
  if (!auth?.startsWith("Bearer ")) return null;
  const token = auth.slice(7).trim();
  return token || null;
}

export async function validateBearer(apiUrl: string, bearer: string): Promise<AuthUser | null> {
  try {
    const client = new Client({ baseUrl: apiUrl, bearer });
    // /api/auth/introspect accepts machine (client_credentials) tokens, which the
    // MCP gateway uses; /api/auth/me rejects them with 403.
    const principal = await client.get<AuthUser & { ok?: boolean }>("/api/auth/introspect");
    if (!principal || (principal as { ok?: boolean }).ok === false) return null;
    return principal;
  } catch {
    return null;
  }
}

export function createApiClient(apiUrl: string, bearer: string): XcelsiorApiClient {
  return new Client({ baseUrl: apiUrl, bearer });
}