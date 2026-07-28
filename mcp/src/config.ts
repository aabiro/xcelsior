import { loadRateLimitConfig, type RateLimitConfig } from "./rate-limit.js";

export interface AppConfig {
  apiUrl: string;
  host: string;
  port: number;
  mcpPath: string;
  rateLimitPerMinute: number;
  rateLimit: RateLimitConfig;
  publicUrl: string;
  resourceAudience: string;
  oauthIssuer: string;
  oauthJwksUrl: string;
}

export function loadConfig(): AppConfig {
  const apiUrl = (process.env.XCELSIOR_MCP_API_URL || process.env.XCELSIOR_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const rateLimit = loadRateLimitConfig(process.env);
  return {
    apiUrl,
    host: process.env.MCP_HOST || "0.0.0.0",
    port: Number(process.env.MCP_PORT || "8770"),
    mcpPath: process.env.MCP_PATH || "/mcp",
    rateLimitPerMinute: rateLimit.perMinute,
    rateLimit,
    publicUrl: (process.env.XCELSIOR_MCP_PUBLIC_URL || "http://127.0.0.1:8770/mcp").replace(/\/$/, ""),
    resourceAudience: process.env.XCELSIOR_MCP_RESOURCE_AUDIENCE || "https://mcp.xcelsior.ca",
    oauthIssuer: (process.env.XCELSIOR_OAUTH_ISSUER || `${apiUrl}/`).replace(/\/$/, ""),
    oauthJwksUrl: process.env.XCELSIOR_OAUTH_JWKS_URL || `${apiUrl}/.well-known/jwks.json`,
  };
}
