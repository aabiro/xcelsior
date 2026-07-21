import { loadRateLimitConfig, type RateLimitConfig } from "./rate-limit.js";

export interface AppConfig {
  apiUrl: string;
  host: string;
  port: number;
  mcpPath: string;
  rateLimitPerMinute: number;
  rateLimit: RateLimitConfig;
}

export function loadConfig(): AppConfig {
  const apiUrl = (process.env.XCELSIOR_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  const rateLimit = loadRateLimitConfig(process.env);
  return {
    apiUrl,
    host: process.env.MCP_HOST || "0.0.0.0",
    port: Number(process.env.MCP_PORT || "8770"),
    mcpPath: process.env.MCP_PATH || "/mcp",
    rateLimitPerMinute: rateLimit.perMinute,
    rateLimit,
  };
}