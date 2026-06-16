export interface AppConfig {
  apiUrl: string;
  host: string;
  port: number;
  mcpPath: string;
  rateLimitPerMinute: number;
}

export function loadConfig(): AppConfig {
  const apiUrl = (process.env.XCELSIOR_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
  return {
    apiUrl,
    host: process.env.MCP_HOST || "0.0.0.0",
    port: Number(process.env.MCP_PORT || "3100"),
    mcpPath: process.env.MCP_PATH || "/mcp",
    rateLimitPerMinute: Number(process.env.MCP_RATE_LIMIT_PER_MIN || "60"),
  };
}