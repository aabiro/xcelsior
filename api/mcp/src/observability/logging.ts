import pino from "pino";

export const log = pino({
  name: "xcelsior-mcp",
  level: process.env.XCELSIOR_LOG_LEVEL || "info",
  redact: {
    paths: ["req.headers.authorization", "bearer", "token", "*.token", "*.secret", "*.password", "*.init_script", "*.environment"],
    censor: "[REDACTED]",
  },
});
