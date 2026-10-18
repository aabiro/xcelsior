import { Counter, Gauge, Histogram, Registry, collectDefaultMetrics } from "prom-client";

export const metricsRegistry = new Registry();
collectDefaultMetrics({ register: metricsRegistry, prefix: "xcelsior_mcp_" });

export const toolCalls = new Counter({
  name: "xcelsior_mcp_tool_calls_total",
  help: "MCP tool calls by tool and outcome.",
  labelNames: ["tool", "outcome"] as const,
  registers: [metricsRegistry],
});
export const toolDuration = new Histogram({
  name: "xcelsior_mcp_tool_duration_seconds",
  help: "MCP tool latency.",
  labelNames: ["tool"] as const,
  buckets: [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 15, 60],
  registers: [metricsRegistry],
});
export const authFailures = new Counter({
  name: "xcelsior_mcp_auth_failures_total", help: "Bearer authentication failures.",
  registers: [metricsRegistry],
});
export const rateFailures = new Counter({
  name: "xcelsior_mcp_rate_limit_failures_total", help: "Rate policy refusals.",
  labelNames: ["code"] as const, registers: [metricsRegistry],
});
export const activeTransports = new Gauge({
  name: "xcelsior_mcp_active_transports", help: "Active MCP transports.",
  labelNames: ["transport"] as const, registers: [metricsRegistry],
});
export const actionFlow = new Counter({
  name: "xcelsior_mcp_action_flow_total",
  help: "Preview, approval-required, and execute outcomes.",
  labelNames: ["tool", "phase", "outcome"] as const,
  registers: [metricsRegistry],
});
export const idempotentReplays = new Counter({
  name: "xcelsior_mcp_idempotent_replays_total",
  help: "Duplicate calls safely replayed.",
  labelNames: ["tool"] as const,
  registers: [metricsRegistry],
});
export const workloadOutcomes = new Counter({
  name: "xcelsior_mcp_workload_outcomes_total",
  help: "Launch and serverless outcomes.",
  labelNames: ["class", "outcome"] as const,
  registers: [metricsRegistry],
});
export const watchDuration = new Histogram({
  name: "xcelsior_mcp_watch_duration_seconds",
  help: "Duration of watch_instance calls.",
  buckets: [1, 5, 15, 30, 60, 300, 900, 3600],
  registers: [metricsRegistry],
});
export const hygieneRedactions = new Counter({
  name: "xcelsior_mcp_hygiene_redactions_total",
  help: "Fields removed from tool output by the response-hygiene filter. Non-zero means a tool is leaking and the filter is the only thing stopping it.",
  labelNames: ["tool"] as const,
  registers: [metricsRegistry],
});
