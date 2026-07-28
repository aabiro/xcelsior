import { createHash, randomUUID } from "node:crypto";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import type { XcelsiorApiClient } from "../client/api.js";
import {
  actionFlow, idempotentReplays, toolCalls, toolDuration,
  watchDuration, workloadOutcomes,
} from "../observability/metrics.js";
import { traced } from "../observability/tracing.js";
import { DEFAULT_OUTPUT_SCHEMA, TOOL_CONTRACTS } from "../tools/contracts.js";
import { withApiCapture, type ApiCallRecord } from "../client/request-context.js";

const FORBIDDEN = /token|secret|password|credential|environment|env|init_script|registry_password/i;

export function redactedArgumentHash(value: unknown): string {
  const canonical = JSON.stringify(redact(value), Object.keys((value as object) ?? {}).sort());
  return createHash("sha256").update(canonical).digest("hex");
}

function redact(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redact);
  if (value && typeof value === "object") {
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [
      key,
      FORBIDDEN.test(key) ? "[REDACTED]" : redact(item),
    ]));
  }
  return value;
}

/** Wrap tool registration once so every current and future tool is audited. */
export function installToolAudit(
  server: McpServer,
  client: XcelsiorApiClient,
  user: AuthUser | undefined,
  transport: "streamable_http" | "stdio",
): void {
  const target = server as unknown as {
    registerTool: (name: string, config: unknown, callback: (...args: unknown[]) => Promise<unknown>) => unknown;
  };
  const original = target.registerTool.bind(server);
  target.registerTool = (name, toolConfig, callback) => {
    const contract = TOOL_CONTRACTS[name];
    if (!contract) throw new Error(`Tool ${name} has no production contract or required scope`);
    const version = contract.version;
    const config = toolConfig as Record<string, unknown>;
    return original(name, {
      ...config,
      outputSchema: config.outputSchema ?? DEFAULT_OUTPUT_SCHEMA,
      annotations: { ...contract.annotations, ...((config.annotations as object | undefined) ?? {}) },
      _meta: {
        ...((config._meta as object | undefined) ?? {}),
        "xcelsior/toolVersion": version,
        "xcelsior/idempotency": contract.idempotency,
        "xcelsior/timeoutMs": contract.timeoutMs,
        "xcelsior/retry": contract.retry,
        "xcelsior/redaction": contract.redaction,
        "xcelsior/tenantClass": contract.tenantClass,
      },
    }, async (...args: unknown[]) => {
      const started = performance.now();
      let traceId = randomUUID().replaceAll("-", "");
      let outcome = "success";
      let result: unknown;
      const apiCalls: ApiCallRecord[] = [];
      try {
        result = await withApiCapture(
          apiCalls,
          () => traced(`mcp.tool.${name}`, (_traceparent, spanTraceId) => {
            if (spanTraceId) traceId = spanTraceId;
            return callback(...args);
          }),
        );
        const structured = (result as { structuredContent?: Record<string, unknown> })?.structuredContent;
        if (structured?.ok === false || structured?.error) outcome = "error";
        return result;
      } catch (error) {
        outcome = "exception";
        throw error;
      } finally {
        const input = args[0] ?? {};
        const apiCall = apiCalls.at(-1);
        const output = result as Record<string, unknown> | undefined;
        const record = {
          tool_name: name,
          tool_version: version,
          transport,
          client_id: user?.client_id,
          principal_id: user?.subject || user?.user_id || user?.email,
          tenant_id: user?.workspace_id || user?.customer_id,
          team_id: user?.team_id,
          scopes_evaluated: user?.scopes ?? [],
          redacted_args_hash: redactedArgumentHash(input),
          action_plan_id:
            stringField(input, "plan_id")
            ?? findString(output?.structuredContent ?? output, "plan_id"),
          idempotency_key: stringField(input, "idempotency_key"),
          api_route: apiCall?.route,
          api_status: apiCall?.status,
          problem_type: apiCall?.problemType,
          resource_id: findResourceId(output?.structuredContent ?? output),
          latency_ms: Math.max(0, Math.round(performance.now() - started)),
          trace_id: traceId,
          outcome,
          approval_method: findString(output?.structuredContent ?? output, "approval_method"),
        };
        toolCalls.inc({ tool: name, outcome });
        const durationSeconds = Math.max(0, performance.now() - started) / 1000;
        toolDuration.observe({ tool: name }, durationSeconds);
        const structured = output?.structuredContent as Record<string, unknown> | undefined;
        if (name === "watch_instance") watchDuration.observe(durationSeconds);
        if (name === "create_instance" || name === "create_serverless_endpoint") {
          const phase = structured?.preview
            ? structured.approval_required ? "approval_required" : "preview"
            : "execute";
          actionFlow.inc({ tool: name, phase, outcome });
          workloadOutcomes.inc({
            class: name === "create_instance" ? "instance" : "serverless_endpoint",
            outcome,
          });
        } else if (name === "run_serverless_job") {
          workloadOutcomes.inc({ class: "serverless_invocation", outcome });
        }
        if (structured?.idempotent === true) idempotentReplays.inc({ tool: name });
        // Audit failure must be visible but must not turn a completed read into
        // a duplicate-prone retry. The API/outbox remains the durable authority.
        try {
          await client.post("/api/v1/mcp/tool-audit", record, { timeoutMs: 3_000, retry: "none" });
        } catch (error) {
          process.stderr.write(`[xcelsior-mcp] audit write failed tool=${name} trace=${traceId}: ${String(error)}\n`);
        }
      }
    });
  };
}

function findResourceId(value: unknown): string | undefined {
  for (const key of ["job_id", "endpoint_id", "host_id", "command_id", "resource_id"]) {
    const found = findString(value, key);
    if (found) return found;
  }
  return undefined;
}

function findString(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  if (typeof record[key] === "string") return record[key] as string;
  for (const child of Object.values(record)) {
    const found = findString(child, key);
    if (found) return found;
  }
  return undefined;
}

function stringField(value: unknown, key: string): string | undefined {
  if (!value || typeof value !== "object") return undefined;
  const item = (value as Record<string, unknown>)[key];
  return typeof item === "string" && item ? item : undefined;
}
