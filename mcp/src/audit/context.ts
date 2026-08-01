import { createHash, randomUUID } from "node:crypto";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { AuthUser } from "../auth/bearer.js";
import type { XcelsiorApiClient } from "../client/api.js";
import {
  actionFlow, hygieneRedactions, idempotentReplays, toolCalls, toolDuration,
  watchDuration, workloadOutcomes,
} from "../observability/metrics.js";
import { scrubResponse, scrubText } from "../lib/hygiene.js";
import { traced } from "../observability/tracing.js";
import { DEFAULT_OUTPUT_SCHEMA, TOOL_CONTRACTS } from "../tools/contracts.js";
import { TOOL_DESCRIPTIONS } from "../tools/descriptions.js";
import { toolIsVisible, type ToolProfile } from "../tools/profiles.js";
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

/**
 * Strip auth material, debug payloads, and undisclosed user fields from a tool
 * result before it leaves the process.
 *
 * Applied here rather than in each tool because "each tool remembers" is
 * exactly the assumption that fails: a new tool, or an upstream response that
 * grows a field, would otherwise leak by default. The `content` text is
 * regenerated from the scrubbed structure when it was a JSON serialisation of
 * it, so the two halves of the result cannot disagree — a scrubbed
 * `structuredContent` beside an unscrubbed text blob would leak anyway, since
 * most clients show the text.
 */
function applyResponseHygiene(name: string, result: unknown): unknown {
  if (!result || typeof result !== "object") return result;
  const record = result as {
    content?: Array<{ type?: string; text?: string }>;
    structuredContent?: Record<string, unknown>;
  };
  const removed: string[] = [];

  let structured = record.structuredContent;
  let structuredWasJsonText = false;
  let originalJson = "";
  if (structured && typeof structured === "object") {
    originalJson = safeStringify(structured);
    const report = scrubResponse(structured);
    removed.push(...report.removed);
    structured = report.value as Record<string, unknown>;
    structuredWasJsonText = true;
  }

  const content = record.content?.map((item) => {
    if (item?.type !== "text" || typeof item.text !== "string") return item;
    if (structuredWasJsonText && item.text === originalJson) {
      return { ...item, text: safeStringify(structured) };
    }
    const { text, masked } = scrubText(item.text);
    if (masked) removed.push("content.text");
    return { ...item, text };
  });

  if (!removed.length) return result;
  hygieneRedactions.inc({ tool: name }, removed.length);
  // Loud on purpose. The filter working is not the same as the tool being
  // correct — something upstream handed us a field that must never be modelled.
  process.stderr.write(
    `[xcelsior-mcp] response hygiene removed ${removed.length} field(s) from ${name}: ` +
      `${removed.join(", ")}\n`,
  );
  return { ...record, ...(content ? { content } : {}), ...(structured ? { structuredContent: structured } : {}) };
}

function safeStringify(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return "";
  }
}

/**
 * A tool that declares annotations must declare the ones its contract states.
 *
 * Throwing rather than merging: a divergent local annotation is a genuine
 * disagreement about what the tool does, and silently preferring either side
 * hides it. This fires at registration, so it fails the build and the E2E, not
 * a reviewer's spot check.
 */
function assertAnnotationsMatchContract(
  name: string,
  declared: unknown,
  contract: Record<string, boolean>,
): void {
  if (!declared || typeof declared !== "object") return;
  const mismatched = Object.entries(declared as Record<string, unknown>)
    .filter(([key, value]) => key in contract && contract[key] !== value)
    .map(([key, value]) => `${key}: declared ${value}, contract ${contract[key]}`);
  if (mismatched.length) {
    throw new Error(
      `Tool ${name} declares annotations that contradict its contract — ${mismatched.join("; ")}`,
    );
  }
}

/** Wrap tool registration once so every current and future tool is audited. */
export function installToolAudit(
  server: McpServer,
  client: XcelsiorApiClient,
  user: AuthUser | undefined,
  transport: "streamable_http" | "stdio",
  profile: ToolProfile = "customer",
): void {
  const target = server as unknown as {
    registerTool: (name: string, config: unknown, callback: (...args: unknown[]) => Promise<unknown>) => unknown;
  };
  const original = target.registerTool.bind(server);
  target.registerTool = (name, toolConfig, callback) => {
    const contract = TOOL_CONTRACTS[name];
    if (!contract) throw new Error(`Tool ${name} has no production contract or required scope`);
    const description = TOOL_DESCRIPTIONS[name];
    if (!description) {
      throw new Error(
        `Tool ${name} has no reviewed description in src/tools/descriptions.ts — a reviewer ` +
          `calls every tool and compares behaviour to what the description promised`,
      );
    }
    // Out-of-profile tools are never registered, so they cannot appear in
    // tools/list and cannot be called by name either — a filter applied only to
    // the listing would leave the tool callable by anyone who guessed it.
    if (!toolIsVisible(name, profile, user?.scopes)) return undefined;
    const version = contract.version;
    const config = toolConfig as Record<string, unknown>;
    assertAnnotationsMatchContract(name, config.annotations, contract.annotations);
    return original(name, {
      ...config,
      description,
      outputSchema: config.outputSchema ?? DEFAULT_OUTPUT_SCHEMA,
      // The contract wins. Annotations are what a directory reviewer checks
      // against real behaviour, so they have exactly one source of truth; a
      // per-tool override that disagreed used to win silently, which is how an
      // annotation drifts away from what the tool does.
      annotations: { ...contract.annotations },
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
        result = applyResponseHygiene(name, result);
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
