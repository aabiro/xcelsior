import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText, structuredResult } from "../lib/format.js";
import { TOOL_SCOPES, userHasScope } from "../auth/scopes.js";
import type { AuthUser } from "../auth/bearer.js";

function scopeDenied(tool: string, user: AuthUser | undefined) {
  const required = TOOL_SCOPES[tool] || ["api"];
  if (!userHasScope(user?.scopes, required)) {
    return jsonText({
      error: "insufficient_scope",
      required,
      message: `This tool requires one of: ${required.join(", ")}`,
    });
  }
  return null;
}

export function registerMonitoringTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "watch_instance",
    {
      inputSchema: z.object({
        job_id: z.string(),
        duration_minutes: z.number().min(1).max(60).default(5),
        poll_interval_seconds: z.number().int().min(10).max(120).default(30),
        log_tail: z.number().int().min(1).max(100).default(20),
        cursor: z.string().max(512).optional(),
        return_on_phase: z.array(z.string().max(64)).max(12).default(["completed", "failed", "cancelled", "terminated", "preempted"]),
      }),
      outputSchema: z.object({ ok: z.boolean(), job_id: z.string(), final_status: z.string(), cursor: z.string().optional() }).passthrough(),
      annotations: { readOnlyHint: true, destructiveHint: false, idempotentHint: true, openWorldHint: false },
    },
    async ({ job_id, duration_minutes, poll_interval_seconds, log_tail, cursor, return_on_phase }, extra) => {
      const denied = scopeDenied("watch_instance", user);
      if (denied) return denied;

      const deadline = Date.now() + duration_minutes * 60_000;
      const samples: Array<Record<string, unknown>> = [];
      const events: Array<Record<string, unknown>> = [];
      let nextCursor = cursor;
      let lastStatus = "";
      let lastTelemetry: unknown = null;
      let lastLogs: unknown = null;

      try {
        while (Date.now() < deadline) {
          const [instRes, telRes, logRes, eventRes] = await Promise.allSettled([
            client.get(`/instance/${encodeURIComponent(job_id)}`),
            client.get(`/api/instances/${encodeURIComponent(job_id)}/telemetry`),
            client.get(`/instances/${encodeURIComponent(job_id)}/logs`, { limit: log_tail }),
            client.get(`/api/v1/instances/${encodeURIComponent(job_id)}/events`, {
              cursor: nextCursor, limit: 100,
            }),
          ]);

          const instance =
            instRes.status === "fulfilled"
              ? ((instRes.value as { instance?: Record<string, unknown> }).instance ||
                  (instRes.value as Record<string, unknown>))
              : { error: String(instRes.reason) };

          lastStatus = String(instance.status || "");
          lastTelemetry =
            telRes.status === "fulfilled" ? telRes.value : { error: String(telRes.reason) };
          lastLogs =
            logRes.status === "fulfilled" ? logRes.value : { error: String(logRes.reason) };
          if (eventRes.status === "fulfilled") {
            const page = eventRes.value as { events?: Array<Record<string, unknown>>; next_cursor?: string };
            events.push(...(page.events ?? []));
            nextCursor = page.next_cursor ?? nextCursor;
          }

          samples.push({
            at: new Date().toISOString(),
            status: lastStatus,
            telemetry: (lastTelemetry as { telemetry?: unknown })?.telemetry ?? null,
          });
          const progressToken = extra._meta?.progressToken;
          if (progressToken !== undefined) {
            await extra.sendNotification({
              method: "notifications/progress",
              params: {
                progressToken,
                progress: Math.min(duration_minutes * 60, Math.round((Date.now() - (deadline - duration_minutes * 60_000)) / 1000)),
                total: duration_minutes * 60,
                message: lastStatus ? `Instance phase: ${lastStatus}` : "Waiting for instance state",
              },
            });
          }

          if (return_on_phase.includes(lastStatus)) {
            break;
          }

          await abortableDelay(poll_interval_seconds * 1000, extra.signal);
        }

        return structuredResult({
          ok: true,
          job_id,
          final_status: lastStatus,
          samples,
          latest_telemetry: lastTelemetry,
          latest_logs: lastLogs,
          events,
          watched_minutes: duration_minutes,
          cursor: nextCursor,
        }, `Instance ${job_id} is ${lastStatus || "unknown"}.`);
      } catch (e) {
        if (extra.signal.aborted) {
          return structuredResult({
            ok: true,
            job_id,
            final_status: lastStatus,
            watch_cancelled: true,
            instance_cancelled: false,
            message: "The watch stopped; the GPU instance was not cancelled.",
          });
        }
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}

function abortableDelay(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal.aborted) return reject(signal.reason);
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      reject(signal.reason);
    }, { once: true });
  });
}
