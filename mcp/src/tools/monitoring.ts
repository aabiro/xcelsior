import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
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
      description:
        "Poll instance status, telemetry, and recent logs for N minutes; returns an agent-friendly summary.",
      inputSchema: z.object({
        job_id: z.string(),
        duration_minutes: z.number().min(1).max(60).default(5),
        poll_interval_seconds: z.number().int().min(10).max(120).default(30),
        log_tail: z.number().int().min(1).max(100).default(20),
      }),
    },
    async ({ job_id, duration_minutes, poll_interval_seconds, log_tail }) => {
      const denied = scopeDenied("watch_instance", user);
      if (denied) return denied;

      const deadline = Date.now() + duration_minutes * 60_000;
      const samples: Array<Record<string, unknown>> = [];
      let lastStatus = "";
      let lastTelemetry: unknown = null;
      let lastLogs: unknown = null;

      try {
        while (Date.now() < deadline) {
          const [instRes, telRes, logRes] = await Promise.allSettled([
            client.get(`/instance/${encodeURIComponent(job_id)}`),
            client.get(`/api/instances/${encodeURIComponent(job_id)}/telemetry`),
            client.get(`/instances/${encodeURIComponent(job_id)}/logs`, { limit: log_tail }),
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

          samples.push({
            at: new Date().toISOString(),
            status: lastStatus,
            telemetry: (lastTelemetry as { telemetry?: unknown })?.telemetry ?? null,
          });

          if (["completed", "failed", "cancelled", "terminated", "preempted"].includes(lastStatus)) {
            break;
          }

          await new Promise((r) => setTimeout(r, poll_interval_seconds * 1000));
        }

        return jsonText({
          ok: true,
          job_id,
          final_status: lastStatus,
          samples,
          latest_telemetry: lastTelemetry,
          latest_logs: lastLogs,
          watched_minutes: duration_minutes,
        });
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}