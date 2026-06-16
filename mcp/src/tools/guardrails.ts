import { z } from "zod";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { XcelsiorApiClient } from "../client/api.js";
import { formatApiError } from "../client/errors.js";
import { jsonText } from "../lib/format.js";
import { evaluateShouldIRunThis } from "../lib/guardrails.js";
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

export function registerGuardrailTools(
  server: McpServer,
  client: XcelsiorApiClient,
  user?: AuthUser,
): void {
  server.registerTool(
    "should_i_run_this",
    {
      description:
        "Composite guardrail: estimate cost, check wallet balance, and return approval guidance before spend.",
      inputSchema: z.object({
        gpu_model: z.string().default("RTX 4090"),
        duration_hours: z.number().min(0).max(8760).default(1),
        spot: z.boolean().default(false),
        max_hourly_cad: z.number().positive().optional().describe("Reject if hourly rate exceeds this"),
        require_canada: z
          .boolean()
          .default(false)
          .describe("Include Canadian residency guidance in the response"),
      }),
    },
    async (args) => {
      const denied = scopeDenied("should_i_run_this", user);
      if (denied) return denied;
      const cid = user?.customer_id || user?.user_id;
      if (!cid) return jsonText({ error: "customer_id required — authenticate first" });
      try {
        const [estimate, walletRes] = await Promise.all([
          client.post("/api/pricing/estimate", {
            gpu_model: args.gpu_model,
            duration_hours: args.duration_hours,
            spot: args.spot,
          }),
          client.get(`/api/billing/wallet/${encodeURIComponent(cid)}`),
        ]);
        const walletBody = walletRes as { wallet?: { balance_cad?: number } };
        const balance = Number(walletBody.wallet?.balance_cad ?? 0);
        const estimateBody =
          (estimate as { estimate?: Record<string, unknown> }).estimate ||
          (estimate as Record<string, unknown>);
        const result = evaluateShouldIRunThis(args, estimateBody, balance);
        return jsonText(result);
      } catch (e) {
        return jsonText({ error: formatApiError(e) });
      }
    },
  );
}