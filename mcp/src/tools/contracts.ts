import { z } from "zod";
import { TOOL_SCOPES } from "../auth/scopes.js";

const READ_ONLY = new Set([
  "list_available_gpus", "get_spot_prices", "get_pricing_reference", "search_marketplace",
  "list_tiers", "list_instances", "get_instance", "get_instance_logs", "watch_instance",
  "should_i_run_this", "should_i_run_pel_job", "get_serverless_job_status",
  "list_serverless_endpoints", "get_wallet_balance", "estimate_job_cost", "list_invoices",
  "explain_instance_placement", "simulate_instance_placement", "get_instance_timeline",
  "get_active_lease", "get_scheduler_health", "get_host_capacity",
  "list_reconciliation_findings", "get_mcp_action_status",
]);
const DESTRUCTIVE = new Set(["cancel_instance", "terminate_instance", "evict_host_workloads"]);

export interface ToolContract {
  version: string;
  requiredScopes: readonly string[];
  tenantClass: "tenant" | "operator";
  idempotency: "read" | "keyed" | "none";
  timeoutMs: number;
  retry: "safe" | "idempotent" | "none";
  redaction: "classified";
  annotations: {
    readOnlyHint: boolean;
    destructiveHint: boolean;
    idempotentHint: boolean;
    openWorldHint: boolean;
  };
}

export const TOOL_CONTRACTS: Record<string, ToolContract> = Object.fromEntries(
  Object.entries(TOOL_SCOPES).map(([name, scopes]) => {
    const readOnly = READ_ONLY.has(name);
    const keyed = !readOnly && !["run_serverless_job"].includes(name);
    return [name, {
      version: "2.0.0",
      requiredScopes: scopes,
      tenantClass: scopes.some((scope) => scope.startsWith("hosts:") || scope.startsWith("control_plane:"))
        ? "operator" : "tenant",
      idempotency: readOnly ? "read" : keyed ? "keyed" : "none",
      timeoutMs: name === "watch_instance" ? 3_600_000 : 15_000,
      retry: readOnly ? "safe" : keyed ? "idempotent" : "none",
      redaction: "classified",
      annotations: {
        readOnlyHint: readOnly,
        destructiveHint: DESTRUCTIVE.has(name),
        idempotentHint: readOnly || keyed,
        openWorldHint: false,
      },
    } satisfies ToolContract];
  }),
);

export const DEFAULT_OUTPUT_SCHEMA = z.object({}).passthrough();
