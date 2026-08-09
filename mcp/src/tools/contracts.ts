import { z } from "zod";
import { TOOL_SCOPES, scopeUnion } from "../auth/scopes.js";
import type { ScopeRequirement } from "../auth/scopes.js";

const READ_ONLY = new Set([
  "list_available_gpus", "get_spot_prices", "get_pricing_reference", "search_marketplace",
  "list_tiers", "list_instances", "get_instance", "get_instance_logs", "watch_instance",
  "should_i_run_this", "should_i_run_pel_job", "get_serverless_job_status",
  "list_serverless_endpoints", "get_wallet_balance", "estimate_job_cost", "list_invoices",
  "list_payment_methods", "get_spend_envelope",
  "explain_instance_placement", "simulate_instance_placement", "get_instance_timeline",
  "get_active_lease", "get_scheduler_health", "get_host_capacity",
  "list_reconciliation_findings", "get_mcp_action_status",
  "search", "fetch",
  "list_volumes", "get_volume", "get_artifact_expiry",
  "get_promotion_status",
]);

/**
 * Tools whose effect is not undoable by calling the tool again.
 *
 * `drain_host` was reviewed for inclusion here and deliberately left out. The
 * MCP tool posts to `/api/v1/hosts/{id}/drain`, which stops *new* placements
 * and returns "running workloads untouched" — the destructive counterpart is a
 * separate endpoint behind a separate scope, exposed as `evict_host_workloads`,
 * which is flagged. (The *legacy* `/host/{id}/drain` did conflate the two; the
 * versioned endpoint this tool calls does not.) Marking drain destructive would
 * make the annotation disagree with the behaviour, and a reviewer who then
 * called it and observed nothing evicted would be right to distrust every other
 * annotation we publish.
 */
const DESTRUCTIVE = new Set([
  "cancel_instance", "terminate_instance", "evict_host_workloads",
  // `delete_volume` destroys the volume's contents and they cannot be
  // recovered.
  //
  // `detach_volume` is **not** here, and that was reconsidered rather than
  // assumed. It is confirm-gated because the plan puts detach behind approval —
  // it can disrupt a running workload — but the annotation asks a narrower
  // question: is the effect undoable? Re-attaching restores the mount. What is
  // lost is whatever was mid-write, which is a consequence of timing rather
  // than of the operation being irreversible.
  //
  // Same reasoning that keeps `drain_host` out: a reviewer who detaches, then
  // re-attaches, and finds the volume intact would be right to distrust every
  // other annotation we publish. Approval gating and `destructiveHint` are
  // different claims and are kept that way.
  "delete_volume",
  // Deleting an endpoint stops it resolving and cancels what is in flight on
  // it; the endpoint is not recreatable by calling the tool again.
  "delete_serverless_endpoint",
  // Cancelling an inference job ends it. The job cannot be resumed — a new one
  // must be submitted — which is the same shape as cancel_instance.
  "cancel_serverless_job",
]);

/**
 * Tools that read state we do not control.
 *
 * `openWorldHint` is a statement about the *domain*, not about network I/O —
 * every tool here reaches our API over the network. These three surface a live
 * marketplace of independent third-party hosts whose inventory, pricing, and
 * membership change without us doing anything, so a model must not assume a
 * previous answer still holds. Everything else reads our own records, where a
 * repeated call within a conversation is a closed-world read.
 */
const OPEN_WORLD = new Set([
  "list_available_gpus", "get_spot_prices", "search_marketplace",
  // Company knowledge indexes the published documentation site and the live
  // marketplace, both of which change without a deploy on our side.
  "search", "fetch",
]);

/**
 * Tools whose authority is platform-global rather than tenant-scoped.
 *
 * Stated explicitly rather than derived from scope prefixes. The derived
 * version classified `get_mcp_action_status` as operator because `hosts:read`
 * appears in its *alternatives* list (an action plan may concern an instance,
 * an endpoint, or a host) — which would have removed the customer's own
 * "did my launch get approved?" tool from the public profile.
 */
const OPERATOR_TOOLS = new Set([
  "get_scheduler_health",
  "get_host_capacity",
  "list_reconciliation_findings",
  "drain_host",
  "undrain_host",
  "evict_host_workloads",
  "retry_agent_command",
]);

export interface ToolContract {
  version: string;
  /** Advertising and metadata only — never authorize against this. */
  requiredScopes: readonly string[];
  /** Authoritative for authorization. */
  scopeRequirement: ScopeRequirement;
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
  Object.entries(TOOL_SCOPES).map(([name, requirement]) => {
    const readOnly = READ_ONLY.has(name);
    const keyed = !readOnly && !["run_serverless_job"].includes(name);
    return [name, {
      version: "2.0.0",
      // Enforcement uses `scopeRequirement`; this flat union exists only for
      // metadata and advertising. Never authorize against it — flattening
      // `allOf` into a list is exactly how the any-one-of bug read.
      requiredScopes: scopeUnion(requirement),
      scopeRequirement: requirement,
      tenantClass: OPERATOR_TOOLS.has(name) ? "operator" : "tenant",
      idempotency: readOnly ? "read" : keyed ? "keyed" : "none",
      timeoutMs: name === "watch_instance" ? 3_600_000 : 15_000,
      retry: readOnly ? "safe" : keyed ? "idempotent" : "none",
      redaction: "classified",
      annotations: {
        readOnlyHint: readOnly,
        destructiveHint: DESTRUCTIVE.has(name),
        idempotentHint: readOnly || keyed,
        openWorldHint: OPEN_WORLD.has(name),
      },
    } satisfies ToolContract];
  }),
);

export const DEFAULT_OUTPUT_SCHEMA = z.object({}).passthrough();
