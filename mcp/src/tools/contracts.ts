import { z } from "zod";
import { TOOL_SCOPES, scopeUnion } from "../auth/scopes.js";
import type { ScopeRequirement, ToolName } from "../auth/scopes.js";

/**
 * Who a tool is published to.
 *
 * `operator` is platform-global authority; `company-knowledge` is the optional
 * documentation surface behind `XCELSIOR_MCP_COMPANY_KNOWLEDGE`. Everything
 * else is `customer`, and only `customer` tools appear in `tool-surface.json`.
 */
export type ToolAudience = "customer" | "operator" | "company-knowledge";

/**
 * Every policy decision about a tool that is not derivable from its scopes.
 *
 * `readOnly`, `destructive` and `audience` are **required**, and that verbosity
 * is the entire point of this file. They were three hand-maintained `Set`s, so
 * adding a tool and forgetting one silently published a default: a destructive
 * tool advertised as reversible, or an operator tool advertised to customers.
 * Optional-with-a-default reintroduces exactly the drift this table removes, so
 * the compiler asks the question instead.
 *
 * `openWorld` stays optional because being wrong about it costs a stale answer
 * rather than a wrong safety claim, and `false` is right for every tool that
 * reads our own records.
 */
export interface ToolPolicy {
  /** No state changes. A model reads this to decide a call is safe. */
  readOnly: boolean;
  /** The effect is not undone by calling the tool again. */
  destructive: boolean;
  audience: ToolAudience;
  /** Reads a domain that changes without us doing anything. Defaults false. */
  openWorld?: boolean;
  /** Overrides the 15s default. */
  timeoutMs?: number;
  /** Overrides the value derived from `readOnly`. */
  idempotency?: "read" | "keyed" | "none";
}

/**
 * The policy table — one row per tool, keyed by `ToolName`.
 *
 * `Record<ToolName, ToolPolicy>` is where the S1 inversion actually lives.
 * `ToolName` is `keyof typeof TOOL_SCOPES`, so a tool added to the scope
 * registry with no row here **fails to compile**, and a row for a tool that
 * does not exist fails the same way. The 37-vs-39 drift that motivated this is
 * no longer something a test has to notice; it is not expressible.
 */
const TOOL_POLICY: Record<ToolName, ToolPolicy> = {
  list_available_gpus: { readOnly: true, destructive: false, audience: "customer", openWorld: true },
  get_spot_prices: { readOnly: true, destructive: false, audience: "customer", openWorld: true },
  get_pricing_reference: { readOnly: true, destructive: false, audience: "customer" },
  search_marketplace: { readOnly: true, destructive: false, audience: "customer", openWorld: true },
  list_tiers: { readOnly: true, destructive: false, audience: "customer" },
  list_instances: { readOnly: true, destructive: false, audience: "customer" },
  get_instance: { readOnly: true, destructive: false, audience: "customer" },
  get_instance_logs: { readOnly: true, destructive: false, audience: "customer" },
  create_instance: { readOnly: false, destructive: false, audience: "customer" },
  cancel_instance: { readOnly: false, destructive: true, audience: "customer" },
  terminate_instance: { readOnly: false, destructive: true, audience: "customer" },
  should_i_run_this: { readOnly: true, destructive: false, audience: "customer" },
  run_training_job: { readOnly: false, destructive: false, audience: "customer" },
  schedule_under_budget: { readOnly: false, destructive: false, audience: "customer" },
  // The one long timeout: it holds a connection open while an instance changes
  // state, so the 15s default would end every useful call.
  watch_instance: { readOnly: true, destructive: false, audience: "customer", timeoutMs: 3_600_000 },
  register_ssh_key: { readOnly: false, destructive: false, audience: "customer" },
  list_volumes: { readOnly: true, destructive: false, audience: "customer" },
  get_volume: { readOnly: true, destructive: false, audience: "customer" },
  create_volume: { readOnly: false, destructive: false, audience: "customer" },
  promote_artifact_to_volume: { readOnly: false, destructive: false, audience: "customer" },
  get_promotion_status: { readOnly: true, destructive: false, audience: "customer" },
  run_pipeline: { readOnly: false, destructive: false, audience: "customer" },
  get_pipeline_status: { readOnly: true, destructive: false, audience: "customer" },
  attach_volume: { readOnly: false, destructive: false, audience: "customer" },
  // **Not** destructive, and this was reconsidered rather than assumed. It is
  // confirm-gated because the plan puts detach behind approval — it can disrupt
  // a running workload — but `destructiveHint` asks a narrower question: is the
  // effect undoable? Re-attaching restores the mount. What is lost is whatever
  // was mid-write, a consequence of timing rather than of irreversibility.
  // Approval gating and `destructiveHint` are different claims, kept that way.
  detach_volume: { readOnly: false, destructive: false, audience: "customer" },
  // Destroys the volume's contents; they cannot be recovered.
  delete_volume: { readOnly: false, destructive: true, audience: "customer" },
  snapshot_volume: { readOnly: false, destructive: false, audience: "customer" },
  get_artifact_expiry: { readOnly: true, destructive: false, audience: "customer" },
  open_instance_access: { readOnly: false, destructive: false, audience: "customer" },
  list_serverless_endpoints: { readOnly: true, destructive: false, audience: "customer" },
  create_serverless_endpoint: { readOnly: false, destructive: false, audience: "customer" },
  should_i_run_pel_job: { readOnly: true, destructive: false, audience: "customer" },
  // The one tool with no idempotency key. A resubmitted inference job is a new
  // job, so replaying the call is not a no-op and must not be advertised as one.
  run_serverless_job: { readOnly: false, destructive: false, audience: "customer", idempotency: "none" },
  get_serverless_job_status: { readOnly: true, destructive: false, audience: "customer" },
  // Ending an inference job is final — a new one must be submitted. Same shape
  // as cancel_instance.
  cancel_serverless_job: { readOnly: false, destructive: true, audience: "customer" },
  // Deleting an endpoint stops it resolving and cancels what is in flight on it;
  // it is not recreatable by calling the tool again.
  delete_serverless_endpoint: { readOnly: false, destructive: true, audience: "customer" },
  explain_instance_placement: { readOnly: true, destructive: false, audience: "customer" },
  simulate_instance_placement: { readOnly: true, destructive: false, audience: "customer" },
  evaluate_placement_preference: { readOnly: true, destructive: false, audience: "customer" },
  get_instance_timeline: { readOnly: true, destructive: false, audience: "customer" },
  get_active_lease: { readOnly: true, destructive: false, audience: "customer" },
  get_scheduler_health: { readOnly: true, destructive: false, audience: "operator" },
  get_host_capacity: { readOnly: true, destructive: false, audience: "operator" },
  list_reconciliation_findings: { readOnly: true, destructive: false, audience: "operator" },
  // `customer`, not `operator`, and stated rather than derived. A derived
  // version classified this as operator because `hosts:read` appears in its
  // *alternatives* (an action plan may concern an instance, an endpoint or a
  // host) — which would have removed the customer's own "did my launch get
  // approved?" tool from the public profile.
  get_mcp_action_status: { readOnly: true, destructive: false, audience: "customer" },
  retry_instance: { readOnly: false, destructive: false, audience: "customer" },
  reconcile_instance: { readOnly: false, destructive: false, audience: "customer" },
  // Deliberately **not** destructive. The tool posts to
  // `/api/v1/hosts/{id}/drain`, which stops *new* placements and returns
  // "running workloads untouched"; the destructive counterpart is a separate
  // endpoint behind a separate scope, exposed as `evict_host_workloads`. (The
  // *legacy* `/host/{id}/drain` did conflate the two; the versioned endpoint
  // this tool calls does not.) Marking it destructive would make the annotation
  // disagree with the behaviour, and a reviewer who called it and observed
  // nothing evicted would be right to distrust every other annotation we
  // publish.
  drain_host: { readOnly: false, destructive: false, audience: "operator" },
  undrain_host: { readOnly: false, destructive: false, audience: "operator" },
  evict_host_workloads: { readOnly: false, destructive: true, audience: "operator" },
  retry_agent_command: { readOnly: false, destructive: false, audience: "operator" },
  get_wallet_balance: { readOnly: true, destructive: false, audience: "customer" },
  get_spend_envelope: { readOnly: true, destructive: false, audience: "customer" },
  estimate_job_cost: { readOnly: true, destructive: false, audience: "customer" },
  list_invoices: { readOnly: true, destructive: false, audience: "customer" },
  list_payment_methods: { readOnly: true, destructive: false, audience: "customer" },
  top_up_wallet: { readOnly: false, destructive: false, audience: "customer" },
  configure_auto_topup: { readOnly: false, destructive: false, audience: "customer" },
  // Company knowledge, off by default. Both index the published documentation
  // site and the live marketplace, which change without a deploy on our side.
  search: { readOnly: true, destructive: false, audience: "company-knowledge", openWorld: true },
  fetch: { readOnly: true, destructive: false, audience: "company-knowledge", openWorld: true },
};

/** Names published to customers — the surface `tool-surface.json` snapshots. */
export const CUSTOMER_TOOLS: ReadonlySet<string> = new Set(
  Object.entries(TOOL_POLICY).filter(([, p]) => p.audience === "customer").map(([n]) => n),
);
export const OPERATOR_TOOLS: ReadonlySet<string> = new Set(
  Object.entries(TOOL_POLICY).filter(([, p]) => p.audience === "operator").map(([n]) => n),
);
export const COMPANY_KNOWLEDGE_TOOL_NAMES: ReadonlySet<string> = new Set(
  Object.entries(TOOL_POLICY).filter(([, p]) => p.audience === "company-knowledge").map(([n]) => n),
);

/** Policy for a tool, or `undefined` for a name that is not registered. */
export function toolPolicy(name: string): ToolPolicy | undefined {
  return (TOOL_POLICY as Record<string, ToolPolicy>)[name];
}

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
    const policy = (TOOL_POLICY as Record<string, ToolPolicy>)[name];
    const readOnly = policy.readOnly;
    const idempotency = policy.idempotency ?? (readOnly ? "read" : "keyed");
    return [name, {
      version: "2.0.0",
      // Enforcement uses `scopeRequirement`; this flat union exists only for
      // metadata and advertising. Never authorize against it — flattening
      // `allOf` into a list is exactly how the any-one-of bug read.
      requiredScopes: scopeUnion(requirement),
      scopeRequirement: requirement,
      tenantClass: policy.audience === "operator" ? "operator" : "tenant",
      idempotency,
      timeoutMs: policy.timeoutMs ?? 15_000,
      retry: idempotency === "read" ? "safe" : idempotency === "keyed" ? "idempotent" : "none",
      redaction: "classified",
      annotations: {
        readOnlyHint: readOnly,
        destructiveHint: policy.destructive,
        idempotentHint: idempotency !== "none",
        openWorldHint: policy.openWorld ?? false,
      },
    } satisfies ToolContract];
  }),
);

export const DEFAULT_OUTPUT_SCHEMA = z.object({}).passthrough();
