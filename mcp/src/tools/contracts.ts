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
/** The version every tool carries unless it has bumped past it. */
const BASE_TOOL_VERSION = "2.0.0";

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
interface ToolPolicyBase {
  audience: ToolAudience;
  /**
   * Overrides {@link BASE_TOOL_VERSION} for this tool alone.
   *
   * `docs/mcp-tool-versioning.md` says to "bump that tool's version in
   * contracts.ts", and until now there was nothing to bump: every contract read
   * a single shared `"2.0.0"` literal, so the only way past the breaking-change
   * guard was to move every tool at once or to overwrite the snapshot — which
   * the same document tells you not to do. The policy was unimplementable as
   * written.
   */
  version?: string;
  /**
   * Reads a domain that changes without us doing anything. Defaults false.
   *
   * A statement about the **domain**, not about network I/O — every tool here
   * reaches our API over a network. The ones that carry it surface a live
   * marketplace of independent third-party hosts whose inventory, pricing and
   * membership move without any deploy of ours, so a model must not assume a
   * previous answer still holds.
   *
   * The rationale this replaces said "everything else reads our own records",
   * and that sentence is how `schedule_under_budget` was missed: it reads
   * `/api/v2/gpu/available` and `/api/v2/marketplace/spot-prices` — the exact
   * two feeds behind `list_available_gpus` and `get_spot_prices`, both flagged
   * — and then **spends money against the answer**. A summary of a list is not
   * a check of it.
   */
  openWorld?: boolean;
  /**
   * Advisory ceiling, recorded in the audit record's `_meta`. **Not enforced.**
   *
   * Worth stating because the name says otherwise. The request deadline comes
   * from `RequestOptions.timeoutMs` at each call site, defaulting to 15s in
   * `client/api.ts`; nothing reads this field except `audit/context.ts`, which
   * emits it as `xcelsior/timeoutMs`. `watch_instance` declares an hour and
   * every HTTP call inside its polling loop still gets 15s — the hour describes
   * how long the *tool* may run, not how long a request may take.
   *
   * Left as-is rather than renamed or wired up: deciding which of those two
   * things it should mean is a design question, and it reaches no client, since
   * `tool-surface.json` does not publish it.
   */
  timeoutMs?: number;
}

/**
 * A tool is either a read or a write, and the two carry different obligations.
 *
 * Split into a union rather than one optional-heavy interface so the compiler
 * enforces two things it previously could not:
 *
 * 1. **A read cannot be destructive.** `readOnly: true` forbids
 *    `destructive: true` outright, so the contradiction is unrepresentable
 *    instead of being caught by a test after the fact.
 * 2. **A write must declare whether repeating it is safe.** `idempotency` is
 *    required on the write arm. It used to default to `"keyed"` for anything
 *    not read-only, which published `idempotentHint: true` — *"calling this
 *    again has no additional effect"* — for **25 tools when only 4 sent an
 *    idempotency key**. `run_training_job` and `schedule_under_budget` both
 *    POST `/instance` with no key, so a model told the retry was free would
 *    launch a second billed GPU; `open_instance_access` mints a fresh
 *    single-use credential per call. The default was the dangerous claim, and
 *    a default that is only right for some tools is how the annotation stops
 *    meaning anything.
 *
 * `"keyed"` means **repeat-safe**, whether because the client sends an
 * idempotency key (`create_instance`, `top_up_wallet`) or because repeating
 * genuinely has no further effect (`terminate_instance`; `register_ssh_key`,
 * which 409s on a duplicate fingerprint; `promote_artifact_to_volume`, whose
 * route carries its own key and `ON CONFLICT`). The name is kept because it is
 * published in `tool-surface.json` and renaming it is a surface change.
 */
export type ToolPolicy =
  | (ToolPolicyBase & {
      readOnly: true;
      /** A read has no destructive form; the type admits only `false`. */
      destructive: false;
      idempotency?: "read";
    })
  | (ToolPolicyBase & {
      readOnly: false;
      /** The effect is not undone by calling the tool again. */
      destructive: boolean;
      /** Required: forgetting must not publish "retrying is free". */
      idempotency: "keyed" | "none";
    });

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
  create_instance: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  cancel_instance: { readOnly: false, destructive: true, audience: "customer", idempotency: "keyed" },
  terminate_instance: { readOnly: false, destructive: true, audience: "customer", idempotency: "keyed" },
  should_i_run_this: { readOnly: true, destructive: false, audience: "customer" },
  run_training_job: { readOnly: false, destructive: false, audience: "customer", idempotency: "none", version: "2.1.0" },
  schedule_under_budget: { readOnly: false, destructive: false, audience: "customer", idempotency: "none", openWorld: true, version: "2.2.0" },
  // The one long timeout: it holds a connection open while an instance changes
  // state, so the 15s default would end every useful call.
  watch_instance: { readOnly: true, destructive: false, audience: "customer", timeoutMs: 3_600_000 },
  register_ssh_key: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  list_volumes: { readOnly: true, destructive: false, audience: "customer" },
  get_volume: { readOnly: true, destructive: false, audience: "customer" },
  create_volume: { readOnly: false, destructive: false, audience: "customer", idempotency: "none", version: "2.1.0" },
  promote_artifact_to_volume: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  get_promotion_status: { readOnly: true, destructive: false, audience: "customer" },
  run_pipeline: { readOnly: false, destructive: false, audience: "customer", idempotency: "none", version: "2.1.0" },
  get_pipeline_status: { readOnly: true, destructive: false, audience: "customer" },
  attach_volume: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  // **Not** destructive, and this was reconsidered rather than assumed. It is
  // confirm-gated because the plan puts detach behind approval — it can disrupt
  // a running workload — but `destructiveHint` asks a narrower question: is the
  // effect undoable? Re-attaching restores the mount. What is lost is whatever
  // was mid-write, a consequence of timing rather than of irreversibility.
  // Approval gating and `destructiveHint` are different claims, kept that way.
  detach_volume: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  // Destroys the volume's contents; they cannot be recovered.
  delete_volume: { readOnly: false, destructive: true, audience: "customer", idempotency: "keyed" },
  snapshot_volume: { readOnly: false, destructive: false, audience: "customer", idempotency: "none", version: "2.1.0" },
  get_artifact_expiry: { readOnly: true, destructive: false, audience: "customer" },
  open_instance_access: { readOnly: false, destructive: false, audience: "customer", idempotency: "none", version: "2.1.0" },
  list_serverless_endpoints: { readOnly: true, destructive: false, audience: "customer" },
  create_serverless_endpoint: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  should_i_run_pel_job: { readOnly: true, destructive: false, audience: "customer" },
  // A resubmitted inference job is a new job, so replaying the call is not a
  // no-op and must not be advertised as one. It was the only tool declaring
  // this; six others were relying on a default that claimed the opposite.
  run_serverless_job: { readOnly: false, destructive: false, audience: "customer", idempotency: "none" },
  get_serverless_job_status: { readOnly: true, destructive: false, audience: "customer" },
  // Ending an inference job is final — a new one must be submitted. Same shape
  // as cancel_instance.
  cancel_serverless_job: { readOnly: false, destructive: true, audience: "customer", idempotency: "keyed" },
  // Deleting an endpoint stops it resolving and cancels what is in flight on it;
  // it is not recreatable by calling the tool again.
  delete_serverless_endpoint: { readOnly: false, destructive: true, audience: "customer", idempotency: "keyed" },
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
  retry_instance: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  reconcile_instance: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  // Deliberately **not** destructive. The tool posts to
  // `/api/v1/hosts/{id}/drain`, which stops *new* placements and returns
  // "running workloads untouched"; the destructive counterpart is a separate
  // endpoint behind a separate scope, exposed as `evict_host_workloads`. (The
  // *legacy* `/host/{id}/drain` did conflate the two; the versioned endpoint
  // this tool calls does not.) Marking it destructive would make the annotation
  // disagree with the behaviour, and a reviewer who called it and observed
  // nothing evicted would be right to distrust every other annotation we
  // publish.
  drain_host: { readOnly: false, destructive: false, audience: "operator", idempotency: "keyed" },
  undrain_host: { readOnly: false, destructive: false, audience: "operator", idempotency: "keyed" },
  evict_host_workloads: { readOnly: false, destructive: true, audience: "operator", idempotency: "keyed" },
  retry_agent_command: { readOnly: false, destructive: false, audience: "operator", idempotency: "keyed" },
  get_wallet_balance: { readOnly: true, destructive: false, audience: "customer" },
  get_spend_envelope: { readOnly: true, destructive: false, audience: "customer" },
  estimate_job_cost: { readOnly: true, destructive: false, audience: "customer" },
  list_invoices: { readOnly: true, destructive: false, audience: "customer" },
  list_payment_methods: { readOnly: true, destructive: false, audience: "customer" },
  get_auto_topup: { readOnly: true, destructive: false, audience: "customer" },
  list_pending_verifications: { readOnly: true, destructive: false, audience: "customer" },
  top_up_wallet: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  configure_auto_topup: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  // Company knowledge, off by default. Both index the published documentation
  // site and the live marketplace, which change without a deploy on our side.
  // Repeat-safe by nature rather than by key: a second snapshot of the same
  // `name:tag` 409s with "already exists (delete it first to overwrite)", so
  // calling twice has no additional effect. Same shape as `register_ssh_key`.
  create_instance_snapshot: { readOnly: false, destructive: false, audience: "customer", idempotency: "keyed" },
  list_user_images: { readOnly: true, destructive: false, audience: "customer" },
  // Destructive: the record cannot be brought back through the API, and a
  // sweep or launch that referenced the image stops being reproducible.
  delete_user_image: { readOnly: false, destructive: true, audience: "customer", idempotency: "keyed" },
  // Preparing a sweep plan spends nothing, but each call creates another plan
  // awaiting approval — the same shape as `run_pipeline`, and classified the
  // same way. Executing an approved plan *is* repeat-safe; that is the execute
  // route's `mark_consumed`, not this tool.
  create_image_sweep: { readOnly: false, destructive: false, audience: "customer", idempotency: "none" },
  get_image_sweep: { readOnly: true, destructive: false, audience: "customer" },
  search: { readOnly: true, destructive: false, audience: "company-knowledge", openWorld: true },
  fetch: { readOnly: true, destructive: false, audience: "company-knowledge", openWorld: true },
};

export const COMPANY_KNOWLEDGE_TOOL_NAMES: ReadonlySet<string> = new Set(
  Object.entries(TOOL_POLICY).filter(([, p]) => p.audience === "company-knowledge").map(([n]) => n),
);


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
    // No fallback for writes. `?? "keyed"` is what published "retrying is
    // free" for 21 tools that never sent a key; the union makes the write
    // arm declare it, so there is nothing left to default to.
    const idempotency = policy.readOnly ? (policy.idempotency ?? "read") : policy.idempotency;
    return [name, {
      version: policy.version ?? BASE_TOOL_VERSION,
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
