export type McpScope =
  | "instances:read"
  | "instances:write"
  | "instances:operate"
  | "instances:connect"
  | "ssh:read"
  | "ssh:write"
  | "volumes:read"
  | "volumes:write"
  | "artifacts:read"
  | "billing:read"
  | "billing:write"
  | "gpu:read"
  | "marketplace:read"
  | "hosts:read"
  | "hosts:operate"
  | "hosts:evict"
  | "control_plane:read"
  | "control_plane:operate"
  | "mcp_actions:approve"
  | "events:read"
  | "inference:read"
  | "inference:write";

/**
 * A tool's authorization requirement.
 *
 * `allOf` — every scope is required (the default for multi-scope tools).
 * `anyOf` — any one suffices, for a subject that may live in several domains.
 */
export interface ScopeRequirement {
  allOf?: McpScope[];
  anyOf?: McpScope[];
}

/**
 * The tool registry. This object's **key set is the source of truth** for which
 * tools exist, and `ToolName` below is what every other per-tool table is keyed
 * by — policy, annotations, descriptions. Adding an entry here and nowhere else
 * is a compile error rather than a drift a test has to catch.
 *
 * Declared as a `const` with `satisfies` rather than annotated directly: an
 * annotation of `Record<string, ScopeRequirement>` would widen the key type to
 * `string` and `ToolName` would become useless, which is the failure mode that
 * makes an inversion like this quietly do nothing.
 */
const TOOL_SCOPE_REGISTRY = {
  list_available_gpus: { allOf: ["gpu:read"] },
  get_spot_prices: { allOf: ["marketplace:read"] },
  get_pricing_reference: { allOf: ["gpu:read"] },
  search_marketplace: { allOf: ["marketplace:read"] },
  list_tiers: { allOf: ["gpu:read"] },
  list_instances: { allOf: ["instances:read"] },
  get_instance: { allOf: ["instances:read"] },
  get_instance_logs: { allOf: ["instances:read"] },
  create_instance: { allOf: ["instances:write"] },
  cancel_instance: { allOf: ["instances:write"] },
  terminate_instance: { allOf: ["instances:write"] },
  should_i_run_this: { allOf: ["billing:read", "instances:read"] },
  run_training_job: { allOf: ["instances:write", "billing:read"] },
  schedule_under_budget: { allOf: ["instances:write", "gpu:read", "marketplace:read"] },
  watch_instance: { allOf: ["instances:read"] },
  // P2. Adding a key is the step that grants shell access, which is why
  // `ssh:write` is split from `ssh:read` and why its consent text names
  // shell access rather than "manage keys".
  register_ssh_key: { allOf: ["ssh:write"] },
  // P3 — durable state. Reads are `volumes:read`; anything that creates,
  // moves or destroys is `volumes:write`, and every destructive one is
  // confirm-gated in the handler as well.
  list_volumes: { allOf: ["volumes:read"] },
  get_volume: { allOf: ["volumes:read"] },
  create_volume: { allOf: ["volumes:write"] },
  // Promotion reads artifacts and writes a volume, so it needs both. The
  // read half is not decorative: without it a caller could copy artifacts it
  // may not read onto a volume it may write.
  promote_artifact_to_volume: { allOf: ["volumes:write", "artifacts:read"] },
  get_promotion_status: { allOf: ["volumes:read"] },
  // A pipeline requires the **union** of its stages' scopes, computed per
  // graph by the API. This entry is the floor for reaching the endpoint at
  // all; `instances:write` is what quoting a graph costs, and the route then
  // demands each stage's own scope before the plan is created.
  run_pipeline: { allOf: ["instances:write"] },
  get_pipeline_status: { allOf: ["instances:read"] },
  attach_volume: { allOf: ["volumes:write", "instances:read"] },
  detach_volume: { allOf: ["volumes:write"] },
  delete_volume: { allOf: ["volumes:write"] },
  snapshot_volume: { allOf: ["volumes:write"] },
  // The retention clock. Read-only, and the one tool that tells a user their
  // work is about to be deleted.
  get_artifact_expiry: { allOf: ["artifacts:read"] },
  // P2. The scope the consent screen describes as "open a terminal on your
  // running instances" — the same one `/api/terminal/ticket` and the stream,
  // expose and auto-launch routes enforce.
  open_instance_access: { allOf: ["instances:connect", "instances:read"] },
  list_serverless_endpoints: { allOf: ["inference:read"] },
  create_serverless_endpoint: { allOf: ["inference:write"] },
  should_i_run_pel_job: { allOf: ["billing:read", "inference:read"] },
  run_serverless_job: { allOf: ["inference:write"] },
  get_serverless_job_status: { allOf: ["inference:read"] },
  // The exits. GT0 found entrances everywhere and exits missing — an agent
  // could create an endpoint and run jobs on it with no tool to stop either.
  cancel_serverless_job: { allOf: ["inference:write"] },
  delete_serverless_endpoint: { allOf: ["inference:write"] },
  explain_instance_placement: { allOf: ["instances:read"] },
  simulate_instance_placement: { allOf: ["instances:read", "gpu:read"] },
  evaluate_placement_preference: { allOf: ["instances:read", "gpu:read"] },
  get_instance_timeline: { allOf: ["instances:read"] },
  get_active_lease: { allOf: ["instances:read"] },
  get_scheduler_health: { allOf: ["control_plane:read"] },
  get_host_capacity: { allOf: ["hosts:read"] },
  list_reconciliation_findings: { allOf: ["instances:read", "control_plane:read"] },
  get_mcp_action_status: { anyOf: ["instances:read", "inference:read", "hosts:read"] },
  retry_instance: { allOf: ["instances:operate"] },
  reconcile_instance: { allOf: ["instances:operate"] },
  drain_host: { allOf: ["hosts:operate"] },
  undrain_host: { allOf: ["hosts:operate"] },
  evict_host_workloads: { allOf: ["hosts:evict"] },
  retry_agent_command: { allOf: ["control_plane:operate"] },
  get_wallet_balance: { allOf: ["billing:read"] },
  // Reads the wallet *and* the running instances burning it down, so it needs
  // both. `instances:read` is not incidental here: the runway is meaningless
  // without knowing what is consuming it, and the per-instance breakdown names
  // the jobs that will be stopped.
  get_spend_envelope: { allOf: ["billing:read", "instances:read"] },
  estimate_job_cost: { allOf: ["billing:read"] },
  list_invoices: { allOf: ["billing:read"] },
  list_payment_methods: { allOf: ["billing:read"] },
  top_up_wallet: { allOf: ["billing:write"] },
  // Reading which charges stopped for a bank challenge. `billing:read`, not
  // write: it moves nothing and returns no `client_secret`. The resume route
  // that does return one is deliberately not on the tool surface.
  list_pending_verifications: { allOf: ["billing:read"] },
  configure_auto_topup: { allOf: ["billing:write"] },
  // The snapshot half of P7, and the reason the sweep was unusable without it:
  // `create_image_sweep` needs an `image_id` and nothing could produce or find
  // one. Listing is `instances:read`; taking or deleting a snapshot mutates the
  // tenant's images and is `instances:write`.
  create_instance_snapshot: { allOf: ["instances:write"] },
  list_user_images: { allOf: ["instances:read"] },
  delete_user_image: { allOf: ["instances:write"] },
  // P7 — the image sweep. Creating one only *quotes* it: the launch happens
  // when an approved plan is executed, and both halves are `instances:write`
  // because a caller who can quote a sweep must be able to run the one they
  // approved.
  create_image_sweep: { allOf: ["instances:write"] },
  get_image_sweep: { allOf: ["instances:read"] },
  // Company knowledge (optional, off by default). These read published
  // documentation and public pricing — no tenant data — so they take the
  // broad read set rather than a scope of their own. A dedicated scope would
  // make every existing connector token unable to read our own docs, for no
  // security gain over content anyone can already load in a browser.
  search: { anyOf: ["gpu:read", "marketplace:read", "instances:read", "inference:read", "billing:read"] },
  fetch: { anyOf: ["gpu:read", "marketplace:read", "instances:read", "inference:read", "billing:read"] },
} satisfies Record<string, ScopeRequirement>;

/**
 * Every registered tool name, as a union type.
 *
 * This is what makes the other tables complete by construction rather than by
 * inspection. `Record<ToolName, …>` in `contracts.ts` and `descriptions.ts`
 * rejects both a missing tool and one that does not exist.
 */
export type ToolName = keyof typeof TOOL_SCOPE_REGISTRY;

/**
 * Scope requirements, indexed loosely on purpose.
 *
 * The enforcement call sites look up whatever name arrived on the wire, and an
 * unregistered name **must** read as `undefined` so `satisfiesScope` denies it.
 * Narrowing this export to `Record<ToolName, …>` would make that lookup a type
 * error and invite a cast, which is how a deny-by-default becomes an assertion
 * that the name is valid. Completeness is enforced where the tables are
 * *declared*; lookup stays permissive so an unknown tool is denied, not trusted.
 */
export const TOOL_SCOPES: Record<string, ScopeRequirement> = TOOL_SCOPE_REGISTRY;

/**
 * Whether a granted scope set satisfies one required scope.
 *
 * A scope is satisfied only by holding it. There is no wildcard and no broad
 * grant: `api` used to short-circuit every check here, which made scope
 * reduction decorative — a token issued for read-only automation could launch
 * instances. It is removed from the enum entirely rather than narrowed, so
 * there is no value a caller can hold that means "everything".
 */
function grants(granted: ReadonlySet<string>, scope: McpScope): boolean {
  return granted.has(scope);
}

/**
 * Does this principal satisfy a tool's scope requirement?
 *
 * The previous implementation was `required.some(...)`, which accepted **any
 * one** of a tool's scopes. Because every contract also listed `"api"`, and
 * `api` short-circuited to `true` before any check ran, a token holding only
 * `gpu:read` satisfied `schedule_under_budget`'s `instances:write`, and
 * `billing:read` alone satisfied `run_training_job`. Scope reduction — which
 * Quick Connect performs deliberately — had no effect on what a token could do.
 * Both the wildcard and the any-one-of semantics are gone.
 *
 * `allOf` is cumulative and is the default for a tool that genuinely needs
 * several scopes. `anyOf` is for a tool whose subject may live in one of
 * several domains, where holding any one of them is the correct test.
 */
export function satisfiesScope(
  userScopes: string[] | undefined,
  requirement: ScopeRequirement | undefined,
): boolean {
  if (!userScopes?.length || !requirement) return false;
  const hasAllOf = requirement.allOf?.length ?? 0;
  const hasAnyOf = requirement.anyOf?.length ?? 0;
  // An empty requirement is a definition error, not "open to everyone".
  if (!hasAllOf && !hasAnyOf) return false;

  const granted = new Set(userScopes);
  const all = requirement.allOf?.every((scope) => grants(granted, scope)) ?? true;
  const any = !hasAnyOf || requirement.anyOf!.some((scope) => grants(granted, scope));
  return all && any;
}

/** @deprecated Use {@link satisfiesScope}. Retained for existing call sites. */
export function userHasScope(
  userScopes: string[] | undefined,
  required: ScopeRequirement | undefined,
): boolean {
  return satisfiesScope(userScopes, required);
}

/** Flat union of a requirement's scopes, for metadata and advertising only. */
export function scopeUnion(requirement: ScopeRequirement | undefined): McpScope[] {
  return [...new Set([...(requirement?.allOf ?? []), ...(requirement?.anyOf ?? [])])];
}

/** Human-readable requirement, for denial messages. */
export function describeScopeRequirement(requirement: ScopeRequirement | undefined): string {
  if (!requirement) return "an unknown scope (tool has no contract)";
  const parts: string[] = [];
  if (requirement.allOf?.length) parts.push(`all of: ${requirement.allOf.join(", ")}`);
  if (requirement.anyOf?.length) parts.push(`one of: ${requirement.anyOf.join(", ")}`);
  return parts.join("; ") || "an empty requirement (definition error)";
}
