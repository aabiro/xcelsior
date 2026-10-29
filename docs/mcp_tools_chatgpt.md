# Xcelsior MCP Tool-Surface Research and Ship Recommendation

## Executive verdict

The July 29 audit is directionally strong but should **not** be implemented as one approximately 91-tool public connector. The better design is an **84-contract registry deployed as four stable surfaces**:

| Surface | Intended users | Tool count | Directory posture |
|---|---:|---:|---|
| Xcelsior Compute | Consumers | 62 | Public, flagship |
| Xcelsior Provider | Providers, including dual-role users | 18, including shared discovery reads | Separate public listing |
| Xcelsior Control Plane | Operators | 8 | Separate, unlisted |
| Xcelsior Knowledge | Documentation and public marketplace research | 2 | Optional separate read-only listing |

The 84 unique contracts are seven fewer than the prior 91-action target, while covering the missing supply-side lifecycle rather than treating “provider” as six payout functions. The endpoint inventory confirms that the actual backend is far broader than the curated OpenAPI document: 528 operations across 37 route modules, while the public OpenAPI file is intentionally selective. fileciteturn0file2 fileciteturn0file0

The most important conclusions are:

| Finding | Severity | Decision |
|---|---|---|
| `userHasScope` authorizes a tool when the principal has **any one** listed scope, and the broad `api` scope bypasses every narrower requirement | **Release blocker** | Replace scope arrays with explicit `allOf` and `anyOf`; stop treating `api` as authorization |
| Current destructive instance descriptions still use `confirm:true`, despite the stated requirement that destructive operations use approved server-bound plans | **Release blocker** | Put cancel, terminate, delete, detach, payout, reservation, and similar operations behind action plans |
| The prior budget counts operator tools inside a surface whose operators are explicitly served by a separate deployment | Architectural error | Count and evaluate each deployment independently |
| “Payouts / Connect” is not an adequate provider domain | Product gap | Add host registration, admission, listing, spot controls, reputation, SLA, earnings, and payout tools |
| Dynamic `tools/list` by principal is not defensible across directory clients | Compatibility risk | Publish stable, purpose-specific listings |
| URL elicitation is standardized, but client support remains inconsistent and often undocumented | Interoperability risk | Capability-negotiate it and always implement a conspicuously incomplete fallback |
| No evidence supports a universal degradation threshold such as “accuracy falls after exactly 50 tools” | Evidence gap | Keep active flat surfaces near 60 or below and gate changes with Xcelsior’s own selection eval |

The attached plan reports 37 registered tools, while the current descriptions and scope map contain 39 because `search` and `fetch` are present. That drift is itself evidence that tool inventory, descriptions, scopes, implementation registration, and directory snapshots need one generated source of truth. fileciteturn0file1 fileciteturn0file3 fileciteturn0file5

### Security blockers before expansion

The current scope helper is not a least-privilege implementation:

```ts
if (userScopes.includes("api")) return true;
return required.some((s) => userScopes.includes(s));
```

That means a contract such as `run_training_job: ["instances:write", "billing:read", "api"]` is accepted when the principal has only `billing:read`, only `instances:write`, or the broad `api` value. Conversely, `get_mcp_action_status` appears to need alternative resource scopes, not all of them. A single flat array therefore cannot express the policy correctly. fileciteturn0file5

The replacement contract should be structurally explicit:

```ts
type ScopeRequirement = {
  allOf?: McpScope[];
  anyOf?: McpScope[];
};

function satisfies(
  granted: ReadonlySet<McpScope>,
  requirement: ScopeRequirement,
): boolean {
  const all = requirement.allOf?.every((scope) => granted.has(scope)) ?? true;
  const any =
    !requirement.anyOf?.length ||
    requirement.anyOf.some((scope) => granted.has(scope));

  return all && any;
}
```

`api` should be an audience, token class, or resource-server marker—not a universal authority. Tool registration may still use a stable superset, but invocation must reject insufficient authority before parsing sensitive business inputs.

The current contract file also marks only `cancel_instance`, `terminate_instance`, and `evict_host_workloads` destructive, while the expanded surface introduces additional irreversible actions: volume deletion and snapshot restoration, serverless endpoint deletion, payment-method removal, reservation cancellation, provider disconnection, host removal, team deletion, member removal, and privacy erasure. The annotation set must therefore be generated from the same policy registry that determines approval, scopes, idempotency, and retry behavior. fileciteturn0file4

## Recommended candidate surface

### Budget comparison

| Domain | Current | July target | Recommended | Judgment |
|---|---:|---:|---:|---|
| Discovery and pricing | 5 | 8 | 6 | Merge overlapping price reads |
| Instance lifecycle | 13 | 14 | 11 | Replace thin operations with journey tools |
| Serverless and inference | 5 | 10 | 9 | Preserve endpoint, batch, and job journeys |
| Storage and artifacts | 0 | 5 | 6 | Artifacts belong with workload data |
| Billing and payments | 3 | 13 | 11 | Hosted browser flows are prerequisites for two candidates |
| Payouts / Connect | 0 | 6 | Included in 12-tool provider domain | The old domain is too narrow |
| Monitoring and events | 4 | 8 | 5 | `watch_instance` already composes several reads |
| Operator control plane | 7 | 10 | 8 | Separate deployment, not public budget |
| Compliance and residency | 0 | 6 | 6 | Keep customer-relevant, auditable journeys |
| Teams and access | 0 | 5 | 5 | Complete enough without exposing auth internals |
| Meta | 1 | 6 | 3 | Six meta tools would be surface inflation |
| Provider supply operations | 0 | Not budgeted | 7 of the 12 provider tools | Critical omission in the old budget |
| Company knowledge | 2 | Not budgeted | 2 | Separate read-only listing |
| **Unique contracts** | **39** | **91** | **84** | Better coverage with less flat-surface pressure |

Every endpoint below is taken from the full inventory, not inferred from a route name or from the curated public OpenAPI file. fileciteturn0file2

Scope notation uses `allOf(...)` for cumulative requirements and `anyOf(...)` for alternatives. Scopes prefixed **NEW** do not exist in the attached enum and must be added deliberately rather than smuggled through `api`.

### Consumer discovery, compute, inference, storage, and spend

| `tool_name` | `user_class` | `domain` | `endpoints_wrapped` | `journey_it_closes` | `read_or_write` | `required_scope` | `needs_approval` | `elicitation_mode` | `rationale` |
|---|---|---|---|---|---|---|---|---|---|
| `list_available_gpus` | consumer, provider | discovery | `GET /api/v2/gpu/available` fileciteturn0file2 | See live GPU models, regions, VRAM, rates, and quantity | read | `gpu:read` | no | none | Keep current tool; live inventory is a primary selection input |
| `search_marketplace` | consumer, provider | discovery | `POST /api/v2/marketplace/search` fileciteturn0file2 | Find capacity matching GPU, region, trust, and price constraints | read | `marketplace:read` | no | none | One constrained search is better than several filter-specific tools |
| `get_market_price_history` | consumer, provider | discovery | `GET /api/v2/marketplace/spot-prices`; `GET /api/v2/marketplace/spot-prices/{gpu_model}/history` fileciteturn0file2 | Compare current spot rates with recent movement | read | `marketplace:read` | no | none | Merges two closely related price reads |
| `get_pricing_reference` | consumer | discovery | `GET /api/pricing/models`; `GET /api/pricing/reference` fileciteturn0file2 | Explain standard model pricing | read | `gpu:read` | no | none | Stable reference, distinct from a workload quote |
| `quote_workload_cost` | consumer | discovery | `POST /api/pricing/estimate`; `GET /api/pricing/rates`; `GET /api/pricing/spot-quote` fileciteturn0file2 | Produce an on-demand or spot quote with tax and duration assumptions | read | `allOf(billing:read, gpu:read)` | no | none | Replaces multiple price calculators with one user-visible quote |
| `list_reserved_pricing` | consumer | discovery | `GET /api/pricing/reserved-plans`; `GET /api/pricing/reservations` fileciteturn0file2 | Compare commitments and inspect existing reservations | read | `billing:read` | no | none | Keeps commitments visible without mixing creation into a read |
| `list_instances` | consumer | instance lifecycle | `GET /instances` fileciteturn0file2 | Find running, queued, stopped, or completed instances | read | `instances:read` | no | none | Keep current tool |
| `get_instance` | consumer | instance lifecycle | `GET /instance/{job_id}`; `GET /api/v1/instances/{job_id}`; `GET /api/v1/instances/{job_id}/control-plane` fileciteturn0file2 | Inspect one instance’s customer and control-plane state | read | `instances:read` | no | none | One coherent detail view avoids choosing among near-duplicate reads |
| `create_instance` | consumer | instance lifecycle | `POST /api/v1/placements/simulate`; `POST /api/v1/launch-plans`; `GET /api/v1/launch-plans/{plan_id}`; `POST /api/v1/launch-plans/{plan_id}/execute` fileciteturn0file2 | Simulate, plan, approve, and launch GPU capacity | write and spend | `allOf(instances:write, billing:read, gpu:read)` | yes | URL | This is the flagship capability; retain one plan/execute tool rather than REST-shaped phases |
| `run_training_job` | consumer | instance lifecycle | Launch-plan endpoints above; `GET /api/v1/instances/{job_id}`; `GET /instances/{job_id}/logs` fileciteturn0file2 | Launch a repository-based training run and return initial status and logs | write and spend | `allOf(instances:write, billing:read)` | yes | URL | High-value workflow abstraction, not redundant when it removes orchestration burden |
| `watch_instance` | consumer | instance lifecycle | `GET /api/v1/instances/{job_id}/events`; `GET /api/instances/{job_id}/telemetry`; `GET /instances/{job_id}/logs`; `GET /api/v1/instances/{job_id}/active-lease` fileciteturn0file2 | Wait for a phase while observing telemetry, logs, and lease health | read | `allOf(instances:read, events:read)` | no | none | Preserves the existing long-running monitoring journey |
| `open_instance_access` | consumer | instance lifecycle | `GET /instances/{job_id}/auto-launch`; `POST /instances/{job_id}/expose`; `POST /api/instances/{job_id}/stream-ticket` fileciteturn0file2 | Open an authenticated browser or terminal session without placing credentials in model output | write, bounded | `instances:operate` | no | URL | Must return a first-party, short-lived browser URL—not service passwords or bearer tokens |
| `set_instance_power_state` | consumer | instance lifecycle | `POST /instances/{job_id}/start`; `POST /instances/{job_id}/stop`; `POST /instances/{job_id}/restart`; `POST /instances/{job_id}/reset` fileciteturn0file2 | Start, stop, restart, or reset one instance | write; start resumes spend | `instances:operate` | yes | URL | A state enum is easier to select than four neighboring tools; reset must state data impact |
| `set_instance_lock` | consumer | instance lifecycle | `POST /instances/{job_id}/lock`; `POST /instances/{job_id}/unlock` fileciteturn0file2 | Protect or unprotect an instance from accidental changes | write, reversible | `instances:operate` | no | none | Useful guardrail with low selection ambiguity |
| `cancel_instance` | consumer | instance lifecycle | `POST /instances/{job_id}/cancel` fileciteturn0file2 | End a queued or running job and stop billing | destructive | `instances:write` | yes | URL | Current `confirm:true` behavior does not meet the new non-negotiable approval rule |
| `terminate_instance` | consumer | instance lifecycle | `POST /instances/{job_id}/terminate` fileciteturn0file2 | Permanently destroy an instance and local state | destructive | `instances:write` | yes | URL | Keep separate from cancel because irreversibility is materially different |
| `repair_instance` | consumer | instance lifecycle | `POST /api/v1/instances/{job_id}/retry`; `POST /api/v1/instances/{job_id}/reconcile`; `POST /instance/{job_id}/requeue` fileciteturn0file2 | Recover a failed or stale instance using an explicit `strategy` | write | `instances:operate` | retry: yes; reconcile: no | URL for retry | Consolidates recovery while preserving strategy-specific effects and idempotency |
| `list_inference_models` | consumer | serverless | `GET /api/inference/models/available`; `GET /api/v2/serverless/preset-token-pricing` fileciteturn0file2 | Choose a supported model and see its token price | read | `inference:read` | no | none | Model discovery belongs before endpoint creation |
| `list_serverless_endpoints` | consumer | serverless | `GET /api/v2/serverless/endpoints` fileciteturn0file2 | Find deployed inference endpoints | read | `inference:read` | no | none | Keep current tool |
| `get_serverless_endpoint` | consumer | serverless | `GET /api/v2/serverless/endpoints/{endpoint_id}`; `GET .../health`; `GET .../metrics`; `GET .../usage` fileciteturn0file2 | Inspect configuration, health, usage, and cost | read | `inference:read` | no | none | Four reads form one natural endpoint-inspection journey |
| `create_serverless_endpoint` | consumer | serverless | `POST /api/v1/serverless/endpoint-plans`; `POST /api/v1/serverless/endpoint-plans/{plan_id}/execute` fileciteturn0file2 | Plan and create a billable endpoint | write and spend | `allOf(inference:write, billing:read)` | yes | URL | Use the existing server-bound endpoint plan |
| `update_serverless_endpoint` | consumer | serverless | `PATCH /api/v2/serverless/endpoints/{endpoint_id}` fileciteturn0file2 | Change endpoint settings or capacity | write; may change spend | `inference:write` | yes when cost ceiling rises | URL when approval needed | Approval can be conditional on a canonical before/after cost delta |
| `delete_serverless_endpoint` | consumer | serverless | `DELETE /api/v2/serverless/endpoints/{endpoint_id}` fileciteturn0file2 | Permanently remove an endpoint | destructive | `inference:write` | yes | URL | Must not rely on a Boolean confirmation |
| `run_serverless_inference` | consumer | serverless | `POST /v1/serverless/{endpoint_id}/run`; `POST .../runsync`; `GET .../status/{job_id}`; `GET .../stream/{job_id}` fileciteturn0file2 | Submit inference and retrieve or stream the result | write and spend | `inference:write` | yes above a configured spend threshold | URL when approval needed | One tool can choose sync or async while reporting non-idempotent billing clearly |
| `cancel_serverless_job` | consumer | serverless | `POST /v1/serverless/{endpoint_id}/cancel/{job_id}`; `POST /api/v2/serverless/endpoints/{endpoint_id}/jobs/{job_id}/cancel` fileciteturn0file2 | Cancel an outstanding inference job | destructive/bounded | `inference:write` | yes | URL | Canonical wrapper hides duplicate aliases |
| `run_serverless_batch` | consumer | serverless | `POST /api/v2/serverless/endpoints/{endpoint_id}/batches`; `GET /api/v2/serverless/batches/{batch_id}` fileciteturn0file2 | Submit and monitor discounted bulk inference | write and spend | `allOf(inference:write, billing:read)` | yes | URL | Batch is a distinct cost and latency journey |
| `list_volumes` | consumer | storage | `GET /api/v2/volumes`; `GET /api/v2/volumes/available`; `GET /api/v2/volumes/{volume_id}` fileciteturn0file2 | Find volumes and inspect attachment readiness | read | **NEW** `storage:read` | no | none | Merges collection and detail reads |
| `create_volume` | consumer | storage | `POST /api/v2/volumes` fileciteturn0file2 | Create billable persistent storage | write and spend | **NEW** `storage:write` | yes | URL | Cost must be shown in the action plan |
| `set_volume_attachment` | consumer | storage | `POST /api/v2/volumes/{volume_id}/attach`; `POST .../detach` fileciteturn0file2 | Attach or detach a volume from an instance | write | `allOf(storage:write, instances:operate)` | detach: yes | URL for detach | Detachment can disrupt a workload and should be approved |
| `manage_volume_snapshots` | consumer | storage | `GET /api/v2/volumes/{volume_id}/snapshots`; `POST .../snapshots`; `DELETE .../snapshots/{snapshot_id}`; `POST .../restore` fileciteturn0file2 | List, create, delete, or restore snapshots | mixed | **NEW** `storage:write` | delete/restore: yes | URL | Restore is destructive to current state; expose an explicit operation enum |
| `delete_volume` | consumer | storage | `DELETE /api/v2/volumes/{volume_id}` fileciteturn0file2 | Permanently delete detached storage | destructive | **NEW** `storage:write` | yes | URL | Irreversible and potentially data-destroying |
| `transfer_artifacts` | consumer | storage and artifacts | `GET /api/artifacts`; `GET /api/artifacts/{job_id}`; `POST /api/artifacts/upload`; `POST /api/artifacts/finalize`; `POST /api/artifacts/download`; `GET /api/artifacts/{job_id}/expiry` fileciteturn0file2 | List, upload, finalize, or obtain a short-lived artifact download | mixed | **NEW** `artifacts:read/write` | no | URL for browser transfer | Presigned URLs should be short-lived and omitted from durable model-visible logs |
| `get_wallet_summary` | consumer | billing | `GET /api/billing/wallet/{customer_id}`; `GET .../depletion` fileciteturn0file2 | See balance, credits, and projected depletion | read | `billing:read` | no | none | Strong pre-spend guardrail |
| `list_wallet_transactions` | consumer | billing | `GET /api/billing/wallet/{customer_id}/history` fileciteturn0file2 | Explain deposits, charges, refunds, and adjustments | read | `billing:read` | no | none | Separate from balance because the user intent differs |
| `get_usage_summary` | consumer, provider | billing | `GET /api/billing/usage/{customer_id}`; `GET /api/analytics/usage`; `GET /api/analytics/enhanced` fileciteturn0file2 | Analyze compute use and spend over time | read | `billing:read` | no | none | One reporting tool with bounded date filters |
| `manage_invoices` | consumer | billing | `GET /api/billing/invoices/{customer_id}`; `GET /api/billing/invoice/{customer_id}`; `GET .../download` fileciteturn0file2 | List, generate, and retrieve an invoice document | read | `billing:read` | no | URL for document download | Avoid returning a large document inline |
| `list_payment_methods` | consumer | billing | `GET /api/billing/payment-methods` fileciteturn0file2 | View redacted saved payment methods | read | `billing:read` | no | none | Only brand, last four, expiry, and default status should enter context |
| `add_payment_method` | consumer | billing | `POST /api/billing/setup-intent` fileciteturn0file2 | Add a card without card data entering MCP | write | `billing:write` | user authorization | URL | **UNVERIFIED / blocked:** the endpoint creates a SetupIntent, not a safe browser page |
| `remove_payment_method` | consumer | billing | `DELETE /api/billing/payment-methods/{payment_method_id}` fileciteturn0file2 | Detach a saved method and explain auto-top-up effects | destructive | `billing:write` | yes | URL | Must show whether auto-top-up will be disabled |
| `deposit_wallet` | consumer | billing | `POST /api/billing/payment-intent`; PayPal create/capture; crypto and Lightning deposit/status/refresh endpoints fileciteturn0file2 | Fund the wallet through a selected supported rail | write and spend | `billing:write` | user authorization | URL | Card branch is blocked until Xcelsior hosts a payment page; PayPal and crypto require rail-specific completion checks |
| `open_billing_portal` | consumer | billing | `POST /api/billing/portal-session` fileciteturn0file2 | Open Stripe’s customer portal | write outside MCP | `billing:write` | user authorization | URL | Existing endpoint naturally returns a browser destination |
| `configure_auto_topup` | consumer | billing | `GET /api/v2/billing/auto-topup`; `POST /api/v2/billing/auto-topup` fileciteturn0file2 | Inspect, enable, disable, or modify automatic funding | write and future spend | `billing:write` | yes for enable or increased limits | URL | Plan must state threshold, amount, period cap, and payment method |
| `request_refund` | consumer | billing | `POST /api/billing/refund` fileciteturn0file2 | Request or process an eligible failed-job refund | write, money movement | `billing:write` | yes | URL | Approval and idempotency are required even when funds move toward the user |
| `reserve_capacity` | consumer | billing | `POST /api/pricing/reserve`; `POST /api/v2/marketplace/reservations`; `DELETE /api/v2/marketplace/reservations/{reservation_id}` fileciteturn0file2 | Create or cancel a reserved commitment | write and spend | `allOf(billing:write, marketplace:read)` | yes | URL | Commitment creation and early cancellation both have financial consequences |

Stripe documents that PaymentIntent and SetupIntent integrations pass a `client_secret` to browser-side Stripe code; Stripe cautions against logging or embedding that secret in URLs. Therefore, merely exposing Xcelsior’s existing intent endpoints through URL elicitation would not create a compliant browser flow. Xcelsior first needs a first-party hosted page that retrieves the intent server-side or from an authenticated browser session and renders Stripe’s Payment Element. citeturn13search0turn13search1turn13search8

### Consumer observability, compliance, teams, and meta

| `tool_name` | `user_class` | `domain` | `endpoints_wrapped` | `journey_it_closes` | `read_or_write` | `required_scope` | `needs_approval` | `elicitation_mode` | `rationale` |
|---|---|---|---|---|---|---|---|---|---|
| `get_instance_logs` | consumer | monitoring | `GET /instances/{job_id}/logs`; `GET .../logs/download` fileciteturn0file2 | Read a recent tail or obtain a full log file | read | `instances:read` | no | none or URL for download | Keep live waiting in `watch_instance` |
| `get_instance_timeline` | consumer | monitoring | `GET /api/v1/instances/{job_id}/timeline`; `GET .../attempts` fileciteturn0file2 | Explain retries, transitions, and failed attempts | read | `instances:read` | no | none | Combines durable attempt history |
| `get_instance_events` | consumer | monitoring | `GET /api/v1/instances/{job_id}/events`; `GET /api/events/{entity_type}/{entity_id}` fileciteturn0file2 | Page through resumable instance events | read | `allOf(instances:read, events:read)` | no | none | Activates the otherwise unused `events:read` scope |
| `get_instance_audit_trail` | consumer | monitoring | `GET /api/audit/instance/{job_id}`; `GET /api/v1/mcp/tool-audit` fileciteturn0file2 | Review infrastructure events and MCP actions affecting a workload | read | `allOf(instances:read, **NEW** mcp_audit:read)` | no | none | Supports the product’s auditability claim |
| `get_notifications` | consumer, provider | monitoring | Notification list, unread-count, mark-read, read-all, and delete endpoints under `/api/notifications` fileciteturn0file2 | Review and clear account notifications | mixed, non-destructive | **NEW** `notifications:read/write` | no | none | Push-subscription management remains browser-only |
| `search_residency_eligible_capacity` | consumer | compliance | `POST /api/jurisdiction/hosts` fileciteturn0file2 | Find hosts satisfying jurisdiction and trust constraints | read | `allOf(marketplace:read, **NEW** compliance:read)` | no | none | Residency must influence placement before launch |
| `get_residency_trace` | consumer | compliance | `GET /api/jurisdiction/residency-trace/{job_id}` fileciteturn0file2 | Produce an auditable trace of where a workload ran | read | `allOf(instances:read, **NEW** compliance:read)` | no | none | Strong differentiator for regulated workloads |
| `get_compliance_posture` | consumer, provider | compliance | `GET /api/compliance/status`; provinces, tax-rates, trust-tier-requirements; `GET /api/trust-tiers`; `GET /api/billing/attestation` fileciteturn0file2 | Answer procurement, tax, trust, and platform-control questions | read | **NEW** `compliance:read` | no | none | One evidence-oriented tool is better than six catalog reads |
| `check_quebec_transfer_pia` | consumer | compliance | `POST /api/compliance/quebec-pia-check` fileciteturn0file2 | Determine whether a proposed cross-border transfer needs a Québec PIA | read/evaluation | **NEW** `compliance:read` | no | none | A concrete compliance decision tool |
| `get_data_retention_posture` | consumer | compliance | `GET /api/privacy/config/{org_id}`; `GET /api/privacy/retention-policies`; `GET /api/privacy/retention-summary` fileciteturn0file2 | Explain current retention configuration and outstanding data | read | **NEW** `privacy:read` | no | none | Keep configuration writes out until an organization policy model exists |
| `request_privacy_action` | consumer | compliance | `GET /api/auth/me/data-export`; `POST /api/v2/privacy/erase`; `GET /api/v2/privacy/erase/{request_id}` fileciteturn0file2 | Request export or erasure and track completion | mixed; erasure destructive | **NEW** `privacy:write` | erasure: yes | URL | Export should download in-browser; personal data must not be copied into the model transcript |
| `list_teams` | consumer | teams | `GET /api/teams/me`; `GET /api/teams/{team_id}` fileciteturn0file2 | List organizations, members, and roles | read | **NEW** `teams:read` | no | none | Complete account context without exposing auth internals |
| `create_team` | consumer | teams | `POST /api/teams` fileciteturn0file2 | Create an organization and owner membership | write | **NEW** `teams:write` | no | none | Bounded and reversible through separately approved deletion |
| `set_active_team` | consumer | teams | `PATCH /api/teams/active` fileciteturn0file2 | Switch wallet, concurrency, and job context | write, reversible | **NEW** `teams:write` | no | none | Description must warn that subsequent actions use the selected team |
| `manage_team_member` | consumer | teams | `POST /api/teams/{team_id}/members`; `PATCH .../{email}`; `DELETE .../{email}` fileciteturn0file2 | Add, change role, remove, or leave | write | **NEW** `teams:write` | role escalation/removal: yes | URL | Explicit operation and before/after role are required |
| `delete_team` | consumer | teams | `DELETE /api/teams/{team_id}` fileciteturn0file2 | Permanently delete an owned team | destructive | **NEW** `teams:write` | yes | URL | Separate tool prevents accidental selection through a generic manager |
| `get_action_plan_status` | all | meta | `GET /api/v1/launch-plans/{plan_id}` and corresponding serverless plan state fileciteturn0file2 | Check whether a prepared action is pending, approved, expired, executed, or revoked | read | `anyOf(instances:read, inference:read, billing:read, hosts:read)` | no | none | Retains current status journey with correct alternative scopes |
| `revoke_action_plan` | all | meta | `POST /api/v1/launch-plans/{plan_id}/revoke` fileciteturn0file2 | Make an unused plan permanently unexecutable | write, protective | Scope matching plan owner and action domain | no | none | Revocation is a security control, not a destructive business action |
| `get_platform_capabilities` | all | meta | `GET /api/status`; `GET /api/v2/serverless/enabled`; payment-rail enabled probes; `GET /api/v1/openapi.json` metadata only fileciteturn0file2 | Discover enabled product capabilities and outages | read | Narrow authenticated base access | no | none | One capabilities tool is sufficient; six meta tools are not justified |

Do **not** expose `POST /api/v1/launch-plans/{plan_id}/approve` as a model-callable public tool. The unused `mcp_actions:approve` scope should belong to a human-bound approval session or first-party approval UI. Wiring it to the same agent that prepared the plan would collapse the separation the approval mechanism is meant to create. The scope’s current lack of a tool is therefore not automatically a defect. fileciteturn0file1 fileciteturn0file5

### Provider and operator surfaces

| `tool_name` | `user_class` | `domain` | `endpoints_wrapped` | `journey_it_closes` | `read_or_write` | `required_scope` | `needs_approval` | `elicitation_mode` | `rationale` |
|---|---|---|---|---|---|---|---|---|---|
| `get_provider_account` | provider | provider operations | `GET /api/providers`; `GET /api/providers/{provider_id}` fileciteturn0file2 | Inspect the caller’s provider profile and payout readiness | read | **NEW** `providers:read` | no | none | Must tenant-filter despite the endpoint’s admin behavior |
| `onboard_provider` | provider | provider operations | `POST /api/providers/register`; `POST .../resume-onboarding`; `POST .../abandon-onboarding`; `POST .../account-session` fileciteturn0file2 | Register and complete or resume provider onboarding | write | **NEW** `providers:write` | user authorization | URL | One lifecycle tool can issue Stripe-hosted or Xcelsior-hosted onboarding |
| `get_provider_onboarding_status` | provider | provider operations | `GET /api/providers/{provider_id}`; `GET .../paypal`; `POST .../paypal/refresh`; `GET /api/connect/accounts/{account_id}/status` fileciteturn0file2 | Check KYC, payout, and missing-requirement state | read plus safe refresh | **NEW** `providers:read` | no | none | Never infer completion from the browser return alone |
| `disconnect_payout_account` | provider | payouts | Stripe and PayPal disconnect endpoints under `/api/providers/{provider_id}` fileciteturn0file2 | Disconnect the selected payout processor | destructive | **NEW** `payouts:write` | yes | URL | Must explain payout interruption and re-onboarding requirement |
| `get_provider_earnings` | provider | payouts | `GET /api/providers/{provider_id}/earnings` fileciteturn0file2 | See accrued earnings and payout history | read | **NEW** `payouts:read` | no | none | Core provider value |
| `request_provider_payout` | provider | payouts | `POST /api/providers/{provider_id}/payout` fileciteturn0file2 | Settle an eligible job or request payout | money movement | **NEW** `payouts:write` | yes | URL | Must bind job, amount, currency, destination state, and idempotency key |
| `inspect_provider_hosts` | provider | provider supply | `GET /hosts`; `GET /host/{host_id}`; reputation detail/history/breakdown; SLA record and violations endpoints fileciteturn0file2 | Review owned hosts, reputation, uptime, and violations | read | `allOf(hosts:read, **NEW** providers:read)` | no | none | Provider quality and revenue are inseparable |
| `register_or_update_host` | provider | provider supply | `POST /api/hosts/register`; `PUT /host` fileciteturn0file2 | Register a new host or update owned host metadata | write | **NEW** `hosts:write` | yes for material capacity/network changes | URL when needed | Self-reported data remains untrusted until admission |
| `remove_provider_host` | provider | provider supply | `DELETE /host/{host_id}` fileciteturn0file2 | Retire an owned host | destructive | **NEW** `hosts:write` | yes | URL | Must refuse while allocations or unpaid obligations remain |
| `publish_marketplace_capacity` | provider | provider supply | `POST /api/v2/marketplace/offers`; `POST /marketplace/list`; `DELETE /marketplace/{host_id}` fileciteturn0file2 | Publish, update, or unpublish admitted capacity | write | `allOf(**NEW** hosts:write, **NEW** marketplace:write)` | unpublish with running commitments: yes | URL conditionally | This, not payout alone, is the supply-side flagship |
| `configure_host_spot_pricing` | provider | provider supply | `GET /api/hosts/{host_id}/spot-preview`; `PATCH .../spot-settings` fileciteturn0file2 | Preview and set provider floor and spot controls | write, revenue impact | **NEW** `marketplace:write` | yes | URL | Plan should show expected rate band and current allocations |
| `manage_host_admission` | provider | provider supply | `GET /api/hosts/{host_id}/admission`; `POST .../compatibility-sessions`; `POST .../evidence`; `POST .../provider-evidence` fileciteturn0file2 | Open a compatibility session, submit evidence, and inspect outstanding requirements | mixed | **NEW** `hosts:write` | no | none | Operator-signed authoritative evidence remains excluded |
| `get_scheduler_health` | operator | control plane | `GET /api/v1/control-plane/health` fileciteturn0file2 | Diagnose platform-wide control-plane health | read | `control_plane:read` | no | none | Existing operator tool |
| `get_placement_queue` | operator | control plane | `GET /api/v1/control-plane/queue` fileciteturn0file2 | Inspect queued instances and placement blockers | read | `control_plane:read` | no | none | Missing from the current operator journey |
| `get_host_capacity` | operator | control plane | `GET /api/v1/hosts/{host_id}/capacity` fileciteturn0file2 | Inspect redacted capacity and allocations | read | `hosts:read` | no | none | Existing operator tool |
| `get_host_observations` | operator | control plane | `GET /api/v1/hosts/{host_id}/observations` fileciteturn0file2 | Inspect recent worker-reported host observations | read | `hosts:read` | no | none | Useful before drain, admission, or eviction decisions |
| `list_reconciliation_findings` | operator | control plane | `GET /api/v1/control-plane/reconciliation-findings` fileciteturn0file2 | Review durable state inconsistencies | read | `control_plane:read` | no | none | Existing operator tool |
| `set_host_drain_state` | operator | control plane | `POST /api/v1/hosts/{host_id}/drain`; `POST .../undrain` fileciteturn0file2 | Stop or resume new placements | write, reversible | `hosts:operate` | no | none | One explicit state tool reduces naming ambiguity |
| `evict_host_workloads` | operator | control plane | `POST /api/v1/hosts/{host_id}/eviction-plans`; `POST .../{plan_id}/execute`; `POST .../evictions` fileciteturn0file2 | Prepare and execute workload eviction | destructive | `hosts:evict` | yes | URL | Only execute an approved server-bound eviction plan |
| `retry_control_plane_command` | operator | control plane | `POST /api/v1/control-plane/commands/{command_id}/retry` fileciteturn0file2 | Redeliver a failed or dead-lettered command | write | `control_plane:operate` | yes if underlying command is destructive or spending | URL conditionally | Approval inherits the effect classification of the command |
| `search` | all, optional | knowledge | Existing indexed public documentation and marketplace sources fileciteturn0file3 | Find relevant Xcelsior documentation | read | Public-knowledge scope or no tenant scope | no | none | Keep outside the operational connector |
| `fetch` | all, optional | knowledge | Existing document retrieval implementation fileciteturn0file3 | Retrieve a document returned by `search` | read | Public-knowledge scope or no tenant scope | no | none | Separate listing prevents generic names from competing with operational tools |

## Better alternative and competitive extensions

### The surface I would ship

I would ship the 84-contract split above, not the budgeted flat surface.

| Criterion | July budget | Recommended split surface |
|---|---|---|
| Consumer flagship remains central | Yes | Yes |
| Full provider supply journey | No; primarily payouts | Yes |
| Flat tool competition | Approximately 91 | 62 maximum on flagship consumer listing |
| Operator exposure | Mixed into overall budget | Separate unlisted deployment |
| Frozen directory snapshots | Vulnerable to principal-dependent subsetting | Stable per listing |
| Dual-role user | One connector, theoretically variable | Same account authorizes two purpose-specific connectors |
| Scope semantics | Flat scope arrays | Explicit `allOf` / `anyOf` |
| Destructive approval | Mixed action plans and Boolean confirms | Uniform server-bound plans |
| Payment flow readiness | Assumes intent endpoints can become URLs | Blocks card tools until a hosted browser flow exists |
| Company knowledge | Competes with operational tools | Separate read-only surface |

What this gives up is the superficial convenience of “install one connector and see everything.” What it gains is reviewability, stable directory behavior, stronger model selection, independent release cadence, safer scope grants, and a provider experience that is actually about supplying compute.

### Capabilities absent from the old budget

| Rank | Capability idea | Endpoints that make it real | Why it widens the gap |
|---:|---|---|---|
| 1 | **Residency-locked training launch**: search eligible hosts, produce a PIA decision, simulate placement, create a plan, and later issue a residency trace | `POST /api/jurisdiction/hosts`; `POST /api/compliance/quebec-pia-check`; `POST /api/v1/placements/simulate`; launch-plan endpoints; `GET /api/jurisdiction/residency-trace/{job_id}` fileciteturn0file2 | Turns compliance from documentation into an executable workload policy |
| 2 | **Checkpoint-aware spot migration workflow**: inspect spot history, create snapshot, stop, relaunch on cheaper capacity, and verify the new placement | Spot-price history; volume snapshot endpoints; instance power endpoints; launch plans; placement explanation fileciteturn0file2 | Makes agent-managed cost optimization concrete rather than advisory |
| 3 | **Provider yield optimizer**: combine admission state, reputation, SLA, current spot preview, and marketplace statistics to recommend a floor and publish it after approval | Host admission, reputation, SLA, spot-preview/settings, marketplace offers/stats endpoints fileciteturn0file2 | Gives small providers an automated revenue manager |
| 4 | **Procurement evidence packet**: compile supplier attestation, compliance posture, trust-tier requirements, invoice history, residency controls, and audit references into a browser-downloadable package | `GET /api/billing/attestation`; compliance reads; invoices; residency trace; MCP audit endpoints fileciteturn0file2 | Reduces enterprise security and procurement friction |
| 5 | **Failure-to-refund resolution**: collect timeline, logs, lease health, event chain, billed amount, and refund eligibility; prepare a refund plan when the evidence supports it | Instance timeline/logs/lease/events; billing usage/invoice; `POST /api/billing/refund` fileciteturn0file2 | Converts a multi-team support case into an auditable agent workflow |
| 6 | **Workload portability package**: snapshot storage, export artifacts, record image/template metadata, and generate a relaunch specification | Volume snapshot endpoints; artifact endpoints; `POST /instances/{job_id}/snapshot`; user-image reads fileciteturn0file2 | Reduces lock-in while making Xcelsior the easiest marketplace to move across |
| 7 | **Capacity admission copilot**: open a compatibility session, collect signed helper evidence, show missing requirements, and prepare the host for human admission | Compatibility-session and evidence endpoints; admission state; host registration fileciteturn0file2 | Makes provider onboarding a guided technical workflow instead of a dashboard checklist |
| 8 | **Spend runway monitor**: combine wallet depletion, active-instance rates, serverless usage, reservations, and auto-top-up limits into a projected exhaustion alert | Wallet depletion/history; pricing rates; instance list; serverless metrics; reservation and auto-top-up endpoints fileciteturn0file2 | Enables continuous financial governance, not just point-in-time quotes |

These are capabilities rather than necessarily separate permanent tools. Several should become orchestrated workflows over the proposed primitives after selection and safety evals prove that a dedicated name improves completion.

## Exclusions and defensibility

The inventory below is partitioned by route module. In each row, every operation not cited in the candidate table is deliberately excluded under the listed rule. Prefix notation covers every matching inventory row; it is not a proposal to expose arbitrary paths. fileciteturn0file2

| Module | Deliberately non-exposed endpoint classes | Reason |
|---|---|---|
| `action_plans` | `POST /api/v1/launch-plans/{plan_id}/approve`; direct generic approval by the model | Approval must remain human-bound; plan create/status/execute/revoke are exposed only through domain tools |
| `admin` | All 26 operations | Cross-tenant or platform administration; permanently excluded |
| `agent` | All 18 worker-agent command, heartbeat, registration, and callback operations | Internal control channel; exposing it would bypass orchestration and trust boundaries |
| `agent_v2` | All 10 worker-agent operations | Same internal-plane exclusion |
| `artifacts` | No raw presigned-session primitive; only the curated `transfer_artifacts` journey | Prevent URL leakage, abandoned upload sessions, and thin REST mirroring |
| `auth` | Login, registration, password reset, email verification, OAuth callbacks, token endpoints, introspection, JWKS, client registration, secret rotation, agent/API key management, sessions, avatar/profile writes, demo credentials, quick-connect | Authentication protocol and browser settings are not model tools; several return or rotate secrets. Only privacy export is used through URL mode |
| `autoscale` | All six autoscaler configuration and execution operations | Platform policy; possible future operator surface only after separate review |
| `billing` | Direct wallet deposit; free-credit claim/status; reset-testing; bill-one/bill-all; platform billing list; PayPal webhook and marketplace settlement internals; raw intent response surfaces | Direct wallet deposit bypasses payment authority; test/admin/internal settlement endpoints are not tenant tools; client secrets must not enter context |
| `chat` | All 12 operations | Recursive AI/chat layer with no need in an MCP infrastructure server; increases prompt-injection and confused-deputy surface |
| `cloudburst` | The cloudburst operation | Cross-cloud capacity control requires a separate architecture and cost-policy review |
| `compliance` | Platform-wide GST threshold and automatic province detection as standalone tools | Platform-wide financial status is operator-only; request-header location inference is not authoritative enough to drive a tool |
| `control_plane_v1` | `POST /api/v1/mcp/tool-audit`; activation-funnel analytics; direct unplanned `/evictions`; raw versioned OpenAPI retrieval as model content | Audit writes are server middleware responsibilities; analytics are internal; eviction must use plans; schema is not an end-user journey |
| `events` | Global `GET /api/events`; global `GET /api/audit/verify-chain`; host event history on public listings | Cross-tenant platform visibility; tenant instance audit remains exposed |
| `gpu` | None beyond the curated read | The sole endpoint is exposed |
| `health` | Root page; legacy auth; dashboards and HTML; liveness/readiness/startup probes; raw metrics; NFS config; alert config writes; Slurm bridge; build endpoints; SSH key generation/public keys; token generation; stream endpoint | Probes are for infrastructure, HTML is not a tool, metrics can leak topology, key/token operations violate the secret boundary, Slurm/build need separate products |
| `host_admission` | Admin admission queue, admission decisions, authoritative evidence | Operator-only trust anchor; providers may submit advisory evidence but cannot admit themselves |
| `hosts` | Global host list/check in consumer listing; legacy host drain/undrain; maintenance readiness as a separate tool; compute-score global list | Cross-tenant fleet state or legacy duplicates; provider-owned and operator-safe variants are curated |
| `inference` | Direct model-download/cache/admin operations and any endpoint that emits credentials | Internal model management or secret-bearing behavior; only available-model discovery is exposed |
| `instances` | Admin shell reinjection; scheduler processing; failover; generic status patch; internal auto-launch and port reports; internal route resolver; queue processor; raw SSE log stream; direct image-template callbacks; API image mutations not incorporated into a reviewed journey | Admin/internal callbacks and raw control paths; status mutation bypasses state machine; streaming handled by `watch_instance` |
| `jurisdiction` | Queue processors; Canada-mode toggles; raw `/hosts/ca`; `/canada` toggle/read as a separate tool | Operator scheduling policy and legacy geographic shortcuts; use explicit constraints instead |
| `marketplace` | Direct allocation/release; bill-job; platform stats; legacy GET search duplicate | Scheduler and settlement internals or aliases; provider offer and consumer search journeys are exposed |
| `mfa` | All 16 setup, challenge, recovery, and verification operations | Authentication ceremony belongs in trusted first-party browser UI; recovery data must never enter model context |
| `notifications` | Push-subscription create/read/delete | Browser/device-specific state; normal notification reads and acknowledgments are exposed |
| `platform` | Platform configuration mutation, internal metadata, and cross-tenant operational endpoints | Separate platform administration boundary |
| `privacy` | Organization privacy config write; consent writes/revocation through raw primitives; purge-expired maintenance | Policy administration needs a dedicated organization-governance model; purge is a scheduled internal job |
| `providers` | Webhook receiver; incorporation upload-link primitive; arbitrary admin-visible provider listing | Webhooks are inbound infrastructure; document handling should use a future reviewed provider-document journey; tenant filtering is mandatory |
| `reputation` | Global leaderboard as an operational tool; milestone claim; admin verify; global entity lookups in consumer connector | Gamification and admin verification do not close compute journeys; provider-owned quality reads remain curated |
| `serverless` | Duplicate `/api/v2/inference` aliases; all worker heartbeat/claim/complete/event callbacks; key creation that returns a secret; raw key list/revoke until metadata-only response is guaranteed; OpenAI-compatible chat/embedding/model aliases as separate tools; slug aliases; dashboard test duplicates; raw worker log streams | Eliminate aliases and internal worker protocol; never put generated endpoint keys in model context; wrap one canonical execution route |
| `sla` | Global downtime and host summaries; enforcement write; global target catalog as standalone tool | Operator/global data or scheduled enforcement; provider-owned SLA is included in host inspection |
| `spot` | Direct scheduler or market-maker controls not represented by provider-owned preview/settings | Platform pricing policy must remain operator-owned |
| `ssh` | All three key or shell-related endpoints | Raw shell execution is permanently excluded; keys and credentials must not enter context |
| `stripe_connect_v2` | Webhook; HTML dashboard/storefront/success pages as tools; platform product create/list; destination-charge checkout unrelated to provider payout onboarding; unrestricted account list/create | Browser pages are destinations, not tools; commerce/catalog and webhooks are separate products; provider onboarding uses tenant-bound endpoints |
| `teams` | Invitation-token GET/POST acceptance as model calls | Tokens may be bearer capabilities; acceptance should occur in an authenticated browser flow |
| `terminal` | The terminal operation | Raw shell execution is permanently excluded |
| `transparency` | Raw legal-request create/respond operations | Legal process requires counsel/operator authority; a future public aggregate report can be considered separately |
| `verification` | All six direct verification operations | Identity and evidence verification are browser/operator ceremonies, not autonomous model actions |
| `volumes` | Admin encrypted-volume reopen; raw retry as a permanent tool; standalone rename | Admin recovery is operator-only; retry can be folded into support/repair later; rename does not justify selection cost |

### Permanent exclusion tests

An endpoint stays excluded when any of these is true:

| Test | Result |
|---|---|
| Cross-tenant or platform-global authority | Operator deployment or no MCP exposure |
| Worker callback, webhook receiver, queue processor, reconciliation scheduler, or health probe | Internal only |
| Returns or accepts passwords, private keys, API keys, card data, access tokens, client secrets, or standing service credentials | Browser ceremony or no exposure |
| HTML renderer or dashboard page | May be a URL destination, never a tool |
| Raw shell, terminal, SSH key generation, arbitrary command execution | Permanently excluded |
| Alias of an exposed canonical operation | Excluded to reduce selection ambiguity |
| Direct mutation bypasses an existing plan, state machine, scheduler, or settlement authority | Excluded |
| User-visible journey cannot be stated independently of its HTTP mechanism | Merge into a higher-level tool or leave out |
| Destructive or spending operation lacks a server-bound action-plan endpoint | Blocked until the API gains one |

## Provider boundary and client behavior

### Client-by-client assessment

A directory connector should be assumed to have a **stable reviewed tool schema**, unless the client explicitly documents per-principal dynamic discovery.

| Client | Can `tools/list` safely vary by principal for a directory-listed connector? | URL elicitation evidence | Conclusion |
|---|---|---|---|
| Claude / Claude Desktop | **UNVERIFIED and not defensible for directory design.** Anthropic documents that users authenticate individually and can enable specific tools, but does not document that a reviewed listing may expose materially different schemas per user. The Anthropic API MCP connector currently states that only MCP tool calls are supported, which excludes elicitation there. citeturn9search3turn9search14 | No official Claude.ai/Claude Desktop documentation found confirming URL mode support | Use a stable listing; test capability negotiation in each Claude host |
| ChatGPT | **No for reviewed workspace deployment.** OpenAI documents that, after approval, ChatGPT uses a frozen snapshot of tools and inputs; later server changes are not applied until an administrator refreshes and publishes them, and schema mismatches can cause calls to fail. citeturn7view2turn8search13 | No official ChatGPT documentation found confirming `elicitation.url`; interactive Apps SDK UI is not proof of protocol URL elicitation | Principal-varying registration is incompatible with the documented snapshot model |
| Gemini Enterprise | **Do not rely on it.** Custom MCP is still a governed data-store integration, and Google documents administrative enablement and metadata/tool configuration rather than a per-user variable listing contract. In related Google MCP tooling, tool schemas are represented and managed as toolsets and overrides; some integrations require listing without bearer authentication. citeturn12search0turn12search1turn12search2turn12search7 | No official Gemini Enterprise URL-elicitation support documentation found | Stable toolset per data store/listing |
| Microsoft Copilot Studio / M365 Copilot | Copilot Studio documents dynamic reflection when server tools are updated, but that is not evidence that principal-specific schemas are supported or governance-safe. Agent 365 review flows explicitly present declared tools for approval. citeturn17search6turn17search12 | Microsoft’s Copilot Cowork documentation explicitly says it supports form mode and **not yet** URL mode. citeturn8search6turn8search7 | Stable approved listing; do not infer per-user schema variation from refresh support |
| Grok | **UNVERIFIED.** xAI documents that Grok discovers a custom MCP server’s tools and that administrators provision connectors while users authenticate their accounts, but does not document principal-dependent tool-list schemas or frozen-snapshot semantics. citeturn15search0turn15search2 | No official xAI documentation found confirming URL elicitation | Stable listing and runtime authorization errors |

The correct answer is therefore not “all clients forbid dynamic lists” but “no cross-client documentation establishes it as safe, and ChatGPT directly documents a frozen schema.” That is enough evidence to reject shape B for a directory product.

### Recommended provider shape

**Recommend shape C: a separate directory listing at a separate URL.**

Use:

- `mcp.xcelsior.ca` for Xcelsior Compute.
- `provider-mcp.xcelsior.ca` for Xcelsior Provider.
- A separate unlisted operator host.
- The same Xcelsior identity account and authorization server across both public listings.

A person being both consumer and provider does not require one tool catalog. “User identity,” “account capabilities,” and “installed connector purpose” are separate concepts. The same person can authorize both listings, just as one person can use two applications backed by the same identity.

Shape A is acceptable as an interim implementation profile, but if the provider surface is strategically important it should become a real listing, not a hidden environment variable. Shape B has the worst combination: it introduces schema variance, complicates review and caching, exposes inaccessible tools if implemented as a superset, and still does not solve tool-selection pressure cleanly.

### Tool groups and tool-selection scale

There is currently **no stable MCP tool-group or toolset primitive in the core protocol**. The standard Tool object includes name, title, description, schemas, annotations, icons, and execution metadata, but no standardized group field. Community discussions and proposals for primitive groups exist; they are not a portable client convention. Google’s use of “McpToolset” is a product configuration object, not an MCP wire-level grouping standard. citeturn11search8turn11search11turn11search2turn12search7

There is also no defensible universal number at which flat tool selection “measurably degrades.” Published evidence establishes the problem, not one threshold:

- MCP-Atlas evaluates 220 tools and finds frontier-model pass rates only a little above 50%, with failures often involving tool use and task understanding. It does not isolate a universal tool-count breakpoint. citeturn10academia25
- A 2026 semantic-discovery study over 121 tools reported a 97.1% retrieval hit rate at top-three selection, supporting retrieval/gating for large catalogs but not proving that 50, 60, or 100 is the exact flat-surface failure point. citeturn10academia24
- OpenAI reported that tool search over all 36 MCP-Atlas servers reduced tokens by 47% at the same accuracy; that demonstrates context-efficiency benefits, not a universal accuracy cliff. citeturn10search0
- Microsoft’s Work IQ MCP deliberately exposes ten generic tools and discovers resource paths at runtime, explicitly following “fewer tools, more paths” and “introspection over enumeration.” That is relevant design evidence, although Xcelsior should not copy the approach so far that it creates arbitrary API-path tools. citeturn17search0

The practical shipping gate should be empirical:

| Active tools in test condition | Required evaluation |
|---:|---|
| Current 39 | Baseline intent-to-tool accuracy |
| 50 | Measure confusion among nearest semantic neighbors |
| 62 | Proposed consumer listing |
| 75 | Stress condition |
| 84 | Full unsplit registry, diagnostic only |
| 91 | Prior budget, diagnostic only |

Build at least 600 prompts: positive selections, no-tool prompts, adversarial near-neighbor prompts, missing-prerequisite prompts, and multi-step journeys. Measure exact first-tool selection, unsafe write selection, unnecessary tool calls, argument validity, refusal correctness, and end-to-end completion. I would ship the 62-tool consumer surface only if its first-tool accuracy is within two absolute percentage points of the best smaller condition and unsafe write selection does not regress.

## URL mode elicitation and payment architecture

### Specification state

SEP-1036 is a **Final Standards Track** SEP. URL mode is part of the MCP `2025-11-25` specification revision and is explicitly intended for sensitive credentials, third-party OAuth, and payment flows that must not transit the MCP client. citeturn18view0turn7view0

The capability contract is strict:

- A supporting client declares `capabilities.elicitation.url`.
- An empty elicitation capability means form mode only.
- A server must not send a mode the client did not advertise.
- An unsupported request should produce invalid-params behavior, not silent degradation.
- URL-mode `accept` means the user consented to navigation; it **does not** mean the external flow succeeded.
- Completion can be signaled through `notifications/elicitation/complete`, but clients must still permit manual retry or cancellation because the notification might never arrive.
- A server may return `URLElicitationRequiredError` code `-32042` with the required URL elicitations when the original operation cannot proceed. citeturn7view0

The attached project’s SDK 1.29.0 is consistent with the project’s plan to implement URL mode, and the official TypeScript SDK has implemented SEP-1036 support. SDK support is only server-side capability; it says nothing about whether a directory client displays the flow. fileciteturn0file1 citeturn1search3

### Client support summary

| Client | Confirmed behavior as of August 2, 2026 |
|---|---|
| Claude.ai / Claude Desktop | **UNVERIFIED** for URL mode in official public docs |
| Anthropic Messages API MCP connector | Tool calls only; elicitation not supported through that connector path citeturn9search14 |
| ChatGPT | **UNVERIFIED** for protocol URL mode in official public docs |
| Gemini Enterprise | **UNVERIFIED** for protocol URL mode |
| Microsoft Copilot Cowork | Form mode supported; URL mode explicitly “not yet” supported citeturn8search6turn8search7 |
| Grok | **UNVERIFIED** for protocol URL mode |
| Generic SDK/client ecosystem | Official MCP SDKs and some gateways implement the protocol, but that must not be conflated with directory-client support. Amazon’s AgentCore gateway, for example, documents both request- and exception-based URL modes with capability negotiation. citeturn14search2 |

### Correct fallback

When URL mode is not advertised, return an unmistakably incomplete result such as:

```json
{
  "status": "requires_user_action",
  "completed": false,
  "operation_executed": false,
  "reason": "client_does_not_support_url_elicitation",
  "action_type": "payment_method_setup",
  "browser_url": "https://pay.xcelsior.ca/start/opaque-one-time-id",
  "expires_at": "2026-08-02T18:20:00Z",
  "resume": {
    "tool": "add_payment_method",
    "arguments": {
      "setup_request_id": "psr_opaque"
    }
  }
}
```

The natural-language content should start with: **“Not completed. No payment method was added.”** It should never say “success,” “accepted,” “authorized,” or “done” merely because a URL was issued or opened.

The fallback URL must be a first-party Xcelsior URL with an opaque one-time identifier. It must not contain a Stripe client secret, payment credential, provider account details, email address, or other personal data. The server should poll or receive a webhook, verify the processor’s authoritative state, and only then return `completed: true`.

### Security requirements

The MCP specification requires the server to bind elicitation to both the authenticated user and originating client, not to a session ID alone. It prohibits sensitive data in the URL and prohibits pre-authenticated URLs that let possession of the link impersonate the user. Clients must show the full URL, obtain consent, avoid prefetching, and open it so the LLM and client cannot inspect page inputs. citeturn7view0

Xcelsior should enforce the following:

| Stage | Required validation |
|---|---|
| Before URL emission | HTTPS; exact allowlisted origin; no caller-controlled host, scheme, path template, or redirect; no credentials or PII in query/fragment; 128-bit or stronger opaque nonce; short TTL; one-time use; bind nonce to authenticated `sub`, client ID, action type, canonical amount/spec hash, and originating request |
| On browser arrival | Reauthenticate or verify an existing first-party session; never treat link possession alone as authentication; CSRF protection; SameSite and secure cookies; show canonical action details independently of model text |
| Before third-party redirect | Exact redirect allowlist; signed `state`; PKCE where applicable; no open redirect; no user-supplied return URL |
| On callback | Verify state, nonce, processor account, environment, amount/currency, expected object ID, and webhook signature; reject replay and mismatched user/client |
| On completion | Derive success from Stripe/PayPal/Xcelsior authoritative state, not the return redirect; atomically mark nonce consumed; send completion only to the originating client; preserve auditable event linkage |
| On retry | Reuse the same business operation and idempotency key; do not create a second charge, payout, endpoint, or instance |

Stripe’s Connect guidance reinforces that Account Link URLs are temporary and single-use, should be shown only to an authenticated account holder inside the platform, and should not be emailed or otherwise sent outside that application context. Stripe also states that return to the `return_url` does not prove that all onboarding requirements are complete. citeturn13search2turn13search3turn13search11

### Payout onboarding

URL elicitation is the correct **transport mechanism** for payout onboarding, but not the onboarding application itself.

For Stripe:

1. Xcelsior creates or identifies the tenant-bound Connect account.
2. The MCP server creates an opaque elicitation record.
3. URL elicitation points to an authenticated Xcelsior page.
4. That page creates a Stripe Account Link or hosts Stripe’s embedded onboarding component.
5. Stripe collects KYC, identity documents, bank details, and attestations directly.
6. Xcelsior validates account requirements through Stripe state and webhooks.
7. The MCP tool reports `pending_requirements`, `restricted`, or `enabled`; it never returns KYC content.

Stripe-hosted onboarding is specifically designed to collect business and identity-verification information, and Stripe advises platforms to rely on account requirements and backend state rather than assuming the redirect means completion. citeturn13search2turn13search4turn13search9

No verified public example was found of a production MCP directory server completing Stripe Connect onboarding through SEP-1036 and publishing a detailed postmortem. Therefore the answer to “what did shipped servers learn?” is **UNVERIFIED**. What would settle it is a public implementation, directory documentation, or operator report showing capability negotiation, browser flow, callback binding, completion notification, and failure handling. General libraries and gateways demonstrate implementation mechanics, not a shipped Stripe Connect case. citeturn14search2turn14search6turn14search8

## Naming, descriptions, and implementation gates

### Naming convention

Use stable snake_case verbs, with the user-class boundary expressed by the **server/listing**, not repeated on every public tool:

| Listing | Naming examples |
|---|---|
| Xcelsior Compute | `create_instance`, `quote_workload_cost`, `set_volume_attachment` |
| Xcelsior Provider | `onboard_provider`, `publish_marketplace_capacity`, `request_provider_payout` |
| Xcelsior Control Plane | `get_scheduler_health`, `set_host_drain_state`, `evict_host_workloads` |
| Xcelsior Knowledge | `search`, `fetch` |

Do not introduce names such as `consumer_create_instance`; the connector identity already supplies that namespace and the prefix consumes selection signal. Within the operator server, explicit host/control-plane nouns are enough to prevent ambiguity.

MCP’s current tool-name guidance allows case-sensitive ASCII names using letters, digits, underscore, hyphen, and dot, with uniqueness within a server. Snake case therefore remains interoperable. citeturn16search0turn16search1

Use these verb rules:

| Intent | Verb |
|---|---|
| Collection read | `list_` |
| One-resource read | `get_` |
| Constraint search | `search_` |
| Cost computation without effect | `quote_` or `estimate_` |
| New resource | `create_` |
| Explicit lifecycle state | `set_*_state` |
| Multi-operation journey with one object | `manage_` only when an `operation` enum is unavoidable and effects remain closely related |
| Irreversible removal | `delete_` or `terminate_` |
| Financial transfer | `deposit_`, `request_*_payout`, `request_refund` |
| Human browser handoff | Name the business outcome, not `open_url` |

Avoid `do_*`, `execute_*`, `process_*`, `handle_*`, generic `update`, and endpoint-shaped names. The tool name should identify the user’s intended outcome, while the action-plan phase remains an argument or returned state.

### Description template

The current descriptions already use a valuable three-part structure—concrete behavior, “Use when,” and cost/impact—and tests reportedly enforce it. fileciteturn0file3

Extend that to five required sentences:

> **Purpose:** Creates a persistent GPU volume of the requested size and region.  
> **Use when:** Use after choosing an instance region and when data must survive instance termination; do not use for temporary scratch space.  
> **Cost:** Bills the authenticated wallet at the quoted storage rate until deletion.  
> **Approval and effect:** This call first prepares a server-bound action plan; no volume exists and no charge begins until the exact plan is approved and executed.  
> **Refuses when:** Refuses if the region is unsupported, the wallet cannot cover the configured minimum runway, the principal lacks `storage:write`, or the approved plan no longer matches the request.

Every description should state:

| Required element | What the model needs |
|---|---|
| Purpose | The concrete user outcome |
| Positive trigger | When this tool should be selected |
| Negative trigger | The nearest tool or circumstance in which it should not be selected |
| Cost | Free, hourly, storage-time, per-token, one-time, commitment, or money movement |
| Effect and reversibility | What changes, when it begins, and how it can be undone |
| Approval semantics | Preparation versus execution; never imply `confirm` is approval |
| Refusal conditions | Missing scope, ownership, stale version, insufficient balance, unsupported client capability, invalid state, or policy restriction |
| Retry semantics | Safe, idempotency-key required, or never retry automatically |
| Data boundary | Whether output is redacted, browser-only, short-lived, or prohibited from model context |

Tool definitions and descriptions are part of the model’s selection input; the MCP client workflow explicitly sends tool descriptions to the model, so ambiguity here directly affects behavior. citeturn16search19

### Final implementation gates

| Gate | Pass condition |
|---|---|
| Authorization gate | `allOf` / `anyOf` implemented; `api` removed as universal bypass; tenant ownership independently enforced |
| Approval gate | Every destructive, spending, commitment, and money-movement candidate has a persisted canonical plan and human-bound approval |
| Secret gate | Automated tests prove card data, Stripe/PayPal secrets, endpoint keys, SSH material, service credentials, and presigned-session secrets cannot appear in tool results, logs, traces, audits, or error text |
| Idempotency gate | Every write accepts or derives a durable idempotency key; `run_serverless_inference` is not automatically retried until the backend provides deduplication |
| Snapshot gate | Each public listing has a stable generated manifest and content digest; schema changes require versioned review |
| URL gate | Capability negotiation tested for URL-supported, form-only, and no-elicitation clients; fallback always says `completed:false` |
| Description gate | Purpose, positive trigger, negative trigger, cost, effect, approval, refusal, retry, and data boundary present for every tool |
| Selection gate | 39/50/62/75/84/91-tool eval completed across supported model families and clients |
| Exclusion gate | Generated coverage report accounts for all 528 operations as exposed, wrapped, alias-excluded, browser-only, internal, operator-only, or permanently excluded |
| Directory gate | Consumer and provider manifests reviewed independently; no principal-dependent `tools/list`; operator surface unreachable from public connector credentials |

The candidate surface should not expand until the authorization bug, action-plan inconsistency, and hosted payment-flow gap are fixed. Once those gates pass, the split 84-contract design is the version I would ship.