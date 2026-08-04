/**
 * Tool descriptions, in one reviewed place.
 *
 * A directory reviewer calls every tool and compares what happened to what the
 * description promised, so descriptions are a correctness surface, not copy.
 * They live here rather than beside each `registerTool` call for the same
 * reason annotations do: one place to review, one place that can drift, and a
 * registration that throws when a tool has no entry.
 *
 * Every description follows the same three beats, because a model choosing
 * between 30 tools needs the trigger more than the mechanism:
 *
 *   1. What it does, concretely.
 *   2. `Use when …` — the situation that should select this tool over its
 *      neighbours, including when *not* to.
 *   3. Cost and impact — what it spends, what it changes, whether it can be
 *      undone. Read-only tools say so; spending tools say what they spend.
 *
 * `tests/unit/descriptions.test.ts` enforces all three.
 */

export const TOOL_DESCRIPTIONS: Record<string, string> = {
  // ── Discovery ───────────────────────────────────────────────────────────
  list_available_gpus:
    "List GPUs currently offered by independent hosts on the Xcelsior marketplace, with VRAM, " +
    "region, available count, and CAD hourly rate. Use when the user asks what hardware is " +
    "available, or before launching anything, to find a real GPU model and region rather than " +
    "assuming one. Read-only and free. Inventory is third-party and changes continuously, so a " +
    "result from earlier in the conversation may already be stale.",

  get_spot_prices:
    "Get live interruptible (spot) hourly rates in CAD for each GPU model. Use when the workload " +
    "can checkpoint and resume, which is most training and batch work — spot is materially " +
    "cheaper than on-demand and is the right default there. Do not use for a workload that " +
    "cannot tolerate preemption. Read-only and free; prices are set by competing third-party " +
    "hosts and move throughout the day.",

  get_pricing_reference:
    "Get Xcelsior's on-demand hourly rate table in CAD, by GPU model. Use when quoting a price " +
    "or comparing models, instead of estimating a rate from memory. Read-only and free. This is " +
    "the reference table, not a quote for a specific host — call estimate_job_cost for what a " +
    "particular job will actually cost.",

  search_marketplace:
    "Search live marketplace listings by GPU model, minimum VRAM, region, and host reputation, " +
    "sorted by price, reputation, region, or VRAM. Use when the user has a constraint to satisfy " +
    "— a region, a VRAM floor, a trusted host — rather than just wanting to see what " +
    "exists. Read-only and free. Listings belong to independent hosts and change without notice.",

  list_tiers:
    "List the compute tier catalog: tier names and the VRAM bands they cover. Use when the user " +
    "talks in tiers rather than specific GPU models, or to map a VRAM requirement onto a tier. " +
    "Read-only and free.",

  // ── Compute ─────────────────────────────────────────────────────────────
  list_instances:
    "List the authenticated account's GPU instances, optionally filtered by status, with cursor " +
    "pagination. Use when the user asks what is running, what they are being billed for, or to " +
    "find a job_id for another tool. Read-only and free; it does not affect running work.",

  get_instance:
    "Get the full record for one instance by job_id: status, host, GPU, pricing mode, and " +
    "timestamps. Use when the user asks about a specific job, or after a launch to confirm what " +
    "was actually allocated. Read-only and free.",

  get_instance_logs:
    "Get the most recent buffered log lines for an instance. Use when diagnosing why a job " +
    "failed or checking progress on a running one; it returns a snapshot, so call it again for " +
    "newer output rather than expecting a stream. Read-only and free. For a live view over " +
    "several minutes, use watch_instance instead.",

  create_instance:
    "Prepare or execute a server-bound plan to rent a GPU instance. Use when the user has " +
    "decided to launch; call should_i_run_this first if affordability is unsettled. Called " +
    "without a plan_id it only *prepares* a plan and returns the canonical spec, a cost " +
    "estimate, and an approval URL — nothing is allocated and nothing is billed. Execution " +
    "requires that prepared plan_id plus approval; confirm:true expresses your intent but never " +
    "substitutes for approval. Executing starts hourly CAD billing against the wallet until the " +
    "instance is cancelled or terminated.",

  cancel_instance:
    "Cancel a queued or running instance and stop its hourly billing. Use when the user wants to " +
    "stop paying for a job — this is the normal way to end work. Requires confirm:true; " +
    "confirm:false returns a preview and changes nothing. Cancelling ends the run: in-flight " +
    "work not already checkpointed is lost, and the instance cannot be resumed.",

  terminate_instance:
    "Permanently terminate an instance and release its host allocation. Use only when the user " +
    "explicitly wants the instance destroyed rather than stopped — prefer cancel_instance for " +
    "ordinary shutdown. Requires confirm:true; confirm:false returns a preview and changes " +
    "nothing. Stops hourly billing immediately, and is irreversible: instance-local state is " +
    "destroyed and cannot be recovered.",

  // ── Guardrails ──────────────────────────────────────────────────────────
  should_i_run_this:
    "Decide whether a proposed GPU job is affordable: estimates cost, reads the wallet balance, " +
    "checks an optional max_hourly_cad ceiling, and returns an approve/decline verdict with " +
    "reasons. Use this instead of estimate_job_cost whenever you are about to launch — it " +
    "answers whether the job *should* run, not merely what it costs. Read-only and free; it " +
    "never launches anything.",

  // ── Workflows ───────────────────────────────────────────────────────────
  run_training_job:
    "End-to-end training launch: prepares a launch plan from a git repo and init script, waits " +
    "for the instance to reach running, and returns connection details plus a log tail. Use when " +
    "the user wants a repo trained on a GPU in one step rather than orchestrating create + poll " +
    "themselves. Spends money exactly as create_instance does — the same plan preparation and " +
    "approval apply — and blocks for up to several minutes while the instance starts.",

  schedule_under_budget:
    "Find available capacity at or below a maximum CAD hourly rate and, optionally, launch it. " +
    "Use when price is the binding constraint and the user has not picked a specific host. " +
    "Searching is read-only and free; launching spends money through the same plan-and-approval " +
    "path as create_instance, so nothing is allocated without it.",

  // ── Monitoring ──────────────────────────────────────────────────────────
  watch_instance:
    "Poll one instance's status, telemetry, and recent logs for up to 60 minutes, returning as " +
    "soon as it reaches one of the phases you name. Use when the user wants to wait for a job to " +
    "finish or fail rather than checking repeatedly. Read-only: it never cancels or modifies the " +
    "instance, and abandoning the watch leaves the job running and still billing. The instance " +
    "keeps accruing hourly cost for as long as it runs, whether or not you are watching.",

  // ── Serverless ──────────────────────────────────────────────────────────
  list_serverless_endpoints:
    "List the account's serverless inference endpoints and their state. Use when the user asks " +
    "what models they can already call, or to find an endpoint_id before running a job. " +
    "Read-only and free.",

  create_serverless_endpoint:
    "Prepare or execute a server-bound plan to create a serverless inference endpoint for a " +
    "model. Use when the user wants per-token inference with no idle cost, rather than renting a " +
    "dedicated GPU — prefer this over create_instance for bursty or low-volume inference. " +
    "confirm:false returns a preview and creates nothing; execution goes through the same plan " +
    "approval as a GPU launch. Once created, the endpoint bills per million tokens processed and " +
    "costs nothing while idle.",

  should_i_run_pel_job:
    "Decide whether a serverless inference job is affordable: estimates token and GPU cost " +
    "against the wallet balance and returns a verdict. Use before run_serverless_job on anything " +
    "large or repeated. Read-only and free; it never invokes the endpoint.",

  run_serverless_job:
    "Enqueue an asynchronous inference job on an existing serverless endpoint and return a job " +
    "handle. Use when the user wants inference from a model already deployed; create the " +
    "endpoint first if there is none. Bills per million tokens processed. Not idempotent — " +
    "calling it twice enqueues two jobs and bills for both, so do not retry a call whose outcome " +
    "you did not see.",

  get_serverless_job_status:
    "Get the status and, when finished, the result of a serverless inference job. Use to poll a " +
    "handle returned by run_serverless_job. Read-only and free.",

  // ── Diagnostics ─────────────────────────────────────────────────────────
  explain_instance_placement:
    "Explain why the scheduler placed an instance on the host it chose, from persisted placement " +
    "facts. Use when the user asks why a job landed where it did, or why it did not land " +
    "somewhere they expected. Read-only and free.",

  simulate_instance_placement:
    "Run a proposed instance spec through the scheduler and return where it would be placed, " +
    "without allocating anything. Use to test feasibility — a region preference, a VRAM " +
    "requirement, a host preference — before preparing a real launch plan. Read-only and free: " +
    "no capacity is reserved and nothing is billed.",

  get_instance_timeline:
    "Get the durable attempt timeline for an instance: every scheduling attempt, transition, and " +
    "failure, in order. Use when a job is stuck or failed and the current status alone does not " +
    "explain it. Read-only and free.",

  get_active_lease:
    "Get the current attempt and lease health for an instance — which host holds it and whether " +
    "the lease is being renewed. Use when an instance appears running but unresponsive, to tell " +
    "a live job from an abandoned lease. Read-only and free.",

  get_mcp_action_status:
    "Get the current state of an action plan owned by the authenticated principal, including " +
    "whether it has been approved. Use after a tool returns approval_required, to check whether " +
    "the plan is now approved and can be executed. Read-only and free; checking a plan does not " +
    "approve or execute it.",

  // ── Diagnostics, operator profile ───────────────────────────────────────
  get_scheduler_health:
    "Get platform-wide control-plane health and scheduling backlog. Operator tool: it reports on " +
    "the whole platform, not one tenant. Use when diagnosing why placements across the fleet are " +
    "slow or stalled. Read-only and free.",

  get_host_capacity:
    "Get redacted GPU capacity and allocation for one platform host. Operator tool. Use when " +
    "deciding whether a host has room, or investigating why it is not receiving placements. " +
    "Read-only and free.",

  list_reconciliation_findings:
    "List durable reconciliation findings — places where recorded control-plane state and " +
    "observed reality disagree. Operator tool. Use when auditing fleet consistency or before " +
    "acting on a host. Read-only and free; listing a finding does not resolve it.",

  // ── Operations ──────────────────────────────────────────────────────────
  retry_instance:
    "Retry a failed instance, creating a fresh fenced attempt while preserving the job's " +
    "identity and history. Use when an instance failed for a transient reason and the user wants " +
    "it to run again. Requires the instance's expected_version and an idempotency key, so a " +
    "repeated call cannot produce a second attempt. Starts hourly billing again once the new " +
    "attempt runs.",

  reconcile_instance:
    "Force reconciliation of one instance so its recorded state is re-derived from what the host " +
    "actually reports. Use when an instance's status looks wrong or stale — for example, running " +
    "in our records but gone on the host. Requires expected_version and an idempotency key. " +
    "Corrects bookkeeping only: it does not start, stop, or move the workload, and it changes " +
    "nothing about what the instance is billing.",

  drain_host:
    "Stop new placements on a platform host. Operator tool. Use before maintenance, or to stop " +
    "sending work to a host that is misbehaving. Workloads already running on the host keep " +
    "running and keep billing — draining is not eviction. Reversible with undrain_host; to " +
    "actually remove workloads, use evict_host_workloads, which needs a separate scope and its " +
    "own approval.",

  undrain_host:
    "Return a drained host to service so the scheduler places new work on it again. Operator " +
    "tool. Use after maintenance completes. Requires expected_version and an idempotency key. " +
    "Affects future placements only; nothing currently running is changed. No direct cost, but " +
    "the host resumes accepting billable work.",

  evict_host_workloads:
    "Prepare or execute the removal of every running workload from a platform host. Operator " +
    "tool, and the destructive counterpart to drain_host. Use when workloads must leave the host " +
    "— hardware failure or an emergency — and only then. Without confirm:true and a plan_id it " +
    "only prepares an eviction plan and changes nothing. Executing preempts each workload, so " +
    "uncheckpointed work on this host is lost; jobs are requeued and reattempted elsewhere, " +
    "which restarts their billing on a new host.",

  retry_agent_command:
    "Retry a failed or dead-lettered worker command, preserving its identity and audit history. " +
    "Operator tool. Use when a control-plane command failed transiently and must be reissued " +
    "without creating a duplicate. Requires expected_version and an idempotency key, so retrying " +
    "cannot double-apply the command. Costs nothing by itself, but the command it reissues acts " +
    "on real infrastructure.",

  // ── Billing ─────────────────────────────────────────────────────────────
  get_wallet_balance:
    "Get the account's wallet balance and available credits in CAD. Use before committing to " +
    "spend, or when the user asks what they have left. Read-only and free.",

  estimate_job_cost:
    "Estimate what a GPU job will cost in CAD: hourly rate and projected total for the duration, " +
    "on-demand by default or spot when the workload can checkpoint. Use when the user asks what " +
    "something would cost. Read-only and free — it estimates only. If you are about to launch, " +
    "call should_i_run_this instead, which also checks the balance.",

  list_invoices:
    "List the account's billing invoices with their periods and amounts in CAD. Use when the " +
    "user asks what they have been charged or wants a specific invoice. Read-only and free.",

  top_up_wallet:
    "Charge a card already saved on the account and credit the wallet. " +
    "Use this when the user asks to add funds — 'put $20 on my account', 'top me up' — or when a " +
    "balance check shows they will run out mid-job. " +
    "Say the amount in CAD; identify the card the way the user did — 'the Visa', 'the one ending " +
    "4242' — using card_brand and/or card_last4, or omit both to use their default card. Call " +
    "list_payment_methods first if you are unsure what is on file. " +
    "MOVES REAL MONEY: confirm the amount with the user before calling, and never retry a call " +
    "that timed out without passing the same idempotency key — the charge may already have gone " +
    "through. If more than one saved card matches, this refuses and lists them rather than " +
    "guessing; ask which one. If the bank requires verification the card is NOT charged and a link " +
    "is returned for the user to confirm in a browser. " +
    "The wallet is credited when the payment processor confirms, moments later — not when this " +
    "returns, so do not report a new balance you have not read.",

  list_payment_methods:
    "List the cards already saved on the account: brand, last four digits, expiry, and which one " +
    "is the default. Use when the user asks what is on file, or before a top-up so you can name " +
    "the card — 'the Visa ending 4242' — rather than guessing. Read-only and free. " +
    "Cards are added by the user in the dashboard, never here; this reads what is already there " +
    "and returns no card number and no secret.",

  // ── Company knowledge (optional; only registered when enabled) ──────────
  search:
    "Search Xcelsior's published knowledge — product documentation, the platform overview, the " +
    "GPU pricing table, and live marketplace listings — and return citable results. Use when the " +
    "user asks how something works, what a term means, or what Xcelsior offers, rather than " +
    "asking about their own account. Returns an id, a title, and an absolute URL for each hit; " +
    "pass an id to fetch to read the full document. Read-only and free. Covers published " +
    "material only, so results reflect the docs site and can change without notice.",

  fetch:
    "Retrieve the full text of one document returned by search, along with the public URL a " +
    "reader can open. Use after search, with an id it returned — ids are not guessable and an " +
    "unknown one returns not_found rather than a nearby match. Read-only and free.",
};
