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

import type { ToolName } from "../auth/scopes.js";

/**
 * Keyed by `ToolName`, so a tool registered with no description here **fails to
 * compile**, and a description for a tool that no longer exists fails the same
 * way. Previously both were caught — if at all — by a test reading this file
 * with a regex.
 */
const DESCRIPTIONS: Record<ToolName, string> = {
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
    "Get the most recent buffered log lines an instance has printed — its output, stdout and " +
    "stderr. Use when the user asks what a job is printing, what it is doing right now, or why " +
    "it failed; the instance id can come from earlier in the conversation rather than from a " +
    "fresh listing. It returns a snapshot, so call it again for newer output rather than " +
    "expecting a stream. Read-only and free. If the user asks to be told when the job finishes, " +
    "or to be notified on completion or failure, that is watch_instance — this tool returns " +
    "immediately and will not wait for anything.",

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
    "stop paying for a job — this is the normal way to end work. Call it with confirm:false " +
    "first: that returns a preview of what will stop and changes nothing, so you do not need " +
    "get_instance beforehand — the preview comes from the tool that will do the work, which a " +
    "status read does not. Then call again with confirm:true. Cancelling ends the run: in-flight " +
    "work not already checkpointed is lost, and the instance cannot be resumed.",

  terminate_instance:
    "Permanently terminate an instance and release its host allocation. Use only when the user " +
    "explicitly wants the instance destroyed rather than stopped — prefer cancel_instance for " +
    "ordinary shutdown. Call it with confirm:false first: that returns a preview of exactly what " +
    "will be destroyed and changes nothing, so you do not need get_instance beforehand — the " +
    "preview comes from the tool that will do the work, which a status read does not. Then call " +
    "again with confirm:true. Stops hourly billing immediately, and is irreversible: " +
    "instance-local state is destroyed and cannot be recovered.",

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
    "approval apply — and blocks for up to several minutes while the instance starts." +
    " Not idempotent: no idempotency key is sent, so calling it again after a timeout launches a second instance and bills for both. If a call seems to have failed, check list_instances before retrying.",

  schedule_under_budget:
    "Find available capacity at or below a maximum CAD hourly rate and, optionally, launch it. " +
    "Use when price is the binding constraint and the user has not picked a specific host. " +
    "Searching is read-only and free; launching spends money through the same plan-and-approval " +
    "path as create_instance, so nothing is allocated without it." +
    " Not idempotent: no idempotency key is sent, so a retry launches a second billed instance. Check list_instances before calling again.",

  // ── Access ──────────────────────────────────────────────────────────────
  register_ssh_key:
    "Register an SSH *public* key on the account — the contents of a .pub file, one line " +
    "beginning with 'ssh-ed25519', 'ssh-rsa' or 'ecdsa-sha2-'. Use when the user wants a shell " +
    "on an instance and their key is refused, or before launching work they intend to drive " +
    "interactively. Never send a private key: if the user pastes one, tell them it is " +
    "compromised and to generate a new pair, and do not call this tool with it. Free — it moves " +
    "no money and disturbs no running job — but it creates standing access: anyone holding the " +
    "matching private key can open a shell on the user's instances and read the data on them, " +
    "so confirm before adding a key the user did not just name. The key is pushed into running " +
    "interactive instances, so it takes effect without a relaunch. Registering a key that is " +
    "already on the account changes nothing and is safe to repeat.",

  open_instance_access:
    "Get a way in to a running instance: either the SSH endpoint to connect to, or a " +
    "single-use ticket for the browser/websocket terminal. Use when the user wants a shell on " +
    "something already running, after watch_instance reports it reached running. method:'ssh' " +
    "reads the endpoint and is free; method:'terminal' creates a short-lived ticket that the " +
    "first connection consumes and cannot be replayed. Read-only with respect to the instance: " +
    "it never launches, cancels or reconfigures anything, and it does not change what the job " +
    "costs — the instance goes on billing at its hourly rate whether or not anyone connects. " +
    "SSH needs a key the account has already registered; if the connection is refused, " +
    "register_ssh_key is the fix." +
    " Not idempotent: every call mints a new single-use ticket and does not return the previous one. Request it once and use it.",

  cancel_serverless_job:
    "Cancel an inference job that is still running on a serverless endpoint. Use when the user " +
    "wants to stop work in progress — a job that is taking too long, was submitted by mistake, " +
    "or is no longer needed. Call it with confirm:false first for a preview. Cancelling stops " +
    "the job billing for further GPU seconds, and ends the run: whatever it had not yet returned " +
    "is lost and the job cannot be resumed, so submit a new one instead.",

  delete_serverless_endpoint:
    "Permanently delete a serverless endpoint. Use when the user is finished with a deployed " +
    "model — prefer cancel_serverless_job if they only want to stop one run. Call it with " +
    "confirm:false first for a preview. Deleting stops the endpoint's idle cost and cancels jobs " +
    "still in flight on it, and is irreversible: the endpoint_id stops resolving and a " +
    "replacement is a new deployment with a new id.",

  // ── Durable state ───────────────────────────────────────────────────────
  list_volumes:
    "List the account's persistent volumes: id, name, size, region, status, and what each is " +
    "attached to. Use when the user asks what storage they have, before creating another one, " +
    "or to find a volume_id for attach. Read-only and free to call — though the volumes " +
    "themselves bill per GB-month whether attached or not.",

  get_volume:
    "Get one volume by id: size, encryption, region, status, and its current attachment. Use " +
    "when the user asks about a specific volume, or before detaching, to see which instance " +
    "would lose its filesystem. Read-only and free.",

  create_volume:
    "Create a persistent volume that outlives any instance. Use when work must survive a " +
    "relaunch — checkpoints, datasets, model weights — because an instance's own disk is " +
    "destroyed with it. Creates storage that bills per GB-month from the moment it exists, " +
    "attached or not, so size it for what is needed rather than rounding up." +
    " Not idempotent: calling it twice creates two volumes, both billed per GB-month. Check list_volumes before retrying.",

  attach_volume:
    "Attach a volume to a running instance at a mount path. Use after launching, so the job " +
    "writes somewhere durable instead of into the container. Free in itself — the volume was " +
    "already billing — and it changes what the instance can see: a job started before the " +
    "attach may need restarting to notice the new path." +
    " Takes job_id, the instance id list_instances returns; instance_id is a deprecated alias " +
    "that still works.",
  detach_volume:
    "Detach a volume from the instance it is mounted on. Use when moving storage to another " +
    "instance, or before deleting the instance it is attached to. Call it with confirm:false " +
    "first: that returns a preview naming the instance that would lose its filesystem. " +
    "Detaching pulls the mount out from under anything writing to it, so unwritten data is " +
    "lost even though the volume itself is not destroyed — and the volume keeps billing per " +
    "GB-month either way, so detaching saves nothing.",

  delete_volume:
    "Permanently delete a volume and everything stored on it. Use only when the user says the " +
    "data is no longer needed — prefer detach_volume when they merely want it off an instance. " +
    "Call it with confirm:false first for a preview. Stops the per-GB-month billing, and is " +
    "irreversible: the contents cannot be recovered, so snapshot first if there is any doubt.",

  run_pipeline:
    "Quotes a multi-stage job — train, then evaluate, then serve — as ONE approval covering "
    + "every stage, with the total committed spend stated before anything runs. Use when the user "
    + "describes work with steps that depend on each other, instead of launching each and asking "
    + "three times. Each stage declares what happens if it fails (halt, continue, or retry) and "
    + "that choice is fixed when the graph is approved. The pipeline cannot spend more than the "
    + "quoted total: a stage that would exceed it never starts. **Returns as soon as the pipeline "
    + "is quoted, before it has run or even been approved** — say it is awaiting approval, and "
    + "check get_pipeline_status before reporting any stage as done." +
    " Not idempotent: each call creates another plan awaiting approval, so retrying a call that appeared to fail leaves two plans for the same work.",
  get_pipeline_status:
    "Reports which pipeline stage is running, which finished, and which were skipped and why. "
    + "Use when a pipeline was started and the user asks how far along it is, or before claiming "
    + "any stage completed — run_pipeline returns while the work is still ahead of it. Read-only "
    + "and free.",
  promote_artifact_to_volume:
    "Copies a finished run's outputs — weights, checkpoints — from artifact storage onto a "
    + "volume, which has no retention clock. This creates a durable copy — it does not move or "
    + "delete the originals. Use when a run produced something worth keeping "
    + "past its expiry, or when the user asks to save results somewhere permanent. The copied "
    + "data bills per GB-month as volume storage for as long as it is kept, so it is a standing "
    + "cost rather than a one-off. **Returns as soon as the copy STARTS, not when it finishes.** "
    + "A large checkpoint takes minutes; say it is running and check get_promotion_status before "
    + "telling anyone their files are safe. Asking twice for the same job does not copy twice." +
    " Returns a promotion_id; the copy is still running when this returns, so pass that id to " +
    "get_promotion_status before telling the user their files are on the volume.",

  get_promotion_status:
    "Reports whether a promotion started by promote_artifact_to_volume has finished, and how "
    + "much it has copied. Use when you started a promotion and need to know whether the files "
    + "have actually landed — the promotion tool returns while the copy is still running, so "
    + "this is what tells you it is safe to say the work is saved. Read-only and free.",
  snapshot_volume:
    "Creates a point-in-time snapshot of a volume, so its current contents can be restored " +
    "later. Use before anything destructive — a delete, a risky job, a version bump — and when " +
    "the user wants a checkpoint they can return to. The snapshot is new stored data and bills " +
    "per GB-month like the volume it came from." +
    " Not idempotent: each call creates another snapshot and snapshots are billed for storage. List the volume's snapshots before retrying.",

  get_artifact_expiry:
    "Show when a job's artifacts will be deleted: each file with its creation time and the date " +
    "its retention window ends. Use when the user asks how long results are kept, and " +
    "unprompted after a job finishes — the clock is otherwise invisible, which is how finished " +
    "work quietly disappears. Read-only and free. Artifacts expire; a volume does not, so copy " +
    "anything that matters onto one before the window closes.",

  // ── Monitoring ──────────────────────────────────────────────────────────
  watch_instance:
    "Wait for an instance to reach a phase you name — finished, failed, running — polling its " +
    "status, telemetry, and recent logs for up to 60 minutes and returning as soon as it gets " +
    "there. Use when the user asks to be told when something finishes, to be notified when it " +
    "is done, or to wait for a result, rather than calling get_instance_logs repeatedly and " +
    "guessing. Read-only: it never cancels or modifies the instance, and abandoning the watch " +
    "leaves the job running and still billing. The instance keeps accruing hourly cost for as " +
    "long as it runs, whether or not you are watching.",

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
    "Decide whether serverless inference is affordable before committing to it: estimates token " +
    "and GPU cost against the wallet balance and returns a verdict. Use before " +
    "create_serverless_endpoint when deploying a model, and before run_serverless_job on " +
    "anything large or repeated — deploying an endpoint commits to standing capacity, so the " +
    "check belongs there too, not only at invocation. Checking the wallet balance directly " +
    "answers a smaller question: it says what is available, not what this will cost. Read-only " +
    "and free; it never deploys an endpoint or invokes one.",

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
    "Answer whether a proposed spec would actually schedule — run it through the real scheduler " +
    "and get back where it would be placed, or why it would not be, without allocating " +
    "anything. Use when the user asks whether something is possible, whether it would fit, or " +
    "whether a combination of GPU model, count, and region can be satisfied at all. Browsing " +
    "inventory does not answer this: list_available_gpus and search_marketplace show what " +
    "exists, not whether one host can satisfy four of them in the region asked for. Read-only " +
    "and free: no capacity is reserved and nothing is billed.",

  evaluate_placement_preference:
    "Check what a placement preference would actually get you, before committing to anything. " +
    "Takes a spec plus constraints — minimum uptime, minimum reputation tier, verified hosts " +
    "only — and an optional cap on how much more you will pay to satisfy them. Returns the host " +
    "it would pick and the premium over the cheapest eligible host, or a clear refusal naming " +
    "the constraint that failed and the best value actually available, so the user can decide " +
    "whether to relax it. Use whenever a user says they want reliable, verified, or high-uptime " +
    "capacity: this answers whether that is possible right now and what it costs, rather than " +
    "quietly placing on something that does not meet it. Differs from " +
    "simulate_instance_placement, which only answers whether any host is available at all. " +
    "Read-only and free: nothing is reserved and nothing is billed.",

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

  get_spend_envelope:
    "How long the balance lasts at the current burn rate, and which instances are consuming it. " +
    "Returns balance, burn rate per hour, seconds until zero, and a per-instance breakdown. Use " +
    "before starting long unattended work, when the user asks how long they can keep running, or " +
    "when deciding whether to top up. Read-only and free.\n\n" +
    "At zero the platform stops running instances automatically with reason 'low_balance' — work " +
    "is interrupted, not merely billed. If auto-top-up is configured it charges first; check " +
    "get_wallet_balance for that. This projects the *balance*, not the *work*: nothing here " +
    "estimates when a job will finish, because instances run until stopped. Treat the runway as " +
    "'how long before it stops if nobody acts', not 'whether the job will make it'.",

  estimate_job_cost:
    "Estimate what a GPU job will cost in CAD: hourly rate and projected total for the duration, " +
    "on-demand by default or spot when the workload can checkpoint. Use when the user asks what " +
    "something would cost. Read-only and free — it estimates only. If you are about to launch, " +
    "call should_i_run_this instead, which also checks the balance.",

  list_invoices:
    "List the account's billing invoices with their periods and amounts in CAD. Use when the " +
    "user asks what they have been charged or wants a specific invoice. Read-only and free.",

  configure_auto_topup:
    "Set up or change automatic wallet top-ups: when the balance falls below the threshold, the " +
    "saved card is charged for the amount, without anyone present. " +
    "Use this when the user wants to stop running out mid-job, or asks to change, pause or stop " +
    "existing auto top-ups. Call it with enabled=false to turn it off. " +
    "CHANGES UNATTENDED SPENDING: raising the amount or the threshold means larger or more " +
    "frequent charges that happen with nobody watching, so state the new numbers back to the user " +
    "and get their agreement before calling. Lowering them or disabling needs no such care — it " +
    "only ever reduces what can happen. " +
    "Every change is recorded on the account, and the response includes the previous settings so " +
    "you can tell the user exactly what changed.",

  get_auto_topup:
    "Show the current automatic top-up settings: whether it is on, the balance that triggers a " +
    "charge, and how much is charged. Use when the user asks what their auto top-up is set to, " +
    "before changing it so you can state what is changing, or when a wallet was topped up with " +
    "nobody present and they want to know why. Read-only and free — configure_auto_topup is what " +
    "changes them, and calling that to find out the current values would alter the unattended " +
    "spending you were only trying to read.",

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
    "is returned for the user to confirm in a browser — list_pending_verifications shows later " +
    "whether that was ever done, which a balance check cannot tell you. " +
    "The wallet is credited when the payment processor confirms, moments later — not when this " +
    "returns, so do not report a new balance you have not read.",

  list_payment_methods:
    "List the cards already saved on the account: brand, last four digits, expiry, and which one " +
    "is the default. Use when the user asks what is on file, or before a top-up so you can name " +
    "the card — 'the Visa ending 4242' — rather than guessing. Read-only and free. " +
    "Cards are added by the user in the dashboard, never here; this reads what is already there " +
    "and returns no card number and no secret.",

  // ── Company knowledge (optional; only registered when enabled) ──────────
  list_pending_verifications:
    "List top-ups that stopped because the cardholder's bank wants them to confirm — the money " +
    "has not moved and the wallet was not credited. Use when a user asks whether a top-up went " +
    "through, when a balance is lower than they expect after topping up, or when top_up_wallet " +
    "returned a verification link earlier in the conversation and you need to know whether it " +
    "was ever followed. A balance check cannot answer this: an unconfirmed charge looks " +
    "identical to one that was never attempted. Read-only and free, and it returns no " +
    "credential — completing the challenge happens in a browser, from the link the user was " +
    "given, and nothing here can finish a payment on their behalf.",

  create_instance_snapshot:
    "Create a reusable image from a running instance's container, so the exact environment can be " +
    "launched again later or run across several machines at once. Use when the user wants to " +
    "keep a working setup, reproduce a result, or prepare an image to sweep — installing " +
    "packages again on a fresh instance is what this avoids. The image is built on the host in " +
    "the background and is not usable until it reports ready; list_user_images shows the " +
    "status. Costs storage while it exists, and taking the same name and tag twice is refused " +
    "rather than overwriting, so an existing image must be deleted first.",

  list_user_images:
    "List the saved images this account can launch from, with their status, size, and the job " +
    "each was captured from. Use to find an image_id — create_image_sweep and a snapshot-based " +
    "launch both need one and it cannot be guessed — or to check whether a snapshot taken " +
    "earlier has finished building. Read-only and free.",

  delete_user_image:
    "Permanently delete a saved image, stopping its storage cost. Use when the user says an " +
    "image is no longer needed, or when a snapshot must be retaken under a name that is " +
    "already taken, since create_instance_snapshot refuses to overwrite. Irreversible through " +
    "the API: anything that referenced the image — a past sweep, a launch you might want to " +
    "reproduce — can no longer be launched from it. Check list_user_images first if there is " +
    "any doubt about which image is which.",

  create_image_sweep:
    "Find out whether one image actually behaves the same on different machines, by running it " +
    "on several at once and comparing what each container ended up with — package versions, " +
    "environment, entrypoint. Use when something works on one host and not another, when a " +
    "result will not reproduce, or when the user asks whether an environment is really " +
    "identical across nodes; that question cannot be answered from list_instances, which shows " +
    "what is running but not what is inside it. Not the tool for simply wanting several " +
    "instances — that is create_instance called N times, and it produces no record to compare " +
    "across. Spends N times one launch: called without plan_id it only quotes the sweep and " +
    "returns a plan awaiting approval, and nothing runs until that plan is approved and this " +
    "tool is called again with plan_id and confirm:true. The approved plan carries the member " +
    "count, so approving three cannot launch sixty-four. " +
    "Not idempotent: each call without plan_id creates another plan awaiting approval, so " +
    "retrying a call that appeared to fail leaves two plans for the same work. Returns a " +
    "sweep_id once it launches; pass that to get_image_sweep to see whether the members agree.",

  get_image_sweep:
    "Get one sweep: its members, which of them launched, the hosts they landed on, and whether " +
    "their environments are byte-identical. Use after create_image_sweep to see whether the " +
    "members agree, and to find which field differs when they do not — the verdict names the " +
    "differing keys rather than only reporting a mismatch. A member that never reported a " +
    "fingerprint is unknown, never counted as agreeing. Read-only and free.",

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

/**
 * Indexed loosely for the same reason `TOOL_SCOPES` is: `audit/context.ts`
 * looks up whatever name arrived, and a missing entry must read as `undefined`
 * rather than be a type error that invites a cast. The completeness guarantee
 * lives on the declaration above.
 */
export const TOOL_DESCRIPTIONS: Record<string, string> = DESCRIPTIONS;
