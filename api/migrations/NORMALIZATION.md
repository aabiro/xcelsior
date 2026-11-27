# Normalization audit

What the schema stores more than once, which copies are bugs and which are
deliberate, and where each fix stands. Written 2026-11-18 from the live
catalog (177 tables, 52 foreign keys before 164) and the code that reads it;
re-audited 2026-11-26 after migrations 166–170.
Rules for *how* to change the schema are in [README.md](README.md); this file
is about *what* the schema should say once.

## The rule

A fact that can change lives in one row. Anything else that needs it refers
to that row by key, enforced by a foreign key, and reads the fact through the
key. A copy is allowed only when it is a **snapshot**: a record of what was
true at a moment (a charge, an audit event, a receipt), which must not change
when the original does. Every snapshot below says so.

## 1. Users were identified by email — fixed in 164, contracted in 171

`users` was keyed by `email`, and 13 tables copied the email as their
reference to a user, with no foreign key anywhere pointing at `users`. Three
copies of role/admin/name sat on `sessions` and `api_keys` as well.

What that broke, each confirmed in code or data before the fix:

- **Changing your email turned off two-factor authentication.**
  `UserStore.change_email` re-keyed six tables by hand and missed
  `mfa_methods`; the next login found no methods under the new address,
  skipped the second factor and reset `mfa_enabled` to 0. SSH keys, connector
  consent grants and team ownership were stranded under the old address too.
- **A session outlived its user.** Rows in `sessions` and
  `oauth_refresh_tokens` whose user had been deleted still resolved, and the
  request was authorised from the role copied onto the session.
- **Low-balance and suspension notices were never shown.**
  `control_plane/billing_effects.py` stored the wallet's `customer_id` in
  `notifications.user_email`. That matches nobody, so no user ever saw one (330
  such rows in the dev database).
- The AI assistant's notification tool compared the integer `read` column to
  `false` and failed on every call; its "add SSH key" tool omitted the key id
  and failed on every call.

Fix (`164_user_identity_by_user_id.py` and the stores in `core/db.py`):
`users` is keyed by `user_id`, `email` is a unique attribute, and every table
that belongs to a user carries `user_id` with a foreign key (`ON DELETE
CASCADE`, except `teams.owner_user_id`, `SET NULL`). Stores still take an
email from their callers and resolve it; nothing reads or writes the email
copies. Email, role, name and admin flag come from `users` by join. A wallet
notice goes to the wallet's user and to the owner of a team billed to it.

164 was the expand step: the email columns stayed for the release then
serving, kept correct by two `SECURITY DEFINER` triggers. 171 is the contract
step (§6): the copies, the triggers and the rows whose user was gone are
removed, and every key is validated.

## 2. References without a foreign key

Before 164 the schema had 308 reference-shaped columns (`*_id`, emails,
`owner`/`tenant`/`customer` ids) with no foreign key. Not all of them should
have one. They fall into four groups:

**a. Live state that refers to a live row: should have a key.** A missing key
here is how a dangling reference keeps working: agent keys outlived their OAuth
client and still authenticated (118 cleaned that up after the fact), invites
outlived their team and could still be accepted into it, and a removed host's
GPU offers stayed listed.

Done in 165: `team_members.team_id`, `team_invites.team_id` and
`users.team_id` → `teams`; `agent_api_keys.client_id` → `oauth_clients`;
`ai_confirmations.conversation_id` → `ai_conversations`;
`gpu_offers.host_id` → `hosts`; `serverless_batches`,
`serverless_semantic_cache` and `serverless_prefix_affinity` → their endpoint
(and affinity → its worker); `image_sweep_members.sweep_id`;
`volume_promotion_files.promotion_id`.
Also `oauth_clients.created_by_user_id`, which replaces the creator's email as
the owner of a connector. The old copy was never rewritten on an email
change, so a user who changed address lost access to connectors they had
registered.

Done in 171: `receipt_shares.job_id` → `jobs` (`ON DELETE CASCADE`). 169
created the table without it, so a share outlived its run when the privacy
pipeline deleted the job, and the page guarded against that by comparing a
copy of the job's owner (`receipt_shares.owner_id`) with the job. The owner of
a job never changes once set, so the key alone says it; readers no longer
read the copy, which goes in the next contract step (§7).

The other columns 166–170 added are not live references: `benchmarks.host_id`
and `verification_id` are run history (group b), `host_gpu_devices.pci_bus_id`
is the hardware's own id (group c), and `receipt_emails` is a delivery record
whose `owner_id` says whom it was sent to (group b).

Left without a key, each for a reason:

- `serverless_workers.host_id`: a worker cascading with its host would hit
  `serverless_jobs.worker_id`, which blocks, so `remove_host` would start
  failing. Needs a decision on what a removed host's worker rows become.
- `worker_model_cache.worker_id`: `workers/inference.py` stores a *host* id
  there. Two kinds of value in one column (§3).
- `pipeline_stages.plan_id`: `text` against `action_plans.plan_id uuid`. A
  key needs the column converted, which changes what readers get back.
- `gpu_allocations.offer_id`: the marketplace's allocation record, read for
  reporting after the offer is gone (group b).
- `job_logs.job_id`, `telemetry_latest.host_id`: high-volume writes. A key
  takes a share lock on the parent row per write, and both parents are held
  `FOR UPDATE` by the scheduler and admission paths. Telemetry from a removed
  host would also start failing with a 503 the agent retries forever.
- `ai_conversations.user_id`, `ai_confirmations.user_id`: the master API token
  writes `api-admin`, which is not a user.
- `slurm_job_mappings.cluster_id`: `workers/slurm_adapter.py` registers a
  mapping under the sentinel cluster `unknown` when the submission names no
  cluster, and that is never a `slurm_clusters` row. The fix is NULL for "no
  cluster" instead of a sentinel; until then a key would refuse those
  registrations.

**b. History that must outlive what it describes: no key, by design.** A key
would either block deleting the parent or delete the history with it, and
ledger, audit and legal retention forbid both (data-architecture companion
§4.3). Ledgers and billing (`wallet_transactions`, `invoices`,
`billing_cycles`, `payout_ledger`, `payout_splits`, `usage_meters`,
`serverless_token_ledger`, `slurm_billing_segments`, `payment_intents`,
`crypto_deposits`, `ln_deposits`, `fintrac_reports`, `wise_payouts`,
`marketplace_sales`, `stripe_meter_event_outbox`, `queue_bid_events`), audit
and events (`audit_events_v2`, `events`, `event_snapshots`, `outbox_events`,
`projection_deliveries`, `mcp_tool_audit`, `placement_decisions`,
`placement_decision_observations`, `scheduler_shadow_decisions`,
`reconciliation_*`, `reputation_events`, `receipt_emails`, every `*_archive`),
telemetry and
observations (`telemetry_samples`, `telemetry_snapshots`,
`verification_history`, `sla_*`, `benchmarks`, `observed_workloads`,
`job_failure_log`), and privacy/legal records that outlive the subject
(`privacy_deletion_*`, `facebook_deletion_confirmations`, `data_disclosures`,
`casl_consent`, `consent_records`, `retention_records`).

**c. Identifiers from another system: no key possible.** Stripe, PayPal and
Wise ids, Slurm job ids, container and PCI ids, trace/correlation/request ids,
WebAuthn credential ids.

**d. Polymorphic references: no key possible.** `entity_id`, `resource_id`,
`aggregate_id`, `principal_id` (a user or an agent).

## 3. One column name, two kinds of identifier

`tenant_id` holds the billing account (`wallets.customer_id`) in most tables,
but the user id in `hosts`, `agent_api_keys` and the privacy tables (088,
090). `owner_id` likewise holds a customer id on `jobs` and a user id on
`hosts`. `worker_model_cache.worker_id` holds a host id where a serverless
worker id belongs. Each is a reference in principle, but no single key can
cover a column that holds two kinds of value. Making them one kind changes what
tenancy checks compare, which is an authorization change rather than a schema
tidy-up. **Not changed; needs its own decision.**

## 4. The same fact twice in one row

- `jobs.payload` repeats columns. `status`, `host_id`, `phase`,
  `desired_state`, `error_message` and `exit_code` were already read from the
  columns. `wallet_hold_id` now is too: billing and settlement set and clear
  the column alone, so the payload kept whatever hold the job had when it was
  last saved (6 of 400 production jobs disagreed on 2026-11-26).
  `priority`, `submitted_at` and `pricing_mode` were still read from the
  payload by every reader, while the queue orders by the columns and
  settlement charges by the column. `upsert_job` writes both from one dict,
  but the lease and attempt requeues write each in its own SQL and an operator
  or migration can write the column alone, and the next save then copied the
  stale payload value back over the column. Production had no disagreement on
  2026-11-26. Since 171's release every reader lays the columns over the
  payload (`_JOB_COLUMN_OVERLAY` in `core/db.py`, the scheduler's loader, and
  `get_next_job`, which returned the raw payload); the copies leave in the
  next contract step (§7).
- `payload.updated_at` is **not** a copy of `jobs.updated_at`: it is the last
  status transition, which the stuck-job reaper measures from, while the
  column is the last write of any kind. One name for two facts; it should
  become its own column (`status_changed_at`). **Deferred:** about 40 places
  write `jobs.status` in SQL and each would have to set it, and the projection
  triggers that could have done it centrally were retired on purpose. The
  reaper only consults it for legacy rows with no attempt, whose status only
  `update_job_status` changes, and that stamps it. Related: the scheduler
  resets `submitted_at` to now while a job waits for a busy GPU, so the queued
  reaper does not count the wait; the column means "queued since", not
  "submitted".
- `hosts.payload` repeats `region`, `country`, `province`, `capabilities`,
  `registered_at` and the admission and availability states (12 hosts in the
  dev database disagree on `region`). `load_hosts`/`get_host` already read
  the columns; the volume region warning in `workers/volumes.py` read the
  payload and now reads the columns. Since 171's release admission lists a
  host's GPU offers under the location columns too; it read the payload,
  which keeps a field the heartbeat stopped reporting only in the column.
- `users.mfa_enabled` is derived from `mfa_methods`. Kept as a cache: every
  MFA change recomputes it, and login decides the second factor from the
  methods themselves, never from the flag. It went stale only through the
  email re-key that 164 removed.

The duplicated keys leave the payload in a contract release, because the
release still serving reads them (§7).

## 5. Kept on purpose (snapshots)

- `gpu_model`, `region` and prices copied onto billing, pricing and usage
  rows: what the charge was for, at the time.
- `team_invites.email`: an address that may not have an account yet.
- `provider_accounts.email`, `connect_accounts.contact_email`: the contact
  given to Stripe/PayPal at onboarding, which those services hold their own
  copy of.
- `marketplace_sales.customer_email`: the receipt.
- Actor fields on audit-like rows (`created_by`, `approved_by`,
  `requested_by`): who acted, as they were then.

## 6. Contract step for 164 and 165 — done in 171

`171_user_identity_contract.py`, in the release after 164/165 code was serving
everywhere (README rule 5):

1. Rows whose user no longer exists are deleted: a user id naming no user,
   or a NULL user id beside an address (a row 164 could not map). A
   `SET NULL` reference is cleared instead. Every key 164 left `NOT VALID` is
   validated.
2. Each user id column is `NOT NULL` except where NULL is a real value:
   `chat_conversations.user_id` (anonymous chats),
   `oauth_refresh_tokens.user_id` (client-credential tokens),
   `teams.owner_user_id` and `oauth_clients.created_by_user_id` (`ON DELETE
   SET NULL`: a team and a connector outlive their creator). `team_members` is
   keyed by `(team_id, user_id)`.
3. The compatibility triggers and their functions are dropped, then the email
   copies, the copied `role`/`name`/`is_admin` on `sessions` and `api_keys`
   (`api_keys.name` stays: it is the key's own label) and
   `oauth_clients.created_by_email`, with their indexes.
4. The orphans 165 exempted are deleted (on production, 17 agent keys whose
   OAuth client is gone, all already revoked) and its keys validated.

Not done here: stripping the payload copies. The release before 171 reads
`priority`, `submitted_at` and `pricing_mode` from the payload, so a copy
removed while it still serves would be read as 0, now and `on_demand`, and its
next save would write a spot job back as on-demand. 171's release reads the
columns instead; the strip follows (§7).

## 7. Next contract step (the release after 171's code is live)

1. `upsert_job` stops writing `priority`, `submitted_at`, `pricing_mode` and
   `wallet_hold_id` into `jobs.payload`, the lease and attempt requeues stop
   adding `submitted_at` to it, and a migration strips the four keys from
   existing rows.
2. `set_shared` stops writing `receipt_shares.owner_id`, and a migration drops
   the column.
3. The column-backed keys in `hosts.payload` (`region`, `country`,
   `province`, `capabilities`, `registered_at`, the admission and availability
   states). Needs its own reader audit first: `upsert_host` merges the
   confirmed location and admission state back into the payload, and the
   worker's heartbeat sends the payload whole.
