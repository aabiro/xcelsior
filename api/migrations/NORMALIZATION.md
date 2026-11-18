# Normalization audit

What the schema stores more than once, which copies are bugs and which are
deliberate, and where each fix stands. Written 2026-11-18 from the live
catalog (177 tables, 52 foreign keys before 164) and the code that reads it.
Rules for *how* to change the schema are in [README.md](README.md); this file
is about *what* the schema should say once.

## The rule

A fact that can change lives in one row. Anything else that needs it refers
to that row by key, enforced by a foreign key, and reads the fact through the
key. A copy is allowed only when it is a **snapshot**: a record of what was
true at a moment (a charge, an audit event, a receipt), which must not change
when the original does. Every snapshot below says so.

## 1. Users were identified by email — fixed in 164

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

Expand step only. The email columns remain for the release still serving
during a blue-green deploy, kept correct by two `SECURITY DEFINER` triggers,
and are removed by the contract step (§6).

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
`reconciliation_*`, `reputation_events`, every `*_archive`), telemetry and
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
  last saved. `priority`, `submitted_at` and `pricing_mode` are written to
  both from the same dict in one statement, so they cannot disagree; they
  leave the payload in the contract release. `payload.updated_at` is **not** a
  copy of `jobs.updated_at`: it is the last status transition, which the
  stuck-job reaper measures from, while the column is the last write of any
  kind. One name for two facts; it should become its own column
  (`status_changed_at`).
- `hosts.payload` repeats `region`, `country`, `province`, `capabilities`,
  `registered_at` and the admission and availability states (12 hosts in the
  dev database disagree on `region`). `load_hosts`/`get_host` already read
  the columns; the volume region warning in `workers/volumes.py` read the
  payload and now reads the columns.
- `users.mfa_enabled` is derived from `mfa_methods`. Kept as a cache: every
  MFA change recomputes it, and login decides the second factor from the
  methods themselves, never from the flag. It went stale only through the
  email re-key that 164 removed.

The duplicated keys leave the payload in the contract release, because the
release still serving reads them.

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

## 6. Contract step for 164 and 165 (the release after they are live)

Ship in the release after the 164/165 code is serving everywhere (README rule 5):

1. Delete rows whose user no longer exists (`user_id` NULL, or naming no
   user), then `VALIDATE` the keys 164 left `NOT VALID`. 164 logs which those
   are.
2. `SET NOT NULL` on each user id column except `chat_conversations.user_id`
   (anonymous chats) and `oauth_refresh_tokens.user_id`
   (client-credential tokens); make `(team_id, user_id)` the `team_members`
   primary key.
3. Drop the two compatibility triggers and their functions, the email copies
   (`sessions.email`, `api_keys.email`, `oauth_refresh_tokens.email`,
   `oauth_consent_grants.email`, `user_ssh_keys.email`, `mfa_*.email`,
   `team_members.email`, `teams.owner_email`, `notifications.user_email`,
   `web_push_subscriptions.user_email`, `chat_conversations.user_email`), the
   copied `role`/`name`/`is_admin` on `sessions` and `api_keys`, and the email
   indexes. Likewise `oauth_clients.created_by_email` with its trigger and
   `xcl_users_email_sync_oauth_clients()`.
4. Delete the orphans 165 logged and `VALIDATE` its `NOT VALID` keys.
5. Strip `priority`, `submitted_at`, `pricing_mode` and `wallet_hold_id` from
   `jobs.payload`, and the column-backed keys from `hosts.payload`.
