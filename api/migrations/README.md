<!-- residency-guard: documents-removal -->
# Migration ledger

Authoritative rules and numbering history for this Alembic chain.

**Enforced by `tests/test_migration_ledger.py`.** That test is the gate;
this file is the explanation. If you change one, change the other.

Governing documents:
[control-plane blueprint](../docs/xcelsior-production-control-plane-mcp-blueprint.md)
§13, [data-architecture companion](../docs/xcelsior-production-data-architecture-companion.md)
§4.4/§14, and
[Track B](../docs/track-b-implementation-checklist.md) §B1 — which is where
this ledger is required.

---

## 1. Rules

1. **One head, always.** Never create a parallel head. Before authoring,
   run `alembic heads` and read the real head — do not infer it from a
   design document.
2. **Revision id equals the filename prefix**, zero-padded, as a quoted
   string: file `066_foo.py` declares `revision = "066"`. Create with an
   explicit id:

   ```bash
   uv run alembic revision -m "foo" --rev-id 066
   ```

   Without `--rev-id`, Alembic generates a random hash and the chain
   becomes unreadable — see the §3 anomaly.
3. **`down_revision` is the immediately preceding file's revision id.**
   No skipping, no reordering.
4. **Contract cleanup is always the last revision in the chain when it
   declares a final global cleanup.** The reserved filename markers
   `contract_cleanup` and `contract_migration` retain this rule. The actual
   cutover uses staged contractions: 127 follows 126, 130 follows 129,
   131 follows 130, 134 follows 133, 135 follows 134 and 136 follows 135.
   Later migrations can use the resulting native schema. They must never
   assume removed scaffolding still exists. The ledger gate requires all six
   named boundaries and their exact predecessors; the migration/restore and
   native-writer gates check their behavior. This replaces the old assumption
   that every contraction must wait for one final revision. Track B §B16 and
   [the staged handoff](../docs/review/2026-10-23-staged-control-plane-handoff.md)
   own the writer changes and quiescence between these boundaries.
5. **Expand-contract only** (blueprint ADR-009, §13.8; companion §4.4).
   Additive first; backfill in bounded `SKIP LOCKED` batches with hard
   verification that aborts on unmapped rows; constraints added
   `NOT VALID`, verified, then `VALIDATE`; `CREATE INDEX CONCURRENTLY` in
   autocommit blocks on large tables; `lock_timeout` set. Contract only in
   a later release, after the legacy-use metric reads zero.
   **One table's locks at a time**, which is the operative half of "`lock_timeout`
   set". Deploys are blue-green: `scripts/deploy.sh` runs `alembic upgrade head`
   while the live API, scheduler and workers keep serving, so a migration
   competes with traffic by construction. A migration touching several tables
   goes through [`migrations/lock_safe.py`](lock_safe.py) — a transaction per
   table, a short `lock_timeout`, retry on `40P01`/`55P03` — and every statement
   in it is idempotent, because per-table commits make such a migration
   resumable rather than atomic. `migrations/env.py` must keep
   `transaction_per_migration=True`, which `lock_safe` depends on and verifies
   at runtime.

   This paragraph exists because the rest of rule 5 was prose with no gate. The
   first production deploy of `080`–`098` (2026-08-04) failed on `095` with
   `deadlock detected` at `ALTER TABLE jobs ADD COLUMN spot_rate_micros`: no
   `lock_timeout` was set, and one enclosing transaction held `ACCESS EXCLUSIVE`
   on fifteen tables while a live request held a read lock on another. Both
   halves are now gated by `tests/test_migration_lock_discipline.py`, which
   reproduces the deadlock against real PostgreSQL and then shows the per-table
   shape surviving the same contention.
6. **Every migration passes `up → down → up` cleanly** on a dev database
   *and* a production-shaped snapshot before it is applied anywhere else.
7. **Alembic is the only production DDL authority.** No runtime
   `CREATE`/`ALTER`. This is enforced in PostgreSQL itself — no runtime
   role holds `CREATE` on a schema (Track A 11.4) — and by
   `tests/test_no_runtime_ddl.py`.
8. **A migration that creates a table another migration `ALTER`s must
   `CREATE TABLE IF NOT EXISTS` first.** From-empty bootstrap broke on
   exactly this (`agent_commands`, Track A A1.6); guarded by
   `tests/test_from_empty_bootstrap.py`.
9. **Typed columns.** `TIMESTAMPTZ` for time, integer minor units or
   `NUMERIC` for money — never binary floats (companion §4.4 rules 5–6).
   Every tenant-owned table has a non-null `tenant_id` and an index
   beginning with it.

When a migration adds a table, `control_plane/db_roles.py` must assign it
to exactly one logical domain, or the drift guard in
`tests/test_db_service_roles.py` fails.

---

## 2. Numbering: what the design documents say vs. what this repository has

Both governing documents assign migration numbers that this repository has
since spent on other content. The companion anticipated this and instructs
the implementer to inspect the real head and renumber (§14, §22.10). This
table is that renumbering, recorded once so no future work guesses.

**Repository head: `139_connect_mapping_authority.py`.**

138 adds immutable operator disposition receipts for qualified duplicate
historical volume intervals. It changes no balances or source history. The
separate `dispose_historical_billing.py` command previews a private evidence
plan, requalifies it on application and commits one bounded credit with its
ledger row and receipt. Ambiguous evidence or wallet/ledger drift refuses the
operation. Runtime roles cannot rewrite these receipts. The runtime schema
floor at revision 138 remained 137; this additive operator table is not execution authority.

133 expands durable weighted scheduling policy, virtual service credit and
preemption plan records. Default policy enables no evictions. Schema 137
introduced native lifecycle intent and completed host eviction results; the
current runtime floor is 139 after moving Connect bootstrap into Alembic.

132 records explicit historical disposition after qualified recovery. Jobs
with a recovered execution record or unavailable launch evidence retain their
original rows and billing/command links, with hashed disposition receipts.
Historical rows must be terminal and unowned; disposition never invents an
admission or repairs active authority. The validated exclusion prevents these
rows being requeued, including after downgrade. New caller admissions remain
eligible. Terminal records created during the 129-to-native handoff receive
the same frozen qualification assessment before disposition.

Runtime floor is now 132 for the explicit resume guard. First deploy the
126-compatible native writers at `dfacc879` (or a separately qualified
126-compatible descendant), complete handoff and migrations through 132,
then deploy this runtime. A direct old-production-to-head deployment is not
qualified by this migration. Repairs remain `report_only`.

126 adds hash-only attempt callback credentials. 127 verifies and archives inactive
v1 lease history, then drops the obsolete table and its eight columns. Apply 127
only after the native-only runtime handoff; active/unknown leases or unreviewed
columns/dependencies abort it. Its downgrade restores the exact archived rows.
The new runtime requires at least 126 and can run there during that handoff;
127 is the later contraction step. Do not run a one-shot upgrade to 127 while
old API/background/scheduler processes can still query the v1 lease table.

128 repairs missing normalized job/host identity from recorded evidence and
host capabilities from recorded objects. It restores outbox occurrence/digest
and contract classification metadata without changing source payloads/headers
or sealed audit rows. Historical tenant/trace IDs come only from original event
headers. Newly recovered job tenancy can resolve archive attribution. Units
commit per table, use bounded SKIP LOCKED selection, and refuse incomplete
locked work; rerun after contention clears. Downgrade retains recovered facts.

129 (`129_qualify_recorded_job_specs.py`) recovers a historical execution spec
only when its saved job hash agrees with a fenced start command and attempt.
It records migration-owned assessment evidence for recovered and unavailable
history; current payload never becomes an invented approved spec. Each batch
commits independently, detects locked unfinished rows and refuses unrelated
job-state drift. Downgrade retains recovered specs and provenance.

130 contracts `jobs.phase` and `desired_state` to non-null and requires an
actual queue timestamp for pending jobs. Missing projections use recorded
status/submission evidence; unknown history aborts the table unit. The job
projection bridge now fires only for legacy status/priority/submission writes
and changes priority only when the raw priority changes. Claims, observations
and billing metadata no longer erase priority boosts or native intent.
131 removes both projection triggers and functions. The native writers own
job phase, intent, priority and queue state; host registration defaults to
pending administration. A validated host constraint refuses admitted/draining
administration without normalized operator admission. Existing inconsistent
authority refuses the host unit rather than being rewritten. Independent host
and job transactions are resumable; downgrade restores the exact 130/082
definitions and prior default without rewriting rows. Current-schema fixtures
now supply explicit native fields; older migration fixtures retain their
historical shape. The [native writer rehearsal](../docs/review/2026-10-22-native-projection-writers.md)
and the migration up/down/interrupted-unit gates cover the handoff mechanics.

This is a scoped trigger contraction, not final B16.2 cleanup. Runtime floor
remains 126. Production rollout requires 126, the native API/worker/service
handoff, 127, then the later data/column migrations. Before 131, every API,
scheduler and maintenance replica must run the qualified native writers
(`c0f4e4f4` or a qualified descendant). An Alembic upgrade is not that handoff.
The historical payload bridge and unqualified spec histories remain open.

(`125` stores one revoked client-credentials JWT id until that token's own
expiry. Downgrade drops the table.)

(`124` publishes suspended account identifiers to the SSH gateway through a
view. The gateway role does not receive the state table. Downgrade drops the
view and leaves suspension records in place.)

(`123` adds transaction-bound command and lease change requests for the current
job/attempt/host/fence only. Healthy lease extensions and duplicate ACK/release
calls add no work. Requests preserve existing claims and advance, never postpone,
due time. No backfill runs; downgrade removes the new triggers/functions while
retaining requests. Reconciler repairs use nonblocking runtime-row locks to
avoid the queue/runtime-row lock inversion. This is the schema floor.)

(`122` adds a transactional job-change trigger feeding the coalesced
reconciliation queue. Observed-generation acknowledgements do not requeue.
The bounded `job_reconciliation_recovery` task covers existing active jobs;
the migration performs no bulk backfill. Downgrade removes only the trigger
and function, retaining queued work and findings.)

(`121` adds a nullable `agent_api_keys.key_ciphertext`: the dashboard's Quick
Connect MCP and CLI keys are kept encrypted with `XCELSIOR_SECRETS_KEY` so they
can be shown again. Additive; other agent keys stay show-once. It is the schema
requirement inherited by `control_plane/schema_compat.py`, because minting writes the column.)

(`120` deletes `ssh_status.root_password` from instances that are finished or
stopped. Those passwords were stored in plaintext before the API began
encrypting them; the worker sets a fresh one whenever it sets up SSH, so none
of them is needed again. Running instances are left alone. It has no downgrade:
a deleted secret must stay deleted.)

(`119` repeats 090's float-epoch → TIMESTAMPTZ conversion for `casl_consent`
and `user_encryption_keys`, guarded on any numeric type instead of only
`double precision`. Databases whose `casl_consent` predates the first 084 held
REAL columns, slipped past 090's guard, and failed every consent write. It's a no-op where
the columns are already TIMESTAMPTZ (prod, 2026-10-10), it's idempotent, and it has no downgrade.)

(`118` revokes Quick Connect agent keys whose OAuth client no longer exists.
"Regenerate" deleted the system-managed client and minted a key for a new one,
but deleting a client never touched the keys bound to it and validation never
consults `oauth_clients`, so every rotated-away key kept working. The route now
revokes as it rotates; this clears what it left behind. Scoped by key name to
the two Quick Connect surfaces, data only, idempotent, no downgrade.)


(`117` drops 17 indexes that another index already covers. A single-column btree
index is redundant when a second has that column leading with the same
predicate — Postgres uses the second for every lookup the first could serve, and
the duplicate still costs a B-tree write on every insert. Five are exact
duplicates of a UNIQUE constraint's own index, same column and same predicate.
Index-only, so the compatibility floor in `control_plane/schema_compat.py` does
not move. `tests/test_no_index_duplicates_another.py` keeps the count at zero,
so the next composite index arrives together with the removal of the prefix it
supersedes.)

(`116` splits the low-balance warning watermark off `wallets.grace_until`.
That column carried two incompatible meanings: `serverless/service.py` reads it
as a grace deadline, and the warning rate-limiter wrote `now` into it. Safe only
because nothing ever set a *future* value — and a trap for whoever did, because
`now - last` would go negative, compare as "warned recently", and silence the
low-balance notice for the whole window. Nullable, no backfill: the watermark is
ephemeral and copying `grace_until` forward would import exactly the wrong
values.)

(`115` gives a launch somewhere to put a stated placement preference —
min uptime and max premium in **integer basis points**, because that bound
gates a price comparison and a float boundary is not the same number on
every machine. Its own columns rather than `jobs.spec`, since `spec` feeds
`spec_hash` and two identical workloads asking for different reliability
must not hash apart. `min_tier` is shape-checked, not enumerated: the tier
vocabulary is derived from `ReputationTier` and a second hand-written copy
in a CHECK is how it drifts.)

(`114` stores what a sweep member's container reported about its own
environment — both the hash and the raw manifest, with a check constraint
refusing half of one. A hash answers "did these differ" and nothing else; the
first time a sweep goes red the question is *which field*. `JSONB` so that
question is a query rather than a diff by hand. The columns are nullable
because a member that never reported is a different fact from one that reported
an empty environment, and the comparison above them treats missing as
**unknown, never as agreement** — a collector that errors and returns nothing
would otherwise make all N equal and the sweep would report a perfect pass.)

(`113` makes a sweep a record rather than a loop. Gate P7 asks N nodes from one
snapshot to be byte-identical; a route calling the launch path N times and
returning N job ids leaves that as an intention in the caller, unverifiable
afterwards, with partial failure invisible and nothing for a fingerprint
comparison to be compared across. `image_sweeps` pins the digest once —
`NOT NULL` with a `LIKE '%@sha256:%'` check, because a sweep from an unknown
digest cannot support the claim and must be refused rather than launched from
the mutable tag — and `image_sweep_members` records `host_id` per member so
"how many distinct hosts did this actually cover" is a query. Two constraints
carry reasoning: a member that is `launched` must have a `job_id`, or the
partial-failure count would be wrong in the direction that flatters it; and
`job_id`/`host_id` stay nullable because a member that never launched is
precisely the case the table exists to make visible.)

(`112` records a snapshot's manifest digest. Gate P7 asks a sweep of N nodes to
be *byte-identical*, and `_build_image_ref` returns a **mutable tag** — so N
containers launched from it were asked for the same name, not given the same
bytes, and the clause is unprovable in principle from a tag. The worker captures
`repo@sha256:…` with `docker inspect` immediately after the push that produced
it, which is the only moment anything knows the digest belongs to those bytes;
resolving the tag later answers "what does this point at now". Nullable with no
backfill or default: a snapshot that predates this, or whose push succeeded
while the inspect failed, genuinely does not know, and a sweep that cannot pin a
digest must refuse to claim byte-identity rather than fall back to the tag.)

(`111` records what a snapshot was built *from*. Gate P7 asks a snapshot to
record its lineage — "what it was built from, when, by which run" — and three of
those four were already on `user_images`: `created_at`, `source_job_id` and
`host_id`. The base image was not. A snapshot is `docker commit` over a running
container, so the image is a diff on top of whatever base the job launched with;
without it the row says which run produced the image but nothing about what is
underneath the commit, which is the half an audit needs when a CVE lands in a
base image. Nullable with no backfill: existing rows genuinely do not know, and
inferring the base from the job afterwards is the guess this column exists to
prevent.)

(`109` and `110` give the two crypto funding rails an idempotency key —
`crypto_deposits` and `ln_deposits` respectively. Gate P1 clause 2 requires a
replayed funding call to produce exactly one charge, and names *"the crypto
rails"* plural. Neither had a key nor deduplicated anything: a retried request
minted a **second Bitcoin address**, or a **second bolt11 with a second payment
hash**, for one intended deposit. Lightning is the sharper failure — two
addresses at least belong to one wallet and both credit if paid, whereas a
second invoice is a distinct payment request that settles nothing when the
first is paid. Both add `(customer_id, idempotency_key)` partial-unique
indexes: scoped per customer because a caller-chosen key is only meaningful
inside the account that chose it, and partial so the existing keyless corpus
does not collide with itself. The insert carries `ON CONFLICT DO NOTHING` so
the guarantee is held by the index rather than by the timing of a
read-then-insert.)

(`108` adds the reported SSH host-key fingerprint, and its table choice is a
**documented deviation from the plan's prose**. A2 says the value belongs to the
attempt/container — on production that column would be null for the whole fleet
(327 jobs, 0 with an active attempt). A2 also says it is cleared at
`_clear_job_output`, which is gated on `user_initiated`; automatic failover is
the primary way a job changes host and does *not* pass it, so a fingerprint
cleared there would survive onto the new host and verify against the wrong one —
the plan's named mechanism defeating the plan's stated reason. Instead the column
is nulled wherever `host_id` changes, in the same upsert statement, which covers
normal placement, the CRIU migration, and failover without any hook to remember.)

(`107` moves recurrence counting off the WORM table. `placement_decisions`
records every evaluation — the right write policy — but a caller polling a
preference writes the same decision repeatedly, and each row carries a
`candidates` snapshot that scales with the fleet. A `times_seen` column is
unimplementable there because WORM forbids UPDATE, and that constraint is
correct rather than an obstacle: frequency is operational telemetry with a
natural retention policy and no business being immutable. So one WORM row per
distinct decision, and the count here, in a plain prunable table. Keyed by month
so dedupe and partition retention share a boundary — without it, an identical
decision next March would collapse into a row timestamped today and the trail
would say March never happened.)

(`106` partitions `placement_decisions` by month. `105` claimed WORM "like `075`
and `072`" and took the trigger from both and the partitioning from neither —
but `072` is explicit that *partition drops are DDL and are unaffected* by the
trigger, which is the whole mechanism by which an append-only table stays
prunable. `075` needs none of it, being one row per signed checkpoint; `105` is
per request and copied the low-volume precedent, giving a table that grows
without bound and whose own trigger forbids the only statement that could
shrink it. A partitioned table's key must contain the partition key, so this is
a rebuild rather than an ALTER — done now because nothing writes to the table
yet, and it becomes a migration on undeletable data the moment something does.
It refuses to run if the table holds rows.)

(`105` adds the placement decision record Gate P5 clause 3 asks for.
Evidence is **copied, not referenced**:
storing a host id and re-reading its score later answers "what is this host's
reputation now", a different and useless question during an incident review, and
verification makes it concrete because it is revocable. **Refusals are recorded
too, and that is the more useful half** — a preference that refused was honoured
by the refusal, and a successes-only table could not answer "why did nothing
launch last Tuesday". WORM, like `075` and `072`: append-only enforced by a
comment is a convention, enforced by a trigger it is a property. Prices are
integer micros so the premium is recomputed from two exact integers rather than
stored as a rounded percentage nobody can check.)

(`104` adds per-stage execution state for P4 pipelines. The *approval* is one
`action_plans` row whose canonical args carry the graph, so Gate P4's "editing
a stage after approval invalidates it" needs no new mechanism — the existing
`canonical_args_hash` already voids an altered plan. This table is only what
happened to each stage. `max_attempts` is NOT NULL defaulting to 1 because an
unbounded retry inside an approved spend ceiling is a way to spend the whole
ceiling on a stage that cannot succeed.)

(`103` adds per-file promotion progress so a retry resumes instead of
restarting — §3.5: a promotion that restarts from zero after failing at 38 GB
will be retried by a human who then watches it fail again. `done` is
constrained to imply `sha256_verified`, because the resume path skips `done`
files and an unverified copy that gets skipped is worse than no copy: it looks
like a backup.)

(`102` adds `volume_promotions`, the row an artifact→volume copy is keyed on —
A0 of `docs/artifact-promotion-plan.md`. Unique on
`(tenant_id, job_id, idempotency_key)` because Gate P3 asks that a repeated call
produce "one volume, not two", which implies promotion may *create* the volume
and so the key must cover creation rather than only the copy. Nothing copies
yet.)

(`101` adds `gpu_allocations.owner_id`, nullable, backfilled from the job each
allocation was created for. It exists because
`POST /api/v2/marketplace/release/{allocation_id}` passed an allocation id to
`release_allocation(job_id)` and so released nothing while returning
`{"ok": true}`. Correcting that lookup alone would have made the route work
*and* made any holder of `marketplace:write` able to release another tenant's
allocation, since the table had no owner — the no-op was the only thing
preventing it. Rows whose job has since been deleted stay `NULL` and are
releasable by nobody, which is the safe direction for a value that authorizes
an action, and the reason the column is not `NOT NULL`.)

(`100` drops `provider_accounts.total_earned_cad` and `total_paid_out_cad` —
the last two columns in the schema whose name ends in `_cad`. They are
`NUMERIC`, not `DOUBLE PRECISION`, which is why the float-money sweep in
`095`–`097` did not reach them. Dropped rather than converted to micros: the
earnings figure the API serves is computed at read time from
`payout_splits.provider_share_micros`, so converting would have created two
new dead columns to mirror two old ones. `085` deliberately kept them pending
evidence about production; production holds one row with both values zero, so
the financial history it was protecting does not exist. **The schema now holds
no `_cad` column at all.**)

(`099` records which credential registered an SSH public key. It was written on
`feat/mcp-p0-scopes`, applied to a developer database, and then stranded when
that pull request closed — so `alembic current` reported `099` on a machine
whose repository head was `098`, a revision the tree could not explain. It is
in the chain now. `registered_by_client_id` and `registered_by_auth_type` are
nullable with no default because `NULL` is the truth for every row registered
interactively and for every row written before the columns existed.)

`099_ssh_key_client_binding.py` was head before it, and `098_unique_stripe_intent_id.py` before that.
(`079_settlement_meters_reprice.py` was head through the settlement reprice;
`080` added authoritative provider settlement; `081` the durable per-sink
privacy deletion workflow; `082` authoritative host admission and signed
compatibility sessions; `083` durable non-expiring agent API keys; `084`
brought `casl_consent` and `user_encryption_keys` into the chain instead of
being created at runtime by `privacy.py`; `085` dropped fifteen columns no
code reads; `086` finished the auto-top-up money cutover to integer micros
and rewrote the wallet projection trigger that referenced the retired float
columns; `087` removed the remaining float money columns and the four
projection triggers that maintained them, leaving integer micros as the single
representation; `088` and `089` brought `agent_api_keys` and `payout_splits`
into line with the data-architecture companion §4.4 — TIMESTAMPTZ times and a
non-null `tenant_id` with a tenant-leading index; `090` finished that pass over
the four privacy tables. `084` had lifted `casl_consent` and
`user_encryption_keys` out of runtime DDL with their float epoch columns
preserved verbatim, which only made the defect official — `084` now creates
them as TIMESTAMPTZ and `090` converts databases that ran the earlier version,
each conversion guarded on the column's actual type so a from-empty build and
an existing database converge. `090` also gives the two privacy deletion tables
a `tenant_id`; they had been treated as tenant-exempt on the grounds that a
deletion subject must stay unlinkable, which conflated tenant with identity.
The companion keeps the tenant and pseudonymises the identity — the tenant is
the workspace, not the person, and erasure still blanks `subject_email` and
`subject_user_id`.
`092`–`094` are the global-marketplace cutover. `092` drops the columns that
only existed to record or price a location: `gpu_pricing.sovereignty_premium`
(every row `0.0`), `usage_meters.is_canadian_compute`, and the four float CAD
invoice columns the closed AI Compute Access Fund fed, and renames the
`sovereign` pricing tier to `dedicated` — the ladder is `community` → `secure`
→ `dedicated`, each rung naming what the capacity *is*. `093` drops
`storage.artifacts.residency_region`, since storage routing is a durability and
cost decision. `098` makes `payment_intents.stripe_intent_id` unique, because the
Stripe confirmation handler resolves who to credit through that column with a
bare `fetchone()` — two rows sharing an id would make the credited customer and
amount a coin flip. Partial, so rows written before Stripe returns an id do not
all collide on the empty string. `097` drops the last 26 float CAD columns and the mirror triggers, after
verifying float and micros agreed on every row — the schema now holds no float
money at all. `096` drops `users.canada_only_routing`, the last per-user setting that
restricted placement by country. `095` mirrors the last 26 float CAD columns
into integer micros, holding the
pair in step with a trigger so an unconverted writer cannot leave micros stale;
`096` drops the floats once code reads micros. `094` renames
`legal_requests.jurisdiction` to `requesting_country`: that column records which authority demanded data, which
is worth keeping, but the old name kept it surfacing in searches for the
placement model these migrations removed.

`091` is the connector-adoption migration: `oauth_clients` gains the
provenance and containment columns a dynamically identified client needs
(`registration_source`, a `resource_audience` pin so a self-registered
client can never mint a general API token, and `registration_expires_at`
so an unused registration disappears instead of accumulating), the RFC 7591
metadata fields a registration response has to echo back, and the new
`oauth_consent_grants` table that makes a user's approval of a connector a
durable, revocable record rather than a transient UI moment.)
(`069_action_plans_mcp_policy_audit.py` was head through Track B B2.1; B3.1
added `070`, binding `serverless_workers` to their fenced attempt; B3.2 added
`071`, the per-endpoint spend ceiling; B4.1 added `072`, the partitioned
append-only `audit_events_v2` audit stream; B4.3 added `073`, the
`event_contracts` registry; B5 added `076`, binding serverless endpoint
creation to an action plan, `077`, preserving OAuth protected-resource
audience through refresh rotation, and `078`, preserving MCP trace identity
through the action-plan and scheduler pipeline; the production payments path
added `079`, the payout-split settlement queue, the `stripe_meter_event_outbox`
Billing Meters dual-write — claimed by the `billing` domain in
`control_plane/db_roles.py` — and the GPU catalog reprice.)

| Document says | Document intended | This repository actually has | Resolution |
|---|---|---|---|
| Blueprint §13.5 `058` | action plans, MCP policy, MCP audit, wallet holds | `058_scheduler_shadow_decisions` | Wallet holds landed as `063_wallet_holds`. Action plans / MCP policy / MCP audit → **Track B §B2.1**, at the live head. |
| Blueprint §13.6 `059` | partitioned `audit_events_v2` | `059_runtime_projection_triggers` | → **Track B §B4.1**, at the live head. |
| Blueprint §13.7 `060` | contract cleanup | `060_shared_state_to_pg` | Contract cleanup **must remain last** (rule 4) → **Track B §B16.2**, at whatever the head is then. |
| Companion §6.3 `061` | storage catalog | `061_residual_runtime_ddl`; catalog landed as `064_storage_catalog` | Already satisfied by `064`. Residuals → Track B §B9.2. |
| Companion §10.1 `062` | Lightning + Slurm consolidation | `062_usage_meters_attempt_id`; partial consolidation landed as `060_shared_state_to_pg` | `060` created the tables but **not to the companion's contract** → Track B §B9.3. |
| Companion §12.1 `063` | outbox projection delivery contracts | `063_wallet_holds` | → **Track B §B4.4**, at the live head. |
| Companion §12.7 `064` | deletion / export state | `064_storage_catalog` | → **Track B §B12.2**, at the live head. |
| Companion §7.3 `065` | retrieval pgvector | `065_host_agent_tokens` | → **Track B §B10.2**, at the live head. |
| Companion §7.6 `066` | semantic cache v2 | *(unused)* | → **Track B §B10.5**, immediately after B10.2. |

**Reading a `§13.5`-style reference in a design document does not tell you
the revision number to use.** It tells you the design content. Get the
number from `alembic heads`.

---

## 3. Known anomaly: revision `060`

`060_shared_state_to_pg.py` declares `revision = 'a0985327493e'`, an
Alembic-generated hash, instead of `"060"`. It was authored without
`--rev-id`. `061_residual_runtime_ddl.py` therefore has
`down_revision = 'a0985327493e'`.

**This is deliberately not corrected.** Rewriting the id of an already
applied revision is safe only if no database anywhere is stamped at
exactly `a0985327493e`; a database stopped there would fail
`alembic upgrade head` with "Can't locate revision identified by
'a0985327493e'" — an unmigratable production deploy traded for cosmetic
tidiness. Local databases were checked (dev `064`, docker-test `059`,
pytest `065` — none at the hash), but production is a separate runtime and
was not verifiable at the time of writing.

The ledger test accommodates this one file by id and enforces the numeric
rule for every migration from `066` onward. Do not add a second exception.

---

## 4. Authoring checklist

```bash
# 1. Read the real head.
uv run alembic heads

# 2. Create with an explicit, numeric revision id.
uv run alembic revision -m "short_description" --rev-id 066

# 3. Write the migration (expand-only; see rule 5).

# 4. Prove reversibility on a dev database.
uv run alembic upgrade head
uv run alembic downgrade -1
uv run alembic upgrade head

# 5. Prove from-empty still reaches head.
./scripts/bootstrap_pg_from_empty.sh

# 6. Run the gates.
./run-tests.sh tests/test_migration_ledger.py
./run-tests.sh tests/test_from_empty_bootstrap.py
./run-tests.sh tests/test_control_plane_schema.py
./run-tests.sh tests/test_db_service_roles.py   # if the migration adds a table
```

Then update, in the same commit: `EXPECTED_HEAD` in
`tests/test_migration_ledger.py` and `tests/test_from_empty_bootstrap.py`,
the declared range in `control_plane/schema_compat.py` if the minimum
compatible revision moves, the domain assignment in
`control_plane/db_roles.py` if a table was added, and the relevant Track B
item.

Migration 134 moves recorded lifecycle intent to a native column. Quiesce every
API/scheduler/maintenance lifecycle writer for this handoff; apply 134, deploy
all native intent writers, and reopen only after schema/runtime checks. Old
JSONB intent writers cannot run concurrently after authority moves. Downgrade
copies the latest native value (including consumed NULL) back before restoring
old readers. No projection authority trigger is reintroduced.

135 archives four unused placement projection columns and old payload intent
keys before removing them. Deploy the 134-compatible canonical preference
readers and writers that no longer name these columns first; only then apply
135. No job or financial history is deleted. Downgrade retains archive evidence
and restores projection values; lifecycle intent remains owned by the column.


136 archives and removes only the obsolete `reconcile_paused_stopped` schedule.
Deploy maintenance workers without that handler first. A live task claim or
archive mismatch refuses retirement. A database constraint prevents older
workers from registering it again. Downgrade restores the recorded row and
retains its archive; it does not execute the retired handler.


137 permits a succeeded `evict_host_workloads` action to retain a host resource
result without inventing a launch job ID. Every prior approval, consumption,
response and failure constraint remains required. Other action types still need
a job ID. No job, financial row, column or history is deleted. The unit locks
only `action_plans` with the migration lock budget. Up/down/up is qualified on
an isolated database with existing plan rows; downgrade explicitly refuses if
completed native host actions would violate the older constraint, preserving
their evidence. Apply 137 before serving the new runtime (floor 137). The frozen
7757bdc3 image remains the schema-134-compatible intermediate for 134–136.

139 moves Connect account and product mapping bootstrap into Alembic. It retains
existing rows and creates missing tables on fresh databases. Downgrade retains
these financial/provider mappings. API imports no longer attempt DDL, and the
runtime floor is 139. Apply this additive migration with the direct migrator
credential before replacing the schema-138-compatible runtime.
