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
4. **Contract cleanup is always the last revision in the chain.** The
   blueprint assigns §13.7 to "contract cleanup"; that step drops
   transitional columns, triggers, and legacy paths, so anything landing
   after it would be building on removed scaffolding. If a contract-cleanup
   migration exists, it is the head. Track B §B16 owns it.
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

**Repository head: `112_user_image_digest.py`.**

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
