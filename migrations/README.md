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

**Repository head: `071_serverless_endpoint_spend_limit.py`.**
(`069_action_plans_mcp_policy_audit.py` was head through Track B B2.1; B3.1
added `070`, binding `serverless_workers` to their fenced attempt; B3.2 added
`071`, the per-endpoint spend ceiling.)

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
