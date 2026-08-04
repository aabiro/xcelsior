"""Mirror the last 26 float CAD columns into integer micros.

Binary floating point cannot represent most decimal money exactly, so every
arithmetic step on a `double precision` amount accumulates error that eventually
shows up as a cent that does not reconcile. Migration 086 and 087 converted the
hot paths; the remainder were pinned as a downward ratchet at 30, then 26.
`095` and `096` together finish it, after which the ratchet becomes an
assertion of zero rather than a budget.

This is the **mirror** half, and it is deliberately additive: each column gains
`<name>_micros BIGINT`, backfilled as `ROUND(value * 1000000)` — the rounding is
applied once, here, so a value that was already imprecise is pinned to its
nearest micro rather than drifting further. **The float stays.**

Dropping the floats in the same step would break every reader the instant it
applied, with no window in which both representations are valid. Splitting it
means the schema can go first, code can move to micros against a database where
both columns exist, and `096` removes the floats only once nothing reads them.
A trigger keeps the two in step for the duration, so a writer that has not been
converted yet cannot silently leave micros stale.

Money crosses the API boundary as CAD. That is unchanged: response fields keep
their `_cad` names and are derived by dividing micros at the edge. What changes
is that CAD is never the *stored* representation.

The downgrade drops the mirror. Nothing is lost: the float remains the value of
record until `096`.

**Per-table transactions, one connection of their own.** The first production
deploy of this migration died on `ALTER TABLE jobs ADD COLUMN spot_rate_micros`
with `deadlock detected`: it had already taken `ACCESS EXCLUSIVE` on fourteen
other tables in the same transaction, and a live request holding a read lock on
`jobs` wanted one of those fourteen. Fifteen tables locked at once for the
duration of fifteen backfills is not a shape that can coexist with traffic, and
deploys here are blue-green by design. Each table now commits on its own with a
short `lock_timeout` and a retry — see `migrations/lock_safe.py` for why that
makes contention survivable, and note the consequence: this migration is
resumable rather than atomic, which is sound only because every statement below
is idempotent.
"""

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "095"
down_revision = "094"
branch_labels = None
depends_on = None

# (table, float column). The micros column is the same name with `_cad`
# replaced by `_micros`, matching the convention 086/087 established.
COLUMNS = [
    ("billing_cycles", "amount_cad"),
    ("billing_cycles", "token_cost_cad"),
    ("cloud_burst_instances", "budget_spent_cad"),
    ("cloud_burst_instances", "cost_per_hour_cad"),
    ("crypto_deposits", "amount_cad"),
    ("fintrac_reports", "trigger_amount_cad"),
    ("gpu_pricing", "base_rate_cad"),
    ("inference_endpoints", "total_cost_cad"),
    ("invoices", "subtotal_cad"),
    ("invoices", "tax_amount_cad"),
    ("invoices", "total_cad"),
    ("jobs", "spot_rate_cad"),
    ("ln_deposits", "amount_cad"),
    ("payout_ledger", "amount_cad"),
    ("payout_ledger", "platform_fee_cad"),
    ("payout_ledger", "provider_payout_cad"),
    ("reservations", "monthly_rate_cad"),
    ("reserved_commitments", "base_rate_cad"),
    ("reserved_commitments", "discounted_rate_cad"),
    ("serverless_cache_savings", "saved_cost_cad"),
    ("serverless_endpoints", "spend_limit_cad"),
    ("serverless_endpoints", "total_cost_cad"),
    ("serverless_endpoints", "unbilled_token_cost_cad"),
    ("serverless_jobs", "cost_cad"),
    ("serverless_token_ledger", "cost_cad"),
    ("sla_monthly", "credit_cad"),
]


def _micros_name(column: str) -> str:
    return column[: -len("_cad")] + "_micros"


def _mirror(table: str, column: str):
    """Add, backfill and pin one float/micros pair. Idempotent throughout."""
    micros = _micros_name(column)

    def apply(conn) -> None:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {micros} BIGINT"))
        conn.execute(
            text(
                f"UPDATE {table} SET {micros} = ROUND({column} * 1000000)::bigint "
                f"WHERE {column} IS NOT NULL AND {micros} IS NULL"
            )
        )
        # Keep the pair consistent while callers are still on the float. The
        # trigger derives micros from the float on write, so an unconverted
        # writer cannot leave micros stale; 097 drops both trigger and float.
        conn.execute(
            text(
                f"""
            CREATE OR REPLACE FUNCTION mirror_{table}_{micros}() RETURNS trigger AS $$
            BEGIN
              IF NEW.{column} IS DISTINCT FROM OLD.{column} OR TG_OP = 'INSERT' THEN
                IF NEW.{column} IS NOT NULL THEN
                  NEW.{micros} := ROUND(NEW.{column} * 1000000)::bigint;
                END IF;
              END IF;
              RETURN NEW;
            END;
            $$ LANGUAGE plpgsql
            """
            )
        )
        conn.execute(text(f"DROP TRIGGER IF EXISTS trg_mirror_{micros} ON {table}"))
        conn.execute(
            text(
                f"CREATE TRIGGER trg_mirror_{micros} BEFORE INSERT OR UPDATE ON {table} "
                f"FOR EACH ROW EXECUTE FUNCTION mirror_{table}_{micros}()"
            )
        )

    return apply


def _unmirror(table: str, column: str):
    micros = _micros_name(column)

    def apply(conn) -> None:
        conn.execute(text(f"DROP TRIGGER IF EXISTS trg_mirror_{micros} ON {table}"))
        conn.execute(text(f"DROP FUNCTION IF EXISTS mirror_{table}_{micros}()"))
        conn.execute(text(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {micros}"))

    return apply


def upgrade() -> None:
    apply_in_own_transactions(
        [(f"095 mirror {t}.{c}", _mirror(t, c)) for t, c in COLUMNS],
        tables=[t for t, _ in COLUMNS],
    )


def downgrade() -> None:
    apply_in_own_transactions(
        [(f"095 unmirror {t}.{c}", _unmirror(t, c)) for t, c in COLUMNS],
        tables=[t for t, _ in COLUMNS],
    )
