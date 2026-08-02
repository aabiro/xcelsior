"""Drop the last 26 float CAD columns and their mirror triggers.

The second half of the cutover `095` began. `095` added a `_micros` twin for
each float, backfilled it, and installed a `BEFORE INSERT OR UPDATE` trigger so
the pair could not drift while code still wrote the float. Every read and write
has since moved to micros, so the float is now a column nothing consults.

Before writing this, the two representations were compared row by row across all
26 columns: **zero rows disagreed** by more than one micro. That check is the
reason this is a drop rather than another mirror.

After this migration the repository holds **no float money columns at all**, and
`MAX_LEGACY_FLOAT_CAD_COLUMNS` in `tests/test_companion_schema_discipline.py`
becomes an assertion of zero rather than a budget — a new float money column is
a straight failure with nothing to negotiate.

CAD remains the unit at the API boundary. Response bodies keep their `_cad`
field names and are derived by dividing micros at the edge; what changes is that
CAD is never the *stored* representation.

The downgrade restores the float columns and derives them from micros. It is
lossy in principle — a double cannot hold every micro value exactly — which is
the whole reason the cutover happened.
"""

from alembic import op

revision = "097"
down_revision = "096"
branch_labels = None
depends_on = None

# Mirrors the COLUMNS tuple in 095. Kept as its own literal so this migration
# reads independently of the one before it.
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


# Migration 066 installed a projection trigger on `ln_deposits` that derives
# `amount_cad_minor` from the float `amount_cad`. Dropping that column leaves
# the function referencing a field the record no longer has, and *every* insert
# fails with `record "new" has no field "amount_cad"`. The projection is still
# wanted — cents are what the Lightning rails settle in — so it is rewritten to
# derive from micros before the column goes.
_LN_PROJECTION = """
CREATE OR REPLACE FUNCTION ln_deposits_project_typed() RETURNS trigger AS $$
BEGIN
    NEW.amount_cad_minor := COALESCE(
        NEW.amount_cad_minor,
        CASE WHEN NEW.amount_micros IS NULL THEN NULL
             ELSE round(NEW.amount_micros::numeric / 10000)::bigint END
    );
    NEW.btc_cad_rate_exact := COALESCE(NEW.btc_cad_rate_exact, NEW.btc_cad_rate::numeric);
    NEW.created_at_ts := COALESCE(
        NEW.created_at_ts,
        CASE WHEN NEW.created_at IS NULL OR NEW.created_at <= 0 THEN NULL
             ELSE to_timestamp(NEW.created_at) END
    );
    NEW.expires_at_ts := COALESCE(
        NEW.expires_at_ts,
        CASE WHEN NEW.expires_at IS NULL OR NEW.expires_at <= 0 THEN NULL
             ELSE to_timestamp(NEW.expires_at) END
    );
    NEW.paid_at_ts := COALESCE(
        NEW.paid_at_ts,
        CASE WHEN NEW.paid_at IS NULL OR NEW.paid_at <= 0 THEN NULL
             ELSE to_timestamp(NEW.paid_at) END
    );
    NEW.credited_at_ts := COALESCE(
        NEW.credited_at_ts,
        CASE WHEN NEW.credited_at IS NULL OR NEW.credited_at <= 0 THEN NULL
             ELSE to_timestamp(NEW.credited_at) END
    );
    RETURN NEW;
END
$$ LANGUAGE plpgsql
"""


def upgrade() -> None:
    op.execute(_LN_PROJECTION)
    for table, column in COLUMNS:
        micros = _micros_name(column)
        # Last-chance backfill: a row written between 095 and this migration by
        # code that still set only the float would otherwise lose its value.
        op.execute(
            f"UPDATE {table} SET {micros} = ROUND({column} * 1000000)::bigint "
            f"WHERE {column} IS NOT NULL AND {micros} IS NULL"
        )
        op.execute(f"DROP TRIGGER IF EXISTS trg_mirror_{micros} ON {table}")
        op.execute(f"DROP FUNCTION IF EXISTS mirror_{table}_{micros}()")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")


def downgrade() -> None:
    for table, column in COLUMNS:
        micros = _micros_name(column)
        op.execute(
            f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} DOUBLE PRECISION"
        )
        op.execute(
            f"UPDATE {table} SET {column} = {micros} / 1000000.0 WHERE {micros} IS NOT NULL"
        )
