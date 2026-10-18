"""Remove the float money columns and the triggers that maintained them.

Money is stored as integer micros. Every table below also carried a `_cad`
double-precision twin, kept in step by a projection trigger, so each money
write fired PL/pgSQL that recomputed floats on the hot path and every row
carried two representations of the same number.

That is not a compatibility layer worth keeping:

- the trigger functions total ~140 lines of PL/pgSQL executing on every insert
  and update to the busiest tables on the platform;
- two representations invite reads of the wrong one, and a float read is
  silently lossy rather than loudly wrong;
- `payout_splits` had the pairs with **no** trigger at all, so its two halves
  were written independently and could diverge — in the settlement path.

Application code now derives the CAD view from micros where it needs one, so
the stored floats have no readers. The API contract is unchanged: it still
speaks CAD, converted at the boundary by ``money.micros_to_cad``.

Dropping the columns requires dropping the triggers in the same transaction:
PL/pgSQL resolves ``NEW.<field>`` at execution time, so a function left
referencing a dropped column fails every write to its table.
"""

from alembic import op

revision = "087"
down_revision = "086"
branch_labels = None
depends_on = None


# (table, trigger, function, [float columns])
DERIVED = (
    (
        "wallets",
        "trg_wallets_project_money",
        "wallets_project_money",
        ("balance_cad", "total_deposited_cad", "total_spent_cad", "total_refunded_cad"),
    ),
    (
        "wallet_transactions",
        "trg_wallet_transactions_project_money",
        "wallet_transactions_project_money",
        ("amount_cad", "balance_after_cad"),
    ),
    (
        "wallet_holds",
        "trg_wallet_holds_project_money",
        "wallet_holds_project_money",
        ("amount_cad",),
    ),
    (
        "usage_meters",
        "trg_usage_meters_project_total_cost_money",
        "usage_meters_project_total_cost_money",
        ("total_cost_cad",),
    ),
    # No trigger: these were written independently by stripe_connect, which is
    # exactly why they are going.
    (
        "payout_splits",
        None,
        None,
        ("total_cad", "provider_share_cad", "platform_share_cad", "gst_hst_cad"),
    ),
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    for table, trigger, function, columns in DERIVED:
        # Reconcile payout_splits before dropping: with no trigger, its float
        # and integer halves could already disagree, and the integer column is
        # the one the settlement code computes from.
        if table == "payout_splits":
            # ck_payout_splits_exact_money requires the parts to reconcile:
            # source_total + rounding == total, and provider + platform ==
            # total. Setting total alone from the float would violate it, so
            # rebuild every component in one statement and derive the platform
            # share as the remainder — that keeps the identity exact instead of
            # trusting two independently rounded floats to add up.
            op.execute(
                """
                UPDATE payout_splits
                   SET total_micros = ROUND(COALESCE(total_cad, 0)::numeric * 1000000)::bigint,
                       source_total_micros =
                           ROUND(COALESCE(total_cad, 0)::numeric * 1000000)::bigint
                           - COALESCE(rounding_adjustment_micros, 0),
                       provider_share_micros =
                           ROUND(COALESCE(provider_share_cad, 0)::numeric * 1000000)::bigint,
                       platform_share_micros =
                           ROUND(COALESCE(total_cad, 0)::numeric * 1000000)::bigint
                           - ROUND(COALESCE(provider_share_cad, 0)::numeric * 1000000)::bigint,
                       gst_hst_micros =
                           ROUND(COALESCE(gst_hst_cad, 0)::numeric * 1000000)::bigint
                 WHERE total_micros IS NULL
                    OR total_micros <> ROUND(COALESCE(total_cad, 0)::numeric * 1000000)::bigint
                """
            )
        if trigger:
            op.execute(f"DROP TRIGGER IF EXISTS {trigger} ON {table}")
        if function:
            op.execute(f"DROP FUNCTION IF EXISTS {function}()")
        for column in columns:
            op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")


def downgrade() -> None:
    """Restore the columns only.

    The projection triggers are deliberately not recreated: they were the
    mechanism being removed, their bodies are recoverable from migration
    history, and recreating them here would silently reintroduce float writes
    on the hot path. Values are reconstructed from micros, which is
    authoritative.
    """
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    for table, _trigger, _function, columns in DERIVED:
        for column in columns:
            base = column[: -len("_cad")]
            op.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} "
                f"DOUBLE PRECISION DEFAULT 0"
            )
            op.execute(
                f"UPDATE {table} SET {column} = "
                f"COALESCE({base}_micros, 0) / 1000000.0"
            )
