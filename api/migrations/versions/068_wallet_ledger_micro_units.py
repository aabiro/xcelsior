"""Wallet ledger in integer micro-CAD (Track B B9.5a).

Companion §4.4 rule 6: "Monetary values use integer minor units or
``NUMERIC``, never binary floats." The wallet core is where that matters
most, because two of its columns *accumulate*:

- ``wallets.balance_cad`` is incremented and decremented in place;
- ``wallet_transactions.balance_after_cad`` is a running balance.

Measured against this database, 1000 postings of $0.07 sum to
69.99999999999966 rather than 70.00. Per operation the error is
negligible; what breaks is that ``sum(amount)`` over the ledger stops
equalling the stored balance, and that equality is exactly what
`DA§8.7`'s finance reconciliation checks. A permanently noisy
reconciliation is indistinguishable from a broken one.

**Authority direction differs from migration 066.** There, the legacy
float was authoritative and the typed column derived. Here that would be
useless: deriving ``balance_micros`` from an already-drifted
``balance_cad`` produces a rounded copy of a wrong number. The minor
column is authoritative and the float is projected from it, so the
arithmetic happens in integers.

**The trigger is therefore bidirectional.** During a rolling deploy two
writer generations coexist:

- new code writes ``balance_micros`` -> the trigger derives ``balance_cad``;
- an un-upgraded replica writes only ``balance_cad`` -> the trigger derives
  ``balance_micros``.

A one-way trigger would silently discard the old replica's write by
overwriting the float from a stale minor value. Which branch fires is
decided by comparing OLD and NEW, so no writer needs to know about the
other.

Expand-only. The float columns stay until contract phase (Track B B9.5e /
B16.2), where they are dropped and nothing renames — ``*_micros`` is the
permanent name because it states its unit.

Revision ID: 068
Revises: 067
Create Date: 2026-07-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "068"
down_revision: Union[str, None] = "067"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BACKFILL_BATCH_SIZE = 500

# Dollars-as-float -> integer micro-CAD (1e-6). round() before the cast: a
# stored 25.0 can be 24.999999999999996 and ::bigint truncates toward zero.
#
# NOT cents. Xcelsior meters GPU-seconds and tokens, which produce
# sub-cent charges: a real per-tick charge of $0.0073 rounds to $0.01 in
# cents, a 37% overcharge repeated every tick. The ledger's unit must be
# at least as fine as the smallest amount the business meters. 1e-6 CAD
# in a BIGINT tops out around $9.2 trillion, so range is not a concern.
_MINOR = "CASE WHEN {col} IS NULL THEN NULL ELSE round({col}::numeric * 1000000)::bigint END"
# Integer micro-CAD -> dollars, for legacy readers still on the float.
_MAJOR = "CASE WHEN {col} IS NULL THEN NULL ELSE ({col}::numeric / 1000000)::double precision END"

# (table, [(float_column, minor_column), ...], pk)
WALLET_MONEY: tuple[tuple[str, tuple[tuple[str, str], ...], str], ...] = (
    (
        "wallets",
        (
            ("balance_cad", "balance_micros"),
            ("total_deposited_cad", "total_deposited_micros"),
            ("total_spent_cad", "total_spent_micros"),
            ("total_refunded_cad", "total_refunded_micros"),
            ("auto_topup_amount_cad", "auto_topup_amount_micros"),
            ("auto_topup_threshold_cad", "auto_topup_threshold_micros"),
        ),
        "customer_id",
    ),
    (
        "wallet_transactions",
        (
            ("amount_cad", "amount_micros"),
            ("balance_after_cad", "balance_after_micros"),
        ),
        "tx_id",
    ),
    (
        "wallet_holds",
        (("amount_cad", "amount_micros"),),
        "hold_id",
    ),
)


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    for table, pairs, pk in WALLET_MONEY:
        for _, minor_col in pairs:
            op.execute(
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {minor_col} BIGINT"
            )
        _create_projection_trigger(table, pairs)
        _backfill(table, pairs, pk)

    _verify_backfill()

    # A balance may legitimately be negative (overdraft/grace), so the only
    # structural claim is that the accumulating totals never go backwards.
    op.execute(
        """
        ALTER TABLE wallets
        ADD CONSTRAINT ck_wallets_totals_non_negative
        CHECK (
            (total_deposited_micros IS NULL OR total_deposited_micros >= 0)
            AND (total_spent_micros IS NULL OR total_spent_micros >= 0)
            AND (total_refunded_micros IS NULL OR total_refunded_micros >= 0)
        ) NOT VALID
        """
    )
    op.execute("ALTER TABLE wallets VALIDATE CONSTRAINT ck_wallets_totals_non_negative")


def _create_projection_trigger(table: str, pairs: tuple[tuple[str, str], ...]) -> None:
    """Bidirectional float <-> minor projection.

    Precedence is deliberate: a change to the *minor* column wins, because
    that is the authoritative representation. Only when the minor column
    is untouched and the float moved do we treat the write as coming from
    an un-upgraded replica and derive the minor from it.
    """
    branches = []
    for float_col, minor_col in pairs:
        branches.append(
            f"""
            IF NEW.{minor_col} IS DISTINCT FROM OLD.{minor_col} THEN
                NEW.{float_col} := {_MAJOR.format(col=f"NEW.{minor_col}")};
            ELSIF NEW.{float_col} IS DISTINCT FROM OLD.{float_col} THEN
                NEW.{minor_col} := {_MINOR.format(col=f"NEW.{float_col}")};
            END IF;
            """
        )
    insert_branches = []
    for float_col, minor_col in pairs:
        insert_branches.append(
            f"""
            IF NEW.{minor_col} IS NULL THEN
                NEW.{minor_col} := {_MINOR.format(col=f"NEW.{float_col}")};
            ELSE
                NEW.{float_col} := {_MAJOR.format(col=f"NEW.{minor_col}")};
            END IF;
            """
        )

    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION {table}_project_money()
        RETURNS trigger AS $$
        BEGIN
            IF TG_OP = 'INSERT' THEN
                {''.join(insert_branches)}
            ELSE
                {''.join(branches)}
            END IF;
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_project_money ON {table}")
    op.execute(
        f"""
        CREATE TRIGGER trg_{table}_project_money
        BEFORE INSERT OR UPDATE ON {table}
        FOR EACH ROW EXECUTE FUNCTION {table}_project_money()
        """
    )


def _backfill(table: str, pairs: tuple[tuple[str, str], ...], pk: str) -> None:
    """Seed the minor columns from the current float values, in batches.

    Termination invariant (the one migration 054 had to be repaired for):
    the predicate selects rows whose first minor column is NULL, and the
    UPDATE always writes a non-NULL value for it whenever the float source
    is non-NULL — so every claimed row leaves the candidate set. Rows whose
    source is NULL are excluded here and surfaced by `_verify_backfill`.
    """
    first_float, first_micros = pairs[0]
    sets = ", ".join(
        f"{minor_col} = {_MINOR.format(col=f't.{float_col}')}"
        for float_col, minor_col in pairs
    )
    bind = op.get_bind()
    max_passes = 1_000_000
    for _ in range(max_passes):
        result = bind.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT {pk}
                      FROM {table}
                     WHERE {first_micros} IS NULL
                       AND {first_float} IS NOT NULL
                     LIMIT :batch_size
                       FOR UPDATE SKIP LOCKED
                )
                UPDATE {table} t
                   SET {sets}
                  FROM batch
                 WHERE t.{pk} = batch.{pk}
                """
            ),
            {"batch_size": BACKFILL_BATCH_SIZE},
        )
        if result.rowcount == 0:
            return
    raise RuntimeError(
        f"migration 068 {table} backfill did not converge after {max_passes} "
        f"passes; refusing to loop indefinitely."
    )


def _verify_backfill() -> None:
    """Abort rather than leave a money row unconverted, or silently round.

    Blueprint §13.1 requires verified backfills. A row that loses more
    than half a cent converting to integer cents was never a real CAD
    amount, and rounding it away here would change what a customer is owed
    without anybody seeing it.
    """
    bind = op.get_bind()
    for table, pairs, _ in WALLET_MONEY:
        for float_col, minor_col in pairs:
            unconverted = bind.execute(
                sa.text(
                    f"SELECT count(*) FROM {table} "
                    f"WHERE {float_col} IS NOT NULL AND {minor_col} IS NULL"
                )
            ).scalar_one()
            if unconverted:
                raise RuntimeError(
                    f"migration 068 left {unconverted} {table}.{minor_col} "
                    f"values unconverted."
                )
            drift = bind.execute(
                sa.text(
                    f"SELECT count(*) FROM {table} "
                    f" WHERE {minor_col} IS NOT NULL AND {float_col} IS NOT NULL "
                    f"   AND abs({minor_col}::numeric / 1000000 - {float_col}::numeric) "
                    f"       > 0.0000005"
                )
            ).scalar_one()
            if drift:
                raise RuntimeError(
                    f"migration 068: {drift} {table}.{float_col} values lose "
                    "more than half a cent converting to minor units. Inspect "
                    "them; do not round away a ledger discrepancy."
                )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE wallets DROP CONSTRAINT IF EXISTS ck_wallets_totals_non_negative"
    )
    for table, pairs, _ in WALLET_MONEY:
        op.execute(f"DROP TRIGGER IF EXISTS trg_{table}_project_money ON {table}")
        op.execute(f"DROP FUNCTION IF EXISTS {table}_project_money()")
        for _, minor_col in pairs:
            op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {minor_col}")
