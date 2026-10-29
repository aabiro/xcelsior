"""Lightning deposits: typed money and time columns (Track B B9.3a).

Migration ``060_shared_state_to_pg`` moved ``ln_deposits`` out of SQLite
but carried the SQLite-era column types with it: CAD amounts and the
BTC/CAD rate as ``double precision``, and all four lifecycle timestamps as
epoch floats.

The data-architecture companion §4.4 is explicit:

- rule 5 — "Every table has typed timestamps (``TIMESTAMPTZ``) and
  database-generated times where ordering matters";
- rule 6 — "Monetary values use integer minor units or ``NUMERIC``, never
  binary floats".

Binary floats cannot represent most decimal cent values exactly, so a
wallet credit derived from ``amount_cad`` accumulates error and two
independently computed totals need not agree. This is a ledger input, so
the exact type is not a style preference.

Expand-only, and safe for a rolling deploy:

- the new columns are added nullable and backfilled in bounded batches
  with hard verification, aborting rather than leaving a row unprojected;
- a ``BEFORE INSERT OR UPDATE`` trigger derives every new column from the
  legacy one whenever the new value is not supplied, so an **old** API
  replica still writing only floats keeps the typed columns correct for
  the duration of the rollout;
- new code writes the typed columns directly, so a fresh deposit never
  makes a float round trip;
- the legacy float columns remain authoritative for nothing and are
  dropped at contract phase (Track B §B16.2), where ``created_at_ts`` is
  also renamed back to ``created_at``.

Money representation: ``amount_cad_minor`` is CAD **cents** as BIGINT.
``btc_cad_rate_exact`` is ``NUMERIC(20,8)`` — a rate, not an amount, so
minor units do not apply, but it still must not be a binary float.
``amount_btc`` is deliberately *not* given a typed twin: it is a lossy
restatement of ``amount_sats`` (which is already an exact BIGINT), so it
is dropped at contract rather than preserved.

Revision ID: 066
Revises: 065
Create Date: 2026-07-22
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "066"
down_revision: Union[str, None] = "065"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

BACKFILL_BATCH_SIZE = 500

# Epoch-float → TIMESTAMPTZ. The legacy schema uses 0 (not NULL) as the
# "never happened" sentinel for paid_at/credited_at, so 0 must project to
# NULL rather than to 1970-01-01.
_TS_EXPR = "CASE WHEN {col} IS NULL OR {col} <= 0 THEN NULL ELSE to_timestamp({col}) END"

# Dollars-as-float → integer cents. round() before cast: a float 25.0 can
# be 24.999999999999996, and a plain ::bigint cast truncates toward zero.
_MINOR_EXPR = "CASE WHEN {col} IS NULL THEN NULL ELSE round({col}::numeric * 100)::bigint END"


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── typed columns, nullable during expand ────────────────────────
    op.execute(
        "ALTER TABLE ln_deposits ADD COLUMN IF NOT EXISTS amount_cad_minor BIGINT"
    )
    op.execute(
        "ALTER TABLE ln_deposits "
        "ADD COLUMN IF NOT EXISTS btc_cad_rate_exact NUMERIC(20,8)"
    )
    for col in ("created_at_ts", "expires_at_ts", "paid_at_ts", "credited_at_ts"):
        op.execute(f"ALTER TABLE ln_deposits ADD COLUMN IF NOT EXISTS {col} TIMESTAMPTZ")

    # ── projection trigger ───────────────────────────────────────────
    # Derives each typed column from its legacy twin when the caller did
    # not supply one. An old replica writing only floats therefore keeps
    # the typed columns correct, and new code supplying exact values wins
    # (COALESCE takes NEW.<typed> first).
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION ln_deposits_project_typed()
        RETURNS trigger AS $$
        BEGIN
            NEW.amount_cad_minor := COALESCE(
                NEW.amount_cad_minor,
                {_MINOR_EXPR.format(col="NEW.amount_cad")}
            );
            NEW.btc_cad_rate_exact := COALESCE(
                NEW.btc_cad_rate_exact,
                NEW.btc_cad_rate::numeric
            );
            NEW.created_at_ts := COALESCE(
                NEW.created_at_ts, {_TS_EXPR.format(col="NEW.created_at")}
            );
            NEW.expires_at_ts := COALESCE(
                NEW.expires_at_ts, {_TS_EXPR.format(col="NEW.expires_at")}
            );
            NEW.paid_at_ts := COALESCE(
                NEW.paid_at_ts, {_TS_EXPR.format(col="NEW.paid_at")}
            );
            NEW.credited_at_ts := COALESCE(
                NEW.credited_at_ts, {_TS_EXPR.format(col="NEW.credited_at")}
            );
            RETURN NEW;
        END
        $$ LANGUAGE plpgsql
        """
    )
    op.execute("DROP TRIGGER IF EXISTS trg_ln_deposits_project_typed ON ln_deposits")
    op.execute(
        """
        CREATE TRIGGER trg_ln_deposits_project_typed
        BEFORE INSERT OR UPDATE ON ln_deposits
        FOR EACH ROW EXECUTE FUNCTION ln_deposits_project_typed()
        """
    )

    _backfill()
    _verify_backfill()

    # ── constraints: NOT VALID, then validated ───────────────────────
    # A deposit is a positive amount by construction (create_deposit
    # enforces LN_MIN_CAD); a zero or negative one is a data bug.
    op.execute(
        """
        ALTER TABLE ln_deposits
        ADD CONSTRAINT ck_ln_deposits_amount_minor_positive
        CHECK (amount_cad_minor IS NULL OR amount_cad_minor > 0) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE ln_deposits VALIDATE CONSTRAINT ck_ln_deposits_amount_minor_positive"
    )
    op.execute(
        """
        ALTER TABLE ln_deposits
        ADD CONSTRAINT ck_ln_deposits_rate_positive
        CHECK (btc_cad_rate_exact IS NULL OR btc_cad_rate_exact > 0) NOT VALID
        """
    )
    op.execute(
        "ALTER TABLE ln_deposits VALIDATE CONSTRAINT ck_ln_deposits_rate_positive"
    )

    # Operator/reconciliation access path: "paid but not credited" is the
    # companion §10.1 reconciliation case that matters most.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_ln_deposits_status_created
        ON ln_deposits (status, created_at_ts DESC)
        """
    )


def _backfill() -> None:
    """Project legacy float columns into the typed ones, in batches.

    Termination invariant (the same one migration 054 had to be repaired
    for): the batch predicate selects only rows whose projection is
    *guaranteed* to leave the candidate set. ``created_at`` is NOT NULL in
    the legacy schema and a deposit's creation time is never <= 0, so
    ``created_at_ts`` is always set to a non-NULL value by this UPDATE. A
    row that somehow violates that is excluded here and surfaced by
    ``_verify_backfill`` instead of spinning forever.
    """
    bind = op.get_bind()
    max_passes = 1_000_000
    for _ in range(max_passes):
        result = bind.execute(
            sa.text(
                f"""
                WITH batch AS (
                    SELECT deposit_id
                      FROM ln_deposits
                     WHERE created_at_ts IS NULL
                       AND created_at IS NOT NULL
                       AND created_at > 0
                     LIMIT :batch_size
                       FOR UPDATE SKIP LOCKED
                )
                UPDATE ln_deposits d
                   SET amount_cad_minor =
                           {_MINOR_EXPR.format(col="d.amount_cad")},
                       btc_cad_rate_exact = d.btc_cad_rate::numeric,
                       created_at_ts = {_TS_EXPR.format(col="d.created_at")},
                       expires_at_ts = {_TS_EXPR.format(col="d.expires_at")},
                       paid_at_ts = {_TS_EXPR.format(col="d.paid_at")},
                       credited_at_ts = {_TS_EXPR.format(col="d.credited_at")}
                  FROM batch
                 WHERE d.deposit_id = batch.deposit_id
                """
            ),
            {"batch_size": BACKFILL_BATCH_SIZE},
        )
        if result.rowcount == 0:
            return
    raise RuntimeError(
        f"migration 066 ln_deposits backfill did not converge after "
        f"{max_passes} passes; refusing to loop indefinitely."
    )


def _verify_backfill() -> None:
    """Abort rather than leave a money row unprojected (blueprint §13.1)."""
    bind = op.get_bind()

    unprojected = bind.execute(
        sa.text(
            "SELECT count(*) FROM ln_deposits "
            "WHERE created_at_ts IS NULL OR amount_cad_minor IS NULL"
        )
    ).scalar_one()
    if unprojected:
        sample = (
            bind.execute(
                sa.text(
                    "SELECT deposit_id, created_at, amount_cad FROM ln_deposits "
                    "WHERE created_at_ts IS NULL OR amount_cad_minor IS NULL "
                    "LIMIT 10"
                )
            )
            .mappings()
            .all()
        )
        raise RuntimeError(
            f"migration 066 left {unprojected} ln_deposits rows without typed "
            f"money/time columns; sample: {[dict(r) for r in sample]}. "
            "Repair the legacy values and re-run."
        )

    # Money must round-trip. If cents and the legacy float disagree by more
    # than half a cent, the legacy value was not a real CAD amount and
    # silently truncating it would change what a customer is owed.
    drift = bind.execute(
        sa.text(
            "SELECT count(*) FROM ln_deposits "
            "WHERE abs(amount_cad_minor::numeric / 100 - amount_cad::numeric) > 0.005"
        )
    ).scalar_one()
    if drift:
        raise RuntimeError(
            f"migration 066: {drift} ln_deposits rows lose more than half a "
            "cent converting amount_cad to integer minor units. Inspect them "
            "before continuing; do not round away a ledger discrepancy."
        )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")

    op.execute("DROP TRIGGER IF EXISTS trg_ln_deposits_project_typed ON ln_deposits")
    op.execute("DROP FUNCTION IF EXISTS ln_deposits_project_typed()")
    op.execute("DROP INDEX IF EXISTS idx_ln_deposits_status_created")
    op.execute(
        "ALTER TABLE ln_deposits "
        "DROP CONSTRAINT IF EXISTS ck_ln_deposits_amount_minor_positive"
    )
    op.execute(
        "ALTER TABLE ln_deposits DROP CONSTRAINT IF EXISTS ck_ln_deposits_rate_positive"
    )
    for col in (
        "amount_cad_minor",
        "btc_cad_rate_exact",
        "created_at_ts",
        "expires_at_ts",
        "paid_at_ts",
        "credited_at_ts",
    ):
        op.execute(f"ALTER TABLE ln_deposits DROP COLUMN IF EXISTS {col}")
