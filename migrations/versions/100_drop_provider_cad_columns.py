"""Drop the last two ``_cad`` money columns in the schema.

``provider_accounts.total_earned_cad`` and ``total_paid_out_cad`` are the only
columns left anywhere whose name ends in ``_cad``. Every other float money
column went in ``095``–``097``; these two survived because they are ``NUMERIC``
rather than ``DOUBLE PRECISION``, so a sweep aimed at float money did not see
them.

**They are dropped rather than converted, and that is the point.** The obvious
reading of "money is integer micros" is to add ``total_earned_micros`` beside
them and backfill. That would create two *new* dead columns to mirror two old
ones. The earnings figure this platform actually serves is computed at read
time from ``payout_splits.provider_share_micros``:

    stripe_connect.py:1116
        COALESCE(SUM(ps.provider_share_micros) / 1000000.0, 0) AS total_earned_cad

The ``_cad`` there is a **response field name**, not a column read — the API
speaks CAD at its boundary and stores micros, which is the intended design.
These two columns are stale denormalised totals written once by ``014`` and
maintained by nothing since.

**Why now, when ``085`` deliberately kept them.** That migration classified them
as *keep*, and its reasoning was right at the time:

    ``provider_accounts.total_paid_out_cad``, ``jobs_hosted`` and
    ``last_payout_at`` are provider financial history. They are empty in
    development, which says nothing about production, and payout records are the
    kind of thing that has to be retained deliberately rather than dropped
    because no code path happens to read them today.

The missing evidence was what production held. It has now been asked:

    PROD provider_accounts: rows=1 earned_nonzero=0 paid_nonzero=0
                            sum_earned=0.00 sum_paid=0.00

One row, both columns zero. There is no financial history to lose, which is the
condition ``085`` was waiting on and could not check from a development box.
``jobs_hosted`` and ``last_payout_at`` stay — they are not money columns and
this migration is not about them.

Rule 5 permits contraction "after the legacy-use metric reads zero". It reads
zero three ways: no reader or writer in the tree, no non-zero value in
production, and a replacement (``provider_share_micros``) that is already the
authority.

Rule 4 requires contract cleanup to be the last revision in the chain; this is
``100``, the head.

One table, so no ``lock_safe`` fan-out is needed — ``migrations/env.py`` sets the
session ``lock_timeout`` that rule 5 requires, and a single ``ALTER TABLE`` on a
one-row table takes its lock briefly.

``downgrade`` restores both columns with their original type and default. It
cannot restore data, which costs nothing here precisely because there is none —
and that is stated rather than left for someone to assume.
"""

from alembic import op

revision = "100"
down_revision = "099"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS total_earned_cad")
    op.execute("ALTER TABLE provider_accounts DROP COLUMN IF EXISTS total_paid_out_cad")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS total_earned_cad NUMERIC(12, 2) NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE provider_accounts "
        "ADD COLUMN IF NOT EXISTS total_paid_out_cad NUMERIC(12, 2) NOT NULL DEFAULT 0"
    )
