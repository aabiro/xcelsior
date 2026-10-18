"""Partition `placement_decisions`, because WORM without pruning is a leak.

**105 copied half a pattern.** Its docstring claims WORM "like `075` and `072`".
It took the trigger from both and the partitioning from neither, and
`072_audit_events_v2.py` states plainly why that matters:

    WORM: reject any row UPDATE/DELETE. Partition drops (retention) are DDL and
    are unaffected.

That is the whole mechanism by which an append-only table stays prunable:
immutable rows, droppable partitions. `audit_events_v2` has it. `075` does not —
and does not need it, being one row per signed checkpoint. **`105` is per
request** and copied the low-volume precedent, giving a table that grows without
bound and whose own trigger forbids the only statement that could reduce it.

## Why this is a rebuild, and why now

A partitioned table's primary key must contain the partition key, so
`decision_id` alone cannot remain the key — it becomes
`(decision_id, decided_at)`. That is not an ALTER; the table has to be recreated.

Right now that costs least: nothing writes to `placement_decisions` yet, because
the caller is exactly the piece Gate P5 clause 3 is still missing.

**Existing rows are carried across, not refused.** The first draft of this
migration aborted when the table was non-empty, on the theory that dropping an
audit table is never safe. That was over-cautious in a way that would have
bricked every database that had ever run the WORM tests — those deliberately
leave rows behind, because the table refuses DELETE. The move is safe precisely
because of what WORM does and does not block: it rejects UPDATE and DELETE on
rows, while `INSERT … SELECT` and `DROP TABLE` are untouched. So the rows are
copied into the partitioned table and the old one is dropped, which is what an
operator would have been told to do by hand anyway.

**Expand-only in spirit** (rule 5): the object is replaced, not altered, and no
other table is touched.
"""

import datetime as _dt

import sqlalchemy as sa
from alembic import op

revision = "106"
down_revision = "105"
branch_labels = None
depends_on = None

#: This month plus three. The maintenance task extends the window daily; these
#: exist so the first write after deploy never has to create one inline.
_INITIAL_PARTITION_MONTHS = 4


def _month_bounds(start: _dt.date, offset: int) -> tuple[str, str, str]:
    year = start.year + (start.month - 1 + offset) // 12
    month = (start.month - 1 + offset) % 12 + 1
    frm = _dt.date(year, month, 1)
    to = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    return f"{year:04d}{month:02d}", frm.isoformat(), to.isoformat()


def _table_columns() -> str:
    """Identical to 105 except for the primary key, which must carry the range."""
    return """
        decision_id           UUID NOT NULL DEFAULT gen_random_uuid(),
        tenant_id             TEXT NOT NULL,
        job_id                TEXT,
        host_id               TEXT,
        outcome               TEXT NOT NULL,
        refusal_code          TEXT,
        refusal_detail        TEXT,
        asked                 JSONB NOT NULL DEFAULT '{}'::jsonb,
        evidence              JSONB NOT NULL DEFAULT '{}'::jsonb,
        candidate_count       INTEGER NOT NULL DEFAULT 0,
        candidates            JSONB NOT NULL DEFAULT '[]'::jsonb,
        baseline_price_micros BIGINT,
        chosen_price_micros   BIGINT,
        decided_at            TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY (decision_id, decided_at),
        CONSTRAINT ck_placement_outcome CHECK (outcome IN ('placed', 'refused')),
        CONSTRAINT ck_placement_shape CHECK (
            (outcome = 'placed'  AND host_id IS NOT NULL AND refusal_code IS NULL)
         OR (outcome = 'refused' AND refusal_code IS NOT NULL)
        ),
        CONSTRAINT ck_placement_prices CHECK (
            (baseline_price_micros IS NULL OR baseline_price_micros > 0)
        AND (chosen_price_micros  IS NULL OR chosen_price_micros  > 0)
        )
    """


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")

    existed = op.get_bind().execute(
        sa.text("SELECT to_regclass('placement_decisions')")
    ).scalar()

    # Build the partitioned table beside the old one. The **children are given
    # their final names now** — a child's name is independent of its parent's,
    # so renaming the parent later would otherwise leave partitions called
    # `placement_decisions_new_202608`, and the maintenance task would then
    # create `placement_decisions_202608` for a range already covered and fail.
    op.execute(
        f"CREATE TABLE placement_decisions_new ({_table_columns()}) "
        "PARTITION BY RANGE (decided_at)"
    )

    today = _dt.date.today().replace(day=1)
    for offset in range(_INITIAL_PARTITION_MONTHS):
        suffix, frm, to = _month_bounds(today, offset)
        op.execute(
            f"CREATE TABLE placement_decisions_{suffix} "
            f"PARTITION OF placement_decisions_new FOR VALUES FROM ('{frm}') TO ('{to}')"
        )
    # Safety net, and the landing place for any carried-over row older than the
    # window. A write beyond the pre-created range lands here instead of
    # failing: an audit write must never be the thing that fails a placement.
    op.execute(
        "CREATE TABLE placement_decisions_default "
        "PARTITION OF placement_decisions_new DEFAULT"
    )

    if existed:
        # WORM blocks UPDATE and DELETE on rows. INSERT … SELECT and DROP TABLE
        # are untouched, which is exactly why the trail can be moved without
        # ever being mutable.
        op.execute(
            """
            INSERT INTO placement_decisions_new
                (decision_id, tenant_id, job_id, host_id, outcome, refusal_code,
                 refusal_detail, asked, evidence, candidate_count, candidates,
                 baseline_price_micros, chosen_price_micros, decided_at)
            SELECT decision_id, tenant_id, job_id, host_id, outcome, refusal_code,
                   refusal_detail, asked, evidence, candidate_count, candidates,
                   baseline_price_micros, chosen_price_micros, decided_at
              FROM placement_decisions
            """
        )
        op.execute(
            "DROP TRIGGER IF EXISTS trg_placement_decisions_immutable ON placement_decisions"
        )
        op.execute("DROP TABLE placement_decisions")

    op.execute("ALTER TABLE placement_decisions_new RENAME TO placement_decisions")

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_decisions_tenant "
        "ON placement_decisions (tenant_id, decided_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_decisions_job "
        "ON placement_decisions (job_id, decided_at DESC)"
    )

    op.execute(
        """
        CREATE OR REPLACE FUNCTION placement_decisions_immutable() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'placement_decisions is append-only (WORM); % is not permitted', TG_OP
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    # On a partitioned parent this propagates to every partition, existing and
    # future. Partition drops are DDL and remain unaffected — which is the
    # entire point of the rebuild.
    op.execute(
        """
        CREATE TRIGGER trg_placement_decisions_immutable
            BEFORE UPDATE OR DELETE ON placement_decisions
            FOR EACH ROW EXECUTE FUNCTION placement_decisions_immutable()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_placement_decisions_immutable ON placement_decisions")
    op.execute("DROP TABLE IF EXISTS placement_decisions CASCADE")
    op.execute(
        f"CREATE TABLE IF NOT EXISTS placement_decisions ({_table_columns()})"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_decisions_tenant "
        "ON placement_decisions (tenant_id, decided_at DESC)"
    )
    op.execute(
        """
        CREATE TRIGGER trg_placement_decisions_immutable
            BEFORE UPDATE OR DELETE ON placement_decisions
            FOR EACH ROW EXECUTE FUNCTION placement_decisions_immutable()
        """
    )
