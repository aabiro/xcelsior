"""Partitioned, append-only audit_events_v2 (Track B B4.1).

Blueprint §13.6 / companion §4.5: the durable, hash-chained audit stream — a new
partitioned table, **not** a rewrite of the live `events` table. Monthly RANGE
partitions on `created_at`, pre-created ahead of time (never ad hoc in a request
handler), with a DEFAULT safety partition so a write beyond the pre-created
range lands somewhere and pages via the partition-lag metric instead of failing.

Schema: tenant, stream type/id + per-stream sequence, aggregate version, event
type, actor/client/request/trace ids, a redacted immutable payload, and the
per-stream `prev_hash`/`event_hash` hash-chain columns.

**PostgreSQL partitioned-table rule:** a UNIQUE/PRIMARY KEY on a partitioned
table must include the partition key, so `event_id` and `(stream_id,
stream_sequence)` uniqueness carry `created_at`. Global uniqueness is anchored by
UUID `event_id` generation and by the per-stream advisory-lock sequence the
append path uses (the same discipline the existing `events` hash chain uses) —
the DB constraints are the in-partition backstop.

**Immutability (WORM):** a BEFORE UPDATE OR DELETE trigger rejects row mutation,
so once written an audit row can never be altered. Retention drops whole
partitions (DDL), which the row trigger does not block.

Expand-only; nothing reads or writes it until the append/registry path lands
(B4.2+).

Revision ID: 072
Revises: 071
Create Date: 2026-07-24
"""

import datetime as _dt
from typing import Sequence, Union

from alembic import op

revision: str = "072"
down_revision: Union[str, None] = "071"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Pre-create this many monthly partitions from the current month. The
# maintenance task (control_plane/audit_partitions.py) keeps the window full.
_INITIAL_PARTITION_MONTHS = 3


def _month_bounds(start: _dt.date, offset: int) -> tuple[str, str, str]:
    """(suffix, from_iso, to_iso) for the month `offset` months after start."""
    year = start.year + (start.month - 1 + offset) // 12
    month = (start.month - 1 + offset) % 12 + 1
    frm = _dt.date(year, month, 1)
    to = _dt.date(year + 1, 1, 1) if month == 12 else _dt.date(year, month + 1, 1)
    return f"{year:04d}{month:02d}", frm.isoformat(), to.isoformat()


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_events_v2 (
            event_id         UUID NOT NULL DEFAULT gen_random_uuid(),
            tenant_id        TEXT,
            stream_type      TEXT NOT NULL,
            stream_id        TEXT NOT NULL,
            stream_sequence  BIGINT NOT NULL,
            aggregate_version BIGINT NOT NULL DEFAULT 0,
            event_type       TEXT NOT NULL,
            actor_id         TEXT,
            client_id        TEXT,
            request_id       TEXT,
            trace_id         TEXT,
            classification   TEXT NOT NULL DEFAULT 'internal',
            payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
            prev_hash        TEXT,
            event_hash       TEXT NOT NULL,
            created_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (event_id, created_at),
            UNIQUE (stream_id, stream_sequence, created_at)
        ) PARTITION BY RANGE (created_at)
        """
    )

    today = _dt.date.today().replace(day=1)
    for offset in range(_INITIAL_PARTITION_MONTHS):
        suffix, frm, to = _month_bounds(today, offset)
        op.execute(
            f"""
            CREATE TABLE IF NOT EXISTS audit_events_v2_{suffix}
            PARTITION OF audit_events_v2
            FOR VALUES FROM ('{frm}') TO ('{to}')
            """
        )
    # Safety net: a write beyond the pre-created range lands here (and pages via
    # the partition-lag metric) instead of failing the append.
    op.execute(
        "CREATE TABLE IF NOT EXISTS audit_events_v2_default PARTITION OF audit_events_v2 DEFAULT"
    )

    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_v2_stream "
        "ON audit_events_v2 (stream_id, stream_sequence)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_v2_type "
        "ON audit_events_v2 (event_type, created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_events_v2_tenant "
        "ON audit_events_v2 (tenant_id, created_at DESC)"
    )

    # WORM: reject any row UPDATE/DELETE. Partition drops (retention) are DDL and
    # are unaffected.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit_events_v2_immutable() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_events_v2 is append-only (WORM); % is not permitted', TG_OP
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_audit_events_v2_immutable
            BEFORE UPDATE OR DELETE ON audit_events_v2
            FOR EACH ROW EXECUTE FUNCTION audit_events_v2_immutable()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_audit_events_v2_immutable ON audit_events_v2")
    op.execute("DROP FUNCTION IF EXISTS audit_events_v2_immutable()")
    op.execute("DROP TABLE IF EXISTS audit_events_v2 CASCADE")
