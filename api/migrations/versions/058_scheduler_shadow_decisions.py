"""Shadow-mode placement decisions (Track A, blueprint Phase 3).

The shadow scheduler runs the new claim→filter→score pipeline against a
read-only snapshot and persists what it *would* have done — never touching
jobs, attempts, or allocations. A comparator later joins each decision
against what the legacy scheduler actually did and records agreement or a
typed mismatch, which is the Phase 3 exit-gate evidence ("shadow mismatch
reasons are understood and signed off").

Expand-only: nothing but the shadow runner reads or writes this table,
and the legacy scheduler is unaware of it.

Revision ID: 058
Revises: 057
Create Date: 2026-07-17
"""

from alembic import op

revision = "058"
down_revision = "057"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduler_shadow_decisions (
            decision_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            cycle_id         UUID NOT NULL,
            replica_id       TEXT NOT NULL,
            job_id           TEXT NOT NULL,
            -- DB clock at snapshot read; all comparison windows key off this.
            snapshot_at      TIMESTAMPTZ NOT NULL,
            engine           TEXT NOT NULL DEFAULT 'v2',
            policy_version   TEXT NOT NULL,

            -- What the new pipeline decided for this job on this snapshot.
            outcome          TEXT NOT NULL,
            queue_reason_code TEXT,
            selected_host_id TEXT,
            placement_score  BIGINT,
            eligible_host_count INTEGER NOT NULL DEFAULT 0,
            host_count       INTEGER NOT NULL DEFAULT 0,
            -- §3.2 placement explanation: filters, rejections, score
            -- breakdowns. Present for every decision AND non-decision.
            explanation      JSONB NOT NULL,

            -- Filled by the comparator once the legacy scheduler has had
            -- its grace window to act on the same queue state.
            compared_at      TIMESTAMPTZ,
            legacy_status    TEXT,
            legacy_host_id   TEXT,
            comparison       TEXT,

            created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

            CONSTRAINT ck_shadow_outcome CHECK (outcome IN ('place', 'queue')),
            -- Decision shape: a placement names a host; a queue decision
            -- names a reason and no host.
            CONSTRAINT ck_shadow_outcome_shape CHECK (
                (outcome = 'place' AND selected_host_id IS NOT NULL)
                OR
                (outcome = 'queue' AND selected_host_id IS NULL
                 AND queue_reason_code IS NOT NULL)
            ),
            CONSTRAINT ck_shadow_comparison CHECK (
                comparison IS NULL OR comparison IN (
                    'match_place', 'match_queue', 'host_mismatch',
                    'shadow_placed_legacy_queued',
                    'legacy_placed_shadow_queued',
                    'job_missing', 'indeterminate'
                )
            ),
            -- Compared rows always say what they concluded.
            CONSTRAINT ck_shadow_compared_shape CHECK (
                (compared_at IS NULL AND comparison IS NULL)
                OR (compared_at IS NOT NULL AND comparison IS NOT NULL)
            )
        )
        """
    )

    # Comparator work queue: uncompared decisions, oldest snapshot first.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shadow_decisions_uncompared
        ON scheduler_shadow_decisions (snapshot_at)
        WHERE compared_at IS NULL
        """
    )
    # Per-job history (dashboards: "what did shadow think about job X").
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shadow_decisions_job
        ON scheduler_shadow_decisions (job_id, snapshot_at DESC)
        """
    )
    # One scheduling cycle's full decision set.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shadow_decisions_cycle
        ON scheduler_shadow_decisions (cycle_id)
        """
    )
    # Retention pruning + mismatch-rate windows.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shadow_decisions_created
        ON scheduler_shadow_decisions (created_at)
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    op.execute("DROP TABLE IF EXISTS scheduler_shadow_decisions")
