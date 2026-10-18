"""Per-stage execution state for a pipeline. B0 of docs/pipeline-plan.md.

A pipeline is **one `action_plans` row** whose `canonical_args` carries the
graph — that is where the approval lives, and it is why "editing any stage after
approval invalidates it" (Gate P4) needs no new mechanism: the existing
`canonical_args_hash` already voids an altered plan, and that path is exercised
by the promotion work rather than written fresh here.

What the approval substrate has no place for is *what happened* to each stage.
This is that, and nothing more. Approving does not create these rows; executing
does.

**`on_failure` is fixed at approval time**, which is the point of storing it
rather than passing it per call. A user approving a graph approves its failure
behaviour too, and the moment a stage breaks is precisely when that decision is
worst made.

**`max_attempts` is NOT NULL with a default of 1** because unbounded retry
inside an approved spend ceiling is a way to spend the entire ceiling on a stage
that cannot succeed. A retry that is not bounded is not a retry policy.

**Expand-only** (rule 5): one new table, nothing altered.
"""

from alembic import op

revision = "104"
down_revision = "103"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS pipeline_stages (
            plan_id         TEXT NOT NULL,
            stage_index     INTEGER NOT NULL,
            -- Denormalised from the plan. Ledger rule 9 / companion §4.4.10:
            -- a tenant-owned table carries its own tenant_id and leads its
            -- index with it, so a tenant-scoped read never joins back.
            tenant_id       TEXT NOT NULL,
            name            TEXT NOT NULL,
            action_type     TEXT NOT NULL,
            state           TEXT NOT NULL DEFAULT 'pending',
            on_failure      TEXT NOT NULL DEFAULT 'halt',
            max_attempts    INTEGER NOT NULL DEFAULT 1,
            attempt_count   INTEGER NOT NULL DEFAULT 0,
            -- What the stage produced, by reference rather than by value: a job
            -- id, a promotion id. §3.4 — passing artifacts by value would put a
            -- manifest in the plan and make the graph enormous.
            result_ref      TEXT,
            estimate_micros BIGINT NOT NULL DEFAULT 0,
            spent_micros    BIGINT NOT NULL DEFAULT 0,
            failure_code    TEXT,
            started_at      TIMESTAMPTZ,
            finished_at     TIMESTAMPTZ,
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (plan_id, stage_index),
            CONSTRAINT ck_pipeline_stage_state CHECK (
                state IN ('pending', 'running', 'succeeded', 'failed', 'skipped')
            ),
            CONSTRAINT ck_pipeline_stage_on_failure CHECK (
                on_failure IN ('halt', 'continue', 'retry')
            ),
            -- A bounded retry, enforced by the database rather than by the
            -- executor remembering to check.
            CONSTRAINT ck_pipeline_stage_attempts CHECK (
                max_attempts >= 1 AND attempt_count <= max_attempts
            ),
            -- A finished stage has a finish time. Without this, "succeeded with
            -- no finished_at" is representable and the audit chain Gate P4 asks
            -- for has a hole exactly where a reader would look.
            CONSTRAINT ck_pipeline_stage_finished CHECK (
                state NOT IN ('succeeded', 'failed') OR finished_at IS NOT NULL
            )
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_pipeline_stages_tenant "
        "ON pipeline_stages (tenant_id, plan_id, stage_index)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_pipeline_stages_tenant")
    op.execute("DROP TABLE IF EXISTS pipeline_stages")
