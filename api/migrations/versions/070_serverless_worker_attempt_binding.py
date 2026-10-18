"""Serverless workers bind to a fenced attempt (Track B B3.1).

Blueprint §15.3 / §5.6: a ``serverless_workers`` row must reference the fenced
``job_attempts`` row that actually holds its GPU. Today the worker links only to
``scheduler_job_id``; the fenced attempt (and, transitively, its
``gpu_device_allocations``) is created later by the scheduler when it places the
worker's job. This adds the durable ``attempt_id`` back-reference so serverless
capacity is attempt-scoped like compute metering already is — one worker, one
fenced attempt, one allocation set.

Expand-only and additive: the column is nullable (a worker is bound the moment
its attempt is reserved — B3.1 wires ``reserve_and_bind`` to stamp it), the FK
is added ``NOT VALID`` then ``VALIDATE``d (cheap — every existing row is NULL),
and ``ON DELETE SET NULL`` keeps a worker row alive if its attempt is ever
hard-deleted. Nothing reads it until the provisioning path lands.

Revision ID: 070
Revises: 069
Create Date: 2026-07-23
"""

from typing import Sequence, Union

from alembic import op

revision: str = "070"
down_revision: Union[str, None] = "069"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE serverless_workers ADD COLUMN IF NOT EXISTS attempt_id uuid")
    # Additive FK — NOT VALID skips the full-table scan (all rows are NULL),
    # then VALIDATE confirms without a strong lock. ON DELETE SET NULL so a
    # deleted attempt never orphans the worker row.
    op.execute(
        """
        ALTER TABLE serverless_workers
            ADD CONSTRAINT fk_serverless_workers_attempt
            FOREIGN KEY (attempt_id) REFERENCES job_attempts(attempt_id)
            ON DELETE SET NULL
            NOT VALID
        """
    )
    op.execute("ALTER TABLE serverless_workers VALIDATE CONSTRAINT fk_serverless_workers_attempt")
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_serverless_workers_attempt_id "
        "ON serverless_workers (attempt_id) WHERE attempt_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_serverless_workers_attempt_id")
    op.execute("ALTER TABLE serverless_workers DROP CONSTRAINT IF EXISTS fk_serverless_workers_attempt")
    op.execute("ALTER TABLE serverless_workers DROP COLUMN IF EXISTS attempt_id")
