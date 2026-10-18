"""P3/Phase-E/E8 — add error column to user_images.

Used by:
- bg_worker's ``snapshot_queue_retry`` task to mark rows that were
  waiting on a now-missing host (``error='host_missing'``).
- bg_worker's ``user_images_pending_sweeper`` to record why a row was
  auto-failed (``error='registry_down_24h'`` for rows that stayed in
  ``queued_registry_down`` status past the 24h cutoff).
- Future callers that want to explain *why* a snapshot failed without
  stuffing the reason into ``status``.

Idempotent — db.py also adds this column at startup for fresh installs.

Revision ID: 027
Revises: 026
Create Date: 2026-04-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "027"
down_revision: Union[str, None] = "026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE user_images "
        "ADD COLUMN IF NOT EXISTS error TEXT DEFAULT ''"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE user_images DROP COLUMN IF EXISTS error")
