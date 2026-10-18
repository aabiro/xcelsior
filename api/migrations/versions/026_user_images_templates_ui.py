"""Phase E/E6 — add Phase-D UI columns to user_images.

Adds three columns used by the /dashboard/templates surface:

* ``is_public`` — visibility toggle (default private). Public images are
  listed under the "Community" tab for all authenticated users.
* ``labels``    — JSONB array of free-form strings for user-defined
  organization ("prod", "experiment-42", "tf-2.15", …).
* ``starred_at`` — nullable timestamp; set when the user stars an image,
  cleared when un-starred. Enables efficient "Starred only" filter +
  sort-by-recency-of-star.

Two partial indexes keep the public + starred filters fast without
penalizing the dominant "my templates" query that already uses the
``(owner_id, deleted_at)`` index from migration 024.

Revision ID: 026
Revises: 025
Create Date: 2026-04-22
"""

from typing import Sequence, Union

from alembic import op


revision: str = "026"
down_revision: Union[str, None] = "025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE user_images
            ADD COLUMN IF NOT EXISTS is_public  boolean NOT NULL DEFAULT false,
            ADD COLUMN IF NOT EXISTS labels     jsonb   NOT NULL DEFAULT '[]'::jsonb,
            ADD COLUMN IF NOT EXISTS starred_at double precision NULL
        """
    )
    # Partial index: public templates across all users. We filter
    # `deleted_at=0` so soft-deleted rows don't appear in the community
    # tab and don't bloat the index.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_images_public_live
            ON user_images (created_at DESC)
            WHERE is_public = true AND deleted_at = 0
        """
    )
    # Partial index: per-user starred templates, ordered by most-recent
    # star first. Matches the "Starred only" view's default sort.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_images_starred_per_owner
            ON user_images (owner_id, starred_at DESC)
            WHERE starred_at IS NOT NULL AND deleted_at = 0
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_user_images_starred_per_owner")
    op.execute("DROP INDEX IF EXISTS idx_user_images_public_live")
    op.execute(
        """
        ALTER TABLE user_images
            DROP COLUMN IF EXISTS starred_at,
            DROP COLUMN IF EXISTS labels,
            DROP COLUMN IF EXISTS is_public
        """
    )
