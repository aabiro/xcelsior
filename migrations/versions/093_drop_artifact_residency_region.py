"""Drop ``storage.artifacts.residency_region``.

Artifact storage routing is a durability and cost decision — a durable primary
bucket or a cheaper cache bucket — never a jurisdiction one. The column existed
to record which country an object landed in so a "Canada only" storage policy
could be asserted; that policy is gone, and a column that records a guarantee
the platform no longer makes is worse than no column, because reports will keep
being written against it.

Separate from 092 because 092 was already applied; a new revision keeps the
history honest rather than editing an applied migration.
"""

from alembic import op

revision = "093"
down_revision = "092"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE storage.artifacts DROP COLUMN IF EXISTS residency_region")


def downgrade() -> None:
    op.execute("ALTER TABLE storage.artifacts ADD COLUMN IF NOT EXISTS residency_region TEXT")
