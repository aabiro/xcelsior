"""Rename ``legal_requests.jurisdiction`` to ``requesting_country``.

This column is not residency. It records which legal authority demanded data —
the standard axis of a transparency report ("requests received: Canada 3, US 1").
That meaning is worth keeping.

The word is not. "Jurisdiction" is the vocabulary of the Canada-first placement
model that migrations 092 and 093 removed, and leaving it here means every
search for that model keeps returning a hit in a file that has nothing to do
with it. ``requesting_country`` says exactly what the column holds.

Data is preserved: this is a rename, not a drop.
"""

from alembic import op

revision = "094"
down_revision = "093"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE legal_requests RENAME COLUMN jurisdiction TO requesting_country")


def downgrade() -> None:
    op.execute("ALTER TABLE legal_requests RENAME COLUMN requesting_country TO jurisdiction")
