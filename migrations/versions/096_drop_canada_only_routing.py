"""Drop ``users.canada_only_routing``.

A per-user preference that restricted that user's workloads to Canadian hosts.
It is the last piece of the Canada-first placement model: 092 removed the
pricing and recording columns, 093 the artifact routing column, 094 the naming,
and the code paths went with them. This column outlived all of it because it
lived on ``users`` rather than on anything placement-shaped, and the only thing
still reading it was a compliance-checklist item asking whether the user had
turned it on.

Xcelsior is a global marketplace. Capacity is selected on price, availability,
hardware, and host reputation. There is no setting that changes that, so there
is no setting to store.

The downgrade restores the column but not its values. Re-deriving them would
mean reconstructing a preference the platform no longer honours, and a column
full of defaults that nothing reads is worse than an absent one.
"""

from alembic import op

revision = "096"
down_revision = "095"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS canada_only_routing")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS canada_only_routing INTEGER DEFAULT 0"
    )
