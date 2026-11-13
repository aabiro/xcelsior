"""Which permission preset a Quick Connect client was set to.

The dashboard lets a user pick what their MCP or CLI key may do: Read, Write,
or a Custom selection. The scopes themselves already live in
`oauth_clients.scopes`; this records *which choice produced them*, because the
scope list alone cannot say. A key minted before a scope was added to the
Quick Connect set looks exactly like a Custom selection that left it out, and
the two must be treated differently when the key is regenerated: the first
follows the set forward, the second keeps what the user chose.

Blank for every existing row, which means "never chosen" and behaves as Write
did before this column existed. A constant default makes the ADD COLUMN a
catalogue-only change, and a release that does not know the column keeps
working.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "161"
down_revision = "160"
branch_labels = None
depends_on = None


def _permission_preset(conn):
    conn.execute(
        text(
            """
            ALTER TABLE oauth_clients
              ADD COLUMN IF NOT EXISTS permission_preset TEXT NOT NULL DEFAULT ''
            """
        )
    )


def _drop_permission_preset(conn):
    conn.execute(text("ALTER TABLE oauth_clients DROP COLUMN IF EXISTS permission_preset"))


def upgrade():
    apply_in_own_transactions(
        [("oauth_clients permission_preset", _permission_preset)],
        tables=("oauth_clients",),
    )


def downgrade():
    apply_in_own_transactions(
        [("drop oauth_clients permission_preset", _drop_permission_preset)],
        tables=("oauth_clients",),
    )
