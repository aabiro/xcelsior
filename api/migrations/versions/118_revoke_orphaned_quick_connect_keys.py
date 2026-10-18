"""Revoke Quick Connect agent keys whose OAuth client was deleted.

"Regenerate" on /dashboard/mcp deleted the user's system-managed Quick Connect
client and minted a key for a new one. Deleting a client does nothing to the
`agent_api_keys` rows bound to it, and key validation is a single lookup by
hash that never consults `oauth_clients`, so every key a user had rotated away
from kept authenticating. The route now revokes the replaced client's keys as
it rotates; this clears the ones it left behind.

Scoped by key name to the two Quick Connect surfaces. Those keys are only ever
issued against a system-managed client, so a missing client means exactly one
thing: it was rotated away. Keys with other names are left alone — tests and
other callers mint keys for client ids that were never stored rows, and absence
from `oauth_clients` says nothing about them.

One table, one statement, idempotent: a second run matches nothing. There is no
downgrade — un-revoking credentials the user asked to replace is not a state
worth restoring.
"""

from alembic import op

revision = "118"
down_revision = "117"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        UPDATE agent_api_keys AS k
           SET revoked_at = now()
         WHERE k.revoked_at IS NULL
           AND k.name IN ('MCP Quick Connect', 'CLI Skill')
           AND NOT EXISTS (
                 SELECT 1 FROM oauth_clients AS c WHERE c.client_id = k.client_id
           )
        """
    )


def downgrade() -> None:
    pass
