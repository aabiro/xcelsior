"""Keep an encrypted copy of Quick Connect agent keys so the dashboard can show them again.

The MCP and CLI tabs at /dashboard/mcp showed a key in full only on the visit
that minted it; every later visit showed a masked prefix and asked the person
to regenerate, which revokes the key their agent is already using. These two
keys exist to be copied into an agent, so they are now stored encrypted with
``security.encrypt_secret`` (``XCELSIOR_SECRETS_KEY``, the same protection as
serverless endpoint secrets) alongside the hash that authenticates them.

Nullable and additive: other agent keys stay show-once, and keys minted before
this keep no ciphertext, so they stay masked until regenerated once.
"""

from alembic import op

revision = "121"
down_revision = "120"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("ALTER TABLE agent_api_keys ADD COLUMN IF NOT EXISTS key_ciphertext TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE agent_api_keys DROP COLUMN IF EXISTS key_ciphertext")
