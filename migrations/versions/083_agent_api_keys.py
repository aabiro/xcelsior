"""Durable, non-expiring agent API keys for MCP and the Agent Skill.

Agent credentials were minted as RS256 JWTs held in the auth cache. Two
problems followed from that:

- They are ~1 KB of base64, pasted by hand into agent config files.
- The cache is Redis, which is sized and operated as a cache. A flush or an
  eviction silently invalidated every agent's credential at once, and the
  failure surfaced to the user as an unexplained 401 inside their editor.

Agent keys are long-lived by design — they are pasted into a config once and
left there — so the source of truth belongs in Postgres, with Redis free to
act as a read-through cache. This mirrors ``serverless_api_keys`` (037) and
``host_agent_tokens`` (065): store only a SHA-256 hash, keep a display prefix,
and track use and revocation rather than expiry.

``last_used_at`` is load-bearing beyond auditing: it is how the dashboard
knows whether a key was ever actually applied to a live config, and therefore
whether rotating it would break something the user depends on.
"""

from alembic import op

revision = "083"
down_revision = "082"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_api_keys (
            key_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            client_id TEXT NOT NULL,
            name TEXT NOT NULL DEFAULT 'Agent key',
            key_prefix TEXT NOT NULL,
            key_hash TEXT NOT NULL,
            scopes TEXT NOT NULL DEFAULT '',
            audience TEXT NOT NULL DEFAULT '',
            last_used_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            revoked_at DOUBLE PRECISION NOT NULL DEFAULT 0,
            created_at DOUBLE PRECISION NOT NULL,
            CHECK (length(key_hash) = 64)
        )
        """
    )
    # Lookup path for every authenticated agent request: hash the presented
    # secret, find the one live row. Unique so a hash collision or a double
    # insert can never yield two principals for one credential.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_api_keys_hash "
        "ON agent_api_keys (key_hash) WHERE revoked_at = 0"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_api_keys_user "
        "ON agent_api_keys (user_id) WHERE revoked_at = 0"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_user")
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_hash")
    op.execute("DROP TABLE IF EXISTS agent_api_keys")
