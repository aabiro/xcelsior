"""Bring `agent_api_keys` into line with the data-architecture companion.

The table was modelled on `serverless_api_keys` (037), which predates the
companion's schema discipline. Two of its rules were broken:

- **§4.4.5** — every table has typed timestamps (`TIMESTAMPTZ`) and
  database-generated times where ordering matters. The table stored epoch
  seconds as `DOUBLE PRECISION`, the same float-time pattern the companion
  calls out as a defect in the existing schema (§1, §3.1).
- **§4.4.10** — every tenant-owned table has a non-null `tenant_id` and an
  index beginning with it for common access paths. An agent key authorises
  actions inside a workspace, so it is tenant-owned; without the column, a
  tenant-scoped query has to join back through `users`, and cross-tenant
  denial cannot be proven at the storage layer.

The 082 admission tables written alongside this one already follow both rules,
which is what made the divergence obvious.

Safe to rewrite in place: these keys are new in `083` and nothing outside
`AgentKeyStore` reads the columns. `tenant_id` backfills from the owning
user's workspace, falling back to the user id, which is what the wallet and
host paths already treat as the tenant for a personal account.
"""

from alembic import op

revision = "088"
down_revision = "087"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # The existing partial indexes predicate on `revoked_at = 0`, which stops
    # being a valid comparison the moment the column becomes TIMESTAMPTZ.
    # Drop them first and rebuild against the new NULL semantics below.
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_hash")
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_user")

    # ── §4.4.5: typed timestamps ──────────────────────────────────────
    # Epoch seconds convert directly; 0 meant "never" for the nullable ones,
    # so it becomes NULL rather than 1970-01-01, which would sort as "used
    # long ago" instead of "never used".
    op.execute(
        """
        ALTER TABLE agent_api_keys
            ALTER COLUMN created_at TYPE TIMESTAMPTZ
                USING to_timestamp(created_at),
            ALTER COLUMN created_at SET DEFAULT now()
        """
    )
    for column in ("last_used_at", "revoked_at"):
        op.execute(
            f"""
            ALTER TABLE agent_api_keys
                ALTER COLUMN {column} DROP DEFAULT,
                ALTER COLUMN {column} DROP NOT NULL,
                ALTER COLUMN {column} TYPE TIMESTAMPTZ
                    USING CASE WHEN {column} = 0 THEN NULL
                               ELSE to_timestamp({column}) END
            """
        )

    # ── §4.4.10: tenant ownership ─────────────────────────────────────
    op.execute("ALTER TABLE agent_api_keys ADD COLUMN IF NOT EXISTS tenant_id TEXT")
    op.execute(
        """
        UPDATE agent_api_keys k
           SET tenant_id = COALESCE(
                   NULLIF(u.team_id, ''),
                   NULLIF(u.customer_id, ''),
                   k.user_id)
          FROM users u
         WHERE u.user_id = k.user_id
           AND k.tenant_id IS NULL
        """
    )
    # Keys whose user row has since been erased (privacy deletion) still need
    # a tenant; the user id is the correct fallback for a personal workspace.
    op.execute("UPDATE agent_api_keys SET tenant_id = user_id WHERE tenant_id IS NULL")
    op.execute("ALTER TABLE agent_api_keys ALTER COLUMN tenant_id SET NOT NULL")

    # Index begins with tenant_id, per §4.4.10, for tenant-scoped listing.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_api_keys_tenant "
        "ON agent_api_keys (tenant_id, user_id) WHERE revoked_at IS NULL"
    )
    # Rebuilt against the new NULL semantics for revoked_at.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_api_keys_hash "
        "ON agent_api_keys (key_hash) WHERE revoked_at IS NULL"
    )
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_user")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_api_keys_user "
        "ON agent_api_keys (user_id) WHERE revoked_at IS NULL"
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_tenant")
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_hash")
    op.execute("DROP INDEX IF EXISTS idx_agent_api_keys_user")
    op.execute("ALTER TABLE agent_api_keys DROP COLUMN IF EXISTS tenant_id")

    for column in ("last_used_at", "revoked_at"):
        op.execute(
            f"""
            ALTER TABLE agent_api_keys
                ALTER COLUMN {column} TYPE DOUBLE PRECISION
                    USING COALESCE(extract(epoch FROM {column}), 0),
                ALTER COLUMN {column} SET DEFAULT 0,
                ALTER COLUMN {column} SET NOT NULL
            """
        )
    op.execute(
        """
        ALTER TABLE agent_api_keys
            ALTER COLUMN created_at DROP DEFAULT,
            ALTER COLUMN created_at TYPE DOUBLE PRECISION
                USING extract(epoch FROM created_at)
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_agent_api_keys_hash "
        "ON agent_api_keys (key_hash) WHERE revoked_at = 0"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_agent_api_keys_user "
        "ON agent_api_keys (user_id) WHERE revoked_at = 0"
    )
