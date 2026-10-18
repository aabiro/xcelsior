"""Connector client registration: CIMD, RFC 7591 DCR, and recorded consent.

Three things the hosted MCP connector needs before a provider directory can
onboard users onto it (docs/mcp-enterprise-adoption-plan.md, BLOCKER 3):

1. **`oauth_clients` has to remember how a client came to exist.** A client a
   human created in Settings, a client Claude registered dynamically, and a
   client identified by a metadata document are three different trust levels
   and cannot share one undifferentiated row. `registration_source` carries
   that, `resource_audience` pins a dynamically-registered client to the MCP
   resource so it can never be turned into a general API client, and
   `registration_expires_at` is what makes an unused registration disappear
   instead of accumulating forever.

2. **RFC 7591 metadata has to round-trip.** A registration response must echo
   back what was registered; storing only our own subset would silently drop
   fields and fail conformance.

3. **Consent must be a record, not a UI moment.** `oauth_consent_grants` is
   what lets a returning user skip the prompt, what a user revokes when they
   disconnect a connector, and what proves — after the fact — which scopes a
   given client was actually granted.

`oauth_clients` predates the data companion and keeps its float `created_at`;
this migration does not convert it (that is a separate, wider change touching
every reader). The new table follows §4.4 in full: TIMESTAMPTZ time, NOT NULL
tenant, tenant-leading index.
"""

from alembic import op

revision = "091"
down_revision = "090"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    # ── oauth_clients: provenance and containment ─────────────────────
    for stmt in (
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS registration_source "
        "TEXT NOT NULL DEFAULT 'manual'",
        # NULL means "not pinned" (a normal API client). A value means every
        # grant for this client must target exactly that resource.
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS resource_audience TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS registration_expires_at "
        "TIMESTAMPTZ",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS client_uri TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS logo_uri TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS policy_uri TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS tos_uri TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS software_id TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS software_version TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS contacts JSONB "
        "NOT NULL DEFAULT '[]'::jsonb",
    ):
        op.execute(stmt)

    op.execute(
        """
        UPDATE oauth_clients SET registration_source = 'first_party'
         WHERE is_first_party = 1 AND registration_source = 'manual'
        """
    )
    # The sweep of expired, never-used dynamic registrations has to be an index
    # scan; without this it is a sequential scan of every client on the platform.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_clients_registration_expiry "
        "ON oauth_clients (registration_expires_at) "
        "WHERE registration_expires_at IS NOT NULL"
    )

    # ── oauth_consent_grants ──────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_consent_grants (
            grant_id TEXT PRIMARY KEY,
            tenant_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            email TEXT NOT NULL,
            client_id TEXT NOT NULL,
            -- '' rather than NULL: it participates in the uniqueness key, and
            -- NULL would let the same (user, client) accumulate duplicate rows.
            resource TEXT NOT NULL DEFAULT '',
            scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
            -- Which product the user connected from (claude / chatgpt / grok /
            -- copilot-studio / local / …). Recorded here because it is knowable
            -- at consent time and unrecoverable afterwards: every surface can
            -- share one pre-provisioned connector client, so `client_id` alone
            -- cannot answer "which directory produced this activation".
            surface TEXT NOT NULL DEFAULT 'unknown',
            granted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_used_at TIMESTAMPTZ,
            revoked_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_oauth_consent_grants_principal "
        "ON oauth_consent_grants (user_id, client_id, resource)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_consent_grants_tenant "
        "ON oauth_consent_grants (tenant_id, granted_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_consent_grants_client "
        "ON oauth_consent_grants (client_id)"
    )
    # The activation funnel groups by surface over a date range.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_consent_grants_surface "
        "ON oauth_consent_grants (surface, granted_at DESC)"
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute("DROP TABLE IF EXISTS oauth_consent_grants")
    op.execute("DROP INDEX IF EXISTS idx_oauth_clients_registration_expiry")
    # Written out rather than looped over a list of names: a column name
    # interpolated into DDL is the shape `tests/test_sql_injection_guard.py`
    # exists to keep out of the codebase, and there is no reason for a
    # migration to be the exception that teaches the pattern.
    op.execute(
        """
        ALTER TABLE oauth_clients
            DROP COLUMN IF EXISTS registration_source,
            DROP COLUMN IF EXISTS resource_audience,
            DROP COLUMN IF EXISTS registration_expires_at,
            DROP COLUMN IF EXISTS client_uri,
            DROP COLUMN IF EXISTS logo_uri,
            DROP COLUMN IF EXISTS policy_uri,
            DROP COLUMN IF EXISTS tos_uri,
            DROP COLUMN IF EXISTS software_id,
            DROP COLUMN IF EXISTS software_version,
            DROP COLUMN IF EXISTS contacts
        """
    )
