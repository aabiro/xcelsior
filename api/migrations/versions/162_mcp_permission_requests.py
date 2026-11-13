"""An agent's request for a permission its Quick Connect key lacks.

An agent holding a Quick Connect key can ask for more than the key carries
(`request_permission` on the MCP server). It cannot grant itself anything: the
request is a row here, the agent is handed a link, and a person signed in to
the dashboard approves or declines on a page that says what the permission
does. Only then does the key change.

One row per request. `code` is what the link carries. A request is answered
once and stops being answerable at `expires_at`; nothing here is ever a
standing grant, which lives in `oauth_clients.scopes` and
`agent_api_keys.scopes` as before.

Follows the data companion (§4.4): TIMESTAMPTZ time, NOT NULL tenant, and a
tenant-leading index. The unit is idempotent and runs in its own short-lock
transaction (migrations/lock_safe.py).
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "162"
down_revision = "161"
branch_labels = None
depends_on = None


def _requests(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS mcp_permission_requests (
                request_id TEXT PRIMARY KEY,
                code TEXT NOT NULL UNIQUE,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                key_id TEXT NOT NULL,
                surface TEXT NOT NULL DEFAULT 'mcp',
                scopes TEXT NOT NULL,
                reason TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                expires_at TIMESTAMPTZ NOT NULL,
                decided_at TIMESTAMPTZ,
                CONSTRAINT mcp_permission_requests_status_check
                    CHECK (status IN ('pending', 'approved', 'declined')),
                CONSTRAINT mcp_permission_requests_surface_check
                    CHECK (surface IN ('mcp', 'cli'))
            )
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS idx_mcp_permission_requests_tenant_key
                ON mcp_permission_requests (tenant_id, key_id, status)
            """
        )
    )


def _drop_requests(conn):
    conn.execute(text("DROP TABLE IF EXISTS mcp_permission_requests"))


def upgrade():
    apply_in_own_transactions(
        [("mcp_permission_requests", _requests)],
        tables=("mcp_permission_requests",),
    )


def downgrade():
    apply_in_own_transactions(
        [("drop mcp_permission_requests", _drop_requests)],
        tables=("mcp_permission_requests",),
    )
