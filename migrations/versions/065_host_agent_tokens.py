"""Per-host agent bearer tokens (blueprint §19.2 — rotate away the fleet token).

Blueprint §19.2 is explicit: "bearer tokens remain only during migration,
scoped to one host and rotated. Do not use one platform API token for the
fleet long term."

``host_agent_tokens`` is the durable authority for that rotation:

- one **active** token per host (partial unique index), so issuing a new
  credential cannot silently leave two live secrets on one host;
- secrets are stored as SHA-256 hashes only — the plaintext exists once,
  in the issue/rotate response;
- rotation is overlap-safe: the previous token moves to ``superseded``
  with its own ``expires_at`` grace window, so a worker that has already
  fetched a new token but not yet restarted is never locked out;
- ``revoked`` is terminal and immediate (compromise response);
- every verification stamps ``last_used_at``, which is what makes an
  unused/stale credential visible to an operator.

Expand-only: nothing reads this table until
``XCELSIOR_AGENT_HOST_TOKENS`` is enabled.

Revision ID: 065
Revises: 064
Create Date: 2026-07-21
"""

from typing import Sequence, Union

from alembic import op

revision: str = "065"
down_revision: Union[str, None] = "064"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS host_agent_tokens (
            token_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            host_id         TEXT NOT NULL,
            token_prefix    TEXT NOT NULL,
            token_hash      TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'active',
            issued_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            expires_at      TIMESTAMPTZ NOT NULL,
            last_used_at    TIMESTAMPTZ,
            last_used_ip    TEXT,
            superseded_at   TIMESTAMPTZ,
            revoked_at      TIMESTAMPTZ,
            revoked_reason  TEXT,
            rotated_from    UUID REFERENCES host_agent_tokens(token_id) ON DELETE SET NULL,
            issued_by       TEXT,
            issue_reason    TEXT,
            CONSTRAINT ck_host_agent_token_status CHECK (
                status IN ('active', 'superseded', 'revoked', 'expired')
            ),
            CONSTRAINT ck_host_agent_token_revoked_shape CHECK (
                (status = 'revoked') = (revoked_at IS NOT NULL)
            ),
            CONSTRAINT ck_host_agent_token_superseded_shape CHECK (
                status <> 'superseded' OR superseded_at IS NOT NULL
            )
        )
        """
    )

    # A host has at most one issuable credential at a time.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_host_one_active_agent_token
            ON host_agent_tokens (host_id)
         WHERE status = 'active'
        """
    )
    # Hash is the lookup key; a collision would be an authentication bug.
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_host_agent_token_hash
            ON host_agent_tokens (token_hash)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_host_agent_tokens_host
            ON host_agent_tokens (host_id, issued_at DESC)
        """
    )
    # Drives the expiry sweep without scanning revoked history.
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_host_agent_tokens_expiry
            ON host_agent_tokens (expires_at)
         WHERE status IN ('active', 'superseded')
        """
    )

    # Durable periodic sweep: expire past-deadline tokens (056 scheduled_tasks).
    op.execute(
        """
        INSERT INTO scheduled_tasks (task_name, interval_seconds, next_run_at, enabled)
        VALUES ('host_agent_token_expiry', 300, clock_timestamp(), TRUE)
        ON CONFLICT (task_name) DO NOTHING
        """
    )


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("SET statement_timeout = '5min'")
    op.execute("DELETE FROM scheduled_tasks WHERE task_name = 'host_agent_token_expiry'")
    op.execute("DROP INDEX IF EXISTS idx_host_agent_tokens_expiry")
    op.execute("DROP INDEX IF EXISTS idx_host_agent_tokens_host")
    op.execute("DROP INDEX IF EXISTS uq_host_agent_token_hash")
    op.execute("DROP INDEX IF EXISTS uq_host_one_active_agent_token")
    op.execute("DROP TABLE IF EXISTS host_agent_tokens")
