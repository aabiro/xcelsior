"""Residual runtime DDL objects moved into Alembic (expand-only).

Historically these tables/columns were created only by
``db._ensure_pg_tables`` / ``_ensure_oauth_auth_tables`` at pool/auth
startup. This revision makes Alembic the sole production schema authority:
this migration owns the residual objects so production startup can be
seed-only (no CREATE/ALTER).

Objects extracted:
- ``job_logs`` (container log lines)
- ``oauth_clients`` + ``oauth_refresh_tokens`` (OAuth2)
- ``team_invites``
- ``users``: max_concurrent_instances, pending_email / email_change_*
- ``billing_cycles``: token_cost_cad, model_ref

All statements are idempotent (IF NOT EXISTS) so environments that already
received the shape via runtime ensure upgrade cleanly.

Revision ID: 061
Revises: a0985327493e
Create Date: 2026-07-19
"""

from typing import Sequence, Union

from alembic import op

revision: str = "061"
down_revision: Union[str, None] = "a0985327493e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── job_logs ─────────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS job_logs (
            id BIGSERIAL PRIMARY KEY,
            job_id TEXT NOT NULL,
            ts DOUBLE PRECISION NOT NULL,
            level TEXT NOT NULL DEFAULT 'info',
            line TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL
                DEFAULT EXTRACT(EPOCH FROM NOW())
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_job_logs_job_ts "
        "ON job_logs (job_id, ts)"
    )

    # ── billing_cycles residual columns (ensure-only historically) ───
    op.execute(
        "ALTER TABLE billing_cycles "
        "ADD COLUMN IF NOT EXISTS token_cost_cad "
        "DOUBLE PRECISION NOT NULL DEFAULT 0"
    )
    op.execute(
        "ALTER TABLE billing_cycles "
        "ADD COLUMN IF NOT EXISTS model_ref TEXT NOT NULL DEFAULT ''"
    )

    # ── users residual columns ───────────────────────────────────────
    op.execute(
        "ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS max_concurrent_instances INTEGER"
    )
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS pending_email TEXT"
    )
    op.execute(
        "ALTER TABLE users ADD COLUMN IF NOT EXISTS email_change_token TEXT"
    )
    op.execute(
        "ALTER TABLE users "
        "ADD COLUMN IF NOT EXISTS email_change_expires DOUBLE PRECISION"
    )

    # ── oauth_clients ────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id TEXT PRIMARY KEY,
            client_name TEXT NOT NULL,
            client_type TEXT NOT NULL,
            redirect_uris JSONB NOT NULL DEFAULT '[]'::jsonb,
            grant_types JSONB NOT NULL DEFAULT '[]'::jsonb,
            scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
            client_secret_hash TEXT,
            client_secret_salt TEXT,
            client_secret_preview TEXT,
            created_by_email TEXT,
            is_first_party INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'active',
            last_used DOUBLE PRECISION,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL,
            workspace_customer_id TEXT,
            team_id TEXT,
            is_system_managed INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    # Expand columns for DBs that got an older ensure-created shape.
    for stmt in (
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS status "
        "TEXT NOT NULL DEFAULT 'active'",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS last_used "
        "DOUBLE PRECISION",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS "
        "client_secret_preview TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS "
        "workspace_customer_id TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS team_id TEXT",
        "ALTER TABLE oauth_clients ADD COLUMN IF NOT EXISTS "
        "is_system_managed INTEGER NOT NULL DEFAULT 0",
    ):
        op.execute(stmt)
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_clients_workspace "
        "ON oauth_clients (workspace_customer_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_clients_owner "
        "ON oauth_clients (created_by_email)"
    )

    # ── oauth_refresh_tokens ─────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS oauth_refresh_tokens (
            token_id TEXT PRIMARY KEY,
            token_hash TEXT NOT NULL UNIQUE,
            family_id TEXT NOT NULL,
            parent_token_id TEXT,
            session_token TEXT UNIQUE,
            client_id TEXT NOT NULL,
            email TEXT,
            user_id TEXT,
            session_type TEXT NOT NULL DEFAULT 'browser',
            scopes JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            consumed_at DOUBLE PRECISION,
            revoked_at DOUBLE PRECISION,
            replaced_by_token_id TEXT,
            reuse_detected_at DOUBLE PRECISION
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_family "
        "ON oauth_refresh_tokens (family_id)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_email "
        "ON oauth_refresh_tokens (email)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_session "
        "ON oauth_refresh_tokens (session_token)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_oauth_refresh_tokens_expires "
        "ON oauth_refresh_tokens (expires_at)"
    )

    # ── sessions residual (ensure historically dual-sourced these) ───
    # 004 already creates session_type/client_id; IF NOT EXISTS is a no-op
    # when present. last_active is from 009.
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS session_type "
        "TEXT NOT NULL DEFAULT 'legacy'"
    )
    op.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS client_id TEXT"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sessions_email_type "
        "ON sessions (email, session_type, last_active DESC)"
    )

    # ── teams residual ───────────────────────────────────────────────
    op.execute(
        "ALTER TABLE teams "
        "ADD COLUMN IF NOT EXISTS billing_customer_id "
        "TEXT NOT NULL DEFAULT ''"
    )

    # ── team_invites ─────────────────────────────────────────────────
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS team_invites (
            token TEXT PRIMARY KEY,
            team_id TEXT NOT NULL,
            email TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'member',
            invited_by TEXT NOT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_invites_email "
        "ON team_invites (email)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_team_invites_team "
        "ON team_invites (team_id)"
    )


def downgrade() -> None:
    # Expand-only residual; contract drops are best-effort IF EXISTS.
    # Do not drop oauth/job_logs tables in reverse of production data —
    # leave structures; only drop columns that 061 exclusively added when
    # they are unused. Prefer no-op contract for safety.
    op.execute(
        "ALTER TABLE billing_cycles DROP COLUMN IF EXISTS model_ref"
    )
    op.execute(
        "ALTER TABLE billing_cycles DROP COLUMN IF EXISTS token_cost_cad"
    )
    op.execute(
        "ALTER TABLE users DROP COLUMN IF EXISTS email_change_expires"
    )
    op.execute(
        "ALTER TABLE users DROP COLUMN IF EXISTS email_change_token"
    )
    op.execute(
        "ALTER TABLE users DROP COLUMN IF EXISTS pending_email"
    )
    op.execute(
        "ALTER TABLE users DROP COLUMN IF EXISTS max_concurrent_instances"
    )
    # Tables intentionally retained on downgrade (data-preserving).
