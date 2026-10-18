"""Expand-only: durable wallet_holds for launch fund reservation.

Creates ``wallet_holds`` (held amount + status + expiry + idempotency) and
enforces ``jobs.wallet_hold_id`` FK when both sides exist. Available balance
is ledger balance minus active holds — concurrent launch preflights serialize
on the wallet row.

Revision ID: 063
Revises: 062
Create Date: 2026-07-20
"""

from typing import Sequence, Union

from alembic import op

revision: str = "063"
down_revision: Union[str, None] = "062"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS wallet_holds (
            hold_id UUID PRIMARY KEY,
            customer_id TEXT NOT NULL,
            amount_cad DOUBLE PRECISION NOT NULL
                CHECK (amount_cad > 0),
            status TEXT NOT NULL
                CHECK (status IN ('held', 'released', 'consumed', 'expired')),
            job_id TEXT NULL,
            idempotency_key TEXT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION NOT NULL,
            released_at DOUBLE PRECISION NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_wallet_holds_idempotency
            ON wallet_holds (customer_id, idempotency_key)
         WHERE idempotency_key IS NOT NULL
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_wallet_holds_customer_active
            ON wallet_holds (customer_id)
         WHERE status = 'held'
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_wallet_holds_expires
            ON wallet_holds (expires_at)
         WHERE status = 'held'
        """
    )
    # jobs.wallet_hold_id was added in 054 as UUID without FK.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint
                 WHERE conname = 'fk_jobs_wallet_hold'
            ) THEN
                ALTER TABLE jobs
                  ADD CONSTRAINT fk_jobs_wallet_hold
                  FOREIGN KEY (wallet_hold_id)
                  REFERENCES wallet_holds (hold_id)
                  ON DELETE SET NULL
                  NOT VALID;
            END IF;
        END $$
        """
    )
    # Validate only when no orphan job pointers exist.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM jobs j
                 WHERE j.wallet_hold_id IS NOT NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM wallet_holds h
                        WHERE h.hold_id = j.wallet_hold_id
                   )
            ) THEN
                ALTER TABLE jobs VALIDATE CONSTRAINT fk_jobs_wallet_hold;
            END IF;
        EXCEPTION WHEN undefined_object THEN
            NULL;
        END $$
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobs DROP CONSTRAINT IF EXISTS fk_jobs_wallet_hold
        """
    )
    op.execute("DROP INDEX IF EXISTS idx_wallet_holds_expires")
    op.execute("DROP INDEX IF EXISTS idx_wallet_holds_customer_active")
    op.execute("DROP INDEX IF EXISTS uq_wallet_holds_idempotency")
    op.execute("DROP TABLE IF EXISTS wallet_holds")
