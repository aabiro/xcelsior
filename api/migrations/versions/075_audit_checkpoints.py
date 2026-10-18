"""Signed immutable audit checkpoints (Track B B4.5).

Blueprint §13.6 / companion §12.2: a periodic, signed Merkle checkpoint over the
audit stream. Each manifest records the Merkle root over the interval's event
ids/hashes, the row count, the previous manifest's hash (chaining the
checkpoints themselves), the schema versions, the signing key version, and the
signature. Verification recomputes the root from the WORM `audit_events_v2` rows
and checks the signature with the recorded key version — so a tampered manifest,
a missing manifest, or any change to the sealed interval's events is detected,
and key rotation preserves verifiability of older manifests.

The manifest row is itself append-only (WORM trigger) — the authoritative,
tamper-evident record; production additionally uploads it to versioned/
WORM-capable object storage (the signing key stays administratively separate
from bucket administration).

Revision ID: 075
Revises: 074
Create Date: 2026-07-24
"""

from typing import Sequence, Union

from alembic import op

revision: str = "075"
down_revision: Union[str, None] = "074"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_checkpoints (
            checkpoint_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            interval_start      TIMESTAMPTZ NOT NULL,
            interval_end        TIMESTAMPTZ NOT NULL,
            merkle_root         TEXT NOT NULL,
            row_count           BIGINT NOT NULL,
            first_event_id      UUID,
            last_event_id       UUID,
            prev_manifest_hash  TEXT,
            manifest_sha256     TEXT NOT NULL,
            schema_versions     JSONB NOT NULL DEFAULT '{}'::jsonb,
            signing_key_version TEXT NOT NULL,
            signature           TEXT NOT NULL,
            object_uri          TEXT,
            created_at          TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            CHECK (interval_end >= interval_start)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_checkpoints_interval "
        "ON audit_checkpoints (interval_end DESC)"
    )
    # WORM: a signed checkpoint can never be altered or deleted.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION audit_checkpoints_immutable() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_checkpoints is append-only (WORM); % is not permitted', TG_OP
                USING ERRCODE = 'restrict_violation';
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_audit_checkpoints_immutable
            BEFORE UPDATE OR DELETE ON audit_checkpoints
            FOR EACH ROW EXECUTE FUNCTION audit_checkpoints_immutable()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_audit_checkpoints_immutable ON audit_checkpoints")
    op.execute("DROP FUNCTION IF EXISTS audit_checkpoints_immutable()")
    op.execute("DROP TABLE IF EXISTS audit_checkpoints")
