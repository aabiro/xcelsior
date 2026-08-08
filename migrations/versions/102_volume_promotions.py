"""The promotion row an artifact→volume copy is keyed on.

A0 of `docs/artifact-promotion-plan.md`. Nothing copies yet — this is the row a
promotion *is*, so the shape can be reviewed before any bytes move.

**Why the unique key is `(tenant_id, job_id, idempotency_key)`.** Gate P3 asks
that "a repeated call produces one volume, not two", and the phrase implies
promotion may *create* the volume when the caller names none — which is the
natural agent flow ("save this somewhere"). So the key has to cover volume
creation, not merely the copy, which is why `volume_id` is nullable and is not
part of the key.

The default idempotency key is the manifest digest: the same job promoted twice
with the same resolved artifact set is the same promotion. That makes a retry
after a timeout converge instead of duplicating, without the caller having to
invent a key — the failure this prevents is a 40 GB copy running twice because
the first response was lost.

`payment_intents` uses this exact shape (`ON CONFLICT DO NOTHING` plus
`rowcount` to tell new from replayed) and `charge_saved_card` surfaces it as
`replayed`. Promotion says the same thing rather than appearing to succeed
twice.

**Expand-only** (rule 5): one new table, nothing altered, nothing dropped.
"""

from alembic import op

revision = "102"
down_revision = "101"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS volume_promotions (
            promotion_id      TEXT PRIMARY KEY,
            tenant_id         TEXT NOT NULL,
            owner_user_id     TEXT,
            job_id            TEXT NOT NULL,
            -- Nullable: a promotion may create the volume it lands on, so the
            -- row exists before the volume does.
            volume_id         TEXT,
            idempotency_key   TEXT NOT NULL,
            -- The digest of the resolved artifact set. Stored rather than
            -- recomputed so a replay can be shown to refer to the same files,
            -- not merely the same job — a job that produced new artifacts since
            -- the first call is a different promotion.
            manifest_sha256   TEXT NOT NULL,
            file_count        INTEGER NOT NULL DEFAULT 0,
            total_bytes       BIGINT NOT NULL DEFAULT 0,
            state             TEXT NOT NULL DEFAULT 'pending',
            failure_code      TEXT,
            -- TIMESTAMPTZ, not the float epoch used by `gpu_allocations` and
            -- other older tables. Ledger rule 9 — "TIMESTAMPTZ for time …
            -- never binary floats" — and `tests/test_companion_schema_discipline.py`
            -- enforces both halves. The first draft of this migration copied the
            -- float convention from the table next door and was caught by them.
            created_at        TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at        TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            completed_at      TIMESTAMPTZ,
            CONSTRAINT ck_volume_promotions_state CHECK (
                state IN ('pending', 'running', 'succeeded', 'failed', 'abandoned')
            ),
            -- A succeeded promotion has landed somewhere and copied something.
            -- Without this, "succeeded with no volume" is representable, and a
            -- tool would report weights saved to nowhere.
            CONSTRAINT ck_volume_promotions_succeeded CHECK (
                state <> 'succeeded' OR (volume_id IS NOT NULL AND completed_at IS NOT NULL)
            )
        )
        """
    )
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_volume_promotions_idem "
        "ON volume_promotions (tenant_id, job_id, idempotency_key)"
    )
    # "What is happening to my volume" and the stale-promotion sweep both scan
    # by state; the tenant prefix keeps one tenant's backlog off another's read.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_volume_promotions_state "
        "ON volume_promotions (tenant_id, state)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_volume_promotions_state")
    op.execute("DROP INDEX IF EXISTS uq_volume_promotions_idem")
    op.execute("DROP TABLE IF EXISTS volume_promotions")
