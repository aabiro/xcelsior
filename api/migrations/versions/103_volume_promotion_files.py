"""Per-file progress for a promotion, so a retry resumes instead of restarting.

A3 / §3.5 of `docs/artifact-promotion-plan.md`: *"Weights are large and networks
are not. A promotion that restarts from zero after a failure at 38 GB will be
retried by a human who then watches it fail again."*

One row per (promotion, artifact). A resumed promotion skips what is already
`done`, and `done` means **verified** — the digest was checked before the file
was renamed into place — because an unverified copy is worse than no copy: it
looks like a backup.

**Why `sha256_verified` is stored rather than inferred from `state = 'done'`.**
They can disagree, and the disagreement is the interesting case: a file marked
done by an older agent that did not verify must not be skipped by a newer one
that would have. Keeping the flag separate lets a resume re-copy exactly those
rather than trusting a state word whose meaning changed between releases.

**Expand-only** (rule 5): one new table, nothing altered.
"""

from alembic import op

revision = "103"
down_revision = "102"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS volume_promotion_files (
            promotion_id     TEXT NOT NULL,
            -- Denormalised from the promotion. Ledger rule 9 / companion
            -- §4.4.10: every tenant-owned table carries a non-null `tenant_id`
            -- and an index beginning with it, so a tenant-scoped query never
            -- has to join back through another table to find out whose row it
            -- is. The first draft omitted it and left tenancy implied by
            -- `promotion_id`, which is exactly the join the rule forbids.
            tenant_id        TEXT NOT NULL,
            artifact_id      TEXT NOT NULL,
            logical_name     TEXT NOT NULL,
            size_bytes       BIGINT NOT NULL DEFAULT 0,
            bytes_written    BIGINT NOT NULL DEFAULT 0,
            sha256_verified  BOOLEAN NOT NULL DEFAULT false,
            state            TEXT NOT NULL DEFAULT 'pending',
            failure_code     TEXT,
            updated_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (promotion_id, artifact_id),
            CONSTRAINT ck_promotion_files_state CHECK (
                state IN ('pending', 'copying', 'done', 'failed')
            ),
            -- A file is only `done` if its digest was checked. Without this the
            -- resume path could skip a file that was copied but never verified,
            -- which is the one outcome §3.5 rules out: it looks like a backup.
            CONSTRAINT ck_promotion_files_done_is_verified CHECK (
                state <> 'done' OR sha256_verified
            )
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_promotion_files_tenant "
        "ON volume_promotion_files (tenant_id, promotion_id, state)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_promotion_files_tenant")
    op.execute("DROP TABLE IF EXISTS volume_promotion_files")
