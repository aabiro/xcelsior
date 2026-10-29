"""Where a sweep member's environment fingerprint is recorded. Piece 3 of Gate P7.

Both the hash and the raw manifest, and the second column is the one that will
earn its keep. A hash answers *"did these differ"* and nothing else; the first
time a sweep goes red the question is **which field**, and a bare digest cannot
answer it. Storing only the hash would mean the check can fail and no one can
act on the failure.

`fingerprint_manifest` is `JSONB` rather than `TEXT` so the differing key can be
found with a query instead of by pulling every member into a process and
diffing by hand.

## Nullable, and the comparison must care

A member that never reported has `NULL` here, and that is a different fact from
"reported an empty environment". The comparison in `control_plane/image_sweeps.py`
treats a missing fingerprint as **unknown, never as agreement** — because a
collector that errors and returns nothing would otherwise make all N members
equal and the sweep would report a perfect pass. That is the failure the
positive control exists for, and it is worth stating at the schema too: the
column admits unknown so the code above it is forced to handle unknown.
"""

from alembic import op

revision = "114"
down_revision = "113"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        "ALTER TABLE image_sweep_members "
        "  ADD COLUMN IF NOT EXISTS fingerprint_hash TEXT, "
        "  ADD COLUMN IF NOT EXISTS fingerprint_manifest JSONB, "
        "  ADD COLUMN IF NOT EXISTS fingerprint_at TIMESTAMPTZ"
    )
    # A hash without its manifest is a mismatch nobody can diagnose; a manifest
    # without its hash is a comparison nobody can make. Neither half is useful
    # alone, so the row may hold both or neither.
    op.execute(
        """
        ALTER TABLE image_sweep_members
          ADD CONSTRAINT ck_sweep_member_fingerprint_is_whole
          CHECK (
            (fingerprint_hash IS NULL AND fingerprint_manifest IS NULL)
            OR (fingerprint_hash IS NOT NULL AND fingerprint_manifest IS NOT NULL)
          )
        """
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE image_sweep_members "
        "DROP CONSTRAINT IF EXISTS ck_sweep_member_fingerprint_is_whole"
    )
    op.execute(
        "ALTER TABLE image_sweep_members "
        "  DROP COLUMN IF EXISTS fingerprint_hash, "
        "  DROP COLUMN IF EXISTS fingerprint_manifest, "
        "  DROP COLUMN IF EXISTS fingerprint_at"
    )
