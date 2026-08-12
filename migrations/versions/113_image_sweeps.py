"""A sweep is a record, not a loop. Piece 2 of Gate P7.

Gate P7: *"a sweep of N nodes from one snapshot is byte-identical in
environment"*.

A route that calls the existing launch path N times and hands back N job ids is
a **client convenience**. It leaves "these N came from one snapshot" as an
intention in whoever called it, and nothing afterwards can check the claim. What
makes a sweep a sweep is one row the members belong to:

* **Partial failure becomes visible.** Three of five launched is a fact with a
  shape — and *which two did not* is answerable — instead of a caller holding
  three ids and no idea what happened to the rest.
* **The verification has something to hang off.** Piece 3's fingerprints are
  compared *within* a sweep; without the row there is no set to compare across.
* **The claim lands in the database.** "N nodes from one snapshot" stops being
  an assertion about a request that already finished.

## The digest is pinned here, once

`image_digest` is copied onto the sweep at creation and every member launches
against `repo@sha256:…`. It is `NOT NULL` **on purpose**: a snapshot whose
digest is unknown cannot support a byte-identity claim, so the sweep is refused
rather than created against a mutable tag. That refusal is the point of
migration 112 — falling back to the tag would leave the clause unprovable while
looking like it had been met.

Pinned once and stored, not re-resolved per member: resolving per member would
reintroduce exactly the race the digest exists to close, since the tag could
move between the first resolution and the last.

## `host_id` per member

Recorded so "how many distinct hosts did this sweep actually cover" is a query
rather than a guess. A sweep that lands entirely on one host is not wrong, but
it establishes much less than one spread across several, and the difference has
to be visible in the data before anything can report it honestly.
"""

from alembic import op

revision = "113"
down_revision = "112"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS image_sweeps (
            sweep_id        TEXT PRIMARY KEY,
            -- Ledger rule 9 / companion §4.4.10: every tenant-owned table
            -- carries a non-null tenant_id and an index beginning with it, so a
            -- tenant-scoped read never joins back through another table to
            -- learn whose row it is.
            tenant_id       TEXT NOT NULL,
            owner_id        TEXT NOT NULL,
            image_id        TEXT NOT NULL,
            -- Pinned at creation. NOT NULL because a sweep from an unknown
            -- digest cannot support the claim the sweep exists to make.
            image_digest    TEXT NOT NULL,
            requested_count INTEGER NOT NULL,
            state           TEXT NOT NULL DEFAULT 'launching',
            failure_code    TEXT,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            updated_at      TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            completed_at    TIMESTAMPTZ,
            CONSTRAINT ck_image_sweeps_state CHECK (
                state IN ('launching', 'running', 'verified', 'mismatch', 'failed')
            ),
            CONSTRAINT ck_image_sweeps_count CHECK (
                requested_count > 0 AND requested_count <= 64
            ),
            -- A digest, not a tag. The check is the mechanism: without it
            -- "pinned by digest" is a convention, and a convention is what a
            -- future caller passing `name:tag` would quietly break.
            CONSTRAINT ck_image_sweeps_digest CHECK (image_digest LIKE '%@sha256:%')
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_image_sweeps_tenant "
        "ON image_sweeps (tenant_id, created_at DESC)"
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS image_sweep_members (
            sweep_id     TEXT NOT NULL,
            member_index INTEGER NOT NULL,
            tenant_id    TEXT NOT NULL,
            -- Null until the launch returns one. A member that never launched
            -- is the case this table exists to make visible, so it must be
            -- representable rather than absent.
            job_id       TEXT,
            host_id      TEXT,
            state        TEXT NOT NULL DEFAULT 'pending',
            failure_code TEXT,
            updated_at   TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            PRIMARY KEY (sweep_id, member_index),
            CONSTRAINT ck_sweep_member_state CHECK (
                state IN ('pending', 'launched', 'reported', 'failed')
            ),
            -- A member that launched has a job. Without this a row could claim
            -- `launched` with nothing to point at, and the partial-failure
            -- count this table exists for would be wrong in the direction that
            -- flatters it.
            CONSTRAINT ck_sweep_member_launched_has_job CHECK (
                state IN ('pending', 'failed') OR job_id IS NOT NULL
            )
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_sweep_members_tenant "
        "ON image_sweep_members (tenant_id, sweep_id)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS image_sweep_members")
    op.execute("DROP TABLE IF EXISTS image_sweeps")
