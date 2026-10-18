"""Give gpu_allocations an owner, so releasing one can be authorized.

`POST /api/v2/marketplace/release/{allocation_id}` passes an **allocation id**
to `release_allocation(job_id)`, which queries `WHERE job_id = %s`. It matches
nothing, updates nothing, and returns `{"ok": true}` — a route that reports
success for work it did not do.

**Why the obvious one-line fix was wrong.** Changing the lookup to
`WHERE allocation_id = %s` makes the route work, and simultaneously turns it
into a cross-tenant capability: `gpu_allocations` has no owner column, so any
caller holding `marketplace:write` could release *anyone's* allocation by id,
freeing their GPU out from under a running job. The no-op was the only thing
standing between the surface and that, which is why the bug could not be fixed
where it appeared.

So the column comes first, and the lookup is corrected in the same release only
because it is meaningless without it.

**Expand-only** (rule 5). The column is nullable, backfilled from the job it
belongs to, and nothing is dropped. Rows whose job has since been deleted stay
`NULL` and are therefore releasable by nobody — the safe direction for a value
that authorizes an action, and the reason this does not add `NOT NULL`.

The backfill is a single UPDATE against a table that holds one row per
allocation and is indexed on `job_id`; `lock_timeout` keeps it from queueing
behind a long transaction rather than blocking writers indefinitely.
"""

from alembic import op

revision = "101"
down_revision = "100"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("ALTER TABLE gpu_allocations ADD COLUMN IF NOT EXISTS owner_id TEXT")
    # Backfill from the job the allocation was created for. A LEFT JOIN would
    # leave the same NULLs, so this is written as a correlated UPDATE that
    # touches only rows it can actually resolve.
    op.execute(
        """
        UPDATE gpu_allocations a
           SET owner_id = j.owner_id
          FROM jobs j
         WHERE j.job_id = a.job_id
           AND a.owner_id IS NULL
           AND j.owner_id IS NOT NULL
        """
    )
    # Releasing is looked up by (allocation_id, owner_id); allocation_id is
    # already the primary key, so this index serves the "list what I own" reads
    # rather than the release path.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_gpu_alloc_owner ON gpu_allocations (owner_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_gpu_alloc_owner")
    op.execute("ALTER TABLE gpu_allocations DROP COLUMN IF EXISTS owner_id")
