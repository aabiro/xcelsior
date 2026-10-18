"""The reported SSH host-key fingerprint. A2 of docs/host-key-fingerprint-plan.md.

One nullable column on `jobs`, and the table choice is a **deviation from the
plan's prose, made because production falsifies it**.

## Why not the attempt

A2 says the fingerprint *"belongs to the attempt/container, not the job"*. On
production that column would be null for the entire fleet: **327 jobs, 0 with an
active attempt, 1 with any attempt at all.** The fenced path that creates
attempts is not the path real jobs take. An attempt-scoped column ships a feature
that stores nothing — the defect this phase has spent its length removing.

## Why not cleared where the plan says

A2 also says it is *"cleared wherever `_clear_job_output` is called"*. That hook
is called from exactly one place — inside `requeue_job` — and is **gated on
`user_initiated`**, deliberately:

    Gated on `user_initiated` deliberately. Both user-facing doors pass it;
    automatic failover does not, and a failover that erased the logs explaining
    why it failed over would destroy the evidence for the retry it just
    performed.

That gate is right for logs and fatal for a fingerprint. **Automatic failover is
the primary way a job changes host**, and it is precisely the path that skips the
clear. A fingerprint cleared there would survive a failover onto a new host and
then verify against the wrong one — the exact failure the plan's own reasoning
names, arriving through the mechanism the plan nominated to prevent it.

## What is done instead

The column is nulled wherever `host_id` changes, in the *same* upsert — see
`DatabaseOps.upsert_job` and `scheduler.update_job_status`. Every placement path
funnels through that one write:

* `run_job` · `process_queue_filtered` · `process_queue_ranked` — normal placement
* the CRIU checkpoint migration — a job moved to a new host
* the failover path — the case `_clear_job_output` skips

So the fingerprint lives on the same row as the field defining its container and
cannot outlive it: both move atomically or neither does. No dependence on
`user_initiated`, no hook anyone has to remember. `_clear_job_output`'s
log-preservation behaviour is untouched — these are two different things cleared
for two different reasons, and coupling them was the plan's error.

**Expand-only** (rule 5): one nullable column, nothing altered, no backfill. Every
existing row is legitimately `NULL` — those instances never reported one.
"""

from alembic import op

revision = "108"
down_revision = "107"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    # Nullable with no default: "unknown" is the honest state for every row that
    # exists today, and a default would manufacture an answer for all of them.
    op.execute("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS host_key_fingerprint TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN IF EXISTS host_key_fingerprint")
