"""Count recurrences outside the WORM table, so the trail records decisions not polls.

`placement_decisions` records **every evaluation**, which is the right write
policy — a preference that refused was honoured by the refusal, and a
successes-only trail cannot answer "why did nothing launch last Tuesday".

But a caller polling a preference writes the same decision repeatedly, and each
row carries a `candidates` snapshot that scales with the fleet. That is a fleet
snapshot per poll, not a thin log line.

**The fix is not a counter on the row.** WORM forbids UPDATE by design, so a
`times_seen` column on `placement_decisions` is unimplementable — and that is the
correct constraint, not an obstacle to work around. Frequency is *operational
telemetry*: it has a natural retention policy and no business being immutable.
Conflating "what was decided" with "how often we were asked" is what created the
growth problem.

So: one WORM row per **distinct decision**, and the recurrence count here, in a
plain table that can be updated and pruned.

## Why the month is part of the key

Dedupe is scoped to the calendar month, which is the same granularity
`placement_decisions` is partitioned by. Without it, a decision made today and
the identical decision made next March would collapse into one row timestamped
today, and the trail would say the March decision never happened. Scoping to the
month bounds that to at most one row per distinct decision per month, and means
the two tables prune on the same boundary.

**Expand-only** (rule 5): one new table, nothing altered.
"""

from alembic import op

revision = "107"
down_revision = "106"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS placement_decision_observations (
            tenant_id    TEXT NOT NULL,
            -- sha256 over (tenant, job, asked, outcome, refusal_code, and the
            -- sorted candidate states). Two evaluations sharing it decided the
            -- same thing over the same field.
            fingerprint  TEXT NOT NULL,
            month        TEXT NOT NULL,
            times_seen   BIGINT NOT NULL DEFAULT 1,
            first_seen_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
            -- The WORM row this fingerprint was first recorded as. Nullable on
            -- purpose: the observation is written *before* the decision, so a
            -- null here is the honest record of "we saw this and failed to
            -- write it down", which is worth being able to find.
            decision_id  UUID,
            PRIMARY KEY (tenant_id, fingerprint, month),
            CONSTRAINT ck_placement_observations_count CHECK (times_seen >= 1)
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_observations_tenant "
        "ON placement_decision_observations (tenant_id, last_seen_at DESC)"
    )
    # Finding the observations that never got their decision written.
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_placement_observations_unrecorded "
        "ON placement_decision_observations (last_seen_at) WHERE decision_id IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_placement_observations_unrecorded")
    op.execute("DROP INDEX IF EXISTS idx_placement_observations_tenant")
    op.execute("DROP TABLE IF EXISTS placement_decision_observations")
