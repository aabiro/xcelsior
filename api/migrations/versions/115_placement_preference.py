"""Where a stated placement preference lives, so a launch can honour one.

P5's gate is *"a placement preference that cannot be satisfied refuses clearly
rather than silently falling back to the cheapest host"*. Everything needed to
decide that existed — `PlacementPreference`, `evaluate_preference`,
`choose_host(hosts, preference=None)` — and the preference had **nowhere to be
stored**, so it never reached the scheduler. `/api/v1/placements/evaluate` could
answer what *would* happen; a launch could not ask for it.

## Why its own columns and not `jobs.spec`

`PlacementPreferenceIn` says it outright: the preference is deliberately not part
of the spec, because `spec` feeds `canonicalize()` and `spec_hash`. Two identical
workloads asking for different reliability would hash differently, and an
approved plan could no longer be matched to a rerun. The spec is *what to run*;
this is *which host may run it*. Folding them together breaks approval matching,
so they stay apart in the schema too.

## Why basis points rather than percent floats

`max_premium_bps` gates a **price** comparison — "pay at most 15% more than the
cheapest eligible host". As a float, 15% is 15.000000000000002 and a host sitting
exactly on the bound falls either side depending on the binary representation of
a number nobody typed. Integer basis points make the boundary exact and the same
on every machine, which is the same reason money here is `*_micros` and the
platform cut is `platform_cut_bps`.

`min_uptime_bps` is the same axis of care for a different reason: 99.95% and
99.9% are meaningfully different SLAs and a float comparison at three decimal
places is where that distinction quietly stops holding.

Scale is basis points throughout — 1 bps = 0.01%. `99.5%` is `9950`; a 15%
premium cap is `1500`.

## Why `placement_min_tier` has no enumerated CHECK

Tempting, and wrong here. `control_plane/scheduler/preference.py` **derives** the
tier vocabulary and its ordering from `ReputationTier` thresholds, and records
why: an earlier version invented `("unverified", "basic", "verified", "trusted")`
and every `min_tier` constraint refused against real data — "a gate that always
refuses is indistinguishable from a broken one", found by querying production
rather than by any test.

A CHECK listing the tiers would be a second, hand-written copy of a vocabulary
the code deliberately derives, and it would reject a tier inserted between two
others until someone shipped a migration. So the constraint here polices the
*shape* — a short lowercase identifier — and the vocabulary stays owned by the
one place that derives it.

## No index, deliberately

The scheduler reads these columns for a job it already has by primary key; they
are never a filter or a join key. An index would be storage and write cost for a
lookup that is already a PK hit.

`require_verified` is `NOT NULL DEFAULT FALSE` — the absence of a stated
preference must read as "no constraint asked for", never as NULL-shaped unknown
that some later comparison gets to interpret. On PostgreSQL 11+ a non-volatile
default is a catalog change, so this does not rewrite the table.
"""

from alembic import op

revision = "115"
down_revision = "114"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute(
        """
        ALTER TABLE jobs
          ADD COLUMN IF NOT EXISTS placement_min_uptime_bps INTEGER,
          ADD COLUMN IF NOT EXISTS placement_min_tier TEXT,
          ADD COLUMN IF NOT EXISTS placement_require_verified BOOLEAN NOT NULL DEFAULT FALSE,
          ADD COLUMN IF NOT EXISTS placement_max_premium_bps INTEGER
        """
    )

    # 0–100% expressed in basis points. Out of range is a caller bug, and the
    # database is the only place that catches it for every writer at once.
    op.execute(
        """
        ALTER TABLE jobs
          DROP CONSTRAINT IF EXISTS ck_jobs_placement_min_uptime_bps_range
        """
    )
    op.execute(
        """
        ALTER TABLE jobs
          ADD CONSTRAINT ck_jobs_placement_min_uptime_bps_range
          CHECK (placement_min_uptime_bps IS NULL
                 OR (placement_min_uptime_bps >= 0 AND placement_min_uptime_bps <= 10000))
        """
    )

    # A premium bound is unbounded above in principle; the API caps the request
    # at 10,000% and the schema agrees rather than trusting one validator.
    op.execute(
        """
        ALTER TABLE jobs
          DROP CONSTRAINT IF EXISTS ck_jobs_placement_max_premium_bps_range
        """
    )
    op.execute(
        """
        ALTER TABLE jobs
          ADD CONSTRAINT ck_jobs_placement_max_premium_bps_range
          CHECK (placement_max_premium_bps IS NULL
                 OR (placement_max_premium_bps >= 0
                     AND placement_max_premium_bps <= 100000000))
        """
    )

    # Shape, not vocabulary — see the module docstring. An empty string is a
    # caller that meant NULL and said something else; refuse it here so the
    # scheduler never has to decide what `''` means.
    op.execute(
        """
        ALTER TABLE jobs
          DROP CONSTRAINT IF EXISTS ck_jobs_placement_min_tier_shape
        """
    )
    op.execute(
        """
        ALTER TABLE jobs
          ADD CONSTRAINT ck_jobs_placement_min_tier_shape
          CHECK (placement_min_tier IS NULL
                 OR placement_min_tier ~ '^[a-z][a-z_]{0,31}$')
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE jobs
          DROP CONSTRAINT IF EXISTS ck_jobs_placement_min_uptime_bps_range,
          DROP CONSTRAINT IF EXISTS ck_jobs_placement_max_premium_bps_range,
          DROP CONSTRAINT IF EXISTS ck_jobs_placement_min_tier_shape
        """
    )
    op.execute(
        """
        ALTER TABLE jobs
          DROP COLUMN IF EXISTS placement_min_uptime_bps,
          DROP COLUMN IF EXISTS placement_min_tier,
          DROP COLUMN IF EXISTS placement_require_verified,
          DROP COLUMN IF EXISTS placement_max_premium_bps
        """
    )
