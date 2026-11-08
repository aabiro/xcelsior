"""Slurm cluster registry, and the reconciler's columns on slurm_job_mappings.

Phase 1 of the Slurm blueprint (hpc/IMPLEMENTATION_PLAN.md §5a): one batched
reconciler per cluster, which alone polls Slurm and keeps Xcelsior's copy of
every job's state for the dashboard, the API and agents to read.

**`slurm_clusters`** — one row per cluster the reconciler polls. It carries the
legitimacy flags the research made a code requirement: a research-only cluster
(an Alliance allocation, whose terms allow academic research under a PI and
nothing else) can never be marketplace capacity or reachable by agent traffic.
A CHECK enforces that, so it can't be lost to a wrong default. Health is kept
in three parts because they fail separately: the SSH transport, the controller
(slurmctld), and accounting (slurmdbd). `breaker_open_until` lets the
reconciler back off a cluster that keeps failing instead of reconnecting every
pass.

**`slurm_job_mappings`**:

- `observed_state` gains `preempted` (decision D2), `held` and `suspended`.
  The old projection called a held job (REQUEUE_HOLD, SPECIAL_EXIT) failed,
  and a suspended one pending, though it keeps its memory and GPUs.
- Columns the reconciler writes: the raw Slurm state and reason, the exit
  code, start and end times, `missing_since` (the first pass the job was in
  neither squeue nor accounting; never a reason to call it finished) and
  `last_changed_at` (when the observation last changed, as opposed to
  `last_observed_at`, when it was last confirmed).

Every unit is idempotent and runs in its own short-lock transaction
(migrations/lock_safe.py), so a retried deploy resumes.

`downgrade` drops the table and the columns and restores 067's constraint.
Rows observed in one of the new states become `unknown` first: lossy, and
harmless, since the next reconcile pass re-observes them.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "152"
down_revision = "151"
branch_labels = None
depends_on = None

#: The observed_state constraint after this migration.
SLURM_OBSERVED = (
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
    "unknown",
    "preempted",
    "held",
    "suspended",
)
#: Migration 067's constraint, restored by downgrade.
SLURM_OBSERVED_067 = ("pending", "running", "completed", "failed", "cancelled", "unknown")

MAPPING_COLUMNS = (
    ("slurm_state", "TEXT"),
    ("state_reason", "TEXT"),
    ("exit_code", "TEXT"),
    ("started_at", "TIMESTAMPTZ"),
    ("ended_at", "TIMESTAMPTZ"),
    ("missing_since", "TIMESTAMPTZ"),
    ("last_changed_at", "TIMESTAMPTZ"),
)


def _create_clusters(conn):
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS slurm_clusters (
                cluster_id            TEXT PRIMARY KEY,
                profile_key           TEXT NOT NULL,
                display_name          TEXT NOT NULL,
                enabled               BOOLEAN NOT NULL DEFAULT TRUE,
                research_only         BOOLEAN NOT NULL DEFAULT FALSE,
                marketplace           BOOLEAN NOT NULL DEFAULT FALSE,
                agent_access          BOOLEAN NOT NULL DEFAULT FALSE,
                poll_interval_seconds INTEGER NOT NULL DEFAULT 60,
                facts                 JSONB NOT NULL DEFAULT '{}'::jsonb,
                last_poll_at          TIMESTAMPTZ,
                transport_ok_at       TIMESTAMPTZ,
                controller_ok_at      TIMESTAMPTZ,
                accounting_ok_at      TIMESTAMPTZ,
                last_error_code       TEXT,
                last_error            TEXT,
                consecutive_failures  INTEGER NOT NULL DEFAULT 0,
                breaker_open_until    TIMESTAMPTZ,
                created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT ck_slurm_cluster_research_only
                    CHECK (NOT (research_only AND (marketplace OR agent_access))),
                CONSTRAINT ck_slurm_cluster_poll_interval
                    CHECK (poll_interval_seconds BETWEEN 30 AND 3600)
            )
            """
        )
    )


def _add_mapping_columns(conn):
    for name, kind in MAPPING_COLUMNS:
        conn.execute(text(f"ALTER TABLE slurm_job_mappings ADD COLUMN IF NOT EXISTS {name} {kind}"))


def _observed_constraint(states):
    def run(conn):
        allowed = ", ".join(f"'{s}'" for s in states)
        conn.execute(text("ALTER TABLE slurm_job_mappings DROP CONSTRAINT IF EXISTS ck_slurm_observed_state"))
        conn.execute(
            text(
                "ALTER TABLE slurm_job_mappings ADD CONSTRAINT ck_slurm_observed_state "
                f"CHECK (observed_state IS NULL OR observed_state IN ({allowed})) NOT VALID"
            )
        )
        conn.execute(text("ALTER TABLE slurm_job_mappings VALIDATE CONSTRAINT ck_slurm_observed_state"))

    return run


def _unknown_for_new_states(conn):
    new = [s for s in SLURM_OBSERVED if s not in SLURM_OBSERVED_067]
    conn.execute(
        text("UPDATE slurm_job_mappings SET observed_state = 'unknown' WHERE observed_state = ANY(:new)"),
        {"new": new},
    )


def _drop_mapping_columns(conn):
    for name, _kind in MAPPING_COLUMNS:
        conn.execute(text(f"ALTER TABLE slurm_job_mappings DROP COLUMN IF EXISTS {name}"))


def upgrade():
    apply_in_own_transactions(
        [
            ("slurm_clusters", _create_clusters),
            ("slurm_job_mappings columns", _add_mapping_columns),
            ("slurm_job_mappings observed_state", _observed_constraint(SLURM_OBSERVED)),
        ],
        tables=("slurm_clusters", "slurm_job_mappings"),
    )


def downgrade():
    apply_in_own_transactions(
        [
            ("slurm_job_mappings new states", _unknown_for_new_states),
            ("slurm_job_mappings observed_state", _observed_constraint(SLURM_OBSERVED_067)),
            ("slurm_job_mappings columns", _drop_mapping_columns),
            ("slurm_clusters", lambda conn: conn.execute(text("DROP TABLE IF EXISTS slurm_clusters"))),
        ],
        tables=("slurm_clusters", "slurm_job_mappings"),
    )
