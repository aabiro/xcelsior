"""Expand durable weighted queue policy, team service and preemption records.

Requires the native writer handoff and retired projection triggers (131).
Policy revisions are explicit operator inputs; migration enables no evictions.
"""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "133"
down_revision = "132"
branch_labels = None
depends_on = None


def _policy(conn):
    conn.execute(text("""CREATE TABLE IF NOT EXISTS scheduling_virtual_clock (
        singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
        virtual_time NUMERIC(30,9) NOT NULL DEFAULT 0 CHECK (virtual_time>=0)
    )"""))
    conn.execute(
        text("INSERT INTO scheduling_virtual_clock(singleton) VALUES(TRUE) ON CONFLICT DO NOTHING")
    )
    conn.execute(text("""CREATE TABLE IF NOT EXISTS scheduling_team_policies (
        team_key TEXT NOT NULL,
        version INTEGER NOT NULL CHECK (version>0),
        administrative_class INTEGER NOT NULL DEFAULT 0 CHECK (administrative_class BETWEEN 0 AND 100),
        reservation_entitlement INTEGER NOT NULL DEFAULT 0 CHECK (reservation_entitlement BETWEEN 0 AND 100),
        weight NUMERIC(12,6) NOT NULL DEFAULT 1 CHECK (weight BETWEEN 0.001 AND 100000),
        max_concurrent_jobs INTEGER CHECK (max_concurrent_jobs>=0),
        max_allocated_gpus INTEGER CHECK (max_allocated_gpus>=0),
        preemption_enabled BOOLEAN NOT NULL DEFAULT FALSE,
        grace_seconds INTEGER NOT NULL DEFAULT 30 CHECK (grace_seconds BETWEEN 1 AND 300),
        force_after_deadline BOOLEAN NOT NULL DEFAULT FALSE,
        created_by TEXT NOT NULL CHECK (length(created_by)>0),
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        PRIMARY KEY(team_key,version)
    )"""))
    conn.execute(text("""CREATE TABLE IF NOT EXISTS scheduling_team_state (
        team_key TEXT PRIMARY KEY,
        virtual_finish NUMERIC(30,9) NOT NULL DEFAULT 0 CHECK (virtual_finish>=0),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )"""))


def _jobs(conn):
    for name, kind in [
        ("scheduling_team_key", "TEXT"),
        ("queue_policy_version", "INTEGER"),
        ("reservation_entitlement", "INTEGER NOT NULL DEFAULT 0"),
        ("queue_components", "JSONB NOT NULL DEFAULT '{}'::jsonb"),
        ("queue_calculated_at", "TIMESTAMPTZ"),
    ]:
        conn.execute(text(f"ALTER TABLE jobs ADD COLUMN IF NOT EXISTS {name} {kind}"))


def _plans(conn):
    conn.execute(text("""CREATE TABLE IF NOT EXISTS preemption_plans (
        plan_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        nominated_job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
        nominated_generation BIGINT NOT NULL,
        nominated_spec_hash TEXT NOT NULL,
        host_id TEXT NOT NULL REFERENCES hosts(host_id) ON DELETE CASCADE,
        policy_version TEXT NOT NULL CHECK (policy_version='preemption/v1'),
        team_policy_version INTEGER NOT NULL,
        state TEXT NOT NULL DEFAULT 'stopping' CHECK (state IN
            ('stopping','ready','placed','cancelled')),
        grace_deadline TIMESTAMPTZ NOT NULL,
        nomination_expires_at TIMESTAMPTZ NOT NULL,
        force_after_deadline BOOLEAN NOT NULL,
        checkpoint_policy TEXT NOT NULL DEFAULT 'none' CHECK (checkpoint_policy='none'),
        reason_code TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )"""))
    conn.execute(text("""CREATE UNIQUE INDEX IF NOT EXISTS uq_preemption_nominated_active
        ON preemption_plans(nominated_job_id) WHERE state IN ('stopping','ready')"""))
    conn.execute(text("""CREATE TABLE IF NOT EXISTS preemption_plan_victims (
        plan_id UUID NOT NULL REFERENCES preemption_plans(plan_id) ON DELETE CASCADE,
        attempt_id UUID NOT NULL REFERENCES job_attempts(attempt_id) ON DELETE CASCADE,
        job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
        lease_id UUID NOT NULL REFERENCES placement_leases(lease_id) ON DELETE CASCADE,
        fencing_token BIGINT NOT NULL,
        stop_command_id UUID NOT NULL,
        force_command_id UUID,
        released BOOLEAN NOT NULL DEFAULT FALSE,
        PRIMARY KEY(plan_id,attempt_id)
    )"""))
    conn.execute(text("""CREATE UNIQUE INDEX IF NOT EXISTS uq_preemption_victim_active
        ON preemption_plan_victims(attempt_id) WHERE NOT released"""))
    conn.execute(text("""CREATE TABLE IF NOT EXISTS preemption_plan_devices (
        plan_id UUID NOT NULL REFERENCES preemption_plans(plan_id) ON DELETE CASCADE,
        gpu_device_id UUID NOT NULL REFERENCES host_gpu_devices(gpu_device_id) ON DELETE CASCADE,
        held BOOLEAN NOT NULL DEFAULT TRUE,
        PRIMARY KEY(plan_id,gpu_device_id)
    )"""))
    conn.execute(text("""CREATE UNIQUE INDEX IF NOT EXISTS uq_preemption_device_held
        ON preemption_plan_devices(gpu_device_id) WHERE held"""))


def upgrade():
    apply_in_own_transactions(
        [
            ("durable team policy", _policy),
            ("persisted queue components", _jobs),
            ("durable preemption plans", _plans),
        ],
        tables=[
            "jobs",
            "hosts",
            "job_attempts",
            "placement_leases",
            "agent_commands",
            "host_gpu_devices",
            "scheduling_virtual_clock",
            "scheduling_team_policies",
            "scheduling_team_state",
            "preemption_plans",
            "preemption_plan_victims",
            "preemption_plan_devices",
        ],
    )


def downgrade():
    # Expansion is deliberately retained. Dropping active plans or service
    # credit on rollback would erase ownership and fairness evidence.
    pass
