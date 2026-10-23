"""Permit host eviction results without inventing a launch job identifier."""

from sqlalchemy import text
from migrations.lock_safe import apply_in_own_transactions

revision = "137"
down_revision = "136"
branch_labels = None
depends_on = None


def _constraint(conn, *, allow_eviction):
    conn.execute(text("LOCK TABLE action_plans IN ACCESS EXCLUSIVE MODE"))
    result = "job_id IS NOT NULL"
    if allow_eviction:
        result = """(job_id IS NOT NULL OR (action_type='evict_host_workloads'
                    AND resulting_resource_id IS NOT NULL))"""
    else:
        if conn.execute(text("""SELECT 1 FROM action_plans
            WHERE status='succeeded' AND job_id IS NULL LIMIT 1""")).first():
            raise RuntimeError("retain completed native eviction evidence; downgrade refused")
    conn.execute(text("ALTER TABLE action_plans DROP CONSTRAINT ck_action_plans_state_machine"))
    conn.execute(
        text(f"""ALTER TABLE action_plans ADD CONSTRAINT ck_action_plans_state_machine CHECK (
        (status<>'approved' OR approved_at IS NOT NULL)
        AND (status NOT IN ('executing','succeeded','failed_retryable','failed_terminal')
            OR (approved_at IS NOT NULL AND consumed_at IS NOT NULL))
        AND (status<>'succeeded' OR ({result} AND idempotent_response IS NOT NULL))
        AND (status NOT IN ('failed_retryable','failed_terminal') OR failure_code IS NOT NULL)
        AND (status<>'revoked' OR revoked_at IS NOT NULL))""")
    )


def upgrade():
    apply_in_own_transactions(
        [("native host eviction results", lambda c: _constraint(c, allow_eviction=True))],
        tables=["action_plans"],
    )


def downgrade():
    apply_in_own_transactions(
        [("restore launch result requirement", lambda c: _constraint(c, allow_eviction=False))],
        tables=["action_plans"],
    )
