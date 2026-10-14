"""Durably request job reconciliation alongside lifecycle mutations.

The trigger covers legacy writers during their migration to domain services.
No backfill runs under the migration lock: the bounded recovery task covers
existing active jobs. Observed-generation acknowledgements do not requeue.
"""

from alembic import op

revision = "122"
down_revision = "121"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("""
        CREATE FUNCTION request_job_reconciliation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                DELETE FROM reconciliation_queue
                 WHERE resource_type = 'job' AND resource_id = OLD.job_id;
                RETURN OLD;
            END IF;
            IF TG_OP = 'UPDATE' THEN
                IF ROW(NEW.status, NEW.desired_state, NEW.phase, NEW.host_id,
                       NEW.active_attempt_id, NEW.generation,
                       NEW.payload->>'lifecycle_intent') IS NOT DISTINCT FROM
                   ROW(OLD.status, OLD.desired_state, OLD.phase, OLD.host_id,
                       OLD.active_attempt_id, OLD.generation,
                       OLD.payload->>'lifecycle_intent') THEN
                    RETURN NEW;
                END IF;
            END IF;
            INSERT INTO reconciliation_queue
                (resource_type, resource_id, reason, requested_by)
            VALUES ('job', NEW.job_id, 'job_changed', 'jobs_trigger')
            ON CONFLICT (resource_type, resource_id) DO UPDATE
               SET due_at = LEAST(reconciliation_queue.due_at, clock_timestamp()),
                   reason = EXCLUDED.reason, updated_at = clock_timestamp();
            RETURN NEW;
        END;
        $$;
        CREATE TRIGGER jobs_request_reconciliation
        AFTER INSERT OR UPDATE OR DELETE ON jobs
        FOR EACH ROW EXECUTE FUNCTION request_job_reconciliation();
    """)


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP TRIGGER IF EXISTS jobs_request_reconciliation ON jobs")
    op.execute("DROP FUNCTION IF EXISTS request_job_reconciliation()")
