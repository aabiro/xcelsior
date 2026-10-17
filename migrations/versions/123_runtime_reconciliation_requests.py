"""Wake current jobs on command outcomes and lease authority changes.

Requests commit with the mutation, including writes from compatibility paths.
Healthy lease extensions do not enqueue work. No backfill or enforcement flip.
"""

from alembic import op

revision = "123"
down_revision = "122"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("""
        CREATE FUNCTION enqueue_runtime_job_reconciliation(
            changed_job text, changed_attempt uuid, changed_host text,
            changed_fence bigint, change_reason text
        ) RETURNS void LANGUAGE sql AS $$
            INSERT INTO reconciliation_queue
                (resource_type, resource_id, reason, requested_by)
            SELECT 'job', j.job_id, change_reason, 'runtime_trigger'
              FROM jobs j JOIN job_attempts a ON a.attempt_id = j.active_attempt_id
             WHERE j.job_id = changed_job AND a.job_id = j.job_id
               AND a.attempt_id = changed_attempt AND a.host_id = changed_host
               AND j.host_id = a.host_id AND a.fencing_token = changed_fence
            ON CONFLICT (resource_type, resource_id) DO UPDATE
               SET due_at = LEAST(reconciliation_queue.due_at, clock_timestamp()),
                   reason = EXCLUDED.reason, updated_at = clock_timestamp();
        $$;

        CREATE FUNCTION request_runtime_reconciliation() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE
            change_reason text;
        BEGIN
            change_reason := CASE TG_TABLE_NAME
                WHEN 'agent_commands' THEN 'command_changed' ELSE 'lease_changed' END;
            IF TG_OP = 'UPDATE' THEN
                IF ROW(NEW.job_id, NEW.attempt_id, NEW.host_id, NEW.fencing_token, NEW.status)
                   IS NOT DISTINCT FROM
                   ROW(OLD.job_id, OLD.attempt_id, OLD.host_id, OLD.fencing_token, OLD.status)
                THEN
                    IF TG_TABLE_NAME = 'agent_commands' THEN
                        IF NEW.error_code IS NOT DISTINCT FROM OLD.error_code
                           AND NEW.expires_at IS NOT DISTINCT FROM OLD.expires_at THEN
                            RETURN NEW;
                        END IF;
                    ELSE
                        -- Normal renewals extend authority, not reconciliation work.
                        -- A shortened/removed deadline or revival of an expired lease
                        -- changes validity and must be observed immediately.
                        IF NOT (
                            (OLD.expires_at IS NOT NULL AND
                                (NEW.expires_at IS NULL OR NEW.expires_at < OLD.expires_at))
                            OR NEW.claim_deadline < OLD.claim_deadline
                            OR (OLD.status = 'active' AND OLD.expires_at <= clock_timestamp()
                                AND NEW.expires_at > clock_timestamp())
                        ) THEN
                            RETURN NEW;
                        END IF;
                    END IF;
                END IF;
            END IF;
            IF TG_OP = 'DELETE' THEN
                PERFORM enqueue_runtime_job_reconciliation(
                    OLD.job_id, OLD.attempt_id, OLD.host_id, OLD.fencing_token, change_reason);
                RETURN OLD;
            END IF;
            IF TG_OP = 'UPDATE' THEN
                IF ROW(NEW.job_id, NEW.attempt_id, NEW.host_id, NEW.fencing_token)
                   IS DISTINCT FROM
                   ROW(OLD.job_id, OLD.attempt_id, OLD.host_id, OLD.fencing_token) THEN
                    PERFORM enqueue_runtime_job_reconciliation(
                        OLD.job_id, OLD.attempt_id, OLD.host_id, OLD.fencing_token, change_reason);
                END IF;
            END IF;
            PERFORM enqueue_runtime_job_reconciliation(
                NEW.job_id, NEW.attempt_id, NEW.host_id, NEW.fencing_token, change_reason);
            RETURN NEW;
        END;
        $$;
        CREATE TRIGGER agent_commands_request_reconciliation
        AFTER INSERT OR UPDATE OR DELETE ON agent_commands
        FOR EACH ROW EXECUTE FUNCTION request_runtime_reconciliation();
        CREATE TRIGGER placement_leases_request_reconciliation
        AFTER INSERT OR UPDATE OR DELETE ON placement_leases
        FOR EACH ROW EXECUTE FUNCTION request_runtime_reconciliation();
    """)


def downgrade() -> None:
    op.execute("SET lock_timeout = '5s'")
    op.execute("DROP TRIGGER IF EXISTS agent_commands_request_reconciliation ON agent_commands")
    op.execute("DROP TRIGGER IF EXISTS placement_leases_request_reconciliation ON placement_leases")
    op.execute("DROP FUNCTION IF EXISTS request_runtime_reconciliation()")
    op.execute(
        "DROP FUNCTION IF EXISTS enqueue_runtime_job_reconciliation(text, uuid, text, bigint, text)"
    )
