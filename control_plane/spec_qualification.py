"""Aggregate-only evidence for historical spec and column contraction readiness."""

from __future__ import annotations

from datetime import datetime, timezone

from psycopg.rows import tuple_row

from control_plane.job_specs import validate_execution_spec


def audit_spec_qualification(conn, *, expected_database: str, expected_port: int) -> dict:
    with conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
        conn.execute("SET LOCAL statement_timeout='15s'")
        conn.execute("SET LOCAL lock_timeout='1s'")
        with conn.cursor(row_factory=tuple_row) as cur:
            database, port = cur.execute(
                "SELECT current_database(),current_setting('port')::integer"
            ).fetchone()
            if (database, port) != (expected_database, expected_port):
                raise ValueError("spec audit refused an unexpected target")
            counts = dict(
                total=0,
                valid_admission=0,
                valid_recorded=0,
                invalid=0,
                missing=0,
                recorded_without_provenance=0,
            )
            evidence = None
            recovered = {}
            if cur.execute("SELECT to_regclass('public.job_spec_backfill_evidence')").fetchone()[0]:
                evidence = dict(
                    cur.execute(
                        "SELECT outcome,count(*) FROM job_spec_backfill_evidence GROUP BY outcome"
                    ).fetchall()
                )
                recovered = dict(
                    cur.execute(
                        "SELECT job_id,recorded_spec_hash FROM job_spec_backfill_evidence "
                        "WHERE outcome='recovered_execution'"
                    ).fetchall()
                )
            for job_id, spec, digest in cur.execute(
                "SELECT job_id,spec,spec_hash FROM jobs"
            ).fetchall():
                counts["total"] += 1
                if spec is None:
                    counts["missing"] += 1
                    continue
                try:
                    validate_execution_spec(spec, digest)
                except (ValueError, TypeError):
                    counts["invalid"] += 1
                else:
                    key = (
                        "valid_admission" if "execution_spec_version" in spec else "valid_recorded"
                    )
                    counts[key] += 1
                    if key == "valid_recorded" and recovered.get(job_id) != digest:
                        counts["recorded_without_provenance"] += 1
            projection = cur.execute("""SELECT count(*) FILTER (WHERE phase IS NULL),
                    count(*) FILTER (WHERE desired_state IS NULL),
                    count(*) FILTER (WHERE phase='pending' AND queued_at IS NULL)
                FROM jobs""").fetchone()
            triggers = [
                row[0]
                for row in cur.execute(
                    """SELECT tgname FROM pg_trigger
                    WHERE tgname IN ('trg_jobs_control_plane_projection',
                                     'trg_hosts_control_plane_projection') ORDER BY tgname"""
                ).fetchall()
            ]
            required_columns = dict(cur.execute("""SELECT column_name,is_nullable='NO'
                FROM information_schema.columns WHERE table_schema='public' AND table_name='jobs'
                  AND column_name IN ('phase','desired_state')""").fetchall())
            queue_check = cur.execute(
                """SELECT EXISTS (SELECT 1 FROM pg_constraint
                WHERE conrelid='jobs'::regclass AND conname='ck_jobs_pending_queued_at' AND convalidated)"""
            ).fetchone()[0]
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "complete": True,
        "read_only": True,
        "database": database,
        "port": port,
        "specs": counts,
        "backfill_evidence": evidence,
        "normalized_column_gaps": dict(
            zip(("phase", "desired_state", "pending_queued_at"), projection)
        ),
        "transitional_triggers": triggers,
        "required_columns_not_null": required_columns,
        "pending_queue_check_validated": queue_check,
        "all_job_specs_qualified": counts["missing"]
        == counts["invalid"]
        == counts["recorded_without_provenance"]
        == 0,
        "trigger_removal_requires_native_writer_gate": True,
    }
