#!/usr/bin/env python3
"""One-shot partition maintenance using ONLY the migration credential.

Schedule daily outside runtime containers after ownership has been transferred
to xcelsior_migrator. Do not give this DSN to API/scheduler/background services.
"""

import os
from pathlib import Path
import sys

for _p in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    sys.path.insert(0, str(_p))

import psycopg
from control_plane.audit_partitions import (
    PARTITIONED_TABLES,
    TELEMETRY_TABLE,
    ensure_monthly_partitions,
    drop_expired_partitions,
    telemetry_policy,
    WORM_RETENTION_MONTHS,
)
from control_plane.db_roles import SERVICE_ROLES, grant_statements
from scripts.provision_db_roles import _fetch_inventory, sequence_grants


def maintain(conn):
    if conn.execute("SELECT current_user, session_user").fetchone() != (
        "xcelsior_migrator",
        "xcelsior_migrator",
    ):
        raise RuntimeError("partition maintenance requires a direct migration-role login")
    conn.commit()
    for task, tables in (
        ("audit_partition_maintenance", PARTITIONED_TABLES),
        ("telemetry_partition_maintenance", (TELEMETRY_TABLE,)),
    ):
        try:
            for table in tables:
                with conn.transaction():
                    ahead, keep = (
                        telemetry_policy(conn)
                        if table == TELEMETRY_TABLE
                        else (3, WORM_RETENTION_MONTHS)
                    )
                    ensure_monthly_partitions(conn, table, months_ahead=ahead)
                    qualified, storage = _fetch_inventory(conn)
                    for role in SERVICE_ROLES.values():
                        for stmt in grant_statements(
                            role, storage_tables=storage, existing_tables=qualified
                        ):
                            conn.execute(stmt)
                        for stmt in sequence_grants(conn, role):
                            conn.execute(stmt)
                with conn.transaction():
                    drop_expired_partitions(conn, table, retention_months=keep)
            with conn.transaction():
                conn.execute(
                    "UPDATE scheduled_tasks SET last_status='succeeded', last_error=NULL, "
                    "last_run_at=clock_timestamp() WHERE task_name=%s",
                    (task,),
                )
        except Exception:
            with conn.transaction():
                conn.execute(
                    "UPDATE scheduled_tasks SET last_status='failed', "
                    "last_error='external partition maintenance failed', last_run_at=clock_timestamp() "
                    "WHERE task_name=%s",
                    (task,),
                )
            raise


def main():
    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN_MIGRATOR")
    if not dsn:
        raise SystemExit("XCELSIOR_POSTGRES_DSN_MIGRATOR is required; shared fallback refused")
    with psycopg.connect(dsn, options="-c lock_timeout=5000 -c statement_timeout=120000") as conn:
        maintain(conn)
    print("partition maintenance succeeded")


if __name__ == "__main__":
    main()
