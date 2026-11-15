#!/usr/bin/env python3
"""One-shot partition maintenance using ONLY the migration credential.

Schedule daily outside runtime containers after ownership has been transferred
to xcelsior_migrator. Do not give this DSN to API/scheduler/background services.

``--grants-only`` converges the runtime roles' grants and touches no partition.
The deploy runs it right after migrating, from the new release's image, so a
table that release created is usable before its API goes live instead of after
the next nightly run.
"""

import argparse
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


def _require_migrator(conn, what: str) -> None:
    if conn.execute("SELECT current_user, session_user").fetchone() != (
        "xcelsior_migrator",
        "xcelsior_migrator",
    ):
        raise RuntimeError(f"{what} requires a direct migration-role login")


def converge_grants(conn) -> None:
    """Grant each runtime role what TABLE_DOMAINS gives it on every table that
    exists now. Idempotent. The caller owns the transaction."""
    qualified, storage = _fetch_inventory(conn)
    for role in SERVICE_ROLES.values():
        for stmt in grant_statements(role, storage_tables=storage, existing_tables=qualified):
            conn.execute(stmt)
        for stmt in sequence_grants(conn, role):
            conn.execute(stmt)


def converge_runtime_grants(conn) -> None:
    """Grants only, in one transaction. No partition is created or dropped."""
    _require_migrator(conn, "runtime grant convergence")
    conn.commit()
    with conn.transaction():
        converge_grants(conn)


def maintain(conn):
    _require_migrator(conn, "partition maintenance")
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
                    # A new partition is a new table: grant it in the same
                    # transaction that created it.
                    converge_grants(conn)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grants-only",
        action="store_true",
        help="converge the runtime roles' grants; create and drop no partition",
    )
    args = parser.parse_args(argv)
    dsn = os.environ.get("XCELSIOR_POSTGRES_DSN_MIGRATOR")
    if not dsn:
        raise SystemExit("XCELSIOR_POSTGRES_DSN_MIGRATOR is required; shared fallback refused")
    with psycopg.connect(dsn, options="-c lock_timeout=5000 -c statement_timeout=120000") as conn:
        if args.grants_only:
            converge_runtime_grants(conn)
        else:
            maintain(conn)
    print("runtime grants converged" if args.grants_only else "partition maintenance succeeded")


if __name__ == "__main__":
    main()
