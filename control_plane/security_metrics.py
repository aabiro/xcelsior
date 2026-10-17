"""Bounded-cardinality security signals from authoritative database state."""

from __future__ import annotations

import os
import time


def collect(conn) -> dict[str, int | float]:
    revived = conn.execute("""
        SELECT count(*) FROM jobs j JOIN state s
          ON s.namespace = 'security.job.' || j.job_id
         WHERE j.status IN ('queued','assigned','leased','starting','running')
    """).fetchone()[0]
    identities = conn.execute("""
        SELECT count(*) FROM pg_stat_activity
         WHERE datname=current_database()
           AND application_name IN ('xcelsior-api','xcelsior-scheduler','xcelsior-reconciler',
               'xcelsior-billing','xcelsior-projector','xcelsior-retrieval','xcelsior-readonly',
               'xcelsior-ssh_gateway')
           AND usename <> replace(application_name,'-','_')
    """).fetchone()[0]
    return {
        "abuse_jobs_runnable": int(revived),
        "database_identity_mismatches": int(identities),
        "webhook_dedicated_key_configured": int(
            bool(os.environ.get("XCELSIOR_SERVERLESS_WEBHOOK_HMAC_SECRET"))
        ),
        "last_success_timestamp_seconds": time.time(),
    }


def render(snapshot: dict | None) -> list[str]:
    values = {"available": int(snapshot is not None), **(snapshot or {})}
    lines = []
    for key, value in values.items():
        name = f"xcelsior_security_{key}"
        lines.extend([f"# TYPE {name} gauge", f"{name} {value}"])
    return lines
