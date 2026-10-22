"""Populate control-plane metadata from already-recorded evidence.

Resumable per-table units; no guessed users, providers, specs or timestamps.
Known canonical fields win. Downgrade retains corrected facts rather than
silently erasing recovered ownership. No schema/trigger removal is bundled.
"""

import hashlib
import json

from alembic import op
from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "128"
down_revision = "127"
branch_labels = None
depends_on = None


def _recorded(key):
    return (
        f"CASE WHEN jsonb_typeof(payload->'{key}')='string' THEN NULLIF(payload->>'{key}','') END"
    )


JOB_OWNER = _recorded("owner")
JOB_TENANT = f"COALESCE({_recorded('tenant_id')},tenant_id,owner_id,{JOB_OWNER})"
JOB_TEAM = _recorded("team_id")
JOBS_WHERE = f"""(owner_id IS NULL AND {JOB_OWNER} IS NOT NULL)
 OR (tenant_id IS NULL AND {JOB_TENANT} IS NOT NULL)
 OR (team_id IS NULL AND {JOB_TEAM} IS NOT NULL)"""
HOST_OWNER = _recorded("owner")
HOST_TENANT = f"COALESCE({_recorded('tenant_id')},tenant_id,owner_id,{HOST_OWNER})"
HOSTS_WHERE = f"""(owner_id IS NULL AND {HOST_OWNER} IS NOT NULL)
 OR (tenant_id IS NULL AND {HOST_TENANT} IS NOT NULL)
 OR (capabilities='{{}}'::jsonb AND jsonb_typeof(payload->'capabilities')='object'
     AND payload->'capabilities'<>'{{}}'::jsonb)"""


def _backfill(conn, table, primary_key, predicate, assignments, allowed_columns):
    conn.execute(text("SET LOCAL statement_timeout='30s'"))
    allowed = "ARRAY[" + ",".join("'" + column + "'" for column in allowed_columns) + "]::text[]"
    for _ in range(10000):
        changed, drift = conn.execute(text(f"""WITH batch AS (
            SELECT {primary_key},to_jsonb(s)-{allowed} AS before
            FROM {table} s WHERE {predicate}
            ORDER BY {primary_key} LIMIT 500 FOR UPDATE SKIP LOCKED
        ), changed AS (UPDATE {table} t SET {assignments}
          FROM batch b WHERE t.{primary_key}=b.{primary_key}
          RETURNING to_jsonb(t)-{allowed} AS after,b.before)
        SELECT count(*),bool_or(after IS DISTINCT FROM before) FROM changed""")).one()
        if drift:
            raise RuntimeError(
                f"{table} metadata backfill changed unrelated state; transaction refused"
            )
        if changed == 0:
            remaining = conn.execute(
                text(f"SELECT count(*) FROM {table} WHERE {predicate}")
            ).scalar_one()
            if remaining:
                raise RuntimeError(
                    f"identity backfill blocked by {remaining} locked {table} rows; rerun migration"
                )
            return
    raise RuntimeError(f"identity backfill did not converge for {table}")


def _outbox_envelopes(conn):
    conn.execute(text("SET LOCAL statement_timeout='30s'"))
    for _ in range(10000):
        rows = (
            conn.execute(text("""SELECT o.event_id,o.payload,o.headers,c.classification
            FROM outbox_events o LEFT JOIN event_contracts c
              ON c.event_type=o.event_type AND c.version=o.event_version
            WHERE o.occurred_at IS NULL OR o.payload_sha256 IS NULL
            ORDER BY o.created_at,o.event_id LIMIT 500 FOR UPDATE OF o SKIP LOCKED"""))
            .mappings()
            .all()
        )
        if not rows:
            remaining = conn.execute(
                text(
                    "SELECT count(*) FROM outbox_events WHERE occurred_at IS NULL OR payload_sha256 IS NULL"
                )
            ).scalar_one()
            if remaining:
                raise RuntimeError("outbox metadata backfill has locked rows; rerun migration")
            return
        updates = []
        for row in rows:
            headers = row["headers"] or {}

            def recorded(*keys):
                return next(
                    (
                        headers[k]
                        for k in keys
                        if isinstance(headers.get(k), str) and headers[k].strip()
                    ),
                    None,
                )

            updates.append(
                {
                    "event_id": row["event_id"],
                    "digest": hashlib.sha256(
                        json.dumps(
                            row["payload"], sort_keys=True, separators=(",", ":"), default=str
                        ).encode()
                    ).hexdigest(),
                    "classification": row["classification"],
                    # Only occurrence-time headers establish historical attribution.
                    # A job's owner today is not evidence of its owner back then.
                    "tenant": recorded("tenant_id", "workspace_id", "customer_id"),
                    "trace": recorded("trace_id"),
                    "correlation": recorded("correlation_id"),
                    "causation": recorded("causation_id"),
                }
            )
        conn.execute(
            text("""UPDATE outbox_events SET
            occurred_at=COALESCE(occurred_at,created_at),payload_sha256=COALESCE(payload_sha256,:digest),
            classification=COALESCE(classification,:classification),tenant_id=COALESCE(tenant_id,:tenant),
            trace_id=COALESCE(trace_id,:trace),correlation_id=COALESCE(correlation_id,:correlation),
            causation_id=COALESCE(causation_id,:causation) WHERE event_id=:event_id"""),
            updates,
        )
    raise RuntimeError("outbox metadata backfill did not converge")


def upgrade():
    # Qualify references in the UPDATE because the batch repeats the PK only.
    apply_in_own_transactions(
        [
            (
                "jobs recorded identity",
                lambda conn: _backfill(
                    conn,
                    "jobs",
                    "job_id",
                    JOBS_WHERE,
                    f"owner_id=COALESCE(owner_id,{JOB_OWNER}),tenant_id=COALESCE(tenant_id,{JOB_TENANT}),team_id=COALESCE(team_id,{JOB_TEAM})",
                    ("owner_id", "tenant_id", "team_id"),
                ),
            ),
            (
                "hosts recorded identity/capabilities",
                lambda conn: _backfill(
                    conn,
                    "hosts",
                    "host_id",
                    HOSTS_WHERE,
                    f"owner_id=COALESCE(owner_id,{HOST_OWNER}),tenant_id=COALESCE(tenant_id,{HOST_TENANT}),capabilities=CASE WHEN capabilities='{{}}'::jsonb AND jsonb_typeof(payload->'capabilities')='object' THEN payload->'capabilities' ELSE capabilities END",
                    ("owner_id", "tenant_id", "capabilities"),
                ),
            ),
            ("outbox recorded envelope", _outbox_envelopes),
            (
                "archive recovered attribution",
                lambda conn: _backfill(
                    conn,
                    "legacy_lease_archive",
                    "lease_id",
                    "NOT tenant_attributed AND EXISTS(SELECT 1 FROM jobs j WHERE j.job_id=record->>'job_id' AND NULLIF(j.tenant_id,'') IS NOT NULL)",
                    "tenant_id=(SELECT j.tenant_id FROM jobs j WHERE j.job_id=t.record->>'job_id'),tenant_attributed=true",
                    ("tenant_id", "tenant_attributed"),
                ),
            ),
        ],
        tables=["jobs", "hosts", "outbox_events", "legacy_lease_archive"],
    )


def downgrade():
    # Recovered facts are valid at 127 too. Do not turn known identity back
    # into NULL or discard host security capabilities during a code rollback.
    op.execute("SET LOCAL lock_timeout='5s'")
