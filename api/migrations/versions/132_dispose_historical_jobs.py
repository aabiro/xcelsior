"""Retain historical jobs explicitly without granting execution authority.

Billing, commands and observations continue to reference the same job rows.
Recovery evidence is not a customer admission and cannot authorize a retry.
No missing identity, launch input, timestamp or financial value is invented.
"""

import hashlib
import json
from importlib import import_module

from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "132"
down_revision = "131"
branch_labels = None
depends_on = None


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _expand(conn):
    present = conn.execute(
        text("""SELECT EXISTS (SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='jobs' AND column_name='history_disposition')""")
    ).scalar_one()
    if not present:
        conn.execute(text("ALTER TABLE jobs ADD COLUMN IF NOT EXISTS history_disposition TEXT"))
    conn.execute(text("""CREATE TABLE IF NOT EXISTS job_history_dispositions (
        job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
        policy_version TEXT NOT NULL CHECK (policy_version='history/v1'),
        disposition TEXT NOT NULL CHECK (disposition IN
            ('retained_recovered','retained_unavailable')),
        reason_code TEXT NOT NULL,
        identity_evidence TEXT NOT NULL CHECK (identity_evidence IN
            ('recorded','unavailable')),
        recorded_row_sha256 TEXT NOT NULL CHECK (recorded_row_sha256 ~ '^[0-9a-f]{64}$'),
        recorded_payload_sha256 TEXT NOT NULL CHECK (recorded_payload_sha256 ~ '^[0-9a-f]{64}$'),
        recorded_spec_hash TEXT,
        disposed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
    )"""))
    protected = conn.execute(text("""SELECT EXISTS (SELECT 1 FROM pg_constraint
        WHERE conrelid='jobs'::regclass AND conname='ck_jobs_history_disposition')""")).scalar_one()
    if not protected:
        # NOT VALID still enforces every new/updated row. Each independently
        # committed disposition is protected even if a later batch refuses.
        conn.execute(text("""ALTER TABLE jobs ADD CONSTRAINT ck_jobs_history_disposition CHECK (
            history_disposition IS NULL OR
            (history_disposition IN ('retained_recovered','retained_unavailable')
             AND phase IN ('succeeded','failed','stopped') AND active_attempt_id IS NULL)
            ) NOT VALID"""))


def _dispose_batch(conn):
    conn.execute(text("SET LOCAL statement_timeout='30s'"))
    rows = conn.execute(text("""SELECT j.*,to_jsonb(j) AS original,
            e.outcome,e.reason_code AS evidence_reason,e.recorded_spec_hash AS evidence_hash
        FROM jobs j LEFT JOIN job_spec_backfill_evidence e ON e.job_id=j.job_id
        WHERE j.history_disposition IS NULL
          AND (j.spec IS NULL OR NOT (j.spec ? 'execution_spec_version'))
        ORDER BY j.job_id LIMIT 500 FOR UPDATE OF j SKIP LOCKED""")).mappings().all()
    if not rows:
        remaining = conn.execute(text("""SELECT count(*) FROM jobs
            WHERE history_disposition IS NULL
              AND (spec IS NULL OR NOT (spec ? 'execution_spec_version'))""")).scalar_one()
        if remaining:
            raise RuntimeError("historical disposition has locked rows; retry after contention")
        return 0
    for row in rows:
        if row["phase"] not in ("succeeded", "failed", "stopped") or row["active_attempt_id"]:
            raise RuntimeError("historical disposition requires terminal, unowned jobs")
        if not row["outcome"]:
            raise RuntimeError("historical disposition requires recorded qualification evidence")
        if row["spec"] is not None:
            if (
                row["outcome"] != "recovered_execution"
                or row["spec_hash"] != row["evidence_hash"]
                or "sha256:" + _hash(row["spec"]) != row["spec_hash"]
            ):
                raise RuntimeError("historical recovered spec disagrees with its evidence")
            disposition = "retained_recovered"
        else:
            if row["outcome"] != "unavailable":
                raise RuntimeError("historical missing spec disagrees with its evidence")
            disposition = "retained_unavailable"
        # Retrying an interrupted Alembic stamp must not rewrite the receipt.
        conn.execute(
            text("""INSERT INTO job_history_dispositions
            (job_id,policy_version,disposition,reason_code,identity_evidence,
             recorded_row_sha256,recorded_payload_sha256,recorded_spec_hash)
            VALUES (:job_id,'history/v1',:disposition,:reason,:identity,:row_hash,
                    :payload_hash,:spec_hash) ON CONFLICT (job_id) DO NOTHING"""),
            dict(
                job_id=row["job_id"],
                disposition=disposition,
                reason=row["evidence_reason"],
                identity="recorded" if row["owner_id"] and row["tenant_id"] else "unavailable",
                row_hash=_hash(row["original"]),
                payload_hash=_hash(row["payload"]),
                spec_hash=row["spec_hash"],
            ),
        )
        conn.execute(
            text("UPDATE jobs SET history_disposition=:disposition WHERE job_id=:job_id"),
            dict(job_id=row["job_id"], disposition=disposition),
        )
    return len(rows)


def _contract(conn):
    conn.execute(text("SET LOCAL statement_timeout='5min'"))
    conn.execute(text("ALTER TABLE jobs VALIDATE CONSTRAINT ck_jobs_history_disposition"))


def upgrade():
    tables = ["jobs", "job_spec_backfill_evidence", "job_history_dispositions"]
    # Cover terminal records written during the 129 -> native-writer handoff.
    # Reuse the frozen migration's evidence algorithm, never current app code
    # or payload-based admission. Existing assessments remain immutable.
    qualification = import_module("migrations.versions.129_qualify_recorded_job_specs")
    for batch in range(10000):
        assessed = []

        def qualify(conn):
            assessed.append(qualification._qualify_batch(conn))

        apply_in_own_transactions(
            [(f"handoff history qualification {batch}", qualify)],
            tables=["jobs", "job_attempts", "agent_commands", "job_spec_backfill_evidence"],
        )
        if assessed[-1] == 0:
            break
    else:
        raise RuntimeError("handoff history qualification exceeded its bounded batch count")
    apply_in_own_transactions([("historical disposition evidence", _expand)], tables=tables)
    for batch in range(10000):
        count = []

        def apply(conn):
            count.append(_dispose_batch(conn))

        apply_in_own_transactions([(f"historical disposition batch {batch}", apply)], tables=tables)
        if count[-1] == 0:
            break
    else:
        raise RuntimeError("historical disposition exceeded its bounded batch count")
    apply_in_own_transactions([("historical execution exclusion", _contract)], tables=tables)


def downgrade():
    # Keep both evidence and execution exclusion on rollback. Removing the
    # constraint would let an older runtime requeue history without admission.
    pass
