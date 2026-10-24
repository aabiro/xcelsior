"""Recover recorded execution specs and retain explicit missing-history evidence.

Only a matching job/command/attempt hash and fenced command binding authorize a
backfill. Current payload is never promoted to an approved or executed spec.
Independent batches are resumable; a failed batch leaves no partial evidence.
"""

import hashlib
import json

from alembic import op
from sqlalchemy import text

from migrations.lock_safe import apply_in_own_transactions

revision = "129"
down_revision = "128"
branch_labels = None
depends_on = None

_TERMINAL = {"completed", "failed", "cancelled", "stopped", "paused", "terminated"}


def _hash(value):
    # Frozen wire contract, shared with the historical worker hash algorithm.
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _create_evidence(conn):
    conn.execute(text("""CREATE TABLE IF NOT EXISTS job_spec_backfill_evidence (
        job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
        outcome TEXT NOT NULL CHECK (outcome IN ('recovered_execution','unavailable')),
        reason_code TEXT NOT NULL,
        recorded_row_sha256 TEXT NOT NULL CHECK (recorded_row_sha256 ~ '^[0-9a-f]{64}$'),
        recorded_payload_sha256 TEXT NOT NULL CHECK (recorded_payload_sha256 ~ '^[0-9a-f]{64}$'),
        recorded_spec_hash TEXT,
        source_command_id TEXT,
        source_attempt_id UUID,
        assessed_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
        CHECK ((outcome='recovered_execution' AND recorded_spec_hash IS NOT NULL
                AND source_command_id IS NOT NULL AND source_attempt_id IS NOT NULL)
            OR (outcome='unavailable' AND source_command_id IS NULL AND source_attempt_id IS NULL))
    )"""))


def _recorded_spec(conn, job):
    if job["status"] not in _TERMINAL:
        return None, "nonterminal_requires_qualification"
    if not job["spec_hash"]:
        return None, "no_recorded_execution_hash"
    candidates = (
        conn.execute(
            text("""SELECT c.command_id,c.attempt_id,c.args,c.fencing_token
                FROM agent_commands c JOIN job_attempts a ON a.attempt_id=c.attempt_id
                WHERE c.job_id=:job_id AND a.job_id=c.job_id AND c.command='start_attempt'
                  AND c.host_id=a.host_id AND c.fencing_token=a.fencing_token
                  AND c.spec_hash=:spec_hash AND a.spec_hash=c.spec_hash
                  AND c.args->>'spec_hash'=c.spec_hash
                  AND a.job_generation=:generation
                ORDER BY c.created_at,c.command_id LIMIT 1001"""),
            job,
        )
        .mappings()
        .all()
    )
    if len(candidates) > 1000:
        raise RuntimeError("recorded spec qualification exceeds the bounded command history")
    for candidate in candidates:
        args = candidate["args"]
        if not isinstance(args, dict):
            continue
        spec = args.get("spec")
        if not isinstance(spec, dict):
            continue
        if (
            args.get("job_id") != job["job_id"]
            or args.get("attempt_id") != str(candidate["attempt_id"])
            or type(args.get("fencing_token")) is not int
            or args["fencing_token"] != candidate["fencing_token"]
            or (spec.get("job_id") is not None and spec["job_id"] != job["job_id"])
            or _hash(spec) != job["spec_hash"]
        ):
            continue
        return (spec, candidate), "matching_fenced_command_and_attempt"
    return None, "matching_execution_evidence_unavailable"


def _qualify_batch(conn):
    conn.execute(text("SET LOCAL statement_timeout='30s'"))
    rows = conn.execute(text("""SELECT j.job_id,j.status,j.spec_hash,j.payload,j.generation,
                to_jsonb(j) AS original
            FROM jobs j WHERE j.spec IS NULL AND NOT EXISTS (
                SELECT 1 FROM job_spec_backfill_evidence e WHERE e.job_id=j.job_id)
            ORDER BY j.job_id LIMIT 500 FOR UPDATE OF j SKIP LOCKED""")).mappings().all()
    if not rows:
        remaining = conn.execute(
            text("""SELECT count(*) FROM jobs j
            WHERE j.spec IS NULL AND NOT EXISTS (
                SELECT 1 FROM job_spec_backfill_evidence e WHERE e.job_id=j.job_id)""")
        ).scalar_one()
        if remaining:
            raise RuntimeError("spec qualification has locked rows; rerun after contention clears")
        return 0
    for row in rows:
        recovered, reason = _recorded_spec(conn, row)
        source_command = source_attempt = None
        if recovered:
            spec, candidate = recovered
            changed = conn.execute(
                text("""UPDATE jobs SET spec=CAST(:spec AS jsonb)
                    WHERE job_id=:job_id AND spec IS NULL RETURNING to_jsonb(jobs)"""),
                {"spec": json.dumps(spec), "job_id": row["job_id"]},
            ).scalar_one()
            before = {k: v for k, v in row["original"].items() if k != "spec"}
            after = {k: v for k, v in changed.items() if k != "spec"}
            if before != after:
                raise RuntimeError("spec backfill changed unrelated job state; batch refused")
            source_command = str(candidate["command_id"])
            source_attempt = candidate["attempt_id"]
        conn.execute(
            text(
                """INSERT INTO job_spec_backfill_evidence
                (job_id,outcome,reason_code,recorded_row_sha256,recorded_payload_sha256,
                 recorded_spec_hash,source_command_id,source_attempt_id)
                VALUES (:job_id,:outcome,:reason,:row_hash,:payload_hash,:spec_hash,:command,:attempt)"""
            ),
            {
                "job_id": row["job_id"],
                "outcome": "recovered_execution" if recovered else "unavailable",
                "reason": reason,
                "row_hash": _hash(row["original"]).removeprefix("sha256:"),
                "payload_hash": _hash(row["payload"]).removeprefix("sha256:"),
                "spec_hash": row["spec_hash"],
                "command": source_command,
                "attempt": source_attempt,
            },
        )
    return len(rows)


def upgrade():
    touched = ["jobs", "agent_commands", "job_attempts", "job_spec_backfill_evidence"]
    apply_in_own_transactions([("spec backfill evidence", _create_evidence)], tables=touched)
    for batch in range(10000):
        progress = []

        def qualify(conn):
            progress[:] = [_qualify_batch(conn)]

        apply_in_own_transactions(
            [(f"recorded execution specs batch {batch}", qualify)], tables=touched
        )
        if progress == [0]:
            return
    raise RuntimeError("spec qualification did not converge")


def downgrade():
    # Recovered specs and assessment evidence remain valid on the older runtime.
    # Retain them rather than erasing provenance or returning specs to NULL.
    op.execute("SET LOCAL lock_timeout='5s'")
