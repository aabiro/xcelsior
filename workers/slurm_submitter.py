"""Turn pending Slurm submissions into exactly one Slurm job each.

The scheduler writes a ``slurm_submissions`` row before any ``sbatch``
(blueprint 2.2). This task, ``slurm_submit``, is the only thing that calls
``sbatch``. The job's name, ``xcl-<submission_name>``, is the idempotency tag:
``--comment`` reaches accounting only where the site stores job comments, and
the name is stored everywhere.

A pass over one cluster is one ``squeue`` and, for names the controller is not
holding, one bounded ``sacct --name``. A name already there is registered and
never submitted again. A name that is absent is submitted only after this task
has claimed the row (a conditional update, so two passes cannot both submit).
A timed-out or dropped ``sbatch`` leaves the row ``unknown``. That timeout is
not proof the job is absent. Two later lookups, at least one poll interval
apart, have to miss it before another ``sbatch``, and there are at most three
``sbatch`` calls. A request Slurm refuses is ``rejected`` and is never retried.

Nothing here charges a customer. No code path settles a meter.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import slurm_adapter as sa
from control_plane.state import job_phase
from db import pg_connection, pg_transaction

log = logging.getLogger("xcelsior-slurm-submitter")

MAX_SBATCH_CALLS = 3
RESOLVE_LOOKS = 2

# A second sbatch cannot make these succeed. host_key_mismatch is a bad pin.
_NEVER_RETRY = frozenset(
    {
        "unsatisfiable_request",
        "invalid_partition",
        "invalid_gres",
        "invalid_qos",
        "invalid_account",
        "invalid_nodelist",
        "walltime_exceeds_limit",
        "invalid_job_spec",
        "invalid_request",
        "invalid_job_id",
        "cluster_retired",
        "missing_account",
        "sbatch_rejected",
        "host_key_mismatch",
    }
)

_ACTIVE_ATTEMPTS = (
    "reserved",
    "command_pending",
    "lease_offered",
    "lease_claimed",
    "starting",
    "running",
)

_OPEN_SUBMISSION = ("pending", "submitting", "unknown")

# Fields copied onto the sbatch request only when the spec actually carries them.
_SPEC_KEYS = (
    "name",
    "command",
    "image",
    "walltime",
    "mem_gb",
    "nodelist",
    "partition",
    "cpus",
    "vram_needed_gb",
    "priority",
    "tier",
    "gpu_model",
)


def _never_retry(code: str | None) -> bool:
    if not code:
        return False
    return code in _NEVER_RETRY or code.startswith("invalid_")


def _wants_stop(desired_state: str | None, lifecycle_intent: str | None) -> bool:
    if desired_state == "stopped":
        return True
    return lifecycle_intent in ("stop", "cancel", "terminate")


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def _whole_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _job_for_sbatch(row: dict) -> dict:
    """The adapter's job dict. Missing fields stay missing; nothing is invented."""
    spec = _as_dict(row.get("spec"))
    job: dict[str, Any] = {"job_id": row["xcelsior_job_id"]}
    gpus = _whole_int(spec.get("num_gpus"))
    if gpus is not None:
        job["num_gpus"] = gpus
    for key in _SPEC_KEYS:
        value = spec.get(key)
        if value is None or value == "":
            continue
        if key in ("mem_gb", "cpus", "priority"):
            number = _whole_int(value)
            if number is None:
                continue
            value = number
        if key == "command" and not isinstance(value, str):
            continue
        job[key] = value
    return job


def _due_rows() -> list[dict]:
    from psycopg.rows import dict_row

    with pg_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        return list(cur.execute("""
                SELECT s.submission_name, s.cluster_id, s.xcelsior_job_id, s.attempt_id,
                       s.tenant_id, s.state, s.sbatch_calls, s.checks,
                       EXTRACT(EPOCH FROM s.last_check_at) AS last_check_epoch,
                       EXTRACT(EPOCH FROM s.submit_started_at) AS submit_started_epoch,
                       EXTRACT(EPOCH FROM s.created_at) AS created_epoch,
                       j.desired_state, j.lifecycle_intent, j.spec,
                       c.profile_key, c.poll_interval_seconds,
                       c.ssh_host, c.ssh_port, c.remote_user,
                       c.known_hosts_line, c.credential_ref
                  FROM slurm_submissions s
                  JOIN jobs j ON j.job_id = s.xcelsior_job_id
                  JOIN slurm_clusters c ON c.cluster_id = s.cluster_id
                 WHERE s.state IN ('pending', 'submitting', 'unknown')
                   AND c.enabled
                 ORDER BY c.cluster_id, s.created_at
                """).fetchall())


def _finish(
    row: dict,
    *,
    state: str,
    error_code: str,
    error: str | None,
    now: float,
    attempt_status: str,
    job_status: str,
) -> bool:
    """Close a submission and, with it, the attempt it was placed for.

    The attempt stays put unless it is still in the active set, so a later
    pass cannot overwrite a terminal status. The job leaves the schedulable
    pair (pending, running) so the scheduler does not open another attempt.
    """
    phase = job_phase(job_status)
    message = (error or error_code)[:500]
    with pg_transaction() as conn:
        closed = conn.execute(
            """
            UPDATE slurm_submissions
               SET state = %s, error_code = %s, error = %s, resolved_at = to_timestamp(%s)
             WHERE submission_name = %s AND state = ANY(%s)
            """,
            (
                state,
                error_code,
                (error or "")[:500],
                now,
                row["submission_name"],
                list(_OPEN_SUBMISSION),
            ),
        )
        if closed.rowcount != 1:
            return False
        attempt = conn.execute(
            """
            UPDATE job_attempts
               SET status = %s, failure_code = %s, ended_at = clock_timestamp()
             WHERE attempt_id = %s AND status = ANY(%s)
            """,
            (attempt_status, error_code, row["attempt_id"], list(_ACTIVE_ATTEMPTS)),
        )
        if attempt.rowcount == 1:
            conn.execute(
                """
                UPDATE jobs
                   SET status = %s, phase = %s, desired_state = 'stopped',
                       error_message = %s, version = version + 1,
                       updated_at = clock_timestamp()
                 WHERE job_id = %s
                """,
                (job_status, phase, message, row["xcelsior_job_id"]),
            )
    return True


def _reject(row: dict, code: str, error: str | None, now: float) -> dict:
    _finish(
        row,
        state="rejected",
        error_code=code,
        error=error,
        now=now,
        attempt_status="failed",
        job_status="failed",
    )
    log.info("slurm submit %s rejected: %s", row["submission_name"], code)
    return {"submission_name": row["submission_name"], "action": "rejected", "error_code": code}


def _cancel(row: dict, code: str, now: float) -> dict:
    _finish(
        row,
        state="rejected",
        error_code=code,
        error=code,
        now=now,
        attempt_status="cancelled",
        job_status="cancelled",
    )
    log.info("slurm submit %s closed without a running job (%s)", row["submission_name"], code)
    return {"submission_name": row["submission_name"], "action": "cancelled", "error_code": code}


def _lose(row: dict, code: str, now: float) -> dict:
    _finish(
        row,
        state="lost",
        error_code=code,
        error=code,
        now=now,
        attempt_status="lost",
        job_status="failed",
    )
    log.warning("slurm submit %s lost: %s", row["submission_name"], code)
    return {"submission_name": row["submission_name"], "action": "lost", "error_code": code}


def _mark_submitted(row: dict, slurm_job_id: str, script_sha256: str | None, now: float) -> bool:
    with pg_transaction() as conn:
        cur = conn.execute(
            """
            UPDATE slurm_submissions
               SET state = 'submitted',
                   slurm_job_id = %s,
                   script_sha256 = COALESCE(%s, script_sha256),
                   submitted_at = COALESCE(submitted_at, to_timestamp(%s)),
                   resolved_at = to_timestamp(%s),
                   error_code = NULL,
                   error = NULL
             WHERE submission_name = %s AND state = ANY(%s)
            """,
            (
                str(slurm_job_id),
                script_sha256,
                now,
                now,
                row["submission_name"],
                list(_OPEN_SUBMISSION),
            ),
        )
    return cur.rowcount == 1


def _claim_sbatch(row: dict, now: float) -> int | None:
    """Take the right to call sbatch once. None when another pass already did."""
    with pg_transaction() as conn:
        claimed = conn.execute(
            """
            UPDATE slurm_submissions
               SET state = 'submitting',
                   sbatch_calls = sbatch_calls + 1,
                   checks = 0,
                   submit_started_at = to_timestamp(%s),
                   last_check_at = NULL,
                   error_code = NULL,
                   error = NULL
             WHERE submission_name = %s
               AND sbatch_calls = %s
               AND state IN ('pending', 'unknown')
            RETURNING sbatch_calls
            """,
            (now, row["submission_name"], int(row["sbatch_calls"])),
        ).fetchone()
    if claimed is None:
        return None
    return int(claimed[0])


def _note_unknown(row: dict, *, code: str, error: str | None, now: float) -> None:
    with pg_transaction() as conn:
        conn.execute(
            """
            UPDATE slurm_submissions
               SET state = 'unknown', checks = 0, last_check_at = to_timestamp(%s),
                   error_code = %s, error = %s
             WHERE submission_name = %s AND state = 'submitting'
            """,
            (now, code, (error or "")[:500], row["submission_name"]),
        )


def _record_look(row: dict, new_checks: int, now: float, error_code: str | None) -> bool:
    with pg_transaction() as conn:
        cur = conn.execute(
            """
            UPDATE slurm_submissions
               SET state = 'unknown', checks = %s, last_check_at = to_timestamp(%s),
                   error_code = COALESCE(%s, error_code)
             WHERE submission_name = %s
               -- pending is included so two ids under a name that was never
               -- submitted still count as looks. Leaving it out keeps that
               -- row pending forever.
               AND state IN ('pending', 'submitting', 'unknown')
               AND checks = %s
            """,
            (new_checks, now, error_code, row["submission_name"], int(row["checks"])),
        )
    return cur.rowcount == 1


def _stop_submitted(row: dict, slurm_job_id: str, runner, profile_key: str) -> None:
    """The job landed and the customer no longer wants it. Ask Slurm to stop it."""
    sa.set_slurm_desired_state(row["xcelsior_job_id"], "cancelled")
    try:
        sa.cancel_slurm_job(slurm_job_id, profile_name=profile_key, runner=runner)
    except Exception:
        log.exception("scancel failed for %s; the reconciler will retry", slurm_job_id)


def _still_wants_stop(row: dict) -> bool:
    with pg_connection() as conn:
        current = conn.execute(
            "SELECT desired_state, lifecycle_intent FROM jobs WHERE job_id = %s",
            (row["xcelsior_job_id"],),
        ).fetchone()
    if current is None:
        return False
    return _wants_stop(current[0], current[1])


def _adopt(
    row: dict,
    slurm_job_id: str,
    *,
    runner,
    profile_key: str,
    now: float,
    just_submitted: bool,
    script_sha256: str | None,
) -> dict:
    try:
        sa.register_slurm_job(
            row["xcelsior_job_id"],
            slurm_job_id,
            cluster_id=row["cluster_id"],
            tenant_id=row.get("tenant_id"),
            attempt_id=str(row["attempt_id"]),
            submission_name=row["submission_name"],
            just_submitted=just_submitted,
        )
    except sa.SlurmAdapterError as exc:
        if exc.code == "mapping_conflict":
            return _reject(row, exc.code, str(exc), now)
        log.warning("slurm submit %s register failed: %s", row["submission_name"], exc.code)
        return {
            "submission_name": row["submission_name"],
            "action": "register_failed",
            "error_code": exc.code,
        }
    except Exception as exc:
        from psycopg.errors import UniqueViolation

        if isinstance(exc, UniqueViolation):
            return _reject(row, "mapping_conflict", "submission is already mapped", now)
        raise
    if not _mark_submitted(row, slurm_job_id, script_sha256, now):
        return {"submission_name": row["submission_name"], "action": "unchanged"}
    if _wants_stop(row.get("desired_state"), row.get("lifecycle_intent")) or _still_wants_stop(row):
        _stop_submitted(row, slurm_job_id, runner, profile_key)
    action = "submitted" if just_submitted else "adopted"
    log.info("slurm submit %s %s as %s", row["submission_name"], action, slurm_job_id)
    return {
        "submission_name": row["submission_name"],
        "action": action,
        "slurm_job_id": str(slurm_job_id),
    }


def _sbatch(row: dict, *, runner, profile_key: str, now: float) -> dict:
    calls = _claim_sbatch(row, now)
    if calls is None:
        return {"submission_name": row["submission_name"], "action": "claimed_elsewhere"}
    # The in-memory row still says the old count. A retry in this same pass
    # must not claim again; the caller does not loop.
    result = sa.submit_to_slurm(
        _job_for_sbatch(row),
        profile_name=profile_key,
        runner=runner,
        submission_name=row["submission_name"],
    )
    if result.get("slurm_job_id"):
        # Register against the registry's cluster, not the profile's. The
        # generic profile reports its own name as cluster_id.
        return _adopt(
            row,
            result["slurm_job_id"],
            runner=runner,
            profile_key=profile_key,
            now=now,
            just_submitted=True,
            script_sha256=result.get("script_sha256"),
        )
    code = str(result.get("error_code") or "adapter_exception")
    error = result.get("error")
    if _never_retry(code):
        return _reject(row, code, error, now)
    _note_unknown(row, code=code, error=error, now=now)
    log.info(
        "slurm submit %s unknown after sbatch (%s); will resolve by name",
        row["submission_name"],
        code,
    )
    return {"submission_name": row["submission_name"], "action": "unknown", "error_code": code}


def _look_anchor(row: dict) -> float | None:
    anchor = row.get("last_check_epoch")
    if anchor is None:
        anchor = row.get("submit_started_epoch")
    if anchor is None:
        return None
    return float(anchor)


def _resolve_absence(
    row: dict, *, runner, profile_key: str, now: float, interval: int, error_code: str | None
) -> dict:
    """One lookup that did not find the name. Two of these, a poll apart, conclude it."""
    anchor = _look_anchor(row)
    if anchor is not None and (now - anchor) < interval:
        return {"submission_name": row["submission_name"], "action": "waiting"}
    new_checks = int(row["checks"]) + 1
    if not _record_look(row, new_checks, now, error_code):
        return {"submission_name": row["submission_name"], "action": "claimed_elsewhere"}
    if new_checks < RESOLVE_LOOKS:
        return {
            "submission_name": row["submission_name"],
            "action": "waiting",
            "checks": new_checks,
        }
    # Two ids under one name is never a job we can safely adopt or cancel.
    if error_code == "ambiguous_name":
        return _lose(row, "ambiguous_name", now)
    if _wants_stop(row.get("desired_state"), row.get("lifecycle_intent")):
        return _cancel(row, "cancelled", now)
    if int(row["sbatch_calls"]) < MAX_SBATCH_CALLS:
        # The claim matches state unknown, which the look just wrote, and the
        # sbatch count this row was read with.
        return _sbatch(row, runner=runner, profile_key=profile_key, now=now)
    return _lose(row, "lost", now)


def _advance(
    row: dict, found_ids: list[str], *, runner, profile_key: str, now: float, interval: int
) -> dict:
    name = row["submission_name"]
    if len(found_ids) > 1:
        return _resolve_absence(
            row,
            runner=runner,
            profile_key=profile_key,
            now=now,
            interval=interval,
            error_code="ambiguous_name",
        )
    if len(found_ids) == 1:
        return _adopt(
            row,
            found_ids[0],
            runner=runner,
            profile_key=profile_key,
            now=now,
            just_submitted=False,
            script_sha256=None,
        )
    # Not in the controller and not in accounting.
    if (
        row["state"] == "pending"
        and int(row["sbatch_calls"]) == 0
        and _wants_stop(row.get("desired_state"), row.get("lifecycle_intent"))
    ):
        return _cancel(row, "cancelled_before_submit", now)
    if row["state"] == "pending" and int(row["sbatch_calls"]) == 0:
        return _sbatch(row, runner=runner, profile_key=profile_key, now=now)
    return _resolve_absence(
        row, runner=runner, profile_key=profile_key, now=now, interval=interval, error_code=None
    )


def _names_from_records(records: dict) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for sid, runs in records.items():
        for run in runs:
            name = run.get("name") or ""
            bucket = found.setdefault(name, [])
            if sid not in bucket:
                bucket.append(sid)
    return found


def submit_cluster(rows: list[dict], *, runner, now: float) -> list[dict]:
    """One pass for one cluster's due submissions. ``rows`` share a cluster."""
    profile_key = rows[0]["profile_key"]
    interval = max(30, int(rows[0]["poll_interval_seconds"] or 60))
    polled = sa.poll_cluster_jobs(profile_name=profile_key, runner=runner)
    if "error" in polled:
        log.warning(
            "slurm submit %s: squeue failed (%s)", rows[0]["cluster_id"], polled.get("error_code")
        )
        return [
            {
                "submission_name": row["submission_name"],
                "action": "poll_failed",
                "error_code": polled.get("error_code"),
            }
            for row in rows
        ]
    by_name: dict[str, list[str]] = {}
    for sid, info in polled["jobs"].items():
        by_name.setdefault(info.get("name") or "", []).append(sid)

    missing = [row for row in rows if f"xcl-{row['submission_name']}" not in by_name]
    accounting_ok = True
    if missing:
        oldest = min(float(row["created_epoch"] or now) for row in missing)
        since = max(1, int((now - oldest) / 60) + 2)
        names = [row["submission_name"] for row in missing]
        for start in range(0, len(names), sa.MAX_IDS_PER_SACCT):
            chunk = names[start : start + sa.MAX_IDS_PER_SACCT]
            acct = sa.accounting_by_name(
                chunk, since_minutes=since, profile_name=profile_key, runner=runner
            )
            if "error" in acct:
                accounting_ok = False
                log.warning(
                    "slurm submit %s: sacct failed (%s); names not in squeue are left untouched",
                    rows[0]["cluster_id"],
                    acct.get("error_code"),
                )
                break
            for name, ids in _names_from_records(acct["records"]).items():
                bucket = by_name.setdefault(name, [])
                for sid in ids:
                    if sid not in bucket:
                        bucket.append(sid)

    results = []
    for row in rows:
        slurm_name = f"xcl-{row['submission_name']}"
        if slurm_name not in by_name and not accounting_ok:
            results.append(
                {
                    "submission_name": row["submission_name"],
                    "action": "accounting_unavailable",
                }
            )
            continue
        ids = list(dict.fromkeys(by_name.get(slurm_name, ())))
        results.append(
            _advance(row, ids, runner=runner, profile_key=profile_key, now=now, interval=interval)
        )
    return results


def submit_due(*, runner_for=None, now: float | None = None) -> list[dict]:
    """Submit every due row. ``now`` is the epoch tests use instead of sleeping.

    ``runner_for(cluster)`` is a context manager yielding the transport. It
    defaults to the reconciler's registry runner. Disabled clusters are left
    pending: a paused cluster is not a lost job.
    """
    if runner_for is None:
        from slurm_reconciler import _runner_for as runner_for
    when = time.time() if now is None else float(now)
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for row in _due_rows():
        cid = row["cluster_id"]
        if cid not in groups:
            order.append(cid)
            groups[cid] = []
        groups[cid].append(row)
    results: list[dict] = []
    for cid in order:
        rows = groups[cid]
        try:
            with runner_for(rows[0]) as runner:
                results.extend(submit_cluster(rows, runner=runner, now=when))
        except Exception:
            log.exception("slurm submit %s failed; the next pass retries", cid)
            results.append({"cluster_id": cid, "action": "pass_failed"})
    return results
