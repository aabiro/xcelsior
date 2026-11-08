"""Batched Slurm reconciler: the only code that polls Slurm for job state.

Dashboards, the API and agents read Xcelsior's copy of each job's state
(slurm_job_mappings); none of them talks to a cluster. However many of them
are watching, a cluster sees the same small, predictable load: per pass, one
``squeue`` and at most one bounded ``sacct``, plus a ``scancel`` for each job
Xcelsior wants stopped (hpc/IMPLEMENTATION_PLAN.md §5a).

A pass over one cluster:

1. ``squeue --me -t all`` once. A job the controller still holds has its
   state taken from there, whatever that state is.
2. ``sacct -D -X`` once, over the open mappings missing from squeue, bounded
   by the oldest one's age. A job gone from the controller whose latest
   accounting record is finished is *settled*: ``terminal_at`` is stamped.
   That is the only way a mapping is closed. A requeued NODE_FAIL or
   PREEMPTED job comes back to PENDING under the same id, so a finished
   state seen in squeue alone is never final.
3. A job in neither keeps its last observation and gets ``missing_since``.
   Accounting can lag (slurmdbd down or backlogged), so silence is never a
   verdict; after an hour it reads ``unknown``, still open.
4. A mapping whose ``desired_state`` is ``cancelled`` and that isn't
   finished gets a ``scancel``.

Rows are written only when the observation changes, each guarded by its
``version``, so two reconcilers (a blue and a green worker, should the
scheduled-task lease ever lapse) can't write over each other. Unchanged rows
get their ``last_observed_at`` refreshed in one statement.

Cluster health is kept in three parts on ``slurm_clusters``: transport (SSH),
controller (slurmctld) and accounting (slurmdbd). After a failed pass the
cluster is skipped for a jittered, doubling interval (the breaker), so an
outage costs one connection attempt per interval, not one per pass.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

import slurm_adapter as sa

log = logging.getLogger("xcelsior-slurm-reconciler")

#: A job in neither squeue nor accounting this long reads ``unknown``.
MISSING_UNKNOWN_AFTER_SEC = 3600
#: The breaker never waits longer than this between attempts.
BREAKER_MAX_SEC = 900
#: Accounting window margin beyond the oldest open mapping.
WINDOW_MARGIN_MIN = 60
#: Job names the adapter gives its jobs (``xcl-<xcelsior id>``).
ADAPTER_JOB_PREFIX = "xcl-"

_TRANSPORT_ERRORS = {"transport_error", "insecure_ssh_config", "unsafe_remote_argument"}


@dataclass
class PassResult:
    """What one pass over one cluster saw and did."""

    cluster_id: str
    ok: bool = True
    error_code: str | None = None
    error: str | None = None
    open_mappings: int = 0
    in_controller: int = 0
    changed: list = field(default_factory=list)  # (xcelsior_job_id, slurm_job_id, old, new)
    settled: list = field(default_factory=list)  # xcelsior_job_ids closed this pass
    missing: list = field(default_factory=list)  # slurm_job_ids in neither source
    orphans: list = field(default_factory=list)  # adapter-named jobs with no mapping
    cancel_requested: list = field(default_factory=list)
    accounting_error: str | None = None
    accounting_checked: bool = False  # sacct ran this pass (it only runs when needed)
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ── Registry ─────────────────────────────────────────────────────────


def register_cluster(profile_key: str, *, enabled: bool = True) -> dict:
    """Add (or refresh) a cluster from its adapter profile.

    The legitimacy flags come from the profile, never from the caller: an
    Alliance profile is research-only, and the table's CHECK refuses any row
    that is research-only and also marketplace or agent-reachable.
    """
    from db import pg_transaction

    profile = sa._get_profile(profile_key)
    research_only = bool(profile.get("research_only"))
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO slurm_clusters
                (cluster_id, profile_key, display_name, enabled, research_only,
                 marketplace, agent_access, poll_interval_seconds)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cluster_id) DO UPDATE
               SET profile_key = EXCLUDED.profile_key,
                   display_name = EXCLUDED.display_name,
                   enabled = EXCLUDED.enabled,
                   research_only = EXCLUDED.research_only,
                   marketplace = EXCLUDED.marketplace,
                   agent_access = EXCLUDED.agent_access,
                   poll_interval_seconds = EXCLUDED.poll_interval_seconds,
                   updated_at = clock_timestamp()
            """,
            (
                profile["cluster_id"],
                profile_key,
                profile["name"],
                enabled,
                research_only,
                bool(profile.get("marketplace")) and not research_only,
                bool(profile.get("agent_access")) and not research_only,
                int(profile.get("poll_interval_seconds") or 60),
            ),
        )
    return {
        "cluster_id": profile["cluster_id"],
        "profile_key": profile_key,
        "research_only": research_only,
    }


def list_clusters() -> list[dict]:
    from psycopg.rows import dict_row

    from db import pg_connection

    with pg_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        return cur.execute("SELECT * FROM slurm_clusters ORDER BY cluster_id").fetchall()


def _due_clusters() -> list[dict]:
    from psycopg.rows import dict_row

    from db import pg_connection

    with pg_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        return cur.execute(
            """
            SELECT cluster_id, profile_key, poll_interval_seconds, consecutive_failures
              FROM slurm_clusters
             WHERE enabled
               AND (breaker_open_until IS NULL OR breaker_open_until <= clock_timestamp())
               AND (last_poll_at IS NULL
                    OR last_poll_at + make_interval(secs => poll_interval_seconds)
                       <= clock_timestamp())
             ORDER BY cluster_id
            """
        ).fetchall()


def _record_health(cluster_id: str, result: PassResult, interval: int, failures: int) -> None:
    """Write the pass's outcome to the cluster row, opening the breaker on failure."""
    from db import pg_transaction

    if result.ok:
        breaker = None
        failures = 0
    else:
        failures += 1
        backoff = min(interval * 2 ** (failures - 1), BREAKER_MAX_SEC)
        breaker = backoff * random.uniform(0.8, 1.2)
    transport_ok = result.ok or result.error_code not in _TRANSPORT_ERRORS
    error_code = result.error_code or ("accounting_error" if result.accounting_error else None)
    with pg_transaction() as conn:
        conn.execute(
            """
            UPDATE slurm_clusters
               SET last_poll_at = clock_timestamp(),
                   transport_ok_at = CASE WHEN %s THEN clock_timestamp() ELSE transport_ok_at END,
                   controller_ok_at = CASE WHEN %s THEN clock_timestamp() ELSE controller_ok_at END,
                   accounting_ok_at = CASE WHEN %s THEN clock_timestamp() ELSE accounting_ok_at END,
                   last_error_code = %s,
                   last_error = %s,
                   consecutive_failures = %s,
                   breaker_open_until = CASE WHEN %s::float8 IS NULL THEN NULL
                        ELSE clock_timestamp() + make_interval(secs => %s::float8) END,
                   facts = jsonb_set(facts, '{orphans}', to_jsonb(%s::text[])),
                   updated_at = clock_timestamp()
             WHERE cluster_id = %s
            """,
            (
                transport_ok,
                result.ok,
                result.ok and result.accounting_checked and result.accounting_error is None,
                error_code,
                result.error or result.accounting_error,
                failures,
                breaker,
                breaker,
                result.orphans,
                cluster_id,
            ),
        )


# ── One pass ─────────────────────────────────────────────────────────


def _open_mappings(cluster_id: str) -> list[dict]:
    from psycopg.rows import dict_row

    from db import pg_connection

    with pg_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        return cur.execute(
            """
            SELECT xcelsior_job_id, slurm_job_id, observed_state, desired_state,
                   slurm_state, state_reason, exit_code, version,
                   EXTRACT(EPOCH FROM missing_since) AS missing_since,
                   EXTRACT(EPOCH FROM COALESCE(submitted_at, to_timestamp(created_at)))
                       AS submitted_epoch
              FROM slurm_job_mappings
             WHERE terminal_at IS NULL AND cluster_id = %s
            """,
            (cluster_id,),
        ).fetchall()


def _mapped_ids(cluster_id: str, slurm_ids: list[str]) -> set[str]:
    from db import pg_connection

    if not slurm_ids:
        return set()
    with pg_connection() as conn:
        rows = conn.execute(
            "SELECT slurm_job_id FROM slurm_job_mappings WHERE cluster_id = %s AND slurm_job_id = ANY(%s)",
            (cluster_id, slurm_ids),
        ).fetchall()
    return {r[0] for r in rows}


def _observation(row: dict, squeue_jobs: dict, records: dict, now: float) -> dict:
    """The observation this pass makes of one open mapping."""
    sid = row["slurm_job_id"]
    seen = squeue_jobs.get(sid)
    if seen is not None:
        return {
            "observed_state": sa.SLURM_OBSERVED_STATE.get(seen["state"], "unknown"),
            "slurm_state": seen["state"],
            "state_reason": seen["reason"] if seen["reason"] not in ("", "None") else None,
            "exit_code": row["exit_code"],
            "started_at": None,
            "ended_at": None,
            "missing_since": None,
            "settled": False,
        }
    runs = records.get(sid)
    if runs:
        latest = runs[-1]
        finished = latest["observed_state"] in sa.FINISHED_OBSERVED_STATES
        return {
            "observed_state": latest["observed_state"],
            "slurm_state": latest["state"],
            "state_reason": None,
            "exit_code": latest["exit_code"] or row["exit_code"],
            "started_at": latest["started_at"],
            "ended_at": latest["ended_at"],
            "missing_since": None,
            # Gone from the controller and finished in accounting: settled.
            # Not finished there means accounting is behind the controller,
            # or the job belongs to another account; either way, keep it open.
            "settled": finished,
        }
    missing_since = float(row["missing_since"]) if row["missing_since"] else now
    observed = row["observed_state"]
    if now - missing_since >= MISSING_UNKNOWN_AFTER_SEC:
        observed = "unknown"
    return {
        "observed_state": observed,
        "slurm_state": row["slurm_state"],
        "state_reason": row["state_reason"],
        "exit_code": row["exit_code"],
        "started_at": None,
        "ended_at": None,
        "missing_since": missing_since,
        "settled": False,
    }


def _changed(row: dict, obs: dict) -> bool:
    if obs["settled"]:
        return True
    before = (row["observed_state"], row["slurm_state"], row["state_reason"], row["exit_code"])
    after = (obs["observed_state"], obs["slurm_state"], obs["state_reason"], obs["exit_code"])
    return before != after or (row["missing_since"] is None) != (obs["missing_since"] is None)


def _write(rows_and_obs: list, unchanged_ids: list[str]) -> list[str]:
    """Apply changed observations (version-guarded) and refresh the rest.

    Returns the ids whose guarded write lost to a concurrent writer.
    """
    from db import pg_transaction

    lost = []
    with pg_transaction() as conn:
        for row, obs in rows_and_obs:
            cur = conn.execute(
                """
                UPDATE slurm_job_mappings
                   SET observed_state = %s,
                       slurm_state = %s,
                       state_reason = %s,
                       exit_code = %s,
                       started_at = COALESCE(to_timestamp(%s::float8), started_at),
                       ended_at = COALESCE(to_timestamp(%s::float8), ended_at),
                       missing_since = to_timestamp(%s::float8),
                       terminal_at = CASE WHEN %s THEN clock_timestamp() ELSE NULL END,
                       last_observed_at = clock_timestamp(),
                       last_changed_at = clock_timestamp(),
                       version = version + 1
                 WHERE xcelsior_job_id = %s AND version = %s AND terminal_at IS NULL
                """,
                (
                    obs["observed_state"],
                    obs["slurm_state"],
                    obs["state_reason"],
                    obs["exit_code"],
                    obs["started_at"],
                    obs["ended_at"],
                    obs["missing_since"],
                    obs["settled"],
                    row["xcelsior_job_id"],
                    row["version"],
                ),
            )
            if cur.rowcount == 0:
                lost.append(row["xcelsior_job_id"])
        if unchanged_ids:
            conn.execute(
                "UPDATE slurm_job_mappings SET last_observed_at = clock_timestamp() "
                "WHERE xcelsior_job_id = ANY(%s) AND terminal_at IS NULL",
                (unchanged_ids,),
            )
    return lost


def reconcile_cluster(
    cluster_id: str,
    profile_key: str,
    *,
    runner=None,
    cancel: bool = True,
    cancel_callback=None,
    update_callback=None,
) -> PassResult:
    """One reconcile pass over one cluster. Never raises for cluster trouble.

    ``cancel_callback(slurm_job_id)`` replaces the built-in scancel;
    ``update_callback(xcelsior_job_id, job_state)`` is told of each changed
    observation, in SLURM_STATE_MAP's job vocabulary.
    """
    started = time.monotonic()
    result = PassResult(cluster_id=cluster_id)
    rows = _open_mappings(cluster_id)
    result.open_mappings = len(rows)

    polled = sa.poll_cluster_jobs(profile_name=profile_key, runner=runner)
    if "error" in polled:
        # No observation at all: change nothing, report, let the breaker work.
        result.ok = False
        result.error_code = polled.get("error_code")
        result.error = polled["error"]
        result.duration_ms = int((time.monotonic() - started) * 1000)
        log.warning("slurm reconcile %s: %s (%s)", cluster_id, result.error, result.error_code)
        return result
    squeue_jobs = polled["jobs"]
    result.in_controller = sum(1 for r in rows if r["slurm_job_id"] in squeue_jobs)

    now = time.time()
    gone = [r for r in rows if r["slurm_job_id"] not in squeue_jobs]
    records: dict = {}
    if gone:
        oldest = min(float(r["submitted_epoch"] or now) for r in gone)
        since_min = int((now - oldest) / 60) + WINDOW_MARGIN_MIN
        for i in range(0, len(gone), sa.MAX_IDS_PER_SACCT):
            chunk = [r["slurm_job_id"] for r in gone[i : i + sa.MAX_IDS_PER_SACCT]]
            acct = sa.accounting_window(
                chunk, since_minutes=since_min, profile_name=profile_key, runner=runner
            )
            if "error" in acct:
                # squeue answered, so jobs still in the controller are applied;
                # the rest keep their last observation until accounting answers.
                result.accounting_error = acct["error"]
                records = {}
                gone_ids = {r["slurm_job_id"] for r in gone}
                rows = [r for r in rows if r["slurm_job_id"] not in gone_ids]
                break
            records.update(acct["records"])
            result.accounting_checked = True

    changes, unchanged, observed = [], [], {}
    for row in rows:
        obs = _observation(row, squeue_jobs, records, now)
        observed[row["xcelsior_job_id"]] = obs["observed_state"]
        if obs["missing_since"] is not None:
            result.missing.append(row["slurm_job_id"])
        if _changed(row, obs):
            changes.append((row, obs))
        else:
            unchanged.append(row["xcelsior_job_id"])
    lost = set(_write(changes, unchanged))

    for row, obs in changes:
        if row["xcelsior_job_id"] in lost:
            continue
        if row["observed_state"] != obs["observed_state"]:
            result.changed.append(
                (
                    row["xcelsior_job_id"],
                    row["slurm_job_id"],
                    row["observed_state"],
                    obs["observed_state"],
                )
            )
            if update_callback and obs["slurm_state"]:
                update_callback(row["xcelsior_job_id"], sa.xcelsior_state_for(obs["slurm_state"]))
        if obs["settled"]:
            result.settled.append(row["xcelsior_job_id"])
    if lost:
        log.info(
            "slurm reconcile %s: %d rows changed concurrently; next pass re-reads them",
            cluster_id,
            len(lost),
        )

    # desired → cluster: Xcelsior wants it stopped and it isn't finished.
    if cancel or cancel_callback:
        for row in rows:
            state = observed[row["xcelsior_job_id"]]
            if row["desired_state"] == "cancelled" and state not in sa.FINISHED_OBSERVED_STATES:
                try:
                    if cancel_callback:
                        cancel_callback(row["slurm_job_id"])
                    else:
                        sa.cancel_slurm_job(
                            row["slurm_job_id"], profile_name=profile_key, runner=runner
                        )
                    result.cancel_requested.append(row["slurm_job_id"])
                except Exception as e:  # one bad cancel must not end the pass
                    log.error(
                        "slurm cancel failed for %s on %s: %s", row["slurm_job_id"], cluster_id, e
                    )

    # Adapter-named jobs with no mapping: a crash between sbatch and the
    # mapping insert leaves one. Reported, never cancelled from here.
    named = [sid for sid, j in squeue_jobs.items() if j["name"].startswith(ADAPTER_JOB_PREFIX)]
    result.orphans = sorted(set(named) - _mapped_ids(cluster_id, named))
    if result.orphans:
        log.warning(
            "slurm reconcile %s: adapter jobs with no mapping: %s", cluster_id, result.orphans
        )

    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


def reconcile_due_clusters() -> list[dict]:
    """The scheduled task: one pass over every registered cluster that is due."""
    results = []
    for cluster in _due_clusters():
        try:
            result = reconcile_cluster(cluster["cluster_id"], cluster["profile_key"])
        except sa.SlurmAdapterError as e:  # a profile that no longer resolves
            result = PassResult(
                cluster_id=cluster["cluster_id"], ok=False, error_code=e.code, error=str(e)
            )
        _record_health(
            cluster["cluster_id"],
            result,
            int(cluster["poll_interval_seconds"]),
            int(cluster["consecutive_failures"]),
        )
        results.append(result.as_dict())
    return results


def reconcile_open_mappings(update_callback=None, cancel_callback=None) -> list:
    """A pass over every cluster that has open mappings, registered or not.

    Backs ``slurm_adapter.sync_slurm_statuses``; returns its
    ``(xcelsior_job_id, slurm_job_id, old_observed, new_observed)`` list.
    Cancels only through ``cancel_callback``, as that function always did.
    """
    from db import pg_connection

    with pg_connection() as conn:
        clusters = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT cluster_id FROM slurm_job_mappings WHERE terminal_at IS NULL"
            ).fetchall()
        ]
    changes = []
    for cluster_id in clusters:
        profile_key = sa._profile_key_for_cluster(cluster_id)
        if not profile_key:
            log.warning(
                "slurm reconcile: no profile for cluster %r; its mappings stay as they are",
                cluster_id,
            )
            continue
        result = reconcile_cluster(
            cluster_id,
            profile_key,
            cancel=False,
            cancel_callback=cancel_callback,
            update_callback=update_callback,
        )
        changes.extend(result.changed)
    return changes
