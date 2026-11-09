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
5. Output tails readers asked for (slurm_job_outputs), at most
   ``OUTPUT_FETCHES_PER_PASS`` of them, and one final copy of each after its
   job settles. Readers never fetch output themselves, so however many are
   watching, a job's output is read at most once per pass.

A Slurm id is unique on one cluster only for a while: a controller that loses
its state numbers from FirstJobId again, and ids wrap at MaxJobId. So a job
under a mapping's id is that mapping's job only if its name agrees. The
adapter names every job ``xcl-<xcelsior job id>``; a job carrying another
Xcelsior job's name is someone else's, in squeue and in accounting alike. A
name without the prefix says nothing either way (the job wasn't submitted by
the adapter), and the id is all there is to go on.

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

import contextlib
import logging
import random
import time
from dataclasses import dataclass, field

import slurm_adapter as sa
import slurm_transport as st

log = logging.getLogger("xcelsior-slurm-reconciler")

#: A job in neither squeue nor accounting this long reads ``unknown``.
MISSING_UNKNOWN_AFTER_SEC = 3600
#: The breaker never waits longer than this between attempts.
BREAKER_MAX_SEC = 900
#: Accounting window margin beyond the oldest open mapping.
WINDOW_MARGIN_MIN = 60
#: Job names the adapter gives its jobs (``xcl-<xcelsior id>``).
ADAPTER_JOB_PREFIX = "xcl-"
#: Output tails fetched per pass over one cluster; the rest wait their turn.
OUTPUT_FETCHES_PER_PASS = 10
#: How old a cluster's config facts may get before a pass captures them
#: again. Settings like MinJobAge and MaxJobId change with an upgrade or a
#: reconfigure, and the reconciler's rules lean on them.
FACTS_REFRESH_SEC = 7 * 86400

_TRANSPORT_ERRORS = {
    "transport_error",
    "host_key_mismatch",
    "insecure_ssh_config",
    "unsafe_remote_argument",
}


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
    outputs_fetched: list = field(default_factory=list)  # (xcelsior_job_id, stream)
    outputs_unchanged: list = field(default_factory=list)  # size check: nothing new to read
    output_errors: list = field(default_factory=list)  # (xcelsior_job_id, stream, error_code)
    accounting_error: str | None = None
    accounting_error_code: str | None = None
    accounting_checked: bool = False  # sacct ran this pass (it only runs when needed)
    facts_refreshed: bool = False
    facts_error_code: str | None = None
    duration_ms: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ── Registry ─────────────────────────────────────────────────────────


def register_cluster(
    profile_key: str,
    *,
    enabled: bool = True,
    cluster_id: str | None = None,
    ssh_host: str | None = None,
    ssh_port: int | None = None,
    remote_user: str | None = None,
    known_hosts_line: str | None = None,
    credential_ref: str | None = None,
) -> dict:
    """Add (or refresh) a cluster from its adapter profile.

    The legitimacy flags come from the profile, never from the caller: an
    Alliance profile is research-only, and the table's CHECK refuses any row
    that is research-only and also marketplace or agent-reachable.

    With ``ssh_host`` the cluster is reached through the registry
    (slurm_transport): pass the host key to pin, as a known_hosts or
    ssh-keyscan line verified out of band, and where the key is kept
    (``env:NAME`` or ``file:/path``). Without it the profile's own transport
    is used, as before. Run ``check_cluster_connection`` afterwards to prove
    the connection and capture the cluster's config facts.
    """
    from db import pg_transaction

    profile = sa._get_profile(profile_key)
    research_only = bool(profile.get("research_only"))
    if ssh_host:
        st.pinned_key(known_hosts_line or "")  # refuse an unusable pin now, not at first poll
    with pg_transaction() as conn:
        conn.execute(
            """
            INSERT INTO slurm_clusters
                (cluster_id, profile_key, display_name, enabled, research_only,
                 marketplace, agent_access, poll_interval_seconds,
                 ssh_host, ssh_port, remote_user, known_hosts_line, credential_ref)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (cluster_id) DO UPDATE
               SET profile_key = EXCLUDED.profile_key,
                   display_name = EXCLUDED.display_name,
                   enabled = EXCLUDED.enabled,
                   research_only = EXCLUDED.research_only,
                   marketplace = EXCLUDED.marketplace,
                   agent_access = EXCLUDED.agent_access,
                   poll_interval_seconds = EXCLUDED.poll_interval_seconds,
                   ssh_host = EXCLUDED.ssh_host,
                   ssh_port = EXCLUDED.ssh_port,
                   remote_user = EXCLUDED.remote_user,
                   known_hosts_line = EXCLUDED.known_hosts_line,
                   credential_ref = EXCLUDED.credential_ref,
                   updated_at = clock_timestamp()
            """,
            (
                cluster_id or profile["cluster_id"],
                profile_key,
                profile["name"],
                enabled,
                research_only,
                bool(profile.get("marketplace")) and not research_only,
                bool(profile.get("agent_access")) and not research_only,
                int(profile.get("poll_interval_seconds") or 60),
                ssh_host or None,
                ssh_port if ssh_host else None,
                remote_user if ssh_host else None,
                known_hosts_line if ssh_host else None,
                credential_ref if ssh_host else None,
            ),
        )
    return {
        "cluster_id": cluster_id or profile["cluster_id"],
        "profile_key": profile_key,
        "research_only": research_only,
        "transport": "registry" if ssh_host else "profile",
    }


def _cluster_row(cluster_id: str) -> dict | None:
    from psycopg.rows import dict_row

    from db import pg_connection

    with pg_connection() as conn:
        cur = conn.cursor(row_factory=dict_row)
        return cur.execute(
            "SELECT * FROM slurm_clusters WHERE cluster_id = %s", (cluster_id,)
        ).fetchone()


@contextlib.contextmanager
def _runner_for(cluster: dict | None):
    """The registry's transport when the row has one, else None (the profile's)."""
    if cluster and cluster.get("ssh_host"):
        with st.cluster_runner(cluster) as runner:
            yield runner
    else:
        yield None


def _store_facts(cluster_id: str, facts: dict, *, controller_up: bool = True) -> None:
    from psycopg.types.json import Jsonb

    from db import pg_transaction

    with pg_transaction() as conn:
        conn.execute(
            """
            UPDATE slurm_clusters
               SET facts = facts || jsonb_build_object(
                       'config', %s::jsonb, 'config_captured_at', clock_timestamp()),
                   transport_ok_at = clock_timestamp(),
                   controller_ok_at = CASE WHEN %s THEN clock_timestamp() ELSE controller_ok_at END,
                   updated_at = clock_timestamp()
             WHERE cluster_id = %s
            """,
            (Jsonb(facts), controller_up, cluster_id),
        )


def _facts_due(facts) -> bool:
    """Whether the cluster's config facts are missing or older than FACTS_REFRESH_SEC."""
    import datetime as _dt

    captured = (facts or {}).get("config_captured_at") if isinstance(facts, dict) else None
    try:
        age = _dt.datetime.now(_dt.timezone.utc) - _dt.datetime.fromisoformat(str(captured))
    except (ValueError, TypeError):
        return True  # never captured, or unreadable: capture now
    return age.total_seconds() > FACTS_REFRESH_SEC


def _refresh_facts(cluster: dict, runner, result: PassResult) -> None:
    """Capture the config facts again, on a pass that already reached the cluster."""
    config = sa.cluster_config_facts(profile_name=cluster["profile_key"], runner=runner)
    if "error" in config:
        result.facts_error_code = config.get("error_code") or "unknown"
        log.warning(
            "slurm reconcile %s: config facts not refreshed: %s",
            cluster["cluster_id"],
            config["error"],
        )
        return
    _store_facts(cluster["cluster_id"], config["facts"])
    result.facts_refreshed = True


def check_cluster_connection(cluster_id: str) -> dict:
    """Connect to a registered cluster, ask its controller, keep its config.

    Uses the same transport the reconciler will (the registry's, or the
    profile's), runs ``scontrol ping`` and ``scontrol show config``, and
    stores the CONFIG_FACTS settings in ``facts['config']`` with the time.
    Returns what it found; a failure is returned, not raised.
    """
    cluster = _cluster_row(cluster_id)
    if cluster is None:
        return {"ok": False, "error_code": "not_found", "error": f"no cluster {cluster_id!r}"}
    try:
        with _runner_for(cluster) as runner:
            ping = sa.controller_health(profile_name=cluster["profile_key"], runner=runner)
            config = (
                sa.cluster_config_facts(profile_name=cluster["profile_key"], runner=runner)
                if "error" not in ping
                else ping
            )
    except sa.SlurmAdapterError as e:
        return {"ok": False, "error_code": e.code, "error": str(e)}
    if "error" in config:
        return {"ok": False, "error_code": config.get("error_code"), "error": config["error"]}
    _store_facts(cluster_id, config["facts"], controller_up=ping["controller"] == "up")
    return {
        "ok": ping["controller"] == "up" and not config["missing"],
        "controller": ping,
        "facts": config["facts"],
        "missing": config["missing"],
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
            SELECT cluster_id, profile_key, poll_interval_seconds, consecutive_failures,
                   ssh_host, ssh_port, remote_user, known_hosts_line, credential_ref, facts
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
    # Durable counts by stage and error code, for the metrics (a counter in
    # this process would vanish with it, and the API exports them).
    counts: dict[str, int] = {}
    if not result.ok:
        counts[f"poll:{result.error_code or 'unknown'}"] = 1
    if result.accounting_error:
        key = f"accounting:{result.accounting_error_code or 'unknown'}"
        counts[key] = counts.get(key, 0) + 1
    for _xid, _stream, code in result.output_errors:
        key = f"output:{code or 'unknown'}"
        counts[key] = counts.get(key, 0) + 1
    if result.facts_error_code:
        key = f"facts:{result.facts_error_code}"
        counts[key] = counts.get(key, 0) + 1
    with pg_transaction() as conn:
        for key, n in counts.items():
            conn.execute(
                "UPDATE slurm_clusters SET failure_counts = jsonb_set(failure_counts, ARRAY[%s], "
                "to_jsonb(COALESCE((failure_counts->>%s)::bigint, 0) + %s)) WHERE cluster_id = %s",
                (key, key, n, cluster_id),
            )
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
            SELECT mapping_id, submission_name, xcelsior_job_id, slurm_job_id,
                   observed_state, desired_state,
                   slurm_state, state_reason, exit_code, derived_exit_code,
                   time_limit_seconds, version,
                   EXTRACT(EPOCH FROM missing_since) AS missing_since,
                   EXTRACT(EPOCH FROM COALESCE(submitted_at, to_timestamp(created_at)))
                       AS submitted_epoch
              FROM slurm_job_mappings
             WHERE terminal_at IS NULL AND cluster_id = %s
            """,
            (cluster_id,),
        ).fetchall()


def _open_names(cluster_id: str, slurm_ids: list[str]) -> set[tuple[str, str]]:
    """(slurm id, adapter job name) for each open mapping among these ids."""
    from db import pg_connection

    if not slurm_ids:
        return set()
    with pg_connection() as conn:
        rows = conn.execute(
            "SELECT slurm_job_id, submission_name FROM slurm_job_mappings "
            "WHERE cluster_id = %s AND slurm_job_id = ANY(%s) AND terminal_at IS NULL",
            (cluster_id, slurm_ids),
        ).fetchall()
    return {(r[0], ADAPTER_JOB_PREFIX + r[1]) for r in rows}


def _owns(row: dict, job_name: str | None) -> bool:
    """Whether a job seen under this mapping's Slurm id is this mapping's job."""
    name = job_name or ""
    return not name.startswith(ADAPTER_JOB_PREFIX) or name == (
        ADAPTER_JOB_PREFIX + row["submission_name"]
    )


def _in_controller(row: dict, squeue_jobs: dict) -> dict | None:
    seen = squeue_jobs.get(row["slurm_job_id"])
    return seen if seen is not None and _owns(row, seen["name"]) else None


def _observation(row: dict, squeue_jobs: dict, records: dict, now: float) -> dict:
    """The observation this pass makes of one open mapping."""
    sid = row["slurm_job_id"]
    seen = _in_controller(row, squeue_jobs)
    if seen is not None:
        observed = sa.SLURM_OBSERVED_STATE.get(seen["state"], "unknown")
        started = None
        if observed == "running" and row["observed_state"] != "running":
            # Entering RUNNING (again, after a requeue): its start is now
            # minus what squeue says it has run, for scheduler progress.
            elapsed = sa.duration_seconds(seen.get("elapsed"))
            started = now - elapsed if elapsed is not None else now
        return {
            "observed_state": observed,
            "slurm_state": seen["state"],
            "state_reason": seen["reason"] if seen["reason"] not in ("", "None") else None,
            "exit_code": row["exit_code"],
            "derived_exit_code": row.get("derived_exit_code"),
            "time_limit_seconds": sa.duration_seconds(seen.get("time_limit"))
            or row.get("time_limit_seconds"),
            "started_at": started,
            "ended_at": None,
            "missing_since": None,
            "settled": False,
        }
    runs = [run for run in records.get(sid, ()) if _owns(row, run["name"])]
    if runs:
        latest = runs[-1]
        finished = latest["observed_state"] in sa.FINISHED_OBSERVED_STATES
        return {
            "observed_state": latest["observed_state"],
            "slurm_state": latest["state"],
            "state_reason": None,
            "exit_code": latest["exit_code"] or row["exit_code"],
            "derived_exit_code": latest.get("derived_exit_code") or row.get("derived_exit_code"),
            "time_limit_seconds": row.get("time_limit_seconds"),
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
        "derived_exit_code": row.get("derived_exit_code"),
        "time_limit_seconds": row.get("time_limit_seconds"),
        "started_at": None,
        "ended_at": None,
        "missing_since": missing_since,
        "settled": False,
    }


def _changed(row: dict, obs: dict) -> bool:
    if obs["settled"]:
        return True
    before = (
        row["observed_state"],
        row["slurm_state"],
        row["state_reason"],
        row["exit_code"],
        row.get("derived_exit_code"),
        row.get("time_limit_seconds"),
    )
    after = (
        obs["observed_state"],
        obs["slurm_state"],
        obs["state_reason"],
        obs["exit_code"],
        obs["derived_exit_code"],
        obs["time_limit_seconds"],
    )
    return before != after or (row["missing_since"] is None) != (obs["missing_since"] is None)


def _mark_controller_presence(in_controller: list[str], gone: list[str]) -> None:
    """Stamp ``left_controller_at`` the first pass a job is gone from the
    controller, and clear it if the job comes back.

    Separate from the observation writes on purpose: when accounting fails,
    the gone jobs' observations aren't written, and those are exactly the
    jobs the ``ending_awaiting_accounting`` alert must see.
    """
    from db import pg_transaction

    if not in_controller and not gone:
        return
    with pg_transaction() as conn:
        if gone:
            conn.execute(
                "UPDATE slurm_job_mappings SET left_controller_at = clock_timestamp() "
                "WHERE mapping_id = ANY(%s) AND left_controller_at IS NULL "
                "AND terminal_at IS NULL",
                (gone,),
            )
        if in_controller:
            conn.execute(
                "UPDATE slurm_job_mappings SET left_controller_at = NULL "
                "WHERE mapping_id = ANY(%s) AND left_controller_at IS NOT NULL",
                (in_controller,),
            )


def _write(rows_and_obs: list, unchanged_ids: list) -> list:
    """Apply changed observations (version-guarded) and refresh the rest.

    Keyed by ``mapping_id``: one submission each (migration 157). Returns
    the mapping ids whose guarded write lost to a concurrent writer.
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
                       derived_exit_code = %s,
                       time_limit_seconds = %s,
                       started_at = COALESCE(to_timestamp(%s::float8), started_at),
                       ended_at = COALESCE(to_timestamp(%s::float8), ended_at),
                       missing_since = to_timestamp(%s::float8),
                       terminal_at = CASE WHEN %s THEN clock_timestamp() ELSE NULL END,
                       last_observed_at = clock_timestamp(),
                       last_changed_at = clock_timestamp(),
                       version = version + 1
                 WHERE mapping_id = %s AND version = %s AND terminal_at IS NULL
                """,
                (
                    obs["observed_state"],
                    obs["slurm_state"],
                    obs["state_reason"],
                    obs["exit_code"],
                    obs["derived_exit_code"],
                    obs["time_limit_seconds"],
                    obs["started_at"],
                    obs["ended_at"],
                    obs["missing_since"],
                    obs["settled"],
                    row["mapping_id"],
                    row["version"],
                ),
            )
            if cur.rowcount == 0:
                lost.append(row["mapping_id"])
        if unchanged_ids:
            conn.execute(
                "UPDATE slurm_job_mappings SET last_observed_at = clock_timestamp() "
                "WHERE mapping_id = ANY(%s) AND terminal_at IS NULL",
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
    result.in_controller = sum(1 for r in rows if _in_controller(r, squeue_jobs))

    now = time.time()
    gone = [r for r in rows if not _in_controller(r, squeue_jobs)]
    _mark_controller_presence(
        [r["mapping_id"] for r in rows if _in_controller(r, squeue_jobs)],
        [r["mapping_id"] for r in gone],
    )
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
                result.accounting_error_code = acct.get("error_code")
                records = {}
                gone_ids = {r["slurm_job_id"] for r in gone}
                rows = [r for r in rows if r["slurm_job_id"] not in gone_ids]
                break
            records.update(acct["records"])
            result.accounting_checked = True

    changes, unchanged, observed = [], [], {}
    for row in rows:
        obs = _observation(row, squeue_jobs, records, now)
        observed[row["mapping_id"]] = obs["observed_state"]
        if obs["missing_since"] is not None:
            result.missing.append(row["slurm_job_id"])
        if _changed(row, obs):
            changes.append((row, obs))
        else:
            unchanged.append(row["mapping_id"])
    lost = set(_write(changes, unchanged))

    for row, obs in changes:
        if row["mapping_id"] in lost:
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
            state = observed[row["mapping_id"]]
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

    # Adapter-named jobs no open mapping accounts for: a crash between sbatch
    # and the mapping insert leaves one, and so does a registration refused
    # because an old mapping still holds the id. Matched by name as well as
    # id, since the id alone may be an old mapping's. Reported, never
    # cancelled from here.
    named = {
        sid: j["name"] for sid, j in squeue_jobs.items() if j["name"].startswith(ADAPTER_JOB_PREFIX)
    }
    accounted = _open_names(cluster_id, list(named))
    result.orphans = sorted(sid for sid, name in named.items() if (sid, name) not in accounted)
    if result.orphans:
        log.warning(
            "slurm reconcile %s: adapter jobs with no mapping: %s", cluster_id, result.orphans
        )

    _refresh_outputs(cluster_id, profile_key, runner, result)

    result.duration_ms = int((time.monotonic() - started) * 1000)
    return result


# ── Output cache ─────────────────────────────────────────────────────


def _due_outputs(cluster_id: str, limit: int) -> list[tuple]:
    """Tails to fetch this pass: asked for since the last attempt, or settled
    since it (one final copy). Oldest request first."""
    from db import pg_connection

    with pg_connection() as conn:
        return conn.execute(
            """
            SELECT o.mapping_id, m.xcelsior_job_id, m.submission_name, o.stream,
                   m.slurm_job_id, m.terminal_at IS NOT NULL, o.size_bytes,
                   o.fetched_at IS NOT NULL AND o.slurm_job_id = m.slurm_job_id
              FROM slurm_job_outputs o
              JOIN slurm_job_mappings m ON m.mapping_id = o.mapping_id
             WHERE m.cluster_id = %s AND NOT o.final
               AND (o.requested_at > COALESCE(o.attempted_at, '-infinity'::timestamptz)
                    OR (m.terminal_at IS NOT NULL
                        AND COALESCE(o.attempted_at, '-infinity'::timestamptz) < m.terminal_at))
             ORDER BY o.requested_at NULLS LAST, o.mapping_id, o.stream
             LIMIT %s
            """,
            (cluster_id, limit),
        ).fetchall()


def _record_output_error(mapping_id, stream: str, got: dict, size: int | None = None) -> None:
    from db import pg_transaction

    with pg_transaction() as conn:
        conn.execute(
            "UPDATE slurm_job_outputs SET attempted_at = clock_timestamp(), "
            "error_code = %s, error = %s, size_bytes = COALESCE(%s, size_bytes) "
            "WHERE mapping_id = %s AND stream = %s",
            (got.get("error_code"), got["error"][:500], size, mapping_id, stream),
        )


def _refresh_outputs(cluster_id: str, profile_key: str, runner, result: PassResult) -> None:
    from db import pg_transaction

    profile = sa._get_profile(profile_key)
    check_size = sa.output_size_check(profile)
    reads_whole_file = profile.get("output_reader") == "cat"
    for mapping_id, xid, name, stream, sid, settled, known_size, have_copy in _due_outputs(
        cluster_id, OUTPUT_FETCHES_PER_PASS
    ):
        size = None
        if check_size:
            # Where reading means cat, ask ls -l first: read only when the
            # file changed, and never past the cap.
            sized = sa.get_slurm_job_output_size(
                sid, name, profile_name=profile_key, runner=runner, stream=stream
            )
            if "error" in sized:
                _record_output_error(mapping_id, stream, sized)
                result.output_errors.append((xid, stream, sized.get("error_code")))
                if sized.get("error_code") in _TRANSPORT_ERRORS:
                    break
                continue
            size = sized["size_bytes"]
            if have_copy and known_size == size:
                with pg_transaction() as conn:
                    conn.execute(
                        "UPDATE slurm_job_outputs SET attempted_at = clock_timestamp(), "
                        "fetched_at = clock_timestamp(), final = %s, error_code = NULL, "
                        "error = NULL WHERE mapping_id = %s AND stream = %s",
                        (settled, mapping_id, stream),
                    )
                result.outputs_unchanged.append((xid, stream))
                continue
            if reads_whole_file and size > sa.OUTPUT_CAT_MAX_BYTES:
                too_big = {
                    "error_code": "output_too_large",
                    "error": f"{size} bytes; this cluster can only cat, and the cap is "
                    f"{sa.OUTPUT_CAT_MAX_BYTES}",
                }
                _record_output_error(mapping_id, stream, too_big, size)
                result.output_errors.append((xid, stream, "output_too_large"))
                continue
        got = sa.get_slurm_job_output(
            sid,
            name,
            profile_name=profile_key,
            runner=runner,
            stream=stream,
            max_bytes=sa.PROGRESS_MAX_BYTES if stream == "progress" else sa.OUTPUT_TAIL_BYTES,
        )
        if "error" in got:
            _record_output_error(mapping_id, stream, got)
            result.output_errors.append((xid, stream, got.get("error_code")))
            if got.get("error_code") in _TRANSPORT_ERRORS:
                break  # the cluster stopped answering; the rest wait for the next pass
            continue
        # PostgreSQL text can't hold NUL, and a job may print one.
        content = got["content"].replace("\x00", "")
        with pg_transaction() as conn:
            conn.execute(
                """
                UPDATE slurm_job_outputs
                   SET content = %s, truncated = %s, slurm_job_id = %s, size_bytes = %s,
                       attempted_at = clock_timestamp(), fetched_at = clock_timestamp(),
                       final = %s, error_code = NULL, error = NULL
                 WHERE mapping_id = %s AND stream = %s
                """,
                (
                    content,
                    len(content.encode()) >= sa.OUTPUT_TAIL_BYTES,
                    sid,
                    size,
                    settled,
                    mapping_id,
                    stream,
                ),
            )
        result.outputs_fetched.append((xid, stream))


def reconcile_due_clusters() -> list[dict]:
    """The scheduled task: one pass over every registered cluster that is due."""
    results = []
    for cluster in _due_clusters():
        try:
            with _runner_for(cluster) as runner:
                result = reconcile_cluster(cluster["cluster_id"], cluster["profile_key"], runner=runner)
                if result.ok and _facts_due(cluster.get("facts")):
                    _refresh_facts(cluster, runner, result)
        except sa.SlurmAdapterError as e:  # a profile or registry row that doesn't resolve
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
        try:
            with _runner_for(_cluster_row(cluster_id)) as runner:
                result = reconcile_cluster(
                    cluster_id,
                    profile_key,
                    runner=runner,
                    cancel=False,
                    cancel_callback=cancel_callback,
                    update_callback=update_callback,
                )
        except sa.SlurmAdapterError as e:
            log.warning("slurm reconcile %s: %s (%s)", cluster_id, e, e.code)
            continue
        changes.extend(result.changed)
    return changes
