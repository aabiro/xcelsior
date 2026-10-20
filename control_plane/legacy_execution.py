"""Fence compatibility execution and measure paths awaiting retirement."""

from contextlib import contextmanager
import os

from fastapi import HTTPException
from prometheus_client import Counter, Gauge, REGISTRY

from control_plane.scheduler.config import SchedulerConfig, SchedulerMode

try:
    _uses = Counter(
        "xcelsior_legacy_execution_total",
        "Authenticated legacy execution requests and effects during retirement",
        ["path", "outcome"],
    )
except ValueError:
    _existing = REGISTRY._names_to_collectors["xcelsior_legacy_execution_total"]
    if not isinstance(_existing, Counter):
        raise
    _uses = _existing

EXECUTION_PATHS = (
    "work_poll",
    "lease_claim",
    "lease_renew",
    "lease_release",
    "command_start",
    "command_pause",
)
LEGACY_PATHS = (*EXECUTION_PATHS, "status_patch", "command_drain")

for _path in LEGACY_PATHS:
    for _outcome in ("requested", "used", "blocked", "failed"):
        _uses.labels(path=_path, outcome=_outcome)


def record_use(path: str, outcome: str, count: int = 1) -> None:
    _uses.labels(path=path, outcome=outcome).inc(count)


class LegacyRequest:
    def __init__(self, path: str):
        self.path = path
        self.effects = 0

    def effect(self, count: int = 1) -> None:
        """Record completed mutator calls or committed command deliveries."""
        record_use(self.path, "used", count)
        self.effects += count


@contextmanager
def legacy_request(path: str):
    if path not in LEGACY_PATHS:
        raise ValueError("Unknown legacy observation path")
    observation = LegacyRequest(path)
    record_use(path, "requested")
    try:
        yield observation
    except Exception as exc:
        # An HTTP failure can follow a committed mutation. Calling that
        # blocked would erase evidence that the compatibility path was used.
        refused = isinstance(exc, HTTPException) and 400 <= exc.status_code < 500
        record_use(path, "blocked" if refused and not observation.effects else "failed")
        raise


def legacy_execution_allowed() -> bool:
    return SchedulerConfig.from_env().mode is not SchedulerMode.ACTIVE


try:
    _enabled = Gauge(
        "xcelsior_legacy_execution_enabled", "Whether scheduler policy permits legacy starts"
    )
except ValueError:
    _existing_gauge = REGISTRY._names_to_collectors["xcelsior_legacy_execution_enabled"]
    if not isinstance(_existing_gauge, Gauge):
        raise
    _enabled = _existing_gauge
_enabled.set_function(lambda: int(legacy_execution_allowed()))


def retired_paths() -> set[str]:
    declared = {
        name.strip()
        for name in os.environ.get("XCELSIOR_RETIRED_LEGACY_PATHS", "").split(",")
        if name.strip()
    }
    if declared.difference(LEGACY_PATHS):
        raise ValueError("XCELSIOR_RETIRED_LEGACY_PATHS contains an unknown path")
    return declared


# Refuse a misspelled policy on startup rather than silently suppress an alert.
retired_paths()
try:
    _retired = Gauge(
        "xcelsior_legacy_path_retired",
        "Whether this replica expects no authenticated calls to a legacy path",
        ["path"],
    )
except ValueError:
    _existing_retired = REGISTRY._names_to_collectors["xcelsior_legacy_path_retired"]
    if not isinstance(_existing_retired, Gauge):
        raise
    _retired = _existing_retired

for _path in LEGACY_PATHS:
    _retired.labels(path=_path).set_function(
        lambda path=_path: int(
            path in retired_paths() or (path in EXECUTION_PATHS and not legacy_execution_allowed())
        )
    )


class LegacyExecutionRejected(ValueError):
    code = "fenced_protocol_required"


def require_legacy_job(conn, job_id: str, *, new_claim: bool = False) -> None:
    """Serialize compatibility lease writes with placement on the job row.

    Fenced history stays fenced even between attempts. A separate history read
    after acquiring the row lock sees a placement that committed while waiting.
    """
    row = conn.execute(
        "SELECT active_attempt_id FROM jobs WHERE job_id=%s FOR UPDATE", (job_id,)
    ).fetchone()
    active_attempt = (
        row.get("active_attempt_id") if isinstance(row, dict) else row[0] if row else None
    )
    history = conn.execute(
        "SELECT 1 FROM job_attempts WHERE job_id=%s LIMIT 1", (job_id,)
    ).fetchone()
    if active_attempt or history:
        raise LegacyExecutionRejected("Job belongs to the fenced attempt protocol")
    if new_claim and not legacy_execution_allowed():
        raise LegacyExecutionRejected("Legacy execution is disabled while scheduler mode is active")


def eligible_work_ids(conn, host_id: str, job_ids: list[str]) -> set[str]:
    if not job_ids:
        return set()
    rows = conn.execute(
        "SELECT j.job_id FROM jobs j WHERE j.job_id=ANY(%s) AND j.host_id=%s "
        "AND j.status='assigned' AND j.active_attempt_id IS NULL "
        "AND NOT EXISTS (SELECT 1 FROM job_attempts a WHERE a.job_id=j.job_id)",
        (job_ids, host_id),
    ).fetchall()
    return {str(row["job_id"] if isinstance(row, dict) else row[0]) for row in rows}


def require_legacy_container_command(
    conn, *, job_id: str, host_id: str, container_name: str, starting: bool
) -> None:
    """Keep compatibility directives on the same job lock as placement."""
    row = conn.execute(
        "SELECT host_id,payload->>'container_name' AS container_name FROM jobs "
        "WHERE job_id=%s FOR UPDATE",
        (job_id,),
    ).fetchone()
    if row is None:
        raise LegacyExecutionRejected("Legacy container job was not found")
    assigned_host = row.get("host_id") if isinstance(row, dict) else row[0]
    stored_name = row.get("container_name") if isinstance(row, dict) else row[1]
    expected_name = str(stored_name or f"xcl-{job_id}")
    if str(assigned_host or "") != host_id or expected_name != container_name:
        raise LegacyExecutionRejected("Legacy container identity no longer matches the job")
    require_legacy_job(conn, job_id, new_claim=starting)
