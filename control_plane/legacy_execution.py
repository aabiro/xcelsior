"""Fence the remaining compatibility execution paths and expose their use."""

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

for _path in ("work_poll", "lease_claim", "lease_renew", "lease_release"):
    for _outcome in ("requested", "used", "blocked"):
        _uses.labels(path=_path, outcome=_outcome)


def record_use(path: str, outcome: str, count: int = 1) -> None:
    _uses.labels(path=path, outcome=outcome).inc(count)


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
