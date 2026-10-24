"""Reject retired execution transports and count authenticated callers."""

from contextlib import contextmanager
import os

from fastapi import HTTPException
from prometheus_client import Counter, Gauge, REGISTRY

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


@contextmanager
def legacy_request(path: str):
    if path not in LEGACY_PATHS:
        raise ValueError("Unknown legacy observation path")
    record_use(path, "requested")
    try:
        yield
    except Exception as exc:
        refused = isinstance(exc, HTTPException) and 400 <= exc.status_code < 500
        record_use(path, "blocked" if refused else "failed")
        raise


def legacy_execution_allowed() -> bool:
    """Native authority is permanent, including paused and canary modes."""
    return False


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
    # Validate obsolete declarations, but no setting can revive a transport.
    return set(LEGACY_PATHS)


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
    _retired.labels(path=_path).set(1)


def require_legacy_route_enabled() -> None:
    raise HTTPException(
        410,
        detail={
            "error": {
                "code": "v2_protocol_required",
                "message": "This control plane requires the durable v2 worker protocol",
            }
        },
    )
