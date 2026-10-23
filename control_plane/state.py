"""Native lifecycle fields maintained by persistence and controller writers.

The status mapping is for admission and compatibility observations. Workers
report an observed phase; only controller intent changes desired state.
"""

_JOB_PHASES = {
    "queued": "pending",
    "preempted": "pending",
    "assigned": "scheduled",
    "leased": "scheduled",
    "starting": "starting",
    "restarting": "starting",
    "running": "running",
    "stopping": "running",
    "completed": "succeeded",
    "failed": "failed",
    "cancelled": "stopped",
    "stopped": "stopped",
    "paused": "stopped",
    "terminated": "stopped",
}


def job_phase(status: str) -> str:
    try:
        return _JOB_PHASES[status]
    except KeyError:
        raise ValueError(f"unsupported job status: {status!r}") from None


def initial_desired_state(status: str) -> str:
    job_phase(status)
    return (
        "stopped"
        if status in {"stopping", "stopped", "paused", "cancelled", "terminated"}
        else "running"
    )


def host_states(status: str, admission_state: str) -> tuple[str, str]:
    administrative = (
        "disabled"
        if status == "disabled"
        else (
            "draining"
            if admission_state == "admitted" and status == "draining"
            else "admitted" if admission_state == "admitted" else "pending"
        )
    )
    availability = {"active": "ready", "dead": "not_ready"}.get(status, "unknown")
    return administrative, availability
