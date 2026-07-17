"""Scheduler runtime configuration (blueprint §30, Phase 3).

One typed config object, sourced from environment variables, shared by
the shadow runner now and the authoritative scheduler service at the
Phase 4 cutover. ``XCELSIOR_SCHEDULER_MODE`` is the master switch:

- ``paused``  — the new pipeline does nothing (default; legacy owns all).
- ``shadow``  — run the new pipeline read-only against snapshots and
  persist would-be decisions for comparison (this phase).
- ``canary``  — new pipeline owns placement for a scoped canary pool
  (Phase 4; not implemented yet).
- ``active``  — new pipeline owns all standard placement (Phase 4).

Unknown mode strings resolve to ``paused`` and log, never crash: a typo
in an env file must not take the scheduler container down.
"""

from __future__ import annotations

import enum
import logging
import os
import socket
import uuid
from dataclasses import dataclass, field

log = logging.getLogger("xcelsior.control_plane.scheduler.config")


class SchedulerMode(enum.Enum):
    PAUSED = "paused"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"


def _int_env(name: str, default: int, *, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default


def default_replica_id() -> str:
    """Stable-per-process, unique-per-replica identity for claim owners."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"


@dataclass(frozen=True)
class SchedulerConfig:
    mode: SchedulerMode = SchedulerMode.PAUSED
    replica_id: str = field(default_factory=default_replica_id)
    # Shadow cycle cadence. The legacy scheduler ticks every 2s; shadow
    # deliberately runs slower — it is an observer, not a competitor for
    # DB resources.
    shadow_interval_sec: int = 15
    # How long the legacy scheduler gets to act on the same queue state
    # before a shadow decision is compared against reality. Shorter than
    # the legacy tick would misclassify in-flight work as mismatch.
    shadow_compare_grace_sec: int = 30
    # Shadow decision retention (comparator output is long-term evidence,
    # but unbounded growth is not).
    shadow_retention_days: int = 14
    # Hosts whose last heartbeat is older than this are treated as stale
    # by the snapshot (FilterContext.stale_host_ids).
    host_freshness_timeout_sec: int = 300
    # Explanation payload bounds (per-host rejection detail / ranked list).
    explain_max_rejections: int = 25
    explain_max_ranked: int = 10

    @classmethod
    def from_env(cls) -> SchedulerConfig:
        raw_mode = (os.environ.get("XCELSIOR_SCHEDULER_MODE") or "paused").strip().lower()
        try:
            mode = SchedulerMode(raw_mode)
        except ValueError:
            log.warning(
                "XCELSIOR_SCHEDULER_MODE=%r is not one of %s; defaulting to paused",
                raw_mode,
                [m.value for m in SchedulerMode],
            )
            mode = SchedulerMode.PAUSED
        return cls(
            mode=mode,
            replica_id=os.environ.get("XCELSIOR_SCHEDULER_REPLICA_ID")
            or default_replica_id(),
            shadow_interval_sec=_int_env("XCELSIOR_SCHEDULER_SHADOW_INTERVAL_SEC", 15),
            shadow_compare_grace_sec=_int_env(
                "XCELSIOR_SCHEDULER_SHADOW_COMPARE_GRACE_SEC", 30
            ),
            shadow_retention_days=_int_env("XCELSIOR_SCHEDULER_SHADOW_RETENTION_DAYS", 14),
            host_freshness_timeout_sec=_int_env(
                "XCELSIOR_SCHEDULER_HOST_FRESHNESS_TIMEOUT_SEC", 300
            ),
        )
