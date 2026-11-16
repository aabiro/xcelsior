"""Scheduler runtime configuration (blueprint §30, Phase 3).

One typed config object, sourced from environment variables, shared by
the shadow runner and authoritative scheduler service. ``XCELSIOR_SCHEDULER_MODE`` is the master switch:

- ``paused``  — placement is disabled (default).
- ``shadow``  — run the new pipeline read-only against snapshots and
  persist would-be decisions for comparison (this phase).
- ``canary``  — new pipeline owns placement for a scoped canary pool
  (all other work stays queued).
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


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


#: XCELSIOR_SLURM_PLACEMENT modes, in rollout order (plan P2.9). Only the
#: last three charge, and a placement keeps the answer it was made under
#: (slurm_submissions.charge, migration 163).
SLURM_PLACEMENT_MODES = ("off", "shadow", "admin", "allowlist", "all")
SLURM_CHARGING_MODES = frozenset({"admin", "allowlist", "all"})


def slurm_placement_mode(raw: str | None = None) -> str:
    """The Slurm placement mode from XCELSIOR_SLURM_PLACEMENT.

    ``off`` places nothing. ``shadow`` places any job that names a cluster
    and never charges it. ``admin`` places only a platform admin's jobs,
    ``allowlist`` those and the tenants in XCELSIOR_SLURM_TENANTS, ``all``
    everyone's, and those three charge. The old truthy values (``1``,
    ``true``, ``yes``, ``on``) meant "place, don't charge", so they read as
    ``shadow``: a switch set before charging existed must never start
    charging by itself. Anything unrecognised is ``off``.
    """
    value = (os.environ.get("XCELSIOR_SLURM_PLACEMENT") if raw is None else raw) or ""
    value = value.strip().lower()
    if value in SLURM_PLACEMENT_MODES:
        return value
    if value in ("1", "true", "yes", "on"):
        return "shadow"
    if value:
        log.error("XCELSIOR_SLURM_PLACEMENT=%r is not a mode; Slurm placement is off", value)
    return "off"


def slurm_charging_enabled() -> bool:
    return slurm_placement_mode() in SLURM_CHARGING_MODES


def slurm_tenants() -> frozenset[str]:
    """Tenant ids the ``allowlist`` mode admits (XCELSIOR_SLURM_TENANTS)."""
    return _csv_env("XCELSIOR_SLURM_TENANTS")


def _csv_env(name: str) -> frozenset[str]:
    raw = os.environ.get(name) or ""
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


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

    # ── Phase 4 cutover scoping ──────────────────────────────────────
    # Kill switch (§ Phase 4 exit gate): stops NEW claims without touching
    # active attempts/leases — maintenance sweeps keep running.
    claims_enabled: bool = True
    # Canary job scope: jobs whose requested gpu_model (lowercased) is in
    # this set — or that carry payload {"scheduler": "v2"} — are owned by
    # the new scheduler in canary mode. Active mode owns every job.
    canary_gpu_models: frozenset[str] = frozenset()
    # Optional canary host pool; empty = every host is in scope.
    canary_host_ids: frozenset[str] = frozenset()
    # Placement work per tick (bounds one replica's burst).
    tick_max_placements: int = 10
    # Lease shape handed to reservations.
    lease_claim_ttl_sec: int = 60
    lease_renewal_ttl_sec: int = 300
    # Off until a cluster is ready to take jobs. A job that names
    # hpc_cluster_id stays queued (reason slurm_placement_disabled) and is
    # never placed on a host. On, the scheduler places it on that cluster,
    # for the owners slurm_placement_mode admits (see slurm_placement_mode).
    slurm_placement: bool = False
    slurm_placement_mode: str = "shadow"
    # Tenant ids admitted in "allowlist" mode (XCELSIOR_SLURM_TENANTS).
    slurm_tenants: frozenset[str] = frozenset()

    def owns_job(self, job: dict) -> bool:
        """Does the new scheduler own queued→assigned for this job?

        Scope determines which jobs the native service may claim. Work outside
        a canary scope remains queued; it never enters another allocator.
        """
        if self.mode is SchedulerMode.ACTIVE:
            return True
        if self.mode is not SchedulerMode.CANARY:
            return False
        if str(job.get("scheduler") or "").strip().lower() == "v2":
            return True
        model = str(job.get("gpu_model") or "").strip().lower()
        return bool(model) and model in self.canary_gpu_models

    def host_in_scope(self, host_id: str) -> bool:
        if self.mode is SchedulerMode.ACTIVE or not self.canary_host_ids:
            return True
        return str(host_id).strip().lower() in self.canary_host_ids

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
            replica_id=os.environ.get("XCELSIOR_SCHEDULER_REPLICA_ID") or default_replica_id(),
            shadow_interval_sec=_int_env("XCELSIOR_SCHEDULER_SHADOW_INTERVAL_SEC", 15),
            shadow_compare_grace_sec=_int_env("XCELSIOR_SCHEDULER_SHADOW_COMPARE_GRACE_SEC", 30),
            shadow_retention_days=_int_env("XCELSIOR_SCHEDULER_SHADOW_RETENTION_DAYS", 14),
            host_freshness_timeout_sec=_int_env(
                "XCELSIOR_SCHEDULER_HOST_FRESHNESS_TIMEOUT_SEC", 300
            ),
            claims_enabled=_bool_env("XCELSIOR_SCHEDULER_CLAIMS_ENABLED", True),
            canary_gpu_models=_csv_env("XCELSIOR_SCHEDULER_CANARY_GPU_MODELS"),
            canary_host_ids=_csv_env("XCELSIOR_SCHEDULER_CANARY_HOSTS"),
            tick_max_placements=_int_env("XCELSIOR_SCHEDULER_TICK_MAX_PLACEMENTS", 10),
            lease_claim_ttl_sec=_int_env("XCELSIOR_SCHEDULER_LEASE_CLAIM_TTL_SEC", 60),
            lease_renewal_ttl_sec=_int_env("XCELSIOR_SCHEDULER_LEASE_RENEWAL_TTL_SEC", 300),
            slurm_placement=slurm_placement_mode() != "off",
            slurm_placement_mode=slurm_placement_mode(),
            slurm_tenants=slurm_tenants(),
        )
