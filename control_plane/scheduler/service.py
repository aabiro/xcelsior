"""Authoritative transactional scheduling service (blueprint Phase 4).

One tick of the new scheduler: maintenance sweeps (expired schedule
claims, stale leases, undelivered command redelivery), GPU inventory
sync for in-scope hosts, then a bounded placement loop —

    claim (§10.2, canary-scoped) → filter (§10.3) → score (§10.4)
    → reserve_and_bind (§10.5), walking ranked candidates on conflict
    → release the claim with a durable queue reason if nothing fits.

Runs only when ``XCELSIOR_SCHEDULER_MODE`` is ``canary`` or ``active``.
The kill switch (``XCELSIOR_SCHEDULER_CLAIMS_ENABLED=false``) stops new
claims while leaving active attempts, leases, and the maintenance sweeps
untouched — exactly the Phase 4 rollback posture.

The canary partition is exclusive by construction: the claim SQL only
selects jobs :meth:`SchedulerConfig.owns_job` matches, and the legacy
``process_queue`` skips that same set, so a job is only ever placed by
one scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from control_plane.db import RetryBudgetExceeded, run_transaction
from control_plane.commands import redeliver_expired_claims
from control_plane.inventory import sync_host_gpu_inventory
from control_plane.leases import expire_stale_leases
from control_plane.scheduler.claim import (
    ClaimedJob,
    claim_next_job,
    clear_expired_claims,
    release_claim,
)
from control_plane.scheduler.config import SchedulerConfig, SchedulerMode
from control_plane.scheduler.explain import build_explanation
from control_plane.scheduler.filters import FilterContext, filter_hosts
from control_plane.scheduler.reservation import (
    Reservation,
    ReservationConflict,
    reserve_and_bind,
)
from control_plane.scheduler.scoring import rank_candidates
from control_plane.scheduler.snapshot import take_snapshot

log = logging.getLogger("xcelsior.control_plane.scheduler.service")

# Backoff written into next_schedule_at when a claimed job cannot be
# placed; keeps one unplaceable job from being re-claimed every tick.
_NO_CAPACITY_BACKOFF_SEC = 15.0
_CONFLICT_BACKOFF_SEC = 3.0


@dataclass(frozen=True)
class TickReport:
    mode: str
    placements: list[Reservation] = field(default_factory=list)
    released: list[tuple[str, str]] = field(default_factory=list)  # (job_id, reason)
    claims_expired: int = 0
    leases_expired: int = 0
    commands_redelivered: int = 0
    inventory_synced: int = 0


class SchedulerService:
    """One scheduler replica for the transactional pipeline."""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig.from_env()

    # ── maintenance ──────────────────────────────────────────────────

    def _run_sweeps(self) -> tuple[int, int, int]:
        claims = run_transaction(
            lambda c: clear_expired_claims(c), what="sched_claim_sweep"
        )
        leases = run_transaction(
            lambda c: expire_stale_leases(c), what="sched_lease_sweep"
        )
        commands = run_transaction(
            lambda c: redeliver_expired_claims(c), what="sched_command_sweep"
        )
        return claims, len(leases), commands

    def _sync_inventory(self, hosts: list[dict[str, Any]]) -> int:
        changed = 0
        for host in hosts:
            host_id = str(host.get("host_id") or "")
            if not host_id or not self.config.host_in_scope(host_id):
                continue
            if str(host.get("administrative_state") or "") != "admitted":
                continue
            try:
                result = run_transaction(
                    lambda c, h=host: sync_host_gpu_inventory(c, h),
                    what="sched_inventory_sync",
                )
            except Exception:
                log.exception("inventory sync failed for host %s", host_id)
                continue
            if result.changed:
                changed += 1
        return changed

    # ── placement ────────────────────────────────────────────────────

    def _claim_scope(self) -> list[str] | None:
        """None = whole queue (active); a list = canary partition only."""
        if self.config.mode is SchedulerMode.ACTIVE:
            return None
        return sorted(self.config.canary_gpu_models)

    def _place_one(self, claimed: ClaimedJob) -> Reservation | tuple[str, str]:
        """Place a claimed job or release it with a durable reason."""
        cfg = self.config
        job = dict(claimed.payload)
        job["job_id"] = claimed.job_id

        snapshot = run_transaction(
            lambda c: take_snapshot(
                c, host_freshness_timeout_sec=cfg.host_freshness_timeout_sec
            ),
            what="sched_fleet_snapshot",
        )
        hosts = [
            h for h in snapshot.hosts
            if cfg.host_in_scope(str(h.get("host_id") or ""))
        ]
        ctx = FilterContext(stale_host_ids=snapshot.stale_host_ids)
        eligible, rejections = filter_hosts(job, hosts, ctx)
        ranked = rank_candidates(job, eligible) if eligible else []

        if not ranked:
            reason = "no_hosts" if not hosts else "no_eligible_host"
            self._release(claimed, reason, _NO_CAPACITY_BACKOFF_SEC)
            return (claimed.job_id, reason)

        host_by_id = {str(h.get("host_id")): h for h in eligible}
        num_gpus = max(1, int(float(job.get("num_gpus", 1) or 1)))
        vram_mb = int(float(job.get("vram_needed_gb", 0) or 0) * 1024)

        for candidate in ranked:
            host = host_by_id[candidate.host_id]
            explanation = build_explanation(
                job=job,
                host_count=len(hosts),
                rejections=rejections,
                ranked=ranked,
                selected_host_id=candidate.host_id,
                max_rejections=cfg.explain_max_rejections,
                max_ranked=cfg.explain_max_ranked,
            )
            try:
                reservation = run_transaction(
                    lambda c, h=host, cand=candidate, expl=explanation: reserve_and_bind(
                        c,
                        job_id=claimed.job_id,
                        claim_token=claimed.claim_token,
                        replica_id=cfg.replica_id,
                        host_id=cand.host_id,
                        num_gpus=num_gpus,
                        requested_vram_mb=vram_mb,
                        expected_inventory_generation=int(
                            h.get("inventory_generation") or 0
                        ),
                        placement_score=cand.total,
                        placement_explanation=expl,
                        lease_claim_ttl_sec=cfg.lease_claim_ttl_sec,
                        lease_renewal_ttl_sec=cfg.lease_renewal_ttl_sec,
                    ),
                    what="sched_reserve",
                )
            except ReservationConflict as exc:
                if exc.code in ("claim_lost", "job_not_schedulable"):
                    # The job itself is gone from under us — no point in
                    # trying more hosts, and nothing to release.
                    log.info(
                        "reservation abandoned for %s: %s", claimed.job_id, exc.code
                    )
                    return (claimed.job_id, exc.code)
                log.info(
                    "candidate %s rejected for %s (%s); trying next",
                    candidate.host_id, claimed.job_id, exc.code,
                )
                continue
            except RetryBudgetExceeded:
                # Transient contention outlasted the retry budget: back
                # off and let a later tick (or another replica) retry.
                self._release(claimed, "placement_conflict", _CONFLICT_BACKOFF_SEC)
                return (claimed.job_id, "placement_conflict")
            log.info(
                "PLACED job=%s host=%s attempt=%s fence=%d gpus=%s",
                reservation.job_id, reservation.host_id,
                reservation.attempt_id, reservation.fencing_token,
                reservation.gpu_device_ids,
            )
            return reservation

        # Every scored candidate lost its revalidation race.
        self._release(claimed, "placement_conflict", _CONFLICT_BACKOFF_SEC)
        return (claimed.job_id, "placement_conflict")

    def _release(self, claimed: ClaimedJob, reason: str, backoff_sec: float) -> None:
        run_transaction(
            lambda c: release_claim(
                c,
                claimed.job_id,
                claimed.claim_token,
                reason_code=reason,
                requeue_delay_sec=backoff_sec,
            ),
            what="sched_release_claim",
        )

    # ── the tick ─────────────────────────────────────────────────────

    def tick(self) -> TickReport:
        cfg = self.config
        if cfg.mode not in (SchedulerMode.CANARY, SchedulerMode.ACTIVE):
            return TickReport(mode=cfg.mode.value)

        claims_swept, leases_swept, commands_redelivered = self._run_sweeps()
        report_kwargs: dict[str, Any] = {
            "claims_expired": claims_swept,
            "leases_expired": leases_swept,
            "commands_redelivered": commands_redelivered,
        }

        if not cfg.claims_enabled:
            log.info("claims disabled (kill switch) — sweeps only")
            return TickReport(mode=cfg.mode.value, **report_kwargs)

        # Inventory before placement so filters and reservations see the
        # same device truth this tick.
        fleet = run_transaction(
            lambda c: take_snapshot(
                c, host_freshness_timeout_sec=cfg.host_freshness_timeout_sec
            ),
            what="sched_inventory_snapshot",
        )
        report_kwargs["inventory_synced"] = self._sync_inventory(fleet.hosts)

        placements: list[Reservation] = []
        released: list[tuple[str, str]] = []
        scope = self._claim_scope()
        for _ in range(cfg.tick_max_placements):
            claimed = run_transaction(
                lambda c: claim_next_job(
                    c, replica_id=cfg.replica_id, scope_gpu_models=scope
                ),
                what="sched_claim",
            )
            if claimed is None:
                break
            outcome = self._place_one(claimed)
            if isinstance(outcome, Reservation):
                placements.append(outcome)
            else:
                released.append(outcome)

        return TickReport(
            mode=cfg.mode.value,
            placements=placements,
            released=released,
            **report_kwargs,
        )
