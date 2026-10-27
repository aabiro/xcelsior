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

Only this fenced pipeline creates placements. Canary scope restricts the
admitted jobs and hosts it may claim; other work remains queued.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from control_plane.db import RetryBudgetExceeded, run_transaction
from control_plane.commands import redeliver_expired_claims
from control_plane.inventory import sync_host_gpu_inventory
from control_plane.leases import expire_stale_leases, release_stopped_attempts
from control_plane.operational_metrics import ServiceHeartbeat
from control_plane.job_specs import validate_execution_spec
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
from control_plane.scheduler.snapshot import take_snapshot, with_nominated_devices

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
    sweep_errors: int = 0


class SchedulerService:
    """One scheduler replica for the transactional pipeline."""

    def __init__(self, config: SchedulerConfig | None = None):
        self.config = config or SchedulerConfig.from_env()
        self._reconciler_heartbeat = ServiceHeartbeat(
            "reconciler",
            replica_id=self.config.replica_id,
        )
        self._sweep_errors = 0

    # ── maintenance ──────────────────────────────────────────────────

    def _sweep(self, what: str, fn, default):
        """Run one maintenance sweep; its failure never blocks the others or placement.

        Row-level sweeps isolate their items themselves. This is the backstop
        for a failure of the sweep as a whole (its batch query, a retry budget
        exhausted by contention): log it, count it, and keep the tick going.
        """
        try:
            return run_transaction(fn, what=what)
        except Exception:
            log.exception("%s failed; continuing the tick", what)
            self._sweep_errors += 1
            return default

    def _run_sweeps(self) -> tuple[int, int, int]:
        self._sweep_errors = 0
        claims = self._sweep("sched_claim_sweep", lambda c: clear_expired_claims(c), 0)
        leases = self._sweep("sched_lease_sweep", lambda c: expire_stale_leases(c), [])
        self._sweep("sched_stop_evidence_sweep", lambda c: release_stopped_attempts(c), [])
        commands = self._sweep("sched_command_sweep", lambda c: redeliver_expired_claims(c), 0)
        from control_plane.scheduler.preemption import sweep_plans

        self._sweep("sched_preemption_sweep", lambda c: sweep_plans(c), 0)
        from control_plane.host_evictions import sweep as sweep_host_evictions

        self._sweep("sched_host_eviction_sweep", lambda c: sweep_host_evictions(c), 0)
        # Phase 6: desired-vs-observed reconciliation (report-only by default),
        # for job mutations, worker observations, and periodic recovery.
        try:
            from control_plane.reconcile import process_due

            run_transaction(
                lambda c: process_due(c, worker_id=self.config.replica_id),
                what="sched_reconcile",
            )
            try:
                self._reconciler_heartbeat.emit_if_due()
            except Exception:
                log.exception("reconciler heartbeat failed; continuing")
        except Exception:
            log.exception("reconcile sweep failed; continuing")
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
        try:
            job = dict(
                validate_execution_spec(claimed.spec, claimed.spec_hash, require_admission=True)
            )
        except ValueError:
            self._release(claimed, "execution_spec_unqualified", _NO_CAPACITY_BACKOFF_SEC)
            return (claimed.job_id, "execution_spec_unqualified")
        job["job_id"] = claimed.job_id

        snapshot = run_transaction(
            lambda c: take_snapshot(c, host_freshness_timeout_sec=cfg.host_freshness_timeout_sec),
            what="sched_fleet_snapshot",
        )
        hosts = [h for h in snapshot.hosts if cfg.host_in_scope(str(h.get("host_id") or ""))]
        nomination_host = run_transaction(
            lambda c: c.execute(
                """SELECT host_id, plan_id FROM preemption_plans
                WHERE nominated_job_id=%s AND state IN ('stopping','ready')""",
                (claimed.job_id,),
            ).fetchone(),
            what="sched_nomination_host",
        )
        if nomination_host:
            # The nominee may bind only the devices its plan holds; everyone
            # else sees those devices as unavailable.
            hosts = [
                with_nominated_devices(h, str(nomination_host[1]))
                for h in hosts
                if h.get("host_id") == nomination_host[0]
            ]
        ctx = FilterContext(stale_host_ids=snapshot.stale_host_ids)
        eligible, rejections = filter_hosts(job, hosts, ctx)
        if eligible:
            from control_plane.scheduler.native_preference import choose_preferred
            from control_plane.scheduler.preference import PlacementRefused

            try:
                decision = run_transaction(
                    lambda c: choose_preferred(c, job, eligible, now=snapshot.taken_at.timestamp()),
                    what="sched_preference",
                )
            except ValueError:
                self._release(claimed, "placement_preference_invalid", _NO_CAPACITY_BACKOFF_SEC)
                return (claimed.job_id, "placement_preference_invalid")
            if isinstance(decision, PlacementRefused):
                reason = "placement_preference_" + decision.code
                self._release(claimed, reason, _NO_CAPACITY_BACKOFF_SEC, details=decision.as_dict())
                return (claimed.job_id, reason)
            if decision is not None:
                eligible = [decision.host]
        ranked = rank_candidates(job, eligible) if eligible else []

        if not ranked:
            if self._plan_preemption(claimed, hosts, rejections):
                self._release(claimed, "preemption_waiting", _CONFLICT_BACKOFF_SEC)
                return (claimed.job_id, "preemption_waiting")
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
                        expected_inventory_generation=int(h.get("inventory_generation") or 0),
                        placement_score=cand.total,
                        placement_explanation=expl,
                        lease_claim_ttl_sec=cfg.lease_claim_ttl_sec,
                        lease_renewal_ttl_sec=cfg.lease_renewal_ttl_sec,
                        trace_id=str(claimed.payload.get("trace_id") or "") or None,
                        expected_spec_hash=claimed.spec_hash,
                        scope_host_ids=(
                            None
                            if cfg.mode is SchedulerMode.ACTIVE or not cfg.canary_host_ids
                            else cfg.canary_host_ids
                        ),
                        host_freshness_timeout_sec=cfg.host_freshness_timeout_sec,
                    ),
                    what="sched_reserve",
                    isolation_level="SERIALIZABLE",
                )
            except ReservationConflict as exc:
                if exc.code == "execution_spec_unqualified":
                    self._release(claimed, exc.code, _NO_CAPACITY_BACKOFF_SEC)
                    return (claimed.job_id, exc.code)
                if exc.code in (
                    "team_quota_exceeded",
                    "preemption_waiting",
                    "queue_policy_changed",
                ):
                    self._release(claimed, exc.code, _NO_CAPACITY_BACKOFF_SEC)
                    return (claimed.job_id, exc.code)
                if exc.code in ("claim_lost", "job_not_schedulable"):
                    # The job itself is gone from under us — no point in
                    # trying more hosts, and nothing to release.
                    log.info("reservation abandoned for %s: %s", claimed.job_id, exc.code)
                    return (claimed.job_id, exc.code)
                log.info(
                    "candidate %s rejected for %s (%s); trying next",
                    candidate.host_id,
                    claimed.job_id,
                    exc.code,
                )
                continue
            except RetryBudgetExceeded:
                # Transient contention outlasted the retry budget: back
                # off and let a later tick (or another replica) retry.
                self._release(claimed, "placement_conflict", _CONFLICT_BACKOFF_SEC)
                return (claimed.job_id, "placement_conflict")
            log.info(
                "PLACED job=%s host=%s attempt=%s fence=%d gpus=%s",
                reservation.job_id,
                reservation.host_id,
                reservation.attempt_id,
                reservation.fencing_token,
                reservation.gpu_device_ids,
            )
            return reservation

        # Every scored candidate lost its revalidation race.
        self._release(claimed, "placement_conflict", _CONFLICT_BACKOFF_SEC)
        return (claimed.job_id, "placement_conflict")

    def _plan_preemption(self, claimed, hosts, rejections) -> bool:
        from psycopg.errors import LockNotAvailable
        from control_plane.commands import CommandProtocolError
        from control_plane.scheduler.preemption import create_plan

        for host in sorted(hosts, key=lambda h: str(h.get("host_id"))):
            host_id = str(host.get("host_id") or "")
            reasons = rejections.get(host_id, [])
            if not reasons or any(
                r.code not in {"insufficient_gpus", "insufficient_vram"} for r in reasons
            ):
                continue
            try:
                plan = run_transaction(
                    lambda c: create_plan(
                        c,
                        job_id=claimed.job_id,
                        claim_token=claimed.claim_token,
                        host_id=host_id,
                        replica_id=self.config.replica_id,
                        scope_host_ids=(
                            None
                            if self.config.mode is SchedulerMode.ACTIVE
                            or not self.config.canary_host_ids
                            else self.config.canary_host_ids
                        ),
                    ),
                    what="sched_preemption_plan",
                    isolation_level="SERIALIZABLE",
                )
            except (
                LockNotAvailable,
                RetryBudgetExceeded,
                CommandProtocolError,
                ReservationConflict,
            ):
                continue
            if plan:
                return True
        return False

    def _release(
        self, claimed: ClaimedJob, reason: str, backoff_sec: float, *, details=None
    ) -> None:
        run_transaction(
            lambda c: release_claim(
                c,
                claimed.job_id,
                claimed.claim_token,
                reason_code=reason,
                requeue_delay_sec=backoff_sec,
                reason_details=details,
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
            "sweep_errors": self._sweep_errors,
        }

        if not cfg.claims_enabled:
            log.info("claims disabled (kill switch) — sweeps only")
            return TickReport(mode=cfg.mode.value, **report_kwargs)

        # Inventory before placement so filters and reservations see the
        # same device truth this tick.
        fleet = run_transaction(
            lambda c: take_snapshot(c, host_freshness_timeout_sec=cfg.host_freshness_timeout_sec),
            what="sched_inventory_snapshot",
        )
        report_kwargs["inventory_synced"] = self._sync_inventory(fleet.hosts)

        placements: list[Reservation] = []
        released: list[tuple[str, str]] = []
        scope = self._claim_scope()
        for _ in range(cfg.tick_max_placements):
            claimed = run_transaction(
                lambda c: claim_next_job(c, replica_id=cfg.replica_id, scope_gpu_models=scope),
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
