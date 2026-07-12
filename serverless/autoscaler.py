# Xcelsior — Serverless autoscaler (pure reconciliation math)

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from serverless.repo import (
    WORKER_STATE_BOOTING,
    WORKER_STATE_DRAINING,
    WORKER_STATE_IDLE,
    WORKER_STATE_READY,
)
from serverless.forecast_fallback import forecast_queue_depth_with_fallback
from serverless.prewarm_pool import apply_prewarm_to_desired
from serverless.scaling_predictive import predictive_scaling_enabled

ACTIVE_WORKER_STATES = frozenset(
    {WORKER_STATE_BOOTING, WORKER_STATE_READY, WORKER_STATE_IDLE, WORKER_STATE_DRAINING}
)

SCALING_QUEUE_REQUEST_COUNT = "queue_request_count"
SCALING_QUEUE_DELAY = "queue_delay"

SCALE_DOWN_COOLDOWN_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_SCALE_DOWN_COOLDOWN_SEC", "60"))
DRAIN_GRACE_SEC = int(os.environ.get("XCELSIOR_SERVERLESS_DRAIN_GRACE_SEC", "30"))


@dataclass
class AutoscalerInput:
    min_workers: int
    max_workers: int
    max_concurrency: int
    scaling_policy_type: str
    scaling_policy_value: int
    queue_depth: int
    max_queue_wait_sec: float
    workers: list[dict] = field(default_factory=list)
    queue_depth_samples: list[tuple[float, int]] = field(default_factory=list)


def count_active_workers(workers: list[dict]) -> int:
    return sum(1 for w in workers if w.get("state") in ACTIVE_WORKER_STATES)


def free_concurrency_slots(workers: list[dict], max_concurrency: int) -> int:
    total = 0
    for w in workers:
        if w.get("state") not in ACTIVE_WORKER_STATES:
            continue
        used = int(w.get("current_concurrency") or 0)
        total += max(0, max_concurrency - used)
    return total


def should_scale_up(inp: AutoscalerInput) -> bool:
    """True when policy says we need more workers."""
    if inp.queue_depth <= 0:
        return False
    free = free_concurrency_slots(inp.workers, inp.max_concurrency)
    if free > 0:
        return False
    if inp.scaling_policy_type == SCALING_QUEUE_DELAY:
        return inp.max_queue_wait_sec > float(inp.scaling_policy_value)
    # queue_request_count (default)
    return inp.queue_depth > inp.scaling_policy_value


def compute_desired_workers(inp: AutoscalerInput) -> int:
    """
    Deterministic desired worker count for one reconcile tick.

    Scale-up is immediate on policy breach; scale-down is handled separately
    via idle_timeout in the service reconcile loop.
    """
    current = count_active_workers(inp.workers)
    desired = max(inp.min_workers, current)

    forecast_depth = inp.queue_depth
    if predictive_scaling_enabled() and inp.queue_depth_samples:
        fc = forecast_queue_depth_with_fallback(inp.queue_depth_samples, horizon_sec=60.0)
        forecast_depth = max(forecast_depth, int(fc.get("forecast_depth") or 0))

    if should_scale_up(inp):
        effective_depth = max(inp.queue_depth, forecast_depth)
        free = free_concurrency_slots(inp.workers, inp.max_concurrency)
        deficit = max(0, effective_depth - free)
        extra_workers = max(1, math.ceil(deficit / max(1, inp.max_concurrency)))
        desired = current + extra_workers

    desired = apply_prewarm_to_desired(
        desired,
        min_workers=inp.min_workers,
        forecast_depth=forecast_depth,
        max_concurrency=inp.max_concurrency,
        endpoint={"min_workers": inp.min_workers},
        active_workers=current,
    )
    return max(inp.min_workers, min(inp.max_workers, desired))


def scale_down_cooldown_active(workers: list[dict], *, now: float, cooldown_sec: int) -> bool:
    """True when a worker was recently allocated — suppress scale-down to reduce thrash."""
    for w in workers:
        alloc = float(w.get("allocated_at") or 0)
        if alloc and now - alloc < cooldown_sec:
            return True
    return False


def workers_to_mark_draining(
    workers: list[dict],
    *,
    desired: int,
    idle_timeout_sec: int,
    now: float,
) -> list[str]:
    """
    Idle workers eligible to enter DRAINING (scale-down phase 1).
    Skips workers already draining or still serving jobs.
    """
    active = [w for w in workers if w.get("state") in ACTIVE_WORKER_STATES]
    draining_count = sum(1 for w in workers if w.get("state") == WORKER_STATE_DRAINING)
    effective_active = len(active) - draining_count
    excess = effective_active - desired
    if excess <= 0:
        return []

    idle_candidates: list[tuple[float, str]] = []
    for w in active:
        if w.get("state") == WORKER_STATE_DRAINING:
            continue
        if int(w.get("current_concurrency") or 0) > 0:
            continue
        last = float(w.get("last_heartbeat_at") or w.get("updated_at") or 0)
        idle_for = now - last
        if idle_for >= idle_timeout_sec:
            idle_candidates.append((idle_for, str(w["worker_id"])))

    idle_candidates.sort(reverse=True)
    return [wid for _, wid in idle_candidates[:excess]]


def workers_to_reap(
    workers: list[dict],
    *,
    desired: int,
    drain_grace_sec: int,
    now: float,
) -> list[str]:
    """
    DRAINING workers with zero concurrency past grace — ready for deprovision (phase 2).
    """
    active = [w for w in workers if w.get("state") in ACTIVE_WORKER_STATES]
    excess = len(active) - desired
    if excess <= 0:
        return []

    reap_candidates: list[tuple[float, str]] = []
    for w in workers:
        if w.get("state") != WORKER_STATE_DRAINING:
            continue
        if int(w.get("current_concurrency") or 0) > 0:
            continue
        since_drain = now - float(w.get("updated_at") or 0)
        if since_drain >= drain_grace_sec:
            reap_candidates.append((since_drain, str(w["worker_id"])))

    reap_candidates.sort(reverse=True)
    return [wid for _, wid in reap_candidates[:excess]]


def workers_to_scale_down(
    workers: list[dict],
    *,
    desired: int,
    idle_timeout_sec: int,
    now: float,
) -> list[str]:
    """Backward-compatible alias — returns workers ready to reap after drain."""
    return workers_to_reap(
        workers,
        desired=desired,
        drain_grace_sec=0,
        now=now,
    )
