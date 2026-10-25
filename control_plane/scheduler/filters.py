"""Stage C — pure, versioned hard filters (blueprint §10.3).

Every filter is a pure function over job/host snapshots: no I/O, no
clocks (freshness cutoffs are computed by the caller and passed in), no
randomness. A filter returns ``None`` for pass or a typed
:class:`FilterReason` — never a bare boolean — so queued jobs get a
durable, explainable non-placement reason (§8 invariant 10).

``FILTER_POLICY_VERSION`` is persisted with every placement decision so
an operator can tell which rule set produced a historical explanation.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

FILTER_POLICY_VERSION = "filters/v5"


@dataclass(frozen=True)
class FilterReason:
    """One failed hard constraint, with remediation context."""

    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FilterContext:
    """Caller-supplied facts the pure filters may not compute themselves."""

    # Hosts whose last observation is older than this are not schedulable
    # (§9.3); the caller derives it from DB time, filters just compare.
    stale_host_ids: frozenset[str] = frozenset()


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ── Individual hard filters ──────────────────────────────────────────


def host_pin(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    wanted = job.get("preferred_host_id")
    if wanted and str(host.get("host_id")) != str(wanted):
        return FilterReason(
            "host_pin_mismatch",
            "job is pinned to another host",
            {"requested_host_id": wanted},
        )
    return None


def host_admitted(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    state = _norm(host.get("administrative_state") or "admitted")
    if state != "admitted":
        return FilterReason(
            "host_not_admitted",
            f"host is {state}",
            {"administrative_state": state},
        )
    return None


def host_ready(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    if _norm(host.get("status")) not in ("active", "ready"):
        return FilterReason(
            "host_not_ready",
            f"host status is {_norm(host.get('status')) or 'unknown'}",
            {"status": host.get("status")},
        )
    if str(host.get("host_id")) in ctx.stale_host_ids:
        return FilterReason(
            "host_observation_stale",
            "host heartbeat/inventory is stale",
            {"host_id": host.get("host_id")},
        )
    return None


def gpu_model(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    wanted = _norm(job.get("gpu_model"))
    if not wanted:
        return None
    have = _norm(host.get("gpu_model"))
    if wanted not in have:
        return FilterReason(
            "gpu_model_mismatch",
            f"requires {job.get('gpu_model')}, host has {host.get('gpu_model')}",
            {"requested": job.get("gpu_model"), "available": host.get("gpu_model")},
        )
    return None


def gpu_count(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    needed = int(_f(job.get("num_gpus"), 1) or 1)
    free = int(_f(host.get("free_gpu_count", host.get("gpu_count")), 0))
    if free < needed:
        return FilterReason(
            "insufficient_gpus",
            f"needs {needed} GPUs, host has {free} free",
            {"requested": needed, "free": free},
        )
    return None


def vram(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    needed_gb = _f(job.get("vram_needed_gb"))
    if needed_gb <= 0:
        return None
    free_gb = _f(host.get("free_vram_gb"))
    if free_gb < needed_gb:
        return FilterReason(
            "insufficient_vram",
            f"needs {needed_gb:g} GB VRAM, host has {free_gb:g} GB free",
            {"requested_gb": needed_gb, "free_gb": free_gb},
        )
    return None


def region(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    wanted = _norm(job.get("region"))
    if not wanted:
        return None
    have = _norm(host.get("region"))
    if have != wanted:
        return FilterReason(
            "region_mismatch",
            f"requires region {job.get('region')}, host is in {host.get('region') or 'unknown'}",
            {"requested": job.get("region"), "host_region": host.get("region")},
        )
    return None


def egress_isolation(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    # Anything but an explicit, server-set "open" runs on the isolated network,
    # which only hosts that verified it on their last heartbeat can provide.
    if _norm(job.get("egress")) == "open":
        return None
    caps = host.get("capabilities")
    if isinstance(caps, dict) and caps.get("egress_isolation") is True:
        return None
    return FilterReason(
        "egress_isolation_unavailable",
        "host has not verified network isolation for tenant containers",
        {"egress": _norm(job.get("egress")) or "restricted"},
    )


def spot_capacity(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    if _norm(job.get("pricing_mode")) != "spot":
        return None
    if host.get("spot_enabled") is not True:
        return FilterReason("spot_disabled", "host does not enable spot workloads")
    slots, total = host.get("spot_gpu_slots"), host.get("gpu_count")
    if type(slots) is not int or type(total) is not int or min(slots, total) < 0:
        return FilterReason("spot_capacity_unqualified", "host has no qualified spot pool")
    floor_cents = host.get("spot_min_cents", 0)
    rate = job.get("spot_rate_cad")
    if type(floor_cents) is not int or floor_cents < 0:
        return FilterReason("spot_capacity_unqualified", "host spot floor is unqualified")
    if rate is None or isinstance(rate, bool) or _f(rate, -1) < floor_cents / 100:
        return FilterReason(
            "spot_floor_exceeds_admitted", "host spot floor exceeds the admitted rate"
        )
    used, needed = int(host.get("active_spot_gpu_count") or 0), int(job.get("num_gpus") or 1)
    if used + needed > min(slots, total):
        return FilterReason(
            "spot_capacity_exhausted",
            "host spot pool has no free GPU slots",
            {"slots": min(slots, total), "allocated": used, "requested": needed},
        )
    return None


def price(job: dict, host: dict, ctx: FilterContext) -> FilterReason | None:
    raw_maximum = job.get("max_price_per_hour")
    max_price = 0.0 if raw_maximum is None else _f(raw_maximum, -1.0)
    if isinstance(raw_maximum, bool) or not math.isfinite(max_price) or max_price < 0:
        return FilterReason("price_approval_unqualified", "approved maximum price is invalid")
    if max_price <= 0:
        return None
    raw_cost = host.get("cost_per_hour")
    cost = _f(raw_cost, -1.0)
    if isinstance(raw_cost, bool) or not math.isfinite(cost) or cost < 0:
        return FilterReason("host_price_unqualified", "host price is unavailable or invalid")
    if cost > max_price:
        return FilterReason(
            "price_exceeds_approved",
            f"host costs {cost:g}/h, approved maximum is {max_price:g}/h",
            {"cost_per_hour": cost, "max_price_per_hour": max_price},
        )
    return None


# Order matters only for explanation readability; all filters always run
# so the queue reason lists *every* failed constraint, not just the first.
HARD_FILTERS: Sequence[Callable[[dict, dict, FilterContext], FilterReason | None]] = (
    host_pin,
    host_admitted,
    host_ready,
    gpu_model,
    gpu_count,
    vram,
    region,
    egress_isolation,
    spot_capacity,
    price,
)


def evaluate_host(job: dict, host: dict, ctx: FilterContext | None = None) -> list[FilterReason]:
    """All failed hard constraints for this (job, host); empty = eligible."""
    ctx = ctx or FilterContext()
    reasons = []
    for check in HARD_FILTERS:
        reason = check(job, host, ctx)
        if reason is not None:
            reasons.append(reason)
    return reasons


def filter_hosts(
    job: dict, hosts: Sequence[dict], ctx: FilterContext | None = None
) -> tuple[list[dict], dict[str, list[FilterReason]]]:
    """Split hosts into eligible candidates and per-host typed rejections."""
    ctx = ctx or FilterContext()
    eligible: list[dict] = []
    rejections: dict[str, list[FilterReason]] = {}
    for host in hosts:
        reasons = evaluate_host(job, host, ctx)
        if reasons:
            rejections[str(host.get("host_id"))] = reasons
        else:
            eligible.append(host)
    return eligible, rejections


def aggregate_reason(rejections: dict[str, list[FilterReason]]) -> dict[str, Any]:
    """Durable queue-reason payload: which constraints failed, how often."""
    counts: dict[str, int] = {}
    for reasons in rejections.values():
        for reason in reasons:
            counts[reason.code] = counts.get(reason.code, 0) + 1
    return {
        "policy_version": FILTER_POLICY_VERSION,
        "hosts_evaluated": len(rejections),
        "failed_constraints": counts,
    }
