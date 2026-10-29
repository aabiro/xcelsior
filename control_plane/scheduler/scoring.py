"""Stage D — deterministic integer scoring (blueprint §10.4).

Scores are fixed-point integers (scaled by 1000), never floats, so two
replicas ranking the same snapshot always produce the identical order.
Ties break on a stable hash of ``(job_id, host_id, inventory_generation)``
— deterministic across replicas, but different per job so one host does
not win every tie fleet-wide.

Every component value is preserved in the breakdown for the placement
explanation (§3.2 `explain_instance_placement`); the total is what ranks.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

SCORING_POLICY_VERSION = "scoring/v1"

_SCALE = 1000

# Component weights (integer). Chosen to keep cost dominant, matching the
# current scheduler's cheapest-first behavior, with packing and
# reliability as secondary signals.
_WEIGHT_COST = 5
_WEIGHT_PACKING = 3
_WEIGHT_RELIABILITY = 2


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ScoreBreakdown:
    host_id: str
    components: dict[str, int] = field(default_factory=dict)
    total: int = 0
    tie_break: int = 0
    policy_version: str = SCORING_POLICY_VERSION


def deterministic_tie_break(job_id: str, host_id: str, inventory_generation: int) -> int:
    """Stable 32-bit tie-break — sha256, never Python's seeded hash()."""
    digest = hashlib.sha256(
        f"{job_id}|{host_id}|{inventory_generation}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _cost_component(job: dict, host: dict) -> int:
    """Cheaper hosts score higher. Normalized against the approved or
    observed price so the component stays in a stable 0.._SCALE range."""
    cost = _f(host.get("cost_per_hour"), 0.0)
    ceiling = _f(job.get("max_price_per_hour"), 0.0)
    if ceiling <= 0:
        # No approved ceiling: normalize against cost+1 so free hosts get
        # full marks and expensive hosts asymptotically approach zero.
        ceiling = cost + 1.0
    if cost <= 0:
        return _SCALE
    ratio = min(1.0, cost / ceiling)
    return int(round((1.0 - ratio) * _SCALE))


def _packing_component(job: dict, host: dict) -> int:
    """Prefer the host whose free VRAM most tightly fits the request —
    §10.4 fragmentation minimization (leave big slots for big jobs)."""
    needed = _f(job.get("vram_needed_gb"), 0.0)
    free = _f(host.get("free_vram_gb"), 0.0)
    if needed <= 0 or free <= 0:
        return 0
    leftover_ratio = max(0.0, (free - needed) / free)
    return int(round((1.0 - leftover_ratio) * _SCALE))


def _reliability_component(job: dict, host: dict) -> int:
    """Host reliability in [0, 1] from the reputation snapshot, if present."""
    reliability = _f(host.get("reliability_score"), -1.0)
    if reliability < 0:
        return _SCALE // 2  # unknown: neutral, neither rewarded nor punished
    return int(round(max(0.0, min(1.0, reliability)) * _SCALE))


def score_host(job: dict, host: dict) -> ScoreBreakdown:
    host_id = str(host.get("host_id"))
    components = {
        "cost": _WEIGHT_COST * _cost_component(job, host),
        "packing": _WEIGHT_PACKING * _packing_component(job, host),
        "reliability": _WEIGHT_RELIABILITY * _reliability_component(job, host),
    }
    return ScoreBreakdown(
        host_id=host_id,
        components=components,
        total=sum(components.values()),
        tie_break=deterministic_tie_break(
            str(job.get("job_id") or job.get("name") or ""),
            host_id,
            int(_f(host.get("inventory_generation"), 0)),
        ),
    )


def rank_candidates(job: dict, hosts: list[dict]) -> list[ScoreBreakdown]:
    """Best-first ranking; total order is fully deterministic (§10.4)."""
    scored = [score_host(job, host) for host in hosts]
    scored.sort(key=lambda s: (-s.total, -s.tie_break, s.host_id))
    return scored
