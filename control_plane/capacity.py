"""Host free-capacity (VRAM) reconciliation helpers.

Single authority for repairing drifted ``hosts.payload.free_vram_gb`` from
ground-truth running reservations. Does not place, fail, or requeue jobs
and does not touch attempt/lease/allocation rows.
"""

from __future__ import annotations

from typing import Any

# Drift below this (GB) is treated as noise / float noise.
VRAM_DRIFT_TOLERANCE_GB = 0.01


def vram_used_by_host(jobs: list[dict[str, Any]]) -> dict[str, float]:
    """Sum reserved VRAM per host from *running* jobs only.

    Assigned/leased jobs have not reserved VRAM in the DB yet — counting
    their ``vram_needed_gb`` would over-count and block legitimate work.
    """
    used: dict[str, float] = {}
    for j in jobs:
        if j.get("status") != "running":
            continue
        hid = j.get("host_id")
        reserved = float(j.get("vram_reserved_gb", 0) or 0)
        if hid and reserved > 0:
            used[str(hid)] = used.get(str(hid), 0.0) + reserved
    return used


def expected_free_vram_gb(total_vram_gb: float, used_vram_gb: float) -> float:
    total = float(total_vram_gb or 0)
    used = float(used_vram_gb or 0)
    return round(max(0.0, min(total, total - used)), 4)


def free_vram_correction(
    current_free_gb: float,
    expected_free_gb: float,
    *,
    tolerance_gb: float = VRAM_DRIFT_TOLERANCE_GB,
) -> float | None:
    """Return the delta (expected - current) when drift exceeds tolerance.

    None means no write is needed.
    """
    drift = abs(float(current_free_gb) - float(expected_free_gb))
    if drift <= tolerance_gb:
        return None
    return round(float(expected_free_gb) - float(current_free_gb), 4)
