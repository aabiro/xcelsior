# Xcelsior — Predictive scaling hints (Phase 15 backlog starter)
#
# Vast.ai-style rate forecasting: extrapolate queue growth from recent samples
# so the autoscaler can scale up before wait-time SLAs are breached.
# Wired optionally from serverless/autoscaler.py when
# XCELSIOR_SERVERLESS_PREDICTIVE_SCALING=true.

from __future__ import annotations

import os


def predictive_scaling_enabled() -> bool:
    return os.environ.get("XCELSIOR_SERVERLESS_PREDICTIVE_SCALING", "").lower() in (
        "1",
        "true",
        "yes",
    )


def forecast_queue_depth(
    samples: list[tuple[float, int]],
    *,
    horizon_sec: float = 60.0,
) -> int:
    """Linear extrapolation of queue depth from (timestamp, depth) samples.

    Returns predicted depth at now+horizon_sec, clamped to >= 0.
    With fewer than two samples, returns the latest depth or 0.
    """
    if not samples:
        return 0
    if len(samples) == 1:
        return max(0, samples[-1][1])
    t0, d0 = samples[-2]
    t1, d1 = samples[-1]
    dt = t1 - t0
    if dt <= 0:
        return max(0, d1)
    rate = (d1 - d0) / dt
    predicted = d1 + rate * horizon_sec
    return max(0, int(round(predicted)))


def headroom_workers(
    *,
    queue_depth: int,
    active_workers: int,
    max_concurrency: int,
    forecast_depth: int,
) -> int:
    """Extra workers suggested beyond queue_depth / max_concurrency ceiling."""
    if not predictive_scaling_enabled():
        return 0
    effective_depth = max(queue_depth, forecast_depth)
    needed = (effective_depth + max_concurrency - 1) // max_concurrency
    return max(0, needed - active_workers)