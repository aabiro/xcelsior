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
    ewma_alpha: float = 0.3,
) -> int:
    """EWMA-smoothed extrapolation of queue depth from (timestamp, depth) samples.

    A naive two-point derivative (last two samples only) is noisy under
    bursty serverless traffic — a single spike or dip between the two most
    recent reconcile ticks can wildly over- or under-predict growth. This
    instead walks the full sample history (up to service.py's
    `_QUEUE_HISTORY_MAX` = 20 samples), computes the per-interval growth
    rate for every consecutive pair, and folds them into an exponentially
    weighted moving average (recent intervals weighted more via
    `ewma_alpha`) before projecting the latest depth forward by
    horizon_sec at that smoothed rate.

    Returns predicted depth at now+horizon_sec, clamped to >= 0.
    With fewer than two samples, returns the latest depth or 0.
    """
    if not samples:
        return 0
    if len(samples) == 1:
        return max(0, samples[-1][1])
    ordered = sorted(samples, key=lambda s: s[0])
    ewma_rate = 0.0
    have_rate = False
    for (t0, d0), (t1, d1) in zip(ordered, ordered[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        rate = (d1 - d0) / dt
        ewma_rate = rate if not have_rate else ewma_alpha * rate + (1 - ewma_alpha) * ewma_rate
        have_rate = True
    if not have_rate:
        return max(0, ordered[-1][1])
    predicted = ordered[-1][1] + ewma_rate * horizon_sec
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