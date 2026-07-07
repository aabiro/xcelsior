"""Pre-warmed worker pools keyed to predicted demand and model-weight cache."""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger("xcelsior.serverless.prewarm_pool")


def prewarm_enabled() -> bool:
    return os.environ.get("XCELSIOR_PREWARM_POOL_ENABLED", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def model_weight_cached(endpoint: dict, *, active_workers: int) -> bool:
    """True when at least one worker is booted/ready (weights likely resident)."""
    return active_workers > 0 or int(endpoint.get("min_workers") or 0) > 0


def compute_prewarm_floor(
    *,
    min_workers: int,
    forecast_depth: int,
    max_concurrency: int,
    model_cached: bool,
) -> int:
    """Minimum warm workers from demand forecast + weight cache availability."""
    if not prewarm_enabled():
        return min_workers
    base = max(min_workers, 1 if forecast_depth > 0 else 0)
    if model_cached and forecast_depth > max_concurrency:
        extra = max(1, (forecast_depth + max_concurrency - 1) // max_concurrency)
        return min(base + extra, base + 2)
    return base


def apply_prewarm_to_desired(
    desired: int,
    *,
    min_workers: int,
    forecast_depth: int,
    max_concurrency: int,
    endpoint: dict,
    active_workers: int,
) -> int:
    floor = compute_prewarm_floor(
        min_workers=min_workers,
        forecast_depth=forecast_depth,
        max_concurrency=max_concurrency,
        model_cached=model_weight_cached(endpoint, active_workers=active_workers),
    )
    return max(desired, floor)