"""Ops telemetry forecast with Toto 2.0-style fallback when Chronos/EWMA underfits."""

from __future__ import annotations

import os
from typing import Any

from serverless.scaling_predictive import forecast_queue_depth


def toto_fallback_enabled() -> bool:
    return os.environ.get("XCELSIOR_TOTO_FORECAST_FALLBACK", "1").lower() in (
        "1",
        "true",
        "yes",
    )


def _simple_moving_average(samples: list[tuple[float, int]], *, window: int = 5) -> int:
    if not samples:
        return 0
    depths = [d for _, d in sorted(samples, key=lambda s: s[0])[-window:]]
    return max(0, int(round(sum(depths) / len(depths))))


def _ewma_error(samples: list[tuple[float, int]]) -> float:
    """Mean absolute error of one-step EWMA vs actual (underfit signal)."""
    if len(samples) < 4:
        return 0.0
    ordered = sorted(samples, key=lambda s: s[0])
    errors: list[float] = []
    alpha = 0.3
    pred = float(ordered[0][1])
    for (_, actual), (t1, _) in zip(ordered[1:], ordered[2:]):
        errors.append(abs(pred - actual))
        pred = alpha * actual + (1 - alpha) * pred
    return sum(errors) / len(errors) if errors else 0.0


def forecast_queue_depth_with_fallback(
    samples: list[tuple[float, int]],
    *,
    horizon_sec: float = 60.0,
    underfit_threshold: float | None = None,
) -> dict[str, Any]:
    """EWMA forecast with Toto-style SMA fallback when error exceeds threshold."""
    threshold = underfit_threshold
    if threshold is None:
        threshold = float(os.environ.get("XCELSIOR_FORECAST_UNDERFIT_THRESHOLD", "2.0"))
    ewma = forecast_queue_depth(samples, horizon_sec=horizon_sec)
    model = "chronos-ewma"
    if toto_fallback_enabled() and len(samples) >= 4:
        err = _ewma_error(samples)
        if err > threshold:
            sma = _simple_moving_average(samples)
            blended = max(ewma, sma, int(round(sma * 1.1)))
            return {
                "forecast_depth": blended,
                "model": "toto-2.0-sma-fallback",
                "ewma_forecast": ewma,
                "sma_forecast": sma,
                "ewma_error": round(err, 3),
            }
    return {"forecast_depth": ewma, "model": model, "ewma_forecast": ewma}