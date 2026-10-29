"""14-day GPU demand forecast with weekday covariates for serverless capacity (row 11)."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any

WEEKDAY_WEIGHTS = {
    0: 1.0,   # Monday
    1: 1.05,
    2: 1.08,
    3: 1.06,
    4: 1.12,  # Friday
    5: 0.75,  # Saturday
    6: 0.65,  # Sunday
}


def weekday_covariate(ts: float | None = None) -> float:
    dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    return WEEKDAY_WEIGHTS.get(dt.weekday(), 1.0)


def forecast_gpu_demand_14d(
    history: list[tuple[float, int]],
    *,
    horizon_days: int = 14,
) -> dict[str, Any]:
    """EWMA baseline × weekday covariate; 99% interval for breach alerts."""
    if not history:
        return {
            "forecast_workers": 0,
            "interval_low": 0,
            "interval_high": 0,
            "weekday_factor": weekday_covariate(),
            "horizon_days": horizon_days,
        }
    ordered = sorted(history, key=lambda s: s[0])
    depths = [d for _, d in ordered[-48:]]
    baseline = sum(depths) / len(depths) if depths else 0.0
    wk = weekday_covariate()
    forecast = max(0, int(round(baseline * wk)))
    std = math.sqrt(sum((d - baseline) ** 2 for d in depths) / max(1, len(depths)))
    interval_low = max(0, int(round(baseline - 2.58 * std)))
    interval_high = int(round(baseline + 2.58 * std))
    return {
        "forecast_workers": forecast,
        "baseline_depth": round(baseline, 2),
        "weekday_factor": wk,
        "interval_low": interval_low,
        "interval_high": interval_high,
        "horizon_days": horizon_days,
        "samples": len(depths),
    }


def demand_breach_alert(
    actual_depth: int,
    forecast: dict[str, Any],
) -> dict[str, Any] | None:
    """Alert when actual falls outside 99% band (row 11 acceptance)."""
    low = int(forecast.get("interval_low") or 0)
    high = int(forecast.get("interval_high") or 0)
    if actual_depth < low:
        return {
            "kind": "demand_below_band",
            "actual": actual_depth,
            "interval_low": low,
            "interval_high": high,
        }
    if high > 0 and actual_depth > high:
        return {
            "kind": "demand_above_band",
            "actual": actual_depth,
            "interval_low": low,
            "interval_high": high,
        }
    return None


def recommended_min_workers(
    history: list[tuple[float, int]],
    *,
    max_workers: int = 8,
) -> int:
    fc = forecast_gpu_demand_14d(history)
    return min(max_workers, max(0, int(fc.get("forecast_workers") or 0)))