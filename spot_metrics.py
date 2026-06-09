"""Prometheus metrics and observability snapshots for spot instances."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("xcelsior.spot_metrics")

try:
    from prometheus_client import Counter, Gauge

    _PROMETHEUS = True
    _spot_jobs_running = Gauge(
        "xcelsior_spot_jobs_running",
        "Number of running spot (interruptible) jobs",
    )
    _spot_preemptions_total = Counter(
        "xcelsior_spot_preemptions_total",
        "Total spot job preemptions",
    )
    _spot_rate_cad = Gauge(
        "xcelsior_spot_rate_cad",
        "Current unified spot rate in CAD per hour",
        ["gpu_model"],
    )
except Exception:
    _PROMETHEUS = False

    class _NoopMetric:
        def labels(self, *args, **kwargs):
            return self

        def set(self, *args, **kwargs):
            return None

        def inc(self, *args, **kwargs):
            return None

    _spot_jobs_running = _NoopMetric()
    _spot_preemptions_total = _NoopMetric()
    _spot_rate_cad = _NoopMetric()

_preemptions_count = 0


def record_spot_preemption() -> None:
    global _preemptions_count
    _preemptions_count += 1
    _spot_preemptions_total.inc()


def refresh_spot_gauges(jobs: list[dict] | None = None) -> dict[str, Any]:
    """Recompute spot gauges from scheduler state and live quotes."""
    if jobs is None:
        from scheduler import list_jobs

        jobs = list_jobs()

    running_spot = 0
    for job in jobs:
        if job.get("status") != "running":
            continue
        mode = (job.get("pricing_mode") or "").strip().lower()
        if mode == "spot" or job.get("preemptible") or job.get("spot"):
            running_spot += 1

    _spot_jobs_running.set(running_spot)

    rates: dict[str, float] = {}
    try:
        from spot_pricing import get_current_spot_prices

        rates = get_current_spot_prices() or {}
        for gpu_model, rate in rates.items():
            _spot_rate_cad.labels(gpu_model=gpu_model).set(float(rate))
    except Exception as exc:
        log.debug("Spot rate gauge refresh skipped: %s", exc)

    return {
        "jobs_running": running_spot,
        "preemptions_total": _preemptions_count,
        "rates_cad": {k: round(float(v), 4) for k, v in rates.items()},
    }


def get_spot_metrics_snapshot(jobs: list[dict] | None = None) -> dict[str, Any]:
    """Return spot metrics for JSON /metrics snapshot."""
    try:
        return refresh_spot_gauges(jobs)
    except Exception as exc:
        log.debug("Spot metrics snapshot failed: %s", exc)
        return {"jobs_running": 0, "preemptions_total": 0, "rates_cad": {}}