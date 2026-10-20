"""Aggregate worker counters without discarding live/custom gauge collectors.

The standard multiprocess collector cannot evaluate set_function gauges, Info,
Enum or custom collectors. Keep those on their original registry and replace
only additive families with the public multiprocess collector's merged values.
"""

from __future__ import annotations

import math
import os
import time

from prometheus_client import CollectorRegistry, REGISTRY, generate_latest, multiprocess, values
from prometheus_client.metrics_core import GaugeMetricFamily
from prometheus_client.registry import Collector

_IMPORTED_AT = time.time()
_ADDITIVE_TYPES = frozenset({"counter", "histogram", "summary"})
_START_HELP = "Start of this API master metrics generation; changes invalidate a zero-use window"


def validate_multiprocess_setup(workers: int, *, forking_server: bool = False) -> None:
    path = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    valid = (
        bool(path)
        and os.path.isdir(path)
        and bool(getattr(values.ValueClass, "_multiprocess", False))
    )
    if not path and workers <= 1 and not forking_server:
        return
    try:
        started = float(os.environ.get("XCELSIOR_METRICS_GENERATION_STARTED", ""))
    except ValueError:
        started = 0
    if not valid or not math.isfinite(started) or started <= 0:
        raise RuntimeError(
            "Shared API metrics must be initialized before prometheus_client is imported. "
            "Start the API through /bin/sh api/scripts/start_api.sh."
        )


class _MergedRegistry(Collector):
    def __init__(self, registry: CollectorRegistry, path: str, started: float):
        self.registry = registry
        self.path = path
        self.started = started

    def collect(self):
        shared = CollectorRegistry()
        multiprocess.MultiProcessCollector(shared, path=self.path)
        additive = {
            metric.name: metric for metric in shared.collect() if metric.type in _ADDITIVE_TYPES
        }
        if "xcelsior_legacy_execution" not in additive:
            raise RuntimeError("Shared legacy usage counters are missing")
        for metric in self.registry.collect():
            if metric.name not in additive:
                yield metric
        yield from additive.values()
        yield GaugeMetricFamily(
            "xcelsior_metrics_generation_started_timestamp_seconds",
            _START_HELP,
            value=self.started,
        )
        yield GaugeMetricFamily(
            "xcelsior_metrics_aggregation_enabled",
            "Whether this scrape aggregates shared worker counters",
            value=1,
        )


def generate_application_metrics(registry: CollectorRegistry = REGISTRY) -> bytes:
    validate_multiprocess_setup(int(os.environ.get("GUNICORN_WORKERS", "1")))
    path = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if path:
        # The generation is supplied before Python starts, so preload and every
        # worker observe one timestamp, including replacements after SIGKILL.
        started = float(os.environ["XCELSIOR_METRICS_GENERATION_STARTED"])
        merged = CollectorRegistry()
        merged.register(_MergedRegistry(registry, path, started))
        return generate_latest(merged)
    # A single-process development/test server needs no shared file storage.
    result = generate_latest(registry)
    generation = CollectorRegistry()
    generation.register(_SingleGeneration())
    return result + generate_latest(generation)


class _SingleGeneration(Collector):
    def collect(self):
        yield GaugeMetricFamily(
            "xcelsior_metrics_generation_started_timestamp_seconds",
            _START_HELP,
            value=_IMPORTED_AT,
        )
        yield GaugeMetricFamily(
            "xcelsior_metrics_aggregation_enabled",
            "Whether this scrape aggregates shared worker counters",
            value=0,
        )
