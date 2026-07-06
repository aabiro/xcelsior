# Xcelsior — Serverless observability (metrics, structured logs, Prometheus)

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serverless.repo import ServerlessRepo

log = logging.getLogger("xcelsior.serverless.observability")

try:
    from prometheus_client import Counter as _Counter
    from prometheus_client import Gauge as _Gauge

    _queue_depth_gauge = _Gauge(
        "serverless_queue_depth",
        "Queued jobs per serverless endpoint",
        ["endpoint_id"],
    )
    _active_workers_gauge = _Gauge(
        "serverless_active_workers",
        "Non-terminated workers per serverless endpoint",
        ["endpoint_id"],
    )
    _idle_workers_gauge = _Gauge(
        "serverless_idle_workers",
        "Ready/idle workers with zero concurrency",
        ["endpoint_id"],
    )
    _cold_starts_total = _Counter(
        "serverless_cold_starts_total",
        "Cold-start events recorded on first job per worker",
        ["endpoint_id"],
    )
    _job_errors_total = _Counter(
        "serverless_job_errors_total",
        "Terminal serverless job failures",
        ["endpoint_id"],
    )
    _jobs_completed_total = _Counter(
        "serverless_jobs_completed_total",
        "Terminal serverless job completions",
        ["endpoint_id"],
    )
except Exception:

    class _NoopMetric:
        def labels(self, *a, **kw):
            return self

        def set(self, *a, **kw):
            pass

        def inc(self, *a, **kw):
            pass

    _queue_depth_gauge = _NoopMetric()
    _active_workers_gauge = _NoopMetric()
    _idle_workers_gauge = _NoopMetric()
    _cold_starts_total = _NoopMetric()
    _job_errors_total = _NoopMetric()
    _jobs_completed_total = _NoopMetric()

_READY_IDLE = frozenset({"ready", "idle"})


def resolve_correlation_id(
    request_headers: dict[str, str] | None,
    *,
    job_id: str,
) -> str:
    """Prefer client correlation headers; fall back to job_id."""
    if request_headers:
        for key in ("x-correlation-id", "x-request-id", "idempotency-key"):
            val = (request_headers.get(key) or "").strip()
            if val:
                return val[:128]
    return job_id


def log_job_event(
    event: str,
    *,
    correlation_id: str,
    job_id: str,
    endpoint_id: str,
    **fields: Any,
) -> None:
    """Structured per-job log line (grep-friendly key=value)."""
    parts = [
        f"event={event}",
        f"correlation_id={correlation_id}",
        f"job_id={job_id}",
        f"endpoint_id={endpoint_id}",
    ]
    for k, v in sorted(fields.items()):
        if v is not None:
            parts.append(f"{k}={v}")
    log.info("serverless %s", " ".join(parts))


def _job_queue_ms(job: dict) -> float | None:
    queued = float(job.get("queued_at") or 0)
    started = float(job.get("started_at") or 0)
    if queued > 0 and started > 0:
        return max(0.0, (started - queued) * 1000.0)
    return None


def _job_execution_ms(job: dict) -> float | None:
    started = float(job.get("started_at") or 0)
    finished = float(job.get("finished_at") or 0)
    if started > 0 and finished > 0:
        return max(0.0, (finished - started) * 1000.0)
    return None


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1))
    return round(ordered[idx], 2)


def worker_fleet_stats(workers: list[dict]) -> dict[str, int]:
    active_states = {"booting", "ready", "idle", "draining"}
    active = sum(1 for w in workers if str(w.get("state") or "") in active_states)
    idle = sum(
        1
        for w in workers
        if str(w.get("state") or "") in _READY_IDLE
        and int(w.get("current_concurrency") or 0) == 0
    )
    busy = sum(
        1
        for w in workers
        if str(w.get("state") or "") in _READY_IDLE
        and int(w.get("current_concurrency") or 0) > 0
    )
    booting = sum(1 for w in workers if str(w.get("state") or "") == "booting")
    return {
        "active_workers": active,
        "idle_workers": idle,
        "busy_workers": busy,
        "booting_workers": booting,
        "workers_total": len(workers),
    }


def compute_endpoint_metrics(
    ep: dict,
    jobs: list[dict],
    workers: list[dict],
    *,
    queue_depth: int,
    window_sec: float,
) -> dict[str, Any]:
    """Aggregate per-endpoint metrics for dashboard + /metrics API."""
    completed = [j for j in jobs if str(j.get("status")) == "COMPLETED"]
    failed = [j for j in jobs if str(j.get("status")) == "FAILED"]
    cancelled = [j for j in jobs if str(j.get("status")) == "CANCELLED"]
    terminal = completed + failed + cancelled
    window_requests = len(terminal)

    queue_ms_vals = [v for j in terminal if (v := _job_queue_ms(j)) is not None]
    exec_ms_vals = [v for j in terminal if (v := _job_execution_ms(j)) is not None]

    total_out_tokens = sum(int(j.get("output_tokens") or 0) for j in completed)
    total_in_tokens = sum(int(j.get("input_tokens") or 0) for j in completed)
    total_cached_tokens = sum(int(j.get("cached_tokens") or 0) for j in completed)
    total_exec_sec = sum(
        max(1, int(j.get("gpu_seconds") or 0)) for j in completed
    )
    tokens_per_sec = (
        round(total_out_tokens / total_exec_sec, 2) if total_exec_sec > 0 else 0.0
    )
    ttft_vals = [
        float(j.get("ttft_ms") or 0)
        for j in completed
        if int(j.get("ttft_ms") or 0) > 0
    ]
    kv_cache_hit_rate = (
        round(total_cached_tokens / total_in_tokens, 4)
        if total_in_tokens > 0
        else 0.0
    )

    success_rate = (
        round(len(completed) / window_requests, 4) if window_requests else 0.0
    )
    error_rate = round(len(failed) / window_requests, 4) if window_requests else 0.0

    fleet = worker_fleet_stats(workers)
    return {
        "endpoint_id": ep.get("endpoint_id"),
        "window_sec": round(window_sec, 2),
        "total_requests": int(ep.get("total_requests") or 0),
        "window_requests": window_requests,
        "jobs_completed": len(completed),
        "jobs_failed": len(failed),
        "jobs_cancelled": len(cancelled),
        "success_rate": success_rate,
        "error_rate": error_rate,
        "queue_depth": queue_depth,
        "avg_queue_ms": _avg(queue_ms_vals),
        "avg_execution_ms": _avg(exec_ms_vals),
        "avg_gpu_seconds": _avg(
            [float(j.get("gpu_seconds") or 0) for j in completed]
        ),
        "total_gpu_seconds": int(ep.get("total_gpu_seconds") or 0),
        "tokens_per_sec": tokens_per_sec,
        "ttft_p95_ms": _p95(ttft_vals),
        "kv_cache_hit_rate": kv_cache_hit_rate,
        "total_input_tokens": total_in_tokens,
        "total_cached_tokens": total_cached_tokens,
        "total_output_tokens": total_out_tokens,
        "total_cost_cad": float(ep.get("total_cost_cad") or 0),
        **fleet,
    }


def refresh_endpoint_gauges(
    repo: ServerlessRepo,
    endpoint_id: str,
    *,
    workers: list[dict] | None = None,
) -> None:
    """Update Prometheus gauges for one endpoint."""
    depth = repo.queue_depth(endpoint_id)
    if workers is None:
        workers = repo.list_workers(endpoint_id)
    fleet = worker_fleet_stats(workers)
    _queue_depth_gauge.labels(endpoint_id=endpoint_id).set(depth)
    _active_workers_gauge.labels(endpoint_id=endpoint_id).set(fleet["active_workers"])
    _idle_workers_gauge.labels(endpoint_id=endpoint_id).set(fleet["idle_workers"])


def record_cold_start(endpoint_id: str) -> None:
    _cold_starts_total.labels(endpoint_id=endpoint_id).inc()


def record_job_terminal(endpoint_id: str, *, failed: bool) -> None:
    if failed:
        _job_errors_total.labels(endpoint_id=endpoint_id).inc()
    else:
        _jobs_completed_total.labels(endpoint_id=endpoint_id).inc()


def refresh_all_gauges(repo: ServerlessRepo) -> int:
    """Refresh gauges for every active endpoint (reconcile tick)."""
    count = 0
    for ep in repo.list_endpoints_for_reconcile():
        eid = str(ep["endpoint_id"])
        refresh_endpoint_gauges(repo, eid)
        count += 1
    return count