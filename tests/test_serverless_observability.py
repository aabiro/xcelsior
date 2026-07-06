"""Phase 10 — serverless observability (metrics, SSE, Prometheus, logs)."""

import os
import time
import uuid
from unittest.mock import patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.observability import (
    compute_endpoint_metrics,
    record_cold_start,
    record_job_terminal,
    resolve_correlation_id,
    worker_fleet_stats,
)
from serverless.repo import (
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    EndpointCreate,
    ServerlessRepo,
    WORKER_STATE_IDLE,
    WORKER_STATE_READY,
)
from serverless.service import ServerlessService


class TestCorrelationAndMetrics:
    def test_resolve_correlation_id_prefers_header(self):
        cid = resolve_correlation_id(
            {"x-correlation-id": "corr-abc"},
            job_id="sjob-1",
        )
        assert cid == "corr-abc"

    def test_resolve_correlation_id_falls_back_to_job_id(self):
        assert resolve_correlation_id(None, job_id="sjob-99") == "sjob-99"

    def test_compute_endpoint_metrics_window(self):
        now = time.time()
        ep = {
            "endpoint_id": "sep-1",
            "total_requests": 10,
            "total_gpu_seconds": 100,
            "total_cost_cad": 1.5,
        }
        jobs = [
            {
                "status": JOB_STATUS_COMPLETED,
                "queued_at": now - 30,
                "started_at": now - 20,
                "finished_at": now - 10,
                "gpu_seconds": 10,
                "output_tokens": 100,
            },
            {
                "status": JOB_STATUS_FAILED,
                "queued_at": now - 25,
                "started_at": now - 15,
                "finished_at": now - 5,
                "gpu_seconds": 5,
                "output_tokens": 0,
            },
        ]
        workers = [
            {"state": WORKER_STATE_READY, "current_concurrency": 0},
            {"state": WORKER_STATE_IDLE, "current_concurrency": 1},
        ]
        m = compute_endpoint_metrics(
            ep,
            jobs,
            workers,
            queue_depth=2,
            window_sec=3600,
        )
        assert m["window_requests"] == 2
        assert m["jobs_completed"] == 1
        assert m["jobs_failed"] == 1
        assert m["success_rate"] == 0.5
        assert m["error_rate"] == 0.5
        assert m["queue_depth"] == 2
        assert m["avg_queue_ms"] == 10000.0
        assert m["avg_execution_ms"] == 10000.0
        assert m["tokens_per_sec"] > 0
        assert m["idle_workers"] == 1
        assert m["busy_workers"] == 1

    def test_compute_endpoint_metrics_includes_token_ledger(self):
        ep = {"endpoint_id": "sep-ledger", "total_requests": 5}
        ledger = [
            {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cached_tokens": 400,
                "ttft_ms": 120,
                "latency_ms": 800,
            },
            {
                "input_tokens": 500,
                "output_tokens": 100,
                "cached_tokens": 100,
                "ttft_ms": 90,
                "latency_ms": 600,
            },
        ]
        m = compute_endpoint_metrics(
            ep,
            [],
            [],
            queue_depth=0,
            window_sec=3600,
            ledger_rows=ledger,
        )
        assert m["window_requests"] == 2
        assert m["total_input_tokens"] == 1500
        assert m["total_cached_tokens"] == 500
        assert m["kv_cache_hit_rate"] == round(500 / 1500, 4)
        assert m["ttft_p95_ms"] > 0
        assert m["tokens_per_sec"] > 0

    def test_worker_fleet_stats(self):
        stats = worker_fleet_stats(
            [
                {"state": "booting", "current_concurrency": 0},
                {"state": WORKER_STATE_READY, "current_concurrency": 0},
                {"state": WORKER_STATE_IDLE, "current_concurrency": 2},
            ]
        )
        assert stats["active_workers"] == 3
        assert stats["idle_workers"] == 1
        assert stats["busy_workers"] == 1


class TestPrometheusCounters:
    def test_record_counters_do_not_raise(self):
        record_cold_start("sep-metrics")
        record_job_terminal("sep-metrics", failed=False)
        record_job_terminal("sep-metrics", failed=True)


def _pg_ready() -> bool:
    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT tablename FROM pg_catalog.pg_tables "
            "WHERE schemaname = 'public' AND tablename = 'serverless_endpoints'"
        ).fetchone()
        return row is not None


@pytest.mark.skipif(not _pg_ready(), reason="Postgres test DB required")
class TestServiceObservabilityIntegration:
    def test_get_endpoint_metrics_from_repo(self):
        repo = ServerlessRepo()
        svc = ServerlessService(repo)
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="obs", mode="custom", image_ref="img")
        )
        ep_id = str(ep["endpoint_id"])
        now = time.time()
        job = repo.enqueue_job(ep_id, owner, {"x": 1})
        repo.complete_job(
            str(job["job_id"]),
            output={"ok": True},
            gpu_seconds=5,
            input_tokens=10,
            output_tokens=20,
        )
        repo.get_job(str(job["job_id"]))
        # backdate finished_at for window query
        with repo._conn() as conn:
            conn.execute(
                "UPDATE serverless_jobs SET finished_at = %s, started_at = %s, queued_at = %s WHERE job_id = %s",
                (now, now - 5, now - 10, job["job_id"]),
            )

        metrics = svc.get_endpoint_metrics(ep_id, since=now - 3600)
        assert metrics["endpoint_id"] == ep_id
        assert metrics["jobs_completed"] >= 1
        assert "avg_queue_ms" in metrics
        assert "active_workers" in metrics
        repo.soft_delete_endpoint(ep_id, owner)

    def test_sse_dot_notation_on_create(self):
        repo = ServerlessRepo()
        svc = ServerlessService(repo)
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        events: list[tuple[str, dict]] = []

        def _capture(event_type: str, data: dict) -> None:
            events.append((event_type, data))

        with (
            patch("routes._deps.broadcast_sse", side_effect=_capture),
            patch.object(svc, "wallet_preflight"),
        ):
            ep = svc.create_endpoint(
                EndpointCreate(owner_id=owner, name="sse", mode="custom", image_ref="img"),
                emit_sse=True,
            )
        assert any(t == "serverless_endpoint.created" for t, _ in events)
        repo.soft_delete_endpoint(str(ep["endpoint_id"]), owner)