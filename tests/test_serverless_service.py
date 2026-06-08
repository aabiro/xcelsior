"""Serverless service layer unit tests (no GPU required)."""

import os
import time
import uuid
from unittest.mock import patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.autoscaler import (
    AutoscalerInput,
    compute_desired_workers,
    free_concurrency_slots,
    should_scale_up,
    workers_to_mark_draining,
    workers_to_reap,
)
from serverless.keys import create_endpoint_key, key_has_scope, validate_key
from serverless.metering import compute_gpu_seconds, estimate_cost_cad, token_cost_metadata
from serverless.openai_proxy import (
    OpenAIProxyError,
    capability_gate,
    normalize_route,
    worker_base_url,
)
from serverless.repo import (
    EndpointCreate,
    ServerlessRepo,
    WORKER_STATE_DRAINING,
    WORKER_STATE_IDLE,
    WORKER_STATE_READY,
)
from serverless.service import ServerlessService, WalletPreflightError
from serverless.streams import format_done_chunk, format_sse_event


class TestAutoscalerMath:
    def test_no_scale_when_queue_empty(self):
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=4,
            max_concurrency=4,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=2,
            queue_depth=0,
            max_queue_wait_sec=0,
            workers=[{"state": "ready", "current_concurrency": 0}],
        )
        assert compute_desired_workers(inp) == 1

    def test_scale_up_on_queue_depth(self):
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=4,
            max_concurrency=2,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=1,
            queue_depth=5,
            max_queue_wait_sec=0,
            workers=[{"state": "ready", "current_concurrency": 2}],
        )
        assert should_scale_up(inp) is True
        assert compute_desired_workers(inp) >= 2

    def test_scale_up_on_queue_delay(self):
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=4,
            max_concurrency=4,
            scaling_policy_type="queue_delay",
            scaling_policy_value=10,
            queue_depth=3,
            max_queue_wait_sec=15.0,
            workers=[{"state": "ready", "current_concurrency": 4}],
        )
        assert should_scale_up(inp) is True

    def test_free_concurrency_slots(self):
        workers = [
            {"state": "ready", "current_concurrency": 2},
            {"state": "ready", "current_concurrency": 0},
        ]
        assert free_concurrency_slots(workers, 4) == 6

    def test_workers_drain_then_reap_idle(self):
        now = time.time()
        workers = [
            {
                "worker_id": "w1",
                "state": WORKER_STATE_IDLE,
                "current_concurrency": 0,
                "last_heartbeat_at": now - 400,
            },
            {
                "worker_id": "w2",
                "state": WORKER_STATE_READY,
                "current_concurrency": 1,
                "last_heartbeat_at": now - 400,
            },
        ]
        mark = workers_to_mark_draining(workers, desired=0, idle_timeout_sec=300, now=now)
        assert mark == ["w1"]
        draining = [
            {
                "worker_id": "w1",
                "state": WORKER_STATE_DRAINING,
                "current_concurrency": 0,
                "updated_at": now - 60,
            },
            workers[1],
        ]
        reap = workers_to_reap(draining, desired=0, drain_grace_sec=30, now=now)
        assert reap == ["w1"]


class TestMetering:
    def test_compute_gpu_seconds_ceiling(self):
        assert compute_gpu_seconds(100.0, 100.5) == 1
        assert compute_gpu_seconds(100.0, 102.0) == 2
        assert compute_gpu_seconds(None, 200.0) == 0

    def test_estimate_cost(self):
        # 3600s @ $1/hr = $1
        assert estimate_cost_cad(3600, 1.0, 1) == 1.0
        assert estimate_cost_cad(0, 1.0) == 0.0

    def test_token_metadata_observability_only(self):
        meta = token_cost_metadata(1_000_000, 1_000_000)
        assert meta["input_tokens"] == 1_000_000
        assert meta["total_token_cost_cad"] > 0


class TestOpenAIProxy:
    def test_normalize_route(self):
        assert normalize_route("/openai/v1/chat/completions") == "chat/completions"
        assert normalize_route("v1/models") == "models"

    def test_capability_gate_rejects_embeddings(self):
        ep = {"mode": "preset", "model_ref": "llama"}
        with pytest.raises(OpenAIProxyError) as exc:
            capability_gate("embeddings", ep, {})
        assert exc.value.status_code == 501

    def test_capability_gate_rejects_vision(self):
        ep = {"mode": "preset"}
        body = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]}
        with pytest.raises(OpenAIProxyError) as exc:
            capability_gate("chat/completions", ep, body)
        assert exc.value.code == "vision_not_supported"

    def test_worker_base_url(self):
        assert worker_base_url({"host_ip": "10.0.0.5"}, 8080) == "http://10.0.0.5:8080"

    def test_worker_base_url_uses_mapped_host_port(self):
        row = {
            "host_ip": "10.0.0.5",
            "job_payload": {"http_ports": {"8080": 55123}},
        }
        assert worker_base_url(row, 8080) == "http://10.0.0.5:55123"


class TestStreams:
    def test_sse_format(self):
        out = format_sse_event({"token": "hi"}, event="token", seq_no=1)
        assert "event: token" in out
        assert "data:" in out
        assert format_done_chunk().strip().endswith("[DONE]")


class TestServiceValidation:
    def test_validate_preset_requires_model(self):
        svc = ServerlessService()
        with pytest.raises(ValueError, match="model_ref"):
            svc.validate_endpoint_spec(EndpointCreate(owner_id="c1", mode="preset"))

    def test_wallet_preflight_suspended(self):
        mock_wallet = {"status": "suspended", "balance_cad": 10.0, "grace_until": 0}
        with patch("billing.get_billing_engine") as mock_be:
            mock_be.return_value.get_wallet.return_value = mock_wallet
            with pytest.raises(WalletPreflightError) as exc:
                ServerlessService.wallet_preflight("cust-1")
            assert exc.value.status_code == 402


def _pg_tables() -> set[str]:
    from db import _get_pg_pool

    pool = _get_pg_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            "SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public'"
        ).fetchall()
        return {r["tablename"] if isinstance(r, dict) else r[0] for r in rows}


@pytest.fixture(scope="module")
def repo():
    required = {"serverless_endpoints", "serverless_workers", "serverless_jobs"}
    if required - _pg_tables():
        pytest.skip("Run alembic upgrade head on test DB")
    return ServerlessRepo()


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


class TestDispatcherIntegration:
    def test_dispatch_assigns_job_to_worker(self, repo: ServerlessRepo, owner_id: str):
        from serverless.dispatcher import ServerlessDispatcher

        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner_id,
                name="dispatch-test",
                mode="custom",
                image_ref="vllm/vllm-openai:latest",
                max_concurrency=2,
            )
        )
        worker = repo.create_worker(
            str(ep["endpoint_id"]),
            scheduler_job_id=f"job-{uuid.uuid4().hex[:8]}",
        )
        repo.update_worker(str(worker["worker_id"]), state=WORKER_STATE_READY)

        job = repo.enqueue_job(str(ep["endpoint_id"]), owner_id, {"prompt": "x"})
        disp = ServerlessDispatcher(repo)
        claimed = disp.dispatch_for_endpoint(ep)
        assert claimed is not None
        assert claimed["job_id"] == job["job_id"]
        assert claimed["status"] == "IN_PROGRESS"

        w2 = repo.get_worker(str(worker["worker_id"]))
        assert int(w2["current_concurrency"]) == 1

        repo.soft_delete_endpoint(str(ep["endpoint_id"]), owner_id)


class TestKeysIntegration:
    def test_create_and_validate_key(self, repo: ServerlessRepo, owner_id: str):
        raw, row = create_endpoint_key(repo, owner_id, name="ci-key")
        assert raw.startswith("xcel_")
        assert row.get("key_id")
        validated = validate_key(repo, raw)
        assert validated is not None
        assert validated["key_id"] == row["key_id"]
        assert key_has_scope(validated, "inference:write")
        repo.revoke_api_key(str(row["key_id"]), owner_id)
        assert validate_key(repo, raw) is None