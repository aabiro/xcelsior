"""Serverless worker SDK + internal worker callback routes."""

import asyncio
import os
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

os.environ["XCELSIOR_ENV"] = "test"
os.environ["XCELSIOR_PERSISTENT_AUTH"] = "true"
os.environ["XCELSIOR_SKIP_CUDA_CHECK"] = "1"

from api import app
from serverless.repo import EndpointCreate, ServerlessRepo, WORKER_STATE_READY
from serverless.service import ServerlessService
from serverless.worker_sdk import (
    WorkerError,
    error_envelope,
    handler,
    progress_event,
    run_fitness_checks,
    run_worker,
)
from serverless.worker_sdk.fitness import FitnessConfig
from serverless.worker_sdk.handler import invoke_handler, normalize_handler_result

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)


def _fund_and_create_endpoint() -> tuple[str, str]:
    email = f"wsdk-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Wsdk"},
    )
    assert reg.status_code == 200
    login = client.post("/api/auth/login", json={"email": email, "password": "StrongPass123!"})
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    customer_id = reg.json()["user"]["customer_id"]
    dep = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0},
        headers=headers,
    )
    assert dep.status_code == 200
    created = client.post(
        "/api/v2/serverless/endpoints",
        headers=headers,
        json={
            "name": "wsdk-echo",
            "mode": "custom",
            "image_ref": "xcelsior/serverless-base:cuda12.4-py3.12",
            "gpu_type": "",
            "min_workers": 0,
            "max_workers": 1,
        },
    )
    assert created.status_code == 200, created.text[:300]
    ep_id = created.json()["endpoint"]["endpoint_id"]
    return ep_id, headers["Authorization"]


def _create_worker(ep_id: str) -> str:
    repo = ServerlessRepo()
    worker = repo.create_worker(ep_id, scheduler_job_id=f"job-{uuid.uuid4().hex[:8]}")
    repo.update_worker(str(worker["worker_id"]), state=WORKER_STATE_READY)
    return str(worker["worker_id"])


class TestFitnessAndErrors:
    def test_fitness_passes_without_cuda(self):
        cfg = FitnessConfig(require_cuda=False, min_disk_gb=0.01, min_free_mem_gb=0.01)
        assert run_fitness_checks(cfg) == []

    def test_error_envelope_shape(self):
        err = WorkerError(422, "bad_input", "missing prompt", retryable=False)
        assert error_envelope(err)["error"]["code"] == "bad_input"


class TestHandlerNormalization:
    def test_dict_handler(self):
        async def fn(job):
            return {"echo": job["input"]}

        out, events, usage = asyncio.run(invoke_handler(fn, {"input": {"x": 1}}))
        assert out == {"echo": {"x": 1}}
        assert events == []
        assert usage == {}

    def test_sync_generator(self):
        def gen():
            yield progress_event("half", pct=50)
            yield {"done": True}

        out, events, usage = normalize_handler_result(gen())
        assert events[0]["type"] == "progress"
        assert out == {"done": True}
        assert usage == {}

    def test_usage_extraction(self):
        out, events, usage = normalize_handler_result({
            "result": "ok",
            "usage": {"input_tokens": 5, "output_tokens": 9},
        })
        assert out == {"result": "ok"}
        assert usage["input_tokens"] == 5
        assert usage["output_tokens"] == 9


class TestWorkerServiceCallbacks:
    def test_claim_and_complete_job(self):
        repo = ServerlessRepo()
        svc = ServerlessService(repo)
        owner = f"cust-{uuid.uuid4().hex[:8]}"
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner, name="svc", mode="custom", image_ref="img")
        )
        ep_id = str(ep["endpoint_id"])
        worker_id = _create_worker(ep_id)
        queued = repo.enqueue_job(ep_id, owner, {"prompt": "hi"})

        claimed = svc.worker_claim_job(worker_id)
        assert claimed is not None
        assert claimed["job_id"] == queued["job_id"]

        completed = svc.worker_complete_job(
            worker_id,
            str(queued["job_id"]),
            output={"echo": "hi"},
        )
        assert completed is not None
        assert completed["status"] == "COMPLETED"
        assert completed["output"]["echo"] == "hi"

        w = repo.get_worker(worker_id)
        assert int(w["current_concurrency"]) == 0


class TestWorkerRoutes:
    def test_worker_callback_roundtrip(self):
        ep_id, _auth = _fund_and_create_endpoint()
        worker_id = _create_worker(ep_id)
        repo = ServerlessRepo()
        owner_row = repo.get_endpoint(ep_id)
        job = repo.enqueue_job(ep_id, str(owner_row["owner_id"]), {"prompt": "route"})

        ready = client.post(f"/api/v2/serverless/workers/{worker_id}/ready", json={"host_id": "h1"})
        assert ready.status_code == 200

        hb = client.post(f"/api/v2/serverless/workers/{worker_id}/heartbeat")
        assert hb.status_code == 200

        claim = client.post(f"/api/v2/serverless/workers/{worker_id}/jobs/claim")
        assert claim.status_code == 200
        assert claim.json()["job"]["job_id"] == job["job_id"]

        ev = client.post(
            f"/api/v2/serverless/workers/{worker_id}/jobs/{job['job_id']}/events",
            json={"event_type": "progress", "payload": {"pct": 10}},
        )
        assert ev.status_code == 200

        done = client.post(
            f"/api/v2/serverless/workers/{worker_id}/jobs/{job['job_id']}/complete",
            json={"output": {"echo": "route"}},
        )
        assert done.status_code == 200
        assert done.json()["job"]["status"] == "COMPLETED"


class TestDeployMaintenance:
    def test_host_maintenance_blocks_on_active_serverless_worker(self):
        from scheduler import submit_job, update_job_status

        host_id = f"h-slv-{uuid.uuid4().hex[:8]}"
        # Spot preemption cycles can evict running jobs in the shared test DB — keep this
        # test deterministic while we assert maintenance blocking semantics.
        with patch("scheduler.preemption_cycle", return_value=({}, [])):
            assert client.put(
                "/host",
                json={
                    "host_id": host_id,
                    "ip": "10.0.0.99",
                    "gpu_model": "RTX 4090",
                    "total_vram_gb": 24,
                    "free_vram_gb": 24,
                },
            ).status_code == 200

            job = submit_job(
                "serverless-sep-test",
                vram_needed_gb=8,
                image="xcelsior/serverless-vllm:12.4",
                job_type="serverless_worker",
                tier="on-demand",
                pricing_mode="on_demand",
            )
            update_job_status(job["job_id"], "assigned", host_id=host_id)
            update_job_status(job["job_id"], "running", host_id=host_id)

            summary = client.get(f"/host/{host_id}/maintenance")
            assert summary.status_code == 200
            data = summary.json()
            assert data["active_serverless_workers"] >= 1
            assert any(w["job_id"] == job["job_id"] for w in data["serverless_workers"])
            assert data["safe_to_maintain"] is False

            drain = client.post(f"/host/{host_id}/drain")
            assert drain.status_code == 200
            assert job["job_id"] in drain.json().get("preempted", [])

            summary = client.get(f"/host/{host_id}/maintenance")
            data = summary.json()
            assert data["draining"] is True
            assert data["active_serverless_workers"] == 0
            assert data["safe_to_maintain"] is True


class TestWorkerClientRetries:
    def test_retries_on_503(self):
        from unittest.mock import MagicMock, patch

        from serverless.worker_sdk.client import WorkerClient

        responses = [
            MagicMock(status_code=503, raise_for_status=MagicMock()),
            MagicMock(
                status_code=200,
                raise_for_status=MagicMock(),
                json=MagicMock(return_value={"ok": True, "job": None}),
            ),
        ]
        mock_client = MagicMock()
        mock_client.post.side_effect = responses

        wc = WorkerClient(base_url="http://test", worker_id="swkr-test", token="t")
        wc._http = mock_client
        with patch("serverless.worker_sdk.client.time.sleep"):
            result = wc.claim_job()
        assert result is None
        assert mock_client.post.call_count == 2


class TestRunWorkerConcurrency:
    def test_reports_token_usage_on_complete(self):
        import serverless.worker_sdk.runtime as rt

        mock_client = MagicMock()
        mock_client.worker_id = "swkr-tok"
        mock_client.is_job_cancelled.return_value = False
        mock_client.signal_ready.return_value = {"ok": True}
        mock_client.heartbeat.return_value = {"ok": True}
        mock_client.claim_job.side_effect = [
            {"job_id": "tok-job", "payload": {"x": 1}, "input": {"x": 1}},
            None,
        ]

        @handler
        def with_usage(job, cancel=None):
            return {"ok": True, "usage": {"input_tokens": 3, "output_tokens": 7}}

        def _complete_and_exit(_job_id, **kwargs):
            rt._draining = True
            return {"ok": True, "kwargs": kwargs}

        mock_client.complete_job.side_effect = lambda job_id, **kw: _complete_and_exit(job_id, **kw)

        rt._draining = False
        with patch("serverless.worker_sdk.runtime.time.sleep"), patch(
            "serverless.worker_sdk.runtime.start_health_server"
        ), patch.object(rt._HeartbeatThread, "start"), patch.object(rt._HeartbeatThread, "stop"):
            run_worker(client=mock_client, skip_fitness=True, poll_interval_sec=0, max_concurrency=2)

        _args, kwargs = mock_client.complete_job.call_args
        assert kwargs.get("input_tokens") == 3
        assert kwargs.get("output_tokens") == 7


class TestRunWorkerLoop:
    def test_run_worker_executes_one_job(self):
        ep_id, _ = _fund_and_create_endpoint()
        worker_id = _create_worker(ep_id)
        repo = ServerlessRepo()
        owner_row = repo.get_endpoint(ep_id)
        repo.enqueue_job(ep_id, str(owner_row["owner_id"]), {"prompt": "loop"})

        mock_client = MagicMock()
        mock_client.worker_id = worker_id
        mock_client.is_job_cancelled.return_value = False
        mock_client.signal_ready.return_value = {"ok": True}
        mock_client.heartbeat.return_value = {"ok": True}
        mock_client.claim_job.side_effect = [
            {"job_id": "sjob-test", "payload": {"prompt": "loop"}, "input": {"prompt": "loop"}},
            None,
        ]
        mock_client.complete_job.return_value = {"ok": True}

        @handler
        def echo(job):
            return {"echo": job["input"]["prompt"]}

        def _complete_and_exit(*_a, **_kw):
            import serverless.worker_sdk.runtime as rt

            rt._draining = True
            return {"ok": True}

        mock_client.complete_job.side_effect = _complete_and_exit

        import serverless.worker_sdk.runtime as rt

        rt._draining = False
        with patch("serverless.worker_sdk.runtime.time.sleep"), patch(
            "serverless.worker_sdk.runtime.start_health_server"
        ):
            run_worker(client=mock_client, skip_fitness=True, poll_interval_sec=0)

        mock_client.complete_job.assert_called_once()
        args, kwargs = mock_client.complete_job.call_args
        assert kwargs.get("output") == {"echo": "loop"}