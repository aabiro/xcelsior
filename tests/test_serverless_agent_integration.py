"""Phase 5 — scheduler + worker-agent integration tests (no GPU)."""

import os
import uuid
from unittest.mock import patch

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.repo import EndpointCreate, ServerlessRepo
from serverless.service import ServerlessService


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def repo():
    return ServerlessRepo()


class TestProvisionWorkerPayload:
    def test_provision_worker_sets_scheduler_fields(self, repo, owner_id):
        ep = repo.create_endpoint(
            EndpointCreate(
                owner_id=owner_id,
                name="agent-payload",
                mode="custom",
                image_ref="xcelsior/serverless-base:cuda12.4-py3.12",
                gpu_tier="RTX 4090",
                http_port=8080,
                health_check_path="/healthz",
                env={"FOO": "bar"},
                min_workers=0,
            )
        )
        svc = ServerlessService(repo)
        with patch.object(svc, "wallet_preflight"), patch(
            "scheduler.submit_job"
        ) as mock_submit, patch("scheduler._set_job_fields") as mock_set:
            mock_submit.return_value = {"job_id": "job1234"}
            worker = svc.provision_worker(str(ep["endpoint_id"]))
        assert worker is not None
        mock_submit.assert_called_once()
        call_kw = mock_submit.call_args[1]
        assert call_kw["job_type"] == "serverless_worker"
        assert call_kw["gpu_model"] == "RTX 4090"
        assert call_kw["exposed_ports"] == [8080]
        mock_set.assert_called_once()
        args, kwargs = mock_set.call_args
        assert args[0] == "job1234"
        assert kwargs["serverless_worker_id"] == worker["worker_id"]
        assert kwargs["serverless_endpoint_id"] == ep["endpoint_id"]
        assert kwargs["environment"] == {"FOO": "bar"}
        assert kwargs["http_port"] == 8080
        assert kwargs["health_check_path"] == "/healthz"


class TestWorkerExited:
    def test_worker_exited_requeues_in_flight_jobs(self, repo, owner_id):
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner_id, name="exit-test", mode="preset", model_ref="m", min_workers=0)
        )
        worker = repo.create_worker(str(ep["endpoint_id"]), scheduler_job_id="sched1")
        job = repo.enqueue_job(str(ep["endpoint_id"]), owner_id, {"x": 1})
        repo.claim_next_job(str(ep["endpoint_id"]), str(worker["worker_id"]))
        svc = ServerlessService(repo)
        result = svc.worker_exited(str(worker["worker_id"]), exit_code=1, error_message="crash")
        assert result is not None
        assert result["state"] == "error"
        requeued = repo.get_job(str(job["job_id"]))
        assert requeued is not None
        assert requeued["status"] == "IN_QUEUE"


class TestRepoSchedulerLookup:
    def test_get_worker_by_scheduler_job_id(self, repo, owner_id):
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner_id, name="lookup", mode="preset", model_ref="m", min_workers=0)
        )
        worker = repo.create_worker(str(ep["endpoint_id"]), scheduler_job_id="sched-lookup-1")
        found = repo.get_worker_by_scheduler_job_id("sched-lookup-1")
        assert found is not None
        assert found["worker_id"] == worker["worker_id"]


class TestWorkerAgentHelpers:
    def test_is_serverless_job(self):
        import worker_agent

        assert worker_agent._is_serverless_job({"job_type": "serverless_worker"})
        assert not worker_agent._is_serverless_job({"job_type": ""})

    def test_allocate_host_port_mappings(self):
        import worker_agent

        port_map, args = worker_agent._allocate_host_port_mappings("abcd1234", [8080, 9090])
        assert "8080" in port_map
        assert "9090" in port_map
        assert any("-p" in a for a in args)