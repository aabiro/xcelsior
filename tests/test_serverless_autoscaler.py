"""Phase 6 — autoscaler runtime, drain-before-reap, reaper, anti-thrash."""

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
    scale_down_cooldown_active,
    workers_to_mark_draining,
    workers_to_reap,
)
from serverless.repo import (
    EndpointCreate,
    ServerlessRepo,
    WORKER_STATE_BOOTING,
    WORKER_STATE_DRAINING,
    WORKER_STATE_IDLE,
    WORKER_STATE_READY,
)
from serverless.service import ServerlessService


class TestPredictiveScaling:
    def test_forecast_increases_desired_workers(self, monkeypatch):
        monkeypatch.setenv("XCELSIOR_SERVERLESS_PREDICTIVE_SCALING", "true")
        now = time.time()
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=8,
            max_concurrency=2,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=1,
            queue_depth=2,
            max_queue_wait_sec=0.0,
            workers=[],
            queue_depth_samples=[(now - 30, 2), (now, 6)],
        )
        desired = compute_desired_workers(inp)
        assert desired >= 3


class TestDrainBeforeReap:
    def test_mark_draining_idle_excess(self):
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

    def test_reap_only_draining_after_grace(self):
        now = time.time()
        workers = [
            {
                "worker_id": "w1",
                "state": WORKER_STATE_DRAINING,
                "current_concurrency": 0,
                "updated_at": now - 60,
            },
            {
                "worker_id": "w2",
                "state": WORKER_STATE_IDLE,
                "current_concurrency": 0,
                "updated_at": now - 60,
            },
        ]
        reap = workers_to_reap(workers, desired=0, drain_grace_sec=30, now=now)
        assert reap == ["w1"]


class TestAntiThrash:
    def test_scale_down_cooldown_blocks_recent_alloc(self):
        now = time.time()
        workers = [{"worker_id": "w1", "allocated_at": now - 10}]
        assert scale_down_cooldown_active(workers, now=now, cooldown_sec=60) is True
        assert scale_down_cooldown_active(workers, now=now, cooldown_sec=5) is False

    def test_burst_scale_up_then_cooldown_prevents_immediate_down(self):
        now = time.time()
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=4,
            max_concurrency=1,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=1,
            queue_depth=0,
            max_queue_wait_sec=0,
            workers=[
                {"state": WORKER_STATE_IDLE, "current_concurrency": 0, "allocated_at": now - 5},
            ],
        )
        assert compute_desired_workers(inp) == 1
        assert scale_down_cooldown_active(inp.workers, now=now, cooldown_sec=60) is True


class TestScaleToZero:
    def test_cold_request_scales_from_zero(self):
        inp = AutoscalerInput(
            min_workers=0,
            max_workers=4,
            max_concurrency=2,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=1,
            queue_depth=3,
            max_queue_wait_sec=0,
            workers=[],
        )
        assert compute_desired_workers(inp) >= 1

    def test_keep_warm_min_workers(self):
        inp = AutoscalerInput(
            min_workers=1,
            max_workers=4,
            max_concurrency=4,
            scaling_policy_type="queue_request_count",
            scaling_policy_value=1,
            queue_depth=0,
            max_queue_wait_sec=0,
            workers=[],
        )
        assert compute_desired_workers(inp) == 1


class TestReconcileDisabled:
    def test_reconcile_all_skips_when_disabled(self):
        svc = ServerlessService()
        with patch.dict(os.environ, {"XCELSIOR_SERVERLESS_RECONCILE": "false"}):
            out = svc.reconcile_all()
        assert out.get("skipped") is True
        assert out.get("reason") == "disabled"


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


class TestReaper:
    def test_stale_heartbeat_reaps_worker(self, owner_id):
        repo = ServerlessRepo()
        ep = repo.create_endpoint(
            EndpointCreate(owner_id=owner_id, name="reap", mode="preset", model_ref="m", min_workers=0)
        )
        worker = repo.create_worker(str(ep["endpoint_id"]), scheduler_job_id="sched-x")
        repo.update_worker(
            str(worker["worker_id"]),
            state=WORKER_STATE_READY,
            last_heartbeat_at=time.time() - 500,
        )
        svc = ServerlessService(repo)
        with patch.object(svc, "deprovision_worker") as mock_dep:
            from serverless.reaper import reap_all

            stats = reap_all(svc)
        assert stats["stale_workers"] >= 1
        mock_dep.assert_called()

    def test_stuck_booting_reprovision(self):
        svc = ServerlessService()
        worker = {
            "worker_id": "swk-stuck-test",
            "endpoint_id": "sep-stuck-test",
            "scheduler_job_id": "sched-boot",
            "state": WORKER_STATE_BOOTING,
        }
        ep = {
            "endpoint_id": "sep-stuck-test",
            "min_workers": 1,
            "max_workers": 2,
            "max_concurrency": 4,
            "scaling_policy_type": "queue_request_count",
            "scaling_policy_value": 1,
        }
        with patch.object(svc.repo, "list_workers_stale_heartbeat", return_value=[]), patch.object(
            svc.repo, "list_jobs_past_request_timeout", return_value=[]
        ), patch.object(svc.repo, "list_stuck_booting_workers", return_value=[worker]), patch.object(
            svc.repo, "list_booting_workers", return_value=[]
        ), patch.object(svc.repo, "get_endpoint", return_value=ep), patch.object(
            svc, "deprovision_worker"
        ), patch.object(svc, "_endpoint_needs_workers", return_value=True), patch.object(
            svc, "provision_worker", return_value={"worker_id": "swk-new"}
        ) as mock_prov:
            from serverless.reaper import reap_all

            stats = reap_all(svc)
        assert stats["stuck_booting"] == 1
        assert stats["reprovisioned"] == 1
        mock_prov.assert_called_once_with("sep-stuck-test")