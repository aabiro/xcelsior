"""Phase 12 — queue dead-letter, TTL requeue, crash recovery, request timeout."""

import os
import time
import uuid
from unittest.mock import patch

import pytest

os.environ["XCELSIOR_ENV"] = "test"
os.environ.setdefault("XCELSIOR_API_TOKEN", "")

from serverless.dispatcher import ServerlessDispatcher
from serverless.repo import (
    EndpointCreate,
    JOB_STATUS_FAILED,
    JOB_STATUS_IN_PROGRESS,
    JOB_STATUS_QUEUED,
    ServerlessRepo,
    WORKER_STATE_ERROR,
    WORKER_STATE_READY,
)
from serverless.service import ServerlessService

JOB_HEARTBEAT_TTL_SEC = 120


@pytest.fixture
def owner_id():
    return f"cust-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def repo():
    return ServerlessRepo()


@pytest.fixture
def endpoint(repo: ServerlessRepo, owner_id: str):
    ep = repo.create_endpoint(
        EndpointCreate(
            owner_id=owner_id,
            name="queue-recovery",
            mode="custom",
            image_ref="xcelsior/serverless-base:cuda12.4-py3.12",
            request_timeout_sec=30,
        )
    )
    yield ep
    repo.soft_delete_endpoint(ep["endpoint_id"], owner_id)


class TestCrashRecovery:
    def test_handle_worker_lost_requeues_inflight_job(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str
    ):
        ep_id = endpoint["endpoint_id"]
        worker = repo.create_worker(ep_id, scheduler_job_id="sched-crash")
        worker_id = str(worker["worker_id"])
        job = repo.enqueue_job(ep_id, owner_id, {"task": 1})
        claimed = repo.claim_next_job(ep_id, worker_id)
        assert claimed is not None
        repo.increment_worker_concurrency(worker_id)

        dispatcher = ServerlessDispatcher(repo)
        requeued = dispatcher.handle_worker_lost(worker_id)
        assert requeued == 1

        refreshed = repo.get_job(job["job_id"], endpoint_id=ep_id)
        assert refreshed is not None
        assert refreshed["status"] == JOB_STATUS_QUEUED
        assert refreshed.get("worker_id") is None
        assert repo.queue_depth(ep_id) == 1

        w = repo.get_worker(worker_id)
        assert w is not None
        assert w["state"] == WORKER_STATE_ERROR

    def test_worker_exited_requeues_and_marks_terminated(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str
    ):
        ep_id = endpoint["endpoint_id"]
        worker = repo.create_worker(ep_id, scheduler_job_id="sched-exit")
        worker_id = str(worker["worker_id"])
        repo.update_worker(worker_id, state=WORKER_STATE_READY)
        job = repo.enqueue_job(ep_id, owner_id, {"task": 2})
        repo.claim_next_job(ep_id, worker_id)

        svc = ServerlessService(repo)
        svc.worker_exited(worker_id, exit_code=1, error_message="OOM")

        refreshed = repo.get_job(job["job_id"], endpoint_id=ep_id)
        assert refreshed is not None
        assert refreshed["status"] == JOB_STATUS_QUEUED
        w = repo.get_worker(worker_id)
        assert w is not None
        assert w["state"] == WORKER_STATE_ERROR


class TestRequestTimeoutReaper:
    def test_timed_out_job_failed_by_reaper(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str, monkeypatch
    ):
        ep_id = endpoint["endpoint_id"]
        worker = repo.create_worker(ep_id, scheduler_job_id="sched-timeout")
        worker_id = str(worker["worker_id"])
        job = repo.enqueue_job(ep_id, owner_id, {"slow": True})
        claimed = repo.claim_next_job(ep_id, worker_id)
        assert claimed is not None

        now = time.time()
        with repo._conn() as conn:
            conn.execute(
                """
                UPDATE serverless_jobs
                SET started_at = %s, updated_at = %s
                WHERE job_id = %s
                """,
                (now - 200, now - 200, job["job_id"]),
            )

        svc = ServerlessService(repo)
        with patch.object(svc.repo, "list_workers_stale_heartbeat", return_value=[]), patch.object(
            svc.repo, "list_stuck_booting_workers", return_value=[]
        ), patch.object(svc.repo, "list_booting_workers", return_value=[]):
            from serverless.reaper import reap_all

            stats = reap_all(svc)

        assert stats["timed_out_jobs"] >= 1
        refreshed = repo.get_job(job["job_id"], endpoint_id=ep_id)
        assert refreshed is not None
        assert refreshed["status"] == JOB_STATUS_FAILED


class TestStaleWorkerReaper:
    def test_stale_heartbeat_requeues_before_deprovision(
        self, repo: ServerlessRepo, endpoint: dict, owner_id: str
    ):
        ep_id = endpoint["endpoint_id"]
        worker = repo.create_worker(ep_id, scheduler_job_id="sched-stale")
        worker_id = str(worker["worker_id"])
        repo.update_worker(
            worker_id,
            state=WORKER_STATE_READY,
            last_heartbeat_at=time.time() - JOB_HEARTBEAT_TTL_SEC - 60,
        )
        job = repo.enqueue_job(ep_id, owner_id, {"x": 1})
        repo.claim_next_job(ep_id, worker_id)

        svc = ServerlessService(repo)
        with patch.object(svc, "deprovision_worker") as mock_dep, patch.object(
            svc.repo, "list_jobs_past_request_timeout", return_value=[]
        ), patch.object(svc.repo, "list_stuck_booting_workers", return_value=[]), patch.object(
            svc.repo, "list_booting_workers", return_value=[]
        ):
            from serverless.reaper import reap_all

            stats = reap_all(svc)

        assert stats["stale_workers"] >= 1
        mock_dep.assert_called()
        refreshed = repo.get_job(job["job_id"], endpoint_id=ep_id)
        assert refreshed is not None
        assert refreshed["status"] == JOB_STATUS_QUEUED


class TestBootFailureHardening:
    def test_recent_boot_failures_counts_consecutive_failures(self):
        from serverless.reaper import _recent_boot_failures

        now = time.time()

        class FakeRepo:
            @staticmethod
            def list_workers(_endpoint_id):
                return [
                    {
                        "worker_id": "w3",
                        "state": "terminated",
                        "created_at": now - 10,
                        "last_heartbeat_at": 0,
                    },
                    {
                        "worker_id": "w2",
                        "state": "error",
                        "created_at": now - 20,
                        "last_heartbeat_at": 0,
                    },
                    {
                        "worker_id": "w1",
                        "state": "ready",
                        "created_at": now - 30,
                        "last_heartbeat_at": now - 25,
                    },
                ]

        class FakeService:
            repo = FakeRepo()

        assert _recent_boot_failures(FakeService(), "sep-1") == 2

    def test_orphan_scheduler_job_without_host_is_cancelled(self):
        from serverless.reaper import _kill_scheduler_job

        job = {"job_id": "sched-orphan", "status": "queued", "host_id": ""}
        with patch("scheduler.list_hosts", return_value=[]), patch(
            "scheduler.kill_job"
        ) as kill_job, patch("scheduler.update_job_status") as update_status:
            assert _kill_scheduler_job(job) is True
        kill_job.assert_not_called()
        update_status.assert_called_once_with("sched-orphan", "cancelled")
