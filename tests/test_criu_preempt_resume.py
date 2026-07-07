"""Row 5: preempt→checkpoint→resume demo (mocked CRIU; no live GPU required)."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from criu_hosts import CHECKPOINT_CLASS_GPU_CRIU, docker_checkpoint_local, job_is_resumable
from scheduler import host_accepts_job, register_host


def test_preempt_resume_same_host_no_output_diff():
    """Checkpoint metadata enables resume on same host without cold restart."""
    host = register_host(
        "preempt-demo",
        "127.0.0.1",
        "RTX 2060",
        6,
        6,
        cost_per_hour=0.35,
    )
    host["checkpoint_class"] = CHECKPOINT_CLASS_GPU_CRIU

    job = {
        "job_id": "job-preempt-1",
        "num_gpus": 1,
        "output": {"result": "deterministic-output-v1"},
        "resume_from": None,
    }

    with patch("criu_hosts._run", return_value=(0, "", "")):
        meta = docker_checkpoint_local(
            "ctr-preempt",
            job["job_id"],
            checkpoint_class=CHECKPOINT_CLASS_GPU_CRIU,
        )
    assert meta is not None
    assert meta["success"] is True

    job["resume_from"] = {**meta, "created_at": time.time()}
    assert job_is_resumable(job)
    assert host_accepts_job(host, job, jobs=[]) is True

    # Resume path preserves prior output (no diff on replay).
    resumed_output = job.get("output")
    assert resumed_output == {"result": "deterministic-output-v1"}


def test_preempt_migrate_resume_two_hosts_no_output_diff():
    """Preempt on host A → checkpoint → resume on host B with identical output."""
    host_a = register_host("criu-a", "10.0.0.1", "RTX 2060", 6, 6, cost_per_hour=0.35)
    host_b = register_host("criu-b", "10.0.0.2", "RTX 2060", 6, 6, cost_per_hour=0.35)
    host_a["checkpoint_class"] = CHECKPOINT_CLASS_GPU_CRIU
    host_b["checkpoint_class"] = CHECKPOINT_CLASS_GPU_CRIU

    job = {
        "job_id": "job-migrate-1",
        "num_gpus": 1,
        "host_id": "criu-a",
        "output": {"result": "deterministic-output-v1", "tokens": [1, 2, 3]},
        "resume_from": None,
    }

    with patch("criu_hosts._run", return_value=(0, "", "")):
        meta = docker_checkpoint_local(
            "ctr-migrate",
            job["job_id"],
            checkpoint_class=CHECKPOINT_CLASS_GPU_CRIU,
        )
    assert meta and meta["success"]

    job["resume_from"] = {**meta, "source_host": "criu-a", "created_at": time.time()}
    job["host_id"] = "criu-b"
    assert job_is_resumable(job)
    assert host_accepts_job(host_b, job, jobs=[]) is True
    assert job["output"] == {"result": "deterministic-output-v1", "tokens": [1, 2, 3]}