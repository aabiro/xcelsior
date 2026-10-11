"""Mining detection must only act on job containers that actually use the flagged GPU.

Regression: the detector read whole-GPU telemetry and, after 5 hits, killed
*every* active job container and reported it failed. A host running its own
GPU service (e.g. pixelspark-gpu on the ASUS worker) tripped it continuously,
which would have killed tenants' jobs for load they did not cause.
"""

from types import SimpleNamespace
from unittest.mock import Mock, mock_open

import pytest

import worker_agent as worker

JOB_CID = "a" * 64
OTHER_CID = "b" * 64
MINING = {"index": 0, "utilization": 100, "memory_util": 19, "pcie_tx_mb_s": 0.5}


def _run_loop(monkeypatch, iterations, telemetry):
    """Run mining_detection_loop for `iterations` passes with IO isolated."""
    calls = {"n": 0}

    def fake_telemetry():
        calls["n"] += 1
        if calls["n"] > iterations:
            worker._shutdown.set()
            return []
        return telemetry

    monkeypatch.setattr(worker, "MINING_CHECK_INTERVAL", 0)
    monkeypatch.setattr(worker, "get_gpu_telemetry", fake_telemetry)
    for name in ("report_mining_alert", "_kill_container", "report_job_status"):
        monkeypatch.setattr(worker, name, Mock())
    worker._shutdown.clear()
    try:
        worker.mining_detection_loop()
    finally:
        worker._shutdown.clear()


@pytest.fixture
def active_jobs(monkeypatch):
    jobs = {"job-on-gpu": "xcl-on-gpu", "job-elsewhere": "xcl-elsewhere"}
    monkeypatch.setattr(worker, "_active_containers", dict(jobs))
    return jobs


def test_cgroup_v2_and_cgroupfs_paths_resolve_to_the_container_id(monkeypatch):
    for content in (f"0::/system.slice/docker-{JOB_CID}.scope\n", f"0::/docker/{JOB_CID}\n"):
        monkeypatch.setattr("builtins.open", mock_open(read_data=content))
        assert worker._container_id_for_pid(123) == JOB_CID
    monkeypatch.setattr("builtins.open", mock_open(read_data="0::/user.slice/session-2.scope\n"))
    assert worker._container_id_for_pid(123) is None


def test_job_containers_on_gpu_matches_by_container_id(monkeypatch):
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: {11, 22})
    monkeypatch.setattr(worker, "_container_id_for_pid", {11: JOB_CID, 22: None}.get)
    ids = {"xcl-on-gpu": JOB_CID, "xcl-elsewhere": OTHER_CID}
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        lambda cmd, **kw: SimpleNamespace(returncode=0, stdout=ids[cmd[-1]] + "\n"),
    )
    active = {"job-on-gpu": "xcl-on-gpu", "job-elsewhere": "xcl-elsewhere"}
    assert worker._job_containers_on_gpu(0, active) == {"job-on-gpu": "xcl-on-gpu"}


def test_attribution_unavailable_is_distinct_from_no_job_on_gpu(monkeypatch):
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: None)
    assert worker._job_containers_on_gpu(0, {"j": "c"}) is None
    monkeypatch.setattr(worker, "get_gpu_compute_pids", lambda idx: {99})
    monkeypatch.setattr(worker, "_container_id_for_pid", lambda pid: None)
    assert worker._job_containers_on_gpu(0, {"j": "c"}) == {}


def test_host_workload_never_alerts_or_kills(monkeypatch, active_jobs):
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: {})
    _run_loop(monkeypatch, iterations=8, telemetry=[MINING])
    worker.report_mining_alert.assert_not_called()
    worker._kill_container.assert_not_called()
    assert worker._active_containers == active_jobs


def test_sustained_mining_kills_only_the_job_on_that_gpu(monkeypatch, active_jobs):
    monkeypatch.setattr(
        worker, "_job_containers_on_gpu", lambda idx, active: {"job-on-gpu": "xcl-on-gpu"}
    )
    _run_loop(monkeypatch, iterations=5, telemetry=[MINING])
    worker._kill_container.assert_called_once_with("xcl-on-gpu")
    worker.report_job_status.assert_called_once_with(
        "job-on-gpu", "terminated", policy_reason="mining_detected"
    )
    assert worker._active_containers == {"job-elsewhere": "xcl-elsewhere"}


def test_no_kill_before_five_consecutive_hits(monkeypatch, active_jobs):
    monkeypatch.setattr(
        worker, "_job_containers_on_gpu", lambda idx, active: {"job-on-gpu": "xcl-on-gpu"}
    )
    _run_loop(monkeypatch, iterations=4, telemetry=[MINING])
    assert worker.report_mining_alert.call_count == 4
    worker._kill_container.assert_not_called()


def test_without_attribution_falls_back_to_all_active_jobs(monkeypatch, active_jobs):
    monkeypatch.setattr(worker, "_job_containers_on_gpu", lambda idx, active: None)
    _run_loop(monkeypatch, iterations=5, telemetry=[MINING])
    assert {c.args[0] for c in worker._kill_container.call_args_list} == set(active_jobs.values())
    assert worker._active_containers == {}


def test_counters_are_per_gpu(monkeypatch, active_jobs):
    # GPU 0 and GPU 1 alternate hits: neither reaches 5 consecutive on its own
    # within 4 passes, so a shared counter (old behaviour: 8 hits) must not fire.
    monkeypatch.setattr(
        worker, "_job_containers_on_gpu", lambda idx, active: {"job-on-gpu": "xcl-on-gpu"}
    )
    two = [dict(MINING, index=0), dict(MINING, index=1)]
    _run_loop(monkeypatch, iterations=4, telemetry=two)
    worker._kill_container.assert_not_called()
