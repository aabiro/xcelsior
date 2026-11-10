"""CVP-01 T-17: one tenant cannot fill Docker's disk, and nobody else pays for it."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import worker_agent as worker

GIB = 1024**3
QUOTA = 100 * GIB
FLOOR = 10 * GIB


def decide(free, sizes):
    return worker.disk_guard_decide(free, sizes, quota_bytes=QUOTA, floor_bytes=FLOOR)


def test_plenty_of_space_and_small_layers_stop_nothing():
    assert decide(500 * GIB, {"a": 3 * GIB, "b": 40 * GIB}) == ([], False)


def test_only_the_container_over_its_quota_is_stopped():
    assert decide(500 * GIB, {"a": 3 * GIB, "b": 101 * GIB}) == (["b"], False)


def test_low_disk_stops_the_largest_layer_and_no_other():
    assert decide(4 * GIB, {"a": 3 * GIB, "b": 60 * GIB, "c": 59 * GIB}) == (["b"], False)


def test_low_disk_no_tenant_caused_stops_nobody():
    # Image pulls or the provider's own files filled the disk.
    assert decide(4 * GIB, {"a": 3 * GIB, "b": 2 * GIB}) == ([], True)
    assert decide(4 * GIB, {}) == ([], True)


def test_switches_off_with_zero():
    assert worker.disk_guard_decide(1, {"a": 900 * GIB}, quota_bytes=0, floor_bytes=0) == (
        [],
        False,
    )


@pytest.fixture
def guard(monkeypatch):
    jobs = {"job-a": "xcl-job-a", "job-b": "xcl-job-b"}
    state = SimpleNamespace(free=500 * GIB, sizes={"xcl-job-a": GIB, "xcl-job-b": GIB}, measured=0)

    def measure(cname):
        state.measured += 1
        return state.sizes.get(cname)

    monkeypatch.setattr(worker, "_active_containers", dict(jobs))
    monkeypatch.setattr(worker, "_v2_attempts", {j: {"container_name": c} for j, c in jobs.items()})
    monkeypatch.setattr(worker, "_docker_storage", {"root": "/var/lib/docker", "quota": False})
    monkeypatch.setattr(worker.shutil, "disk_usage", lambda path: SimpleNamespace(free=state.free))
    monkeypatch.setattr(worker, "_container_writable_bytes", measure)
    monkeypatch.setattr(worker, "CONTAINER_DISK_GB", 100)
    monkeypatch.setattr(worker, "HOST_DISK_FLOOR_GB", 10)
    for name in ("_kill_container", "report_job_status"):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker, "_v2_journal_save", Mock(return_value=True))
    return state


def test_layers_are_not_measured_on_every_pass(guard):
    assert worker.disk_guard_pass(measure=False) is False
    assert guard.measured == 0
    worker._kill_container.assert_not_called()


def test_a_tight_disk_is_measured_at_once_and_again_next_pass(guard):
    guard.free = 15 * GIB
    assert worker.disk_guard_pass(measure=False) is True
    assert guard.measured == 2
    worker._kill_container.assert_not_called()


def test_over_quota_container_is_stopped_with_the_outcome_journaled_first(guard):
    guard.sizes["xcl-job-b"] = 120 * GIB
    order = []
    worker._v2_journal_save.side_effect = lambda: order.append("journal") or True
    worker._kill_container.side_effect = lambda name: order.append("kill")

    worker.disk_guard_pass(measure=True)

    assert order == ["journal", "kill"]
    worker._kill_container.assert_called_once_with("xcl-job-b")
    report = worker.report_job_status.call_args
    assert report.args[:2] == ("job-b", "failed")
    assert report.kwargs["policy_reason"] == "disk_quota_exceeded"
    assert "100 GB" in report.kwargs["error_message"]
    assert worker._v2_attempts["job-b"]["pending_terminal_report"] == {
        "status": "failed",
        "failure_code": "disk_quota_exceeded",
    }
    assert "pending_terminal_report" not in worker._v2_attempts["job-a"]
    assert worker._active_containers == {"job-a": "xcl-job-a"}


def test_full_disk_nobody_caused_ends_no_tenants_work(guard):
    guard.free = 2 * GIB
    worker.disk_guard_pass(measure=False)
    worker._kill_container.assert_not_called()
    worker.report_job_status.assert_not_called()


def test_a_container_replaced_since_measuring_is_left_alone(guard):
    guard.sizes["xcl-job-b"] = 120 * GIB
    worker._v2_attempts["job-b"]["container_name"] = "xcl-job-b-retry"
    worker.disk_guard_pass(measure=True)
    worker._kill_container.assert_not_called()


def test_an_unreadable_size_is_not_treated_as_zero_or_as_guilt(guard):
    guard.free = 2 * GIB
    guard.sizes = {}
    worker.disk_guard_pass(measure=True)
    worker._kill_container.assert_not_called()


def _docker_info(monkeypatch, info_out, mount_out="rw,relatime"):
    def run(cmd, **kwargs):
        out = info_out if cmd[0] == "docker" else mount_out
        return SimpleNamespace(returncode=0, stdout=out)

    monkeypatch.setattr(worker, "_docker_storage", None)
    monkeypatch.setattr(worker.subprocess, "run", run)


def test_docker_quota_is_used_only_where_the_driver_enforces_it(monkeypatch):
    monkeypatch.setattr(worker, "CONTAINER_DISK_GB", 100)
    ext4 = '/var/lib/docker|overlay2|[["Backing Filesystem","extfs"]]'
    xfs = '/data/docker|overlay2|[["Backing Filesystem","xfs"]]'

    _docker_info(monkeypatch, ext4)
    assert worker.container_disk_quota_gb(True) is None

    _docker_info(monkeypatch, xfs, "rw,relatime,noquota")
    assert worker.container_disk_quota_gb(True) is None

    _docker_info(monkeypatch, xfs, "rw,relatime,prjquota")
    assert worker.container_disk_quota_gb(True) == 100
    assert worker._docker_storage["root"] == "/data/docker"
    # Batch containers have a read-only root.
    assert worker.container_disk_quota_gb(False) is None


def test_memory_share_follows_the_gpus_leased(monkeypatch):
    monkeypatch.setattr(worker, "get_gpu_telemetry", lambda: [{}, {}, {}, {}])
    assert worker._memory_share(1) == 0.25
    assert worker._memory_share(2) == 0.5
    assert worker._memory_share(8) == 1.0
    monkeypatch.setattr(worker, "get_gpu_telemetry", lambda: [])
    assert worker._memory_share(1) == 1.0
