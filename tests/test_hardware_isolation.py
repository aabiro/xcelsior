"""Worker-side hardware scope must agree with the control-plane allocation."""

import fcntl
import os
from types import SimpleNamespace

import pytest

import hardware_isolation as isolation
from security import build_secure_docker_args


@pytest.fixture
def hardware(monkeypatch, tmp_path):
    monkeypatch.setenv("XCELSIOR_MANAGED_HARDWARE_ONLY", "true")
    monkeypatch.setenv("XCELSIOR_DEDICATED_HOST_ENABLED", "true")
    state = {"inventory": "GPU-one\nGPU-two\n", "processes": ""}

    def smi(args, **kw):
        assert args[0] == "nvidia-smi" and kw["check"] is True
        return SimpleNamespace(
            stdout=state["inventory"] if "--query-gpu=uuid" in args else state["processes"]
        )

    monkeypatch.setattr(isolation.subprocess, "run", smi)
    return state, str(tmp_path)


def test_gpu_only_ignores_processes_on_other_gpu(hardware):
    state, directory = hardware
    state["processes"] = "GPU-two, 123\n"
    acquired = isolation.acquire_scope(["GPU-one"], lock_dir=directory)
    assert len(acquired["fds"]) == 1
    isolation.release_scope(acquired)
    assert acquired.get("fds", []) == []


def test_busy_allocated_gpu_fails_and_releases_locks(hardware):
    state, directory = hardware
    state["processes"] = "GPU-one, 123\n"
    with pytest.raises(ValueError, match="occupied"):
        isolation.acquire_scope(["GPU-one"], lock_dir=directory)
    state["processes"] = ""
    acquired = isolation.acquire_scope(["GPU-one"], lock_dir=directory)
    isolation.release_scope(acquired)


def test_dedicated_requires_every_physical_gpu(hardware):
    _, directory = hardware
    with pytest.raises(ValueError, match="every physical GPU"):
        isolation.acquire_scope(["GPU-one"], physical_host=True, lock_dir=directory)
    acquired = isolation.acquire_scope(
        ["GPU-one", "GPU-two"], physical_host=True, lock_dir=directory
    )
    assert len(acquired["fds"]) == 2
    isolation.release_scope(acquired)


def test_other_agent_lock_excludes_double_execution(hardware):
    _, directory = hardware
    fd = os.open(os.path.join(directory, "exclusive-GPU-one.lock"), os.O_CREAT | os.O_WRONLY, 0o660)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            isolation.acquire_scope(["GPU-one"], lock_dir=directory)
    finally:
        os.close(fd)


@pytest.mark.parametrize("uuids", [[], ["GPU-one", "GPU-one"], ["0"], ["GPU-one;all"], ["MIG-one"]])
def test_invalid_or_nonphysical_allocation_is_refused(hardware, uuids):
    with pytest.raises(ValueError, match="physical GPU UUIDs"):
        isolation.acquire_scope(uuids, lock_dir=hardware[1])


def test_unreadable_occupancy_is_refused(hardware):
    state, directory = hardware
    state["processes"] = "[Not Supported]"
    with pytest.raises(ValueError, match="could not be verified"):
        isolation.acquire_scope(["GPU-one"], lock_dir=directory)


def test_unqualified_hosts_cannot_sell_exclusivity(hardware, monkeypatch):
    monkeypatch.delenv("XCELSIOR_MANAGED_HARDWARE_ONLY")
    assert isolation.qualified_capabilities() == {
        "gpu_exclusivity": False,
        "dedicated_host": False,
        "gpu_uuid_binding": True,
    }
    with pytest.raises(ValueError, match="not qualified"):
        isolation.acquire_scope(["GPU-one"], lock_dir=hardware[1])


def test_inventory_uses_all_physical_uuids_and_exact_vram(monkeypatch):
    monkeypatch.setattr(
        isolation.subprocess,
        "run",
        lambda *args, **kw: SimpleNamespace(
            stdout="0, GPU-one, NVIDIA GeForce RTX 4090, 24576, 24000\n1, GPU-two, NVIDIA GeForce RTX 4090, 24576, 23000\n"
        ),
    )
    inventory = isolation.collect_inventory()
    assert [gpu["gpu_uuid"] for gpu in inventory] == ["GPU-one", "GPU-two"]
    assert [gpu["total_vram_mb"] for gpu in inventory] == [24576, 24576]


@pytest.mark.parametrize(
    "output", ["", "0, slot:0, GPU, 24576, 24000", "0, GPU-one, GPU, 24576, 25000"]
)
def test_unreadable_or_invalid_inventory_is_not_advertised(monkeypatch, output):
    monkeypatch.setattr(
        isolation.subprocess, "run", lambda *args, **kw: SimpleNamespace(stdout=output)
    )
    with pytest.raises(ValueError):
        isolation.collect_inventory()


def test_docker_uses_the_exact_two_allocated_devices():
    args = build_secure_docker_args(
        "pytorch/pytorch:latest",
        "test-container",
        gpu=True,
        num_gpus=2,
        gpu_device_uuids=["GPU-one", "GPU-two"],
    )
    assert args[args.index("--gpus") + 1] == '"device=GPU-one,GPU-two"'
    assert "all" not in args


def test_missing_v2_gpu_binding_never_falls_back_to_all_devices():
    with pytest.raises(ValueError, match="physical GPU UUIDs"):
        build_secure_docker_args(
            "pytorch/pytorch:latest", "test-container", gpu=True, gpu_device_uuids=[]
        )


def test_adoption_reacquires_only_its_journaled_gpu_locks(hardware):
    state, directory = hardware
    state["processes"] = "GPU-one, 123\n"
    # The survivor already runs on GPU-one; restoring its lock must not
    # mistake its own process for a new competing workload.
    restored = isolation.restore_scope(["GPU-one"], lock_dir=directory)
    with pytest.raises(BlockingIOError):
        isolation.restore_scope(["GPU-one"], lock_dir=directory)
    other = isolation.restore_scope(["GPU-two"], lock_dir=directory)
    isolation.release_scope(other)
    isolation.release_scope(restored)
