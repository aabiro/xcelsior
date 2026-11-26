"""An instance keeps its GPU when the host reloads systemd.

With ``--gpus`` the NVIDIA hook grants the device nodes outside the container's
spec, and the host's next ``systemctl daemon-reload`` revoked them: on
aaryn-tuf-rtx2060 (Docker 29.1.3, runc, systemd cgroups) every running
instance's nvidia-smi said "Failed to initialize NVML: Unknown Error". A CDI
device (``--device nvidia.com/gpu=<UUID>``) is in the spec and survives the
reload; both were checked on that host with PyTorch before and after a reload.
"""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from security import build_secure_docker_args
import worker_agent

UUID = "GPU-f22a551d-2bee-2f17-22af-454481edcdc7"
LISTED = [
    {"Source": "cdi", "ID": "nvidia.com/gpu=0"},
    {"Source": "cdi", "ID": f"nvidia.com/gpu={UUID}"},
    {"Source": "cdi", "ID": "nvidia.com/gpu=all"},
]


def _docker_info(monkeypatch, stdout: str | None = None, raises: Exception | None = None):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        if raises:
            raise raises
        return SimpleNamespace(
            returncode=0, stdout=stdout if stdout is not None else json.dumps(LISTED)
        )

    monkeypatch.setattr(worker_agent.subprocess, "run", run)
    monkeypatch.delenv("XCELSIOR_GPU_CDI", raising=False)
    return calls


def _args(runtime: str = "runc", uuids=(UUID,)) -> list[str]:
    return build_secure_docker_args(
        image="nvcr.io/nvidia/pytorch:24.12-py3",
        container_name="xcl-job-1",
        gpu=True,
        num_gpus=len(uuids),
        gpu_device_uuids=list(uuids),
        runtime=runtime,
        environment={},
        interactive=True,
    )


def test_a_runc_instance_gets_its_gpu_by_cdi(monkeypatch):
    calls = _docker_info(monkeypatch)
    before = _args()
    after = worker_agent._gpu_request_via_cdi(before, [UUID])
    assert "--gpus" not in after
    at = after.index("--device")
    assert after[at : at + 2] == ["--device", f"nvidia.com/gpu={UUID}"]
    # Nothing else moves: same flags, same pinned NVIDIA_VISIBLE_DEVICES.
    rest = before[: before.index("--gpus")] + before[before.index("--gpus") + 2 :]
    assert after[:at] + after[at + 2 :] == rest
    assert f"NVIDIA_VISIBLE_DEVICES={UUID}" in after
    assert calls == [["docker", "info", "--format", "{{json .DiscoveredDevices}}"]]


def test_every_reserved_gpu_becomes_its_own_device(monkeypatch):
    second = "GPU-0000aaaa-bbbb-cccc-dddd-eeeeffff0000"
    _docker_info(
        monkeypatch, json.dumps(LISTED + [{"Source": "cdi", "ID": f"nvidia.com/gpu={second}"}])
    )
    after = worker_agent._gpu_request_via_cdi(_args(uuids=(UUID, second)), [UUID, second])
    devices = [after[i + 1] for i, a in enumerate(after) if a == "--device"]
    assert devices == [f"nvidia.com/gpu={UUID}", f"nvidia.com/gpu={second}"]


@pytest.mark.parametrize(
    "stdout, raises",
    [
        (json.dumps([{"Source": "cdi", "ID": "nvidia.com/gpu=all"}]), None),  # not this GPU
        ("null", None),  # an older Docker with no device discovery
        ("", None),
        ("not json", None),
        (None, OSError("docker missing")),
        (None, subprocess.TimeoutExpired("docker", 10)),
    ],
)
def test_anything_short_of_a_listed_gpu_keeps_the_gpus_flag(monkeypatch, stdout, raises):
    _docker_info(monkeypatch, stdout, raises)
    before = _args()
    assert worker_agent._gpu_request_via_cdi(before, [UUID]) == before


def test_gvisor_keeps_its_own_gpu_path(monkeypatch):
    calls = _docker_info(monkeypatch)
    before = _args(runtime="runsc")
    assert worker_agent._gpu_request_via_cdi(before, [UUID]) == before
    assert calls == [], "no need to ask Docker"


def test_a_gpus_value_for_other_gpus_is_left_alone(monkeypatch):
    _docker_info(monkeypatch)
    before = _args()
    assert worker_agent._gpu_request_via_cdi(before, ["GPU-not-the-reserved-one"]) == before


def test_the_switch_turns_it_off(monkeypatch):
    _docker_info(monkeypatch)
    monkeypatch.setenv("XCELSIOR_GPU_CDI", "0")
    before = _args()
    assert worker_agent._gpu_request_via_cdi(before, [UUID]) == before


def test_the_launch_path_uses_it():
    import inspect

    body = inspect.getsource(worker_agent)
    assert (
        'docker_args = _gpu_request_via_cdi(docker_args, authority.get("gpu_device_uuids"))' in body
    )
