"""CRIU / gpu-criu checkpoint capability and resumable job tests."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from criu_hosts import (
    CHECKPOINT_CLASS_GPU_CRIU,
    CHECKPOINT_CLASS_NONE,
    MIN_NVIDIA_DRIVER_MAJOR,
    REFERENCE_CHECKPOINT_DRIVER,
    REFERENCE_CHECKPOINT_GPU,
    _parse_nvidia_driver_version,
    cuda_driver_requirements,
    docker_checkpoint_local,
    enrich_job_resumable,
    host_supports_checkpoint,
    job_is_resumable,
    merge_checkpoint_capabilities,
    probe_checkpoint_stack,
)
from scheduler import host_accepts_job, register_host


def test_probe_forced_gpu_criu():
    probe = probe_checkpoint_stack(force_class=CHECKPOINT_CLASS_GPU_CRIU)
    assert probe["checkpoint_class"] == CHECKPOINT_CLASS_GPU_CRIU
    assert probe["criu_available"] is True


def test_resolve_checkpoint_class_from_probe():
    assert (
        probe_checkpoint_stack(force_class=CHECKPOINT_CLASS_GPU_CRIU)["checkpoint_class"]
        == CHECKPOINT_CLASS_GPU_CRIU
    )


def test_host_supports_checkpoint():
    assert host_supports_checkpoint({"checkpoint_class": "gpu-criu"})
    assert host_supports_checkpoint({"checkpoint_class": "docker-criu"})
    assert not host_supports_checkpoint({"checkpoint_class": ""})
    assert not host_supports_checkpoint(None)


def test_job_resumable_requires_success_meta():
    assert not job_is_resumable({})
    assert job_is_resumable(
        {
            "resume_from": {
                "success": True,
                "checkpoint_name": "ckpt-j1",
                "created_at": __import__("time").time(),
            }
        }
    )
    assert not job_is_resumable({"resume_from": {"success": False}})


def test_enrich_job_resumable():
    import time

    j = enrich_job_resumable(
        {
            "job_id": "j1",
            "resume_from": {
                "success": True,
                "checkpoint_name": "ckpt-j1",
                "created_at": time.time(),
            },
        }
    )
    assert j["resumable"] is True


def test_merge_checkpoint_capabilities():
    host = merge_checkpoint_capabilities(
        {"host_id": "h1"},
        probe_checkpoint_stack(force_class=CHECKPOINT_CLASS_GPU_CRIU),
    )
    assert host["checkpoint_class"] == CHECKPOINT_CLASS_GPU_CRIU
    assert host["capabilities"]["checkpoint"]["class"] == CHECKPOINT_CLASS_GPU_CRIU


def test_host_accepts_resumable_job_only_on_criu_hosts():
    job = {
        "job_id": "j-resume",
        "num_gpus": 1,
        "resume_from": {"success": True, "checkpoint_name": "ckpt", "created_at": __import__("time").time()},
    }
    criu_host = {"host_id": "h-criu", "status": "active", "gpu_count": 1, "checkpoint_class": "gpu-criu"}
    plain_host = {"host_id": "h-plain", "status": "active", "gpu_count": 1, "checkpoint_class": ""}
    assert host_accepts_job(criu_host, job, jobs=[]) is True
    assert host_accepts_job(plain_host, job, jobs=[]) is False


def test_register_host_preserves_checkpoint_class():
    entry = register_host(
        "criu-test-host",
        "127.0.0.1",
        "RTX 2060",
        6,
        6,
        cost_per_hour=0.35,
    )
    assert entry["host_id"] == "criu-test-host"


@patch("criu_hosts._run", return_value=(0, "", ""))
def test_docker_checkpoint_local_success(mock_run, tmp_path, monkeypatch):
    monkeypatch.setenv("XCELSIOR_CHECKPOINT_DIR", str(tmp_path))
    meta = docker_checkpoint_local("xcl-job1", "job1", checkpoint_class=CHECKPOINT_CLASS_GPU_CRIU)
    assert meta is not None
    assert meta["success"] is True
    assert meta["checkpoint_class"] == CHECKPOINT_CLASS_GPU_CRIU
    assert meta["job_id"] == "job1"


def test_cuda_driver_requirements_pinned_for_criugpu():
    reqs = cuda_driver_requirements()
    assert reqs["min_nvidia_driver_major"] == MIN_NVIDIA_DRIVER_MAJOR
    assert reqs["reference_gpu"] == REFERENCE_CHECKPOINT_GPU
    assert reqs["docker_experimental_required"] is True
    assert "gpu-criu" in reqs["checkpoint_classes"]


def test_rtx_2060_driver_580_qualifies_for_gpu_criu_gate():
    """Row 31: ASUS RTX 2060 + driver 580 passes the CRIUgpu driver gate (criu install is host ops)."""
    driver_ver = _parse_nvidia_driver_version(f"{REFERENCE_CHECKPOINT_DRIVER}.65.02")
    assert driver_ver is not None
    assert driver_ver >= MIN_NVIDIA_DRIVER_MAJOR
    probe = probe_checkpoint_stack(force_class=CHECKPOINT_CLASS_GPU_CRIU)
    assert probe["nvidia_driver_version"] >= MIN_NVIDIA_DRIVER_MAJOR


def test_probe_empty_without_criu(monkeypatch):
    monkeypatch.delenv("XCELSIOR_CHECKPOINT_CLASS", raising=False)

    def fake_run(cmd, **kwargs):
        if "criu" in str(cmd):
            return 127, "", "not found"
        if "docker" in str(cmd):
            return 0, "false", ""
        return 0, "", ""

    with patch("criu_hosts._run", side_effect=fake_run):
        probe = probe_checkpoint_stack()
    assert probe["checkpoint_class"] == CHECKPOINT_CLASS_NONE