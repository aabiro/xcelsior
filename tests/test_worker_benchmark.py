"""The verification benchmark runs in a PyTorch container when the agent's venv has none.

install.sh gives the agent a venv without PyTorch, so before this every installed host
skipped the benchmark, never sent a verification report, and stayed pending.
"""

import json
from types import SimpleNamespace

import worker_agent
import xcelsior_bench

# An xcelsior-bench report shaped like a tower RTX 3060 run. The device fields
# are from the tower's 2026-10-11 benchmark; the sustained phase is built with
# the benchmark's own summary function from synthetic one-second samples.
TOWER_RTX3060 = {
    "schema": xcelsior_bench.SCHEMA,
    "bench_version": xcelsior_bench.BENCH_VERSION,
    "gpu_model": "NVIDIA GeForce RTX 3060",
    "total_vram_gb": 11.61,
    "compute_capability": "8.6",
    "cuda_version": "12.4",
    "driver_version": "590.48.01",
    "fp16_tflops": 24.68,
    "bf16_tflops": 24.1,
    "memory_bandwidth_gbps": 330.0,
    "allocatable_vram_gb": 11.2,
    "pcie_bandwidth_gbps": 5.23,
    "pcie_h2d_gbps": 8.64,
    "pcie_d2h_gbps": 1.82,
    "sustained": xcelsior_bench.summarize_sustained(
        [{"sm_clock_mhz": 1777, "temp_c": 34.0, "power_w": 165.0, "throttle_bits": 4}] * 300,
        300.0,
    ),
    "errors": {},
}
EXPECTED = worker_agent._verification_fields(TOWER_RTX3060)
NO_TORCH = {"error": "no_torch"}


def _done(stdout="", returncode=0):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _install(monkeypatch, venv, *, image_present=True, pull_ok=False):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == worker_agent.sys.executable:
            return _done(json.dumps(venv))
        if cmd[:3] == ["docker", "image", "inspect"]:
            return _done(returncode=0 if image_present else 1)
        if cmd[0] == "docker" and "pull" in cmd:
            return _done(returncode=0 if pull_ok else 1)
        if cmd[:2] == ["docker", "run"]:
            return _done(json.dumps(TOWER_RTX3060))
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(worker_agent.subprocess, "run", run)
    return calls


def test_benchmarks_in_container_when_venv_has_no_torch(monkeypatch):
    calls = _install(monkeypatch, NO_TORCH)
    assert worker_agent.run_compute_benchmark() == EXPECTED
    container = next(c for c in calls if c[:2] == ["docker", "run"])
    assert container[container.index("--gpus") + 1] == "all"
    assert container[container.index("--network") + 1] == "none"
    assert worker_agent.BENCHMARK_IMAGE in container


def test_pulls_the_image_once_when_missing(monkeypatch):
    calls = _install(monkeypatch, NO_TORCH, image_present=False, pull_ok=True)
    assert worker_agent.run_compute_benchmark() == EXPECTED
    assert worker_agent.registry_docker_cmd("pull", worker_agent.BENCHMARK_IMAGE) in calls


def test_the_report_carries_the_fields_the_server_checks_and_the_full_run():
    assert EXPECTED["tflops"] == 24.68
    assert EXPECTED["gpu_temp_celsius"] == 34.0
    assert EXPECTED["gpu_temp_samples"] == 300
    assert EXPECTED["allocatable_vram_gb"] == 11.2
    assert EXPECTED["bench"] == TOWER_RTX3060


def test_no_report_when_the_image_is_unavailable(monkeypatch):
    calls = _install(monkeypatch, NO_TORCH, image_present=False, pull_ok=False)
    assert worker_agent.run_compute_benchmark() is None
    assert not any(c[:2] == ["docker", "run"] for c in calls)


def test_venv_torch_is_used_without_docker(monkeypatch):
    calls = _install(monkeypatch, TOWER_RTX3060)
    assert worker_agent.run_compute_benchmark() == EXPECTED
    assert all(c[0] != "docker" for c in calls)
