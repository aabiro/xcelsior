"""The verification benchmark runs in a PyTorch container when the agent's venv has none.

install.sh gives the agent a venv without PyTorch, so before this every installed host
skipped the benchmark, never sent a verification report, and stayed pending.
"""

import json
from types import SimpleNamespace

import worker_agent

# Output of the containerised benchmark on the tower's RTX 3060, 2026-10-11.
TOWER_RTX3060 = {
    "gpu_model": "NVIDIA GeForce RTX 3060",
    "total_vram_gb": 11.61,
    "compute_capability": "8.6",
    "cuda_version": "12.4",
    "driver_version": "590.48.01",
    "tflops": 24.68,
    "elapsed_s": 0.278,
    "iters": 50,
    "pcie_bandwidth_gbps": 5.23,
    "pcie_h2d_gbps": 8.64,
    "pcie_d2h_gbps": 1.82,
    "gpu_temp_celsius": 34.0,
    "gpu_temp_avg_celsius": 32.9,
    "gpu_temp_samples": 29,
}
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
        if cmd[:2] == ["docker", "pull"]:
            return _done(returncode=0 if pull_ok else 1)
        if cmd[:2] == ["docker", "run"]:
            return _done(json.dumps(TOWER_RTX3060))
        raise AssertionError(f"unexpected command {cmd}")

    monkeypatch.setattr(worker_agent.subprocess, "run", run)
    return calls


def test_benchmarks_in_container_when_venv_has_no_torch(monkeypatch):
    calls = _install(monkeypatch, NO_TORCH)
    assert worker_agent.run_compute_benchmark() == TOWER_RTX3060
    container = next(c for c in calls if c[:2] == ["docker", "run"])
    assert container[container.index("--gpus") + 1] == "all"
    assert container[container.index("--network") + 1] == "none"
    assert worker_agent.BENCHMARK_IMAGE in container


def test_pulls_the_image_once_when_missing(monkeypatch):
    calls = _install(monkeypatch, NO_TORCH, image_present=False, pull_ok=True)
    assert worker_agent.run_compute_benchmark() == TOWER_RTX3060
    assert ["docker", "pull", worker_agent.BENCHMARK_IMAGE] in calls


def test_no_report_when_the_image_is_unavailable(monkeypatch):
    calls = _install(monkeypatch, NO_TORCH, image_present=False, pull_ok=False)
    assert worker_agent.run_compute_benchmark() is None
    assert not any(c[:2] == ["docker", "run"] for c in calls)


def test_venv_torch_is_used_without_docker(monkeypatch):
    calls = _install(monkeypatch, TOWER_RTX3060)
    assert worker_agent.run_compute_benchmark() == TOWER_RTX3060
    assert all(c[0] != "docker" for c in calls)
