"""What the worker sends for verification, and how it says whose fault a failure was.

Phase 0 (2026-11-22). Before this the agent sent no GPU UUID or serial, the NVML
helper reported the UUID as the serial on cards without one, a busy host that
declined a reverification never asked again, and a host-side failure carried no
code the server could tell apart from a renter's crash.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import threading
import types
from pathlib import Path
from unittest import mock

import pytest

import nvml_telemetry
import worker_agent

UUID_A = "GPU-f22a551d-2bee-2f17-22af-454481edcdc7"
UUID_B = "GPU-0b1c2d3e-4f50-6172-8394-a5b6c7d8e9f0"


def _smi(stdout: str, returncode: int = 0):
    return mock.Mock(return_value=subprocess.CompletedProcess([], returncode, stdout, ""))


# ── Identity: UUID always, the board serial only when there is one ─────


def test_nvidia_smi_identity_keeps_the_uuid_and_drops_a_missing_serial(monkeypatch):
    monkeypatch.setattr(nvml_telemetry, "_nvml_initialized", False)
    out = (
        f"0, {UUID_A}, [N/A], 00000000:01:00.0, NVIDIA GeForce RTX 3060, 12288\n"
        f"1, {UUID_B}, 1324021034567, 00000000:02:00.0, NVIDIA A100-SXM4-40GB, 40960\n"
    )
    with mock.patch("subprocess.run", _smi(out)):
        ids = nvml_telemetry.collect_gpu_identities()
    assert [i["gpu_uuid"] for i in ids] == [UUID_A, UUID_B]
    assert [i["serial"] for i in ids] == ["", "1324021034567"]
    assert ids[0]["pci_bus_id"] == "00000000:01:00.0" and ids[1]["total_vram_mb"] == 40960


def test_a_uuid_in_the_serial_column_is_not_a_serial(monkeypatch):
    monkeypatch.setattr(nvml_telemetry, "_nvml_initialized", False)
    out = f"0, {UUID_A}, {UUID_A}, 00000000:01:00.0, NVIDIA GeForce RTX 2060, 6144\n"
    with mock.patch("subprocess.run", _smi(out)):
        assert nvml_telemetry.collect_gpu_identities()[0]["serial"] == ""


def test_an_unreadable_gpu_makes_the_whole_identity_unknown(monkeypatch):
    """A partial list would read as a lost card; nothing reads as unknown."""
    monkeypatch.setattr(nvml_telemetry, "_nvml_initialized", False)
    out = f"0, {UUID_A}, , 00000000:01:00.0, NVIDIA GeForce RTX 3060, 12288\n1, [N/A], , , ,\n"
    with mock.patch("subprocess.run", _smi(out)):
        assert nvml_telemetry.collect_gpu_identities() is None
    with mock.patch("subprocess.run", _smi("", returncode=9)):
        assert nvml_telemetry.collect_gpu_identities() is None


def test_nvml_identity_reads_every_device_and_tolerates_a_card_without_a_serial(monkeypatch):
    class NVMLError(Exception):
        pass

    def serial(handle):
        if handle == 0:
            raise NVMLError("Not Supported")
        return b"1650123456789"

    fake = types.SimpleNamespace(
        nvmlDeviceGetCount=lambda: 2,
        nvmlDeviceGetHandleByIndex=lambda i: i,
        nvmlDeviceGetSerial=serial,
        nvmlDeviceGetUUID=lambda h: (UUID_A if h == 0 else UUID_B).encode(),
        nvmlDeviceGetName=lambda h: b"NVIDIA GeForce RTX 3060",
        nvmlDeviceGetMemoryInfo=lambda h: types.SimpleNamespace(total=12 * 1024**3),
    )
    monkeypatch.setattr(nvml_telemetry, "pynvml", fake)
    monkeypatch.setattr(nvml_telemetry, "_nvml_initialized", True)
    monkeypatch.setattr(
        nvml_telemetry, "get_gpu_pci_info", lambda h: {"bus_id": f"00000000:0{h + 1}:00.0"}
    )
    ids = nvml_telemetry.collect_gpu_identities()
    assert [(i["gpu_uuid"], i["serial"]) for i in ids] == [(UUID_A, ""), (UUID_B, "1650123456789")]
    assert ids[0]["total_vram_mb"] == 12288


def test_the_agent_falls_back_to_nvidia_smi_beside_an_older_nvml_module(monkeypatch):
    """`upgrade_agent` does not refresh nvml_telemetry.py, so it may lack the function."""
    monkeypatch.setitem(__import__("sys").modules, "nvml_telemetry", types.ModuleType("old"))
    out = f"0, {UUID_A}, [N/A], 00000000:01:00.0, NVIDIA GeForce RTX 2060, 6144\n"
    with mock.patch.object(worker_agent.subprocess, "run", _smi(out)):
        ids = worker_agent.collect_gpu_identities_for_verification()
    assert ids == [
        {
            "index": 0,
            "gpu_uuid": UUID_A,
            "serial": "",
            "pci_bus_id": "00000000:01:00.0",
            "model": "NVIDIA GeForce RTX 2060",
            "total_vram_mb": 6144,
        }
    ]


def test_the_verification_cycle_sends_identity_and_versions(monkeypatch):
    sent = {}
    monkeypatch.setattr(worker_agent, "run_compute_benchmark", lambda: {"tflops": 10.0})
    monkeypatch.setattr(worker_agent, "report_benchmark", lambda *a, **k: None)
    monkeypatch.setattr(worker_agent, "run_network_benchmark", lambda: None)
    monkeypatch.setattr(worker_agent, "report_verification", lambda r: sent.update(r))
    monkeypatch.setattr(
        worker_agent,
        "collect_gpu_identities_for_verification",
        lambda: [{"index": 0, "gpu_uuid": UUID_A, "serial": "", "pci_bus_id": "x"}],
    )
    worker_agent.run_verification_cycle(
        {"gpu_model": "RTX 3060", "total_vram_gb": 12.0}, {"runc": "1.4.0"}
    )
    assert sent["gpu_uuid"] == UUID_A and sent["serial"] == ""
    assert sent["gpu_identities"][0]["gpu_uuid"] == UUID_A
    assert sent["versions"] == {"runc": "1.4.0"}


# ── The benchmark ───────────────────────────────────────────────────


def _benchmark_source() -> str:
    return (Path(worker_agent.__file__).parent / worker_agent.BENCH_FILE).read_text()


def test_the_benchmark_names_nothing_it_does_not_define():
    """The old inline script referenced `log`, which the subprocess never defined."""
    tree = ast.parse(_benchmark_source())
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "log" not in assigned


def test_the_benchmark_measures_allocatable_vram_and_reports_a_dead_sensor_as_none():
    import xcelsior_bench

    assert "mem_get_info" in _benchmark_source()
    summary = xcelsior_bench.summarize_sustained(
        [{"sm_clock_mhz": 1800, "temp_c": 0, "power_w": None, "throttle_bits": 0}] * 5, 5.0
    )
    assert summary["temp_peak_c"] is None


def test_the_flat_fields_come_from_the_benchmark_report():
    bench = {
        "bench_version": "1.0.0",
        "gpu_model": "NVIDIA GeForce RTX 3060",
        "allocatable_vram_gb": 11.2,
        "fp16_tflops": 25.0,
        "sustained": {"temp_peak_c": 71, "temp_p50_c": 68.0, "samples": 300},
        "errors": {"pcie": "boom"},
    }
    fields = worker_agent._verification_fields(bench)
    assert fields["tflops"] == 25.0 and fields["allocatable_vram_gb"] == 11.2
    assert fields["gpu_temp_celsius"] == 71 and fields["gpu_temp_samples"] == 300
    assert fields["pcie_error"] == "boom" and fields["pcie_bandwidth_gbps"] == 0
    assert fields["bench"] is bench
    # A sustained phase that read no temperature reports no samples.
    no_temp = worker_agent._verification_fields({"sustained": {"samples": 300}})
    assert no_temp["gpu_temp_samples"] == 0


# ── Busy hosts remember a reverification ─────────────────────────────


@pytest.fixture
def idle_agent(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_agent, "_active_containers", {})
    monkeypatch.setattr(worker_agent, "_v2_attempts", {})
    monkeypatch.setattr(worker_agent, "_v2_starts_in_progress", set())
    monkeypatch.setattr(worker_agent, "_reverify_running", threading.Event())
    monkeypatch.setattr(worker_agent, "_reverify_pending", threading.Event())
    monkeypatch.setattr(worker_agent, "_REVERIFY_PENDING_PATH", str(tmp_path / "reverify-pending"))
    # Nothing outside Xcelsior on the GPU unless a test says so; never this machine's GPU.
    monkeypatch.setattr(worker_agent, "gpu_outside_use", lambda: {})
    monkeypatch.setattr(worker_agent, "_outside_use_deferred_at", None)
    return worker_agent


def test_a_busy_host_defers_the_reverification_and_runs_it_once_idle(idle_agent):
    idle_agent._active_containers["job-1"] = {"container": "xcl-job-1"}
    with mock.patch.object(idle_agent.threading, "Thread") as thread:
        assert idle_agent._start_reverification() is False
        assert idle_agent._reverify_pending.is_set()
        # Still busy: the main loop asks again and is declined again.
        assert idle_agent._run_pending_reverification() is False
        thread.assert_not_called()

        idle_agent._active_containers.clear()
        assert idle_agent._run_pending_reverification() is True
        thread.assert_called_once()
    assert not idle_agent._reverify_pending.is_set()


def test_a_deferred_reverify_command_is_accepted_not_failed(idle_agent):
    """A busy host remembers the request; the command must not be retried to death.

    Prod 2026-11-22: both hosts ran a long instance, declined, nacked, and every
    hourly request ended in dead_letter although the agent had it pending.
    """
    idle_agent._active_containers["job-1"] = {"container": "xcl-job-1"}
    cmd = {"id": "cmd-1", "command": "reverify", "args": {"reason": "scheduled_reverification"}}
    with mock.patch.object(idle_agent.threading, "Thread") as thread:
        assert idle_agent._dispatch_agent_commands([cmd]) == 1
        assert idle_agent._reverify_pending.is_set()
        # A second request while one is already pending is accepted too.
        assert idle_agent._dispatch_agent_commands([cmd]) == 1
        thread.assert_not_called()


def test_a_deferred_reverification_survives_an_agent_restart(idle_agent, monkeypatch):
    """In memory only, an upgrade or reboot forgot it; the idle host then waited
    up to an hour for the sweep to ask again."""
    idle_agent._active_containers["job-1"] = {"container": "xcl-job-1"}
    with mock.patch.object(idle_agent.threading, "Thread"):
        assert idle_agent._start_reverification() is False
    assert Path(idle_agent._REVERIFY_PENDING_PATH).exists()

    # A new process: the event starts clear, the file is still there.
    monkeypatch.setattr(idle_agent, "_reverify_pending", threading.Event())
    idle_agent._active_containers.clear()
    assert idle_agent._restore_reverify_pending() is True
    with mock.patch.object(idle_agent.threading, "Thread") as thread:
        assert idle_agent._run_pending_reverification() is True
        thread.assert_called_once()
    # Started: nothing is left to restore after the next restart.
    assert not Path(idle_agent._REVERIFY_PENDING_PATH).exists()
    assert idle_agent._restore_reverify_pending() is False


def test_nothing_pending_starts_nothing(idle_agent):
    with mock.patch.object(idle_agent.threading, "Thread") as thread:
        assert idle_agent._run_pending_reverification() is False
    thread.assert_not_called()


def test_a_gpu_in_outside_use_defers_the_benchmark_instead_of_failing_it(idle_agent, monkeypatch):
    """Prod 2026-11-25: the tower's 3060 failed every hourly benchmark with
    allocatable_vram_below_listing (10 of 12 GB) because the owner's own
    processes held 2.9 GB, and was de-verified for it. The card was sound."""
    outside = {"GPU-tower": {"processes": 8, "vram_mb": 2908}}
    asked = []

    def listing():
        asked.append(1)
        return outside

    monkeypatch.setattr(idle_agent, "gpu_outside_use", listing)
    with mock.patch.object(idle_agent.threading, "Thread") as thread:
        assert idle_agent._start_reverification() is False
        assert idle_agent._reverify_pending.is_set()
        assert not idle_agent._reverify_running.is_set()  # jobs may start again
        # The main loop runs every few seconds; the GPU is asked again only
        # once per observation interval.
        assert idle_agent._run_pending_reverification() is False
        assert len(asked) == 1
        thread.assert_not_called()

        monkeypatch.setattr(idle_agent, "_outside_use_deferred_at", None)
        outside = {}
        assert idle_agent._run_pending_reverification() is True
        thread.assert_called_once()
    assert not idle_agent._reverify_pending.is_set()


def test_an_unknown_gpu_process_list_does_not_hold_the_benchmark(idle_agent, monkeypatch):
    """Without a process list the benchmark runs; its allocatable-VRAM check still
    catches memory held by something the agent cannot see."""
    monkeypatch.setattr(idle_agent, "gpu_outside_use", lambda: None)
    with mock.patch.object(idle_agent.threading, "Thread") as thread:
        assert idle_agent._start_reverification() is True
        thread.assert_called_once()


# ── GPU use outside Xcelsior ──────────────────────────────────────────

_OWN_CID, _OTHER_CID = "a" * 64, "b" * 64


@pytest.fixture
def gpu_processes(monkeypatch):
    """nvidia-smi, docker ps and /proc as the agent reads them, from a table."""
    state = {
        "apps": "",
        "apps_rc": 0,
        "docker": f"{_OWN_CID}\txcl-job-1\t\n{_OTHER_CID}\tsomeones-db\t\n",
        "docker_rc": 0,
        "docker_calls": 0,
    }
    procs: dict[int, dict] = {}

    def run(cmd, **_kwargs):
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, state["apps_rc"], state["apps"], "")
        if cmd[:2] == ["docker", "ps"]:
            state["docker_calls"] += 1
            return subprocess.CompletedProcess(cmd, state["docker_rc"], state["docker"], "")
        raise AssertionError(f"unexpected command {cmd}")

    def proc_text(pid, name):
        entry = procs.get(pid)
        if entry is None:
            raise FileNotFoundError(pid)
        if entry.get("unreadable"):
            raise PermissionError(pid)
        if name == "cgroup":
            return entry["cgroup"]
        return f"{pid} (python3 (worker)) S {entry['ppid']} {pid} {pid} 0 -1\n"

    monkeypatch.setattr(worker_agent.subprocess, "run", run)
    monkeypatch.setattr(worker_agent, "_proc_text", proc_text)
    return state, procs


def _user(ppid=1):
    return {"cgroup": "0::/user.slice/user-1000.slice/session-2.scope\n", "ppid": ppid}


def _container(cid):
    return {"cgroup": f"0::/system.slice/docker-{cid}.scope\n", "ppid": 1}


def test_outside_use_counts_only_processes_the_agent_did_not_start(gpu_processes):
    state, procs = gpu_processes
    state["apps"] = (
        "GPU-a, 11, 1000\n"  # a job container (xcl-*)
        "GPU-a, 12, 318\n"  # the owner's own script
        "GPU-a, 13, [N/A]\n"  # another project's container
        "GPU-b, 14, 2000\n"  # the benchmark, run in the agent's own Python
        "GPU-b, 15, 500\n"  # exited since nvidia-smi listed it
    )
    procs.update(
        {
            11: _container(_OWN_CID),
            12: _user(),
            13: _container(_OTHER_CID),
            14: {"cgroup": "0::/system.slice/xcelsior-worker.service\n", "ppid": os.getpid()},
        }
    )
    assert worker_agent.gpu_outside_use() == {"GPU-a": {"processes": 2, "vram_mb": 318}}


def test_a_gpu_with_no_compute_processes_has_no_outside_use(gpu_processes):
    state, _ = gpu_processes
    assert worker_agent.gpu_outside_use() == {}
    assert state["docker_calls"] == 0


def test_a_process_list_that_cannot_be_read_is_unknown_not_empty(gpu_processes):
    state, _ = gpu_processes
    state["apps_rc"] = 9
    assert worker_agent.gpu_outside_use() is None
    state["apps_rc"], state["apps"] = 0, "Failed to initialize NVML\n"
    assert worker_agent.gpu_outside_use() is None


def test_whose_process_it_is_falls_to_outside_when_it_cannot_be_read(gpu_processes):
    state, procs = gpu_processes
    state["apps"] = "GPU-a, 11, 1000\nGPU-a, 12, 50\n"
    procs.update({11: _container(_OWN_CID), 12: {**_user(), "unreadable": True}})
    state["docker_rc"] = 1  # Docker cannot say which containers are ours
    assert worker_agent.gpu_outside_use() == {"GPU-a": {"processes": 2, "vram_mb": 1050}}


def test_each_observed_gpu_carries_its_outside_use(monkeypatch):
    monkeypatch.setattr(worker_agent, "is_nvml_available", lambda: True)
    monkeypatch.setattr(
        nvml_telemetry,
        "collect_gpu_inventory",
        lambda: [
            {"gpu_uuid": "GPU-a", "device_index": 0, "model": "RTX 3060", "total_vram_mb": 12288},
            {"gpu_uuid": "GPU-b", "device_index": 1, "model": "RTX 3060", "total_vram_mb": 12288},
        ],
    )
    monkeypatch.setattr(
        worker_agent, "gpu_outside_use", lambda: {"GPU-a": {"processes": 3, "vram_mb": 900}}
    )
    inventory, conditions = worker_agent._observation_inventory()
    assert [item["outside_use"] for item in inventory] == [
        {"processes": 3, "vram_mb": 900},
        {"processes": 0, "vram_mb": 0},
    ]
    assert conditions == {"gpu_inventory": "reported", "gpu_outside_use": "reported"}

    monkeypatch.setattr(worker_agent, "gpu_outside_use", lambda: None)
    inventory, conditions = worker_agent._observation_inventory()
    assert all("outside_use" not in item for item in inventory)
    assert conditions["gpu_outside_use"] == "unavailable"


# ── Host-fault failures carry a code the server classifies ───────────


@pytest.mark.parametrize(
    "last_line, code",
    [
        ("write /var/lib/docker/tmp: no space left on device", "host_disk_full"),
        ("error getting credentials - err: exit status 1", "host_registry_credentials"),
        ("manifest unknown: manifest unknown", "image_not_found"),
        ("pull access denied for acme/private", "image_access_denied"),
        ("dial tcp: lookup registry-1.docker.io: no such host", None),
        ("toomanyrequests: You have reached your pull rate limit", None),
    ],
)
def test_pull_failures_are_coded_by_whose_fault_they_are(last_line, code):
    assert worker_agent._pull_failure_code(last_line) == code


def test_every_host_fault_the_agent_reports_is_one_the_server_counts():
    from failure_fault import CUSTOMER_FAULT_CODES, HOST_FAULT_CODES

    source = Path(worker_agent.__file__).read_text(encoding="utf-8")
    emitted = set(re.findall(r'policy_reason="((?:host|image|container)_[a-z_]+)"', source))
    emitted |= {code for _, code in worker_agent._PULL_FAILURE_CODES}
    host_side = {c for c in emitted if c.startswith("host_")}
    assert host_side, "no host-fault codes found; the pattern moved"
    assert host_side <= HOST_FAULT_CODES, host_side - HOST_FAULT_CODES
    image_side = {c for c in emitted if c.startswith("image_")}
    assert image_side <= CUSTOMER_FAULT_CODES, image_side - CUSTOMER_FAULT_CODES
