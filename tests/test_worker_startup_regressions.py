"""Startup failures observed in production, with machine/network IO isolated."""

from unittest.mock import Mock, ANY
import itertools
import subprocess
import threading

import pytest
import requests

import worker_agent as worker


@pytest.fixture(autouse=True)
def native_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_protocol_negotiated", True)
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_v2_cleanup_pending", {})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "journal.json"))
    monkeypatch.setattr(worker, "_v2_container_state", Mock(return_value=("missing", -1, "")))


def _fenced_job(job_id, **spec):
    cname = f"xcl-{job_id}-attempt"
    worker._v2_attempts[job_id] = {
        "job_id": job_id,
        "attempt_id": "attempt",
        "lease_id": f"lease-{job_id}",
        "fencing_token": 7,
        "container_name": cname,
        "stop": threading.Event(),
    }
    return dict(
        spec, job_id=job_id, _v2_attempt_id="attempt", _v2_fencing_token=7, _v2_container_name=cname
    )


def test_managed_engine_flags_preserve_the_image_entrypoint():
    from serverless.service import _preset_startup_command

    command, overrides = worker._container_command(
        _preset_startup_command("vllm", "Qwen/Qwen3-8B", http_port=9123),
        serverless=True,
    )
    assert overrides == []
    assert command[:2] == ["--model", "Qwen/Qwen3-8B"]
    assert command[command.index("--port") + 1] == "9123"


@pytest.mark.parametrize("serverless", [True, False])
def test_custom_shell_commands_still_override_the_entrypoint(serverless):
    command, overrides = worker._container_command(
        "python app.py --name 'my model'", serverless=serverless
    )
    assert overrides == ["--entrypoint", "sh"]
    assert command == ["-c", "python app.py --name 'my model'"]


def test_explicit_argv_is_not_reinterpreted_as_shell():
    args = ["--model", "my model"]
    assert worker._container_command(args, serverless=True) == (args, [])


def test_partial_volume_mount_failure_cleans_up_without_masking_the_error(monkeypatch):
    for name in ("get_gpu_info", "build_platform_env"):
        monkeypatch.setattr(worker, name, Mock(return_value={}))
    for name in ("_kill_container", "_remove_container", "destroy_encrypted_volume"):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker.time, "sleep", Mock())
    report = Mock(return_value=True)
    monkeypatch.setattr(worker, "report_job_status", report)
    monkeypatch.setattr(worker, "_mount_nfs", Mock(side_effect=[True, False]))
    unmount = Mock()
    monkeypatch.setattr(worker, "_unmount_nfs", unmount)

    worker.run_job(
        _fenced_job(
            "mount-failure-regression",
            image="test/image:1",
            volume_ids=["vol-present", "vol-missing"],
            nfs_server="192.0.2.1",
        )
    )

    report.assert_any_call(
        "mount-failure-regression",
        "failed",
        error_message="required volume mount failed: vol-missing",
        expected_v2_auth=ANY,
    )
    unmount.assert_called_once_with("/mnt/xcelsior-volumes/vol-present")
    worker.destroy_encrypted_volume.assert_not_called()
    assert "mount-failure-regression" not in worker._active_containers


@pytest.mark.parametrize("failure", [None, "http", "short", "second_download", "no_ping"])
def test_network_measurement_requires_real_payload_and_ping_evidence(monkeypatch, failure):
    ping = Mock(
        return_value=subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                "20 packets transmitted, 20 received, 0% packet loss\n"
                "rtt min/avg/max/mdev = 12.0/15.0/18.0/2.0 ms\n"
            ),
        )
    )
    if failure == "no_ping":
        ping.side_effect = FileNotFoundError("ping")
    monkeypatch.setattr(worker.subprocess, "run", ping)
    monkeypatch.setattr(worker, "HOST_ID", "test-host")
    monkeypatch.setattr(worker, "_api_headers", lambda: {"Authorization": "Bearer xat_test"})
    # Four parallel pulls each read the clock twice (first byte, finished); a
    # thread-safe counter makes the span 10.0 → 10.7 whatever the interleaving.
    ticks = itertools.count()
    monkeypatch.setattr(worker.time, "perf_counter", lambda: 10.0 + next(ticks) / 10)
    response = requests.Response()
    response.status_code = 403 if failure == "http" else 200
    response._content = b"x" * (128 if failure == "short" else 8 * 1024 * 1024)
    response._content_consumed = True  # iter_content serves the preloaded body
    get = Mock(return_value=response)
    if failure == "second_download":
        get.side_effect = [response, response, response, requests.Timeout("timed out")]
    monkeypatch.setattr(worker.requests, "get", get)

    report = worker.run_network_benchmark("https://api.example.test/")

    # 4 x 8 MiB over 0.7 s, timed from first byte.
    assert report["throughput_mbps"] == (383.48 if failure in (None, "no_ping") else 0)
    if failure in (None, "no_ping"):
        assert get.call_count == 4
    assert report["packet_loss_pct"] == (100 if failure == "no_ping" else 0)
    for call in get.call_args_list:
        assert call.args[0] == "https://api.example.test/agent/network-download/test-host"
        assert call.kwargs["headers"]["Authorization"] == "Bearer xat_test"


def test_exclusive_gpu_refusal_clears_worker_bookkeeping(monkeypatch):
    for name in ("get_gpu_info", "build_platform_env"):
        monkeypatch.setattr(worker, name, Mock(return_value={}))
    for name in ("_kill_container", "_remove_container"):
        monkeypatch.setattr(worker, name, Mock())
    monkeypatch.setattr(worker.time, "sleep", Mock())
    monkeypatch.setattr(worker, "report_job_status", Mock(return_value=True))
    monkeypatch.setattr(
        worker, "_acquire_exclusive_gpu", Mock(return_value={"ok": False, "error": "busy"})
    )
    worker.run_job(_fenced_job("gpu-busy-regression", image="test/image:1", exclusive_gpu=True))
    assert "gpu-busy-regression" not in worker._active_containers
    worker.report_job_status.assert_any_call(
        "gpu-busy-regression",
        "failed",
        error_message="exclusive_gpu acquire failed: busy",
        expected_v2_auth=ANY,
    )


@pytest.mark.parametrize(
    "status,ready",
    [
        (200, True),
        (204, True),
        (401, False),
        (403, False),
        (404, False),
        (429, False),
        (503, False),
    ],
)
def test_serverless_health_requires_a_success_response(monkeypatch, status, ready):
    import threading
    from types import SimpleNamespace

    clock = itertools.count()
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker.time, "time", lambda: next(clock) / 10)
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    get = Mock(return_value=SimpleNamespace(status_code=status))
    monkeypatch.setattr(worker.requests, "get", get)
    assert worker._wait_serverless_health(8080, "health", timeout_sec=0.15) is ready
    get.assert_called_once_with("http://127.0.0.1:8080/health", timeout=5)
