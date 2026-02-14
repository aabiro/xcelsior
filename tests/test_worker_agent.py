"""Xcelsior Worker Agent unit tests — Phase 7.4

Tests GPU detection, heartbeat, API communication, lease management,
image cache, mining detection helpers, telemetry, Tailscale, and config
validation using monkeypatched subprocess/requests calls.
"""

import json
import signal
import subprocess
import time
import pytest
import requests

import worker_agent


# ── Helpers ───────────────────────────────────────────────────────────

def _patch_base(monkeypatch, host_id="rig-01", url="http://localhost:8000",
                token=None, cost=0.50):
    """Apply the common module-level patches for most tests."""
    monkeypatch.setattr(worker_agent, "HOST_ID", host_id)
    monkeypatch.setattr(worker_agent, "SCHEDULER_URL", url)
    monkeypatch.setattr(worker_agent, "API_TOKEN", token)
    monkeypatch.setattr(worker_agent, "COST_PER_HOUR", cost)


class _FakeResp:
    """Minimal requests.Response stand-in."""
    def __init__(self, status_code=200, body=None, ok=True):
        self.status_code = status_code
        self.ok = ok
        self._body = body or {}
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")
    def json(self):
        return self._body


# ── Config / Startup ─────────────────────────────────────────────────

class TestValidateConfig:
    def test_allows_missing_api_token(self, monkeypatch):
        _patch_base(monkeypatch, token=None)
        worker_agent.validate_config()  # should not exit

    def test_exits_without_host_id(self, monkeypatch):
        _patch_base(monkeypatch)
        monkeypatch.setattr(worker_agent, "HOST_ID", None)
        with pytest.raises(SystemExit):
            worker_agent.validate_config()

    def test_exits_without_scheduler_url(self, monkeypatch):
        _patch_base(monkeypatch)
        monkeypatch.setattr(worker_agent, "SCHEDULER_URL", None)
        with pytest.raises(SystemExit):
            worker_agent.validate_config()


# ── API Helpers ──────────────────────────────────────────────────────

class TestApiHelpers:
    def test_api_headers_no_token(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "API_TOKEN", "")
        h = worker_agent._api_headers()
        assert "Authorization" not in h
        assert h["Content-Type"] == "application/json"

    def test_api_headers_with_token(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "API_TOKEN", "secret-tok")
        h = worker_agent._api_headers()
        assert h["Authorization"] == "Bearer secret-tok"

    def test_api_url(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "http://api:8000/")
        assert worker_agent._api_url("/host") == "http://api:8000/host"

    def test_api_url_strips_trailing_slash(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "SCHEDULER_URL", "http://api:8000///")
        assert worker_agent._api_url("/host") == "http://api:8000/host"


# ── GPU Detection ────────────────────────────────────────────────────

class TestGetGpuInfo:
    def test_parses_nvidia_smi(self, monkeypatch):
        fake_output = "NVIDIA GeForce RTX 4090, 24564, 22000\n"

        def fake_run(*a, **kw):
            return subprocess.CompletedProcess(a, 0, stdout=fake_output, stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        info = worker_agent.get_gpu_info()
        assert info["gpu_model"] == "NVIDIA GeForce RTX 4090"
        assert abs(info["total_vram_gb"] - 23.99) < 0.1
        assert abs(info["free_vram_gb"] - 21.48) < 0.1

    def test_raises_on_empty_output(self, monkeypatch):
        def fake_run(*a, **kw):
            return subprocess.CompletedProcess(a, 0, stdout="\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="No GPU found"):
            worker_agent.get_gpu_info()

    def test_raises_on_bad_format(self, monkeypatch):
        def fake_run(*a, **kw):
            return subprocess.CompletedProcess(a, 0, stdout="only-one-field\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="Unexpected nvidia-smi format"):
            worker_agent.get_gpu_info()

    def test_raises_on_timeout(self, monkeypatch):
        def fake_run(*a, **kw):
            raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="timed out"):
            worker_agent.get_gpu_info()

    def test_raises_on_nonzero_exit(self, monkeypatch):
        def fake_run(*a, **kw):
            raise subprocess.CalledProcessError(1, "nvidia-smi", stderr="driver error")

        monkeypatch.setattr(subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="nvidia-smi failed"):
            worker_agent.get_gpu_info()


# ── Host IP Detection ────────────────────────────────────────────────

class TestGetHostIp:
    def test_from_hostname(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", False)

        def fake_run(cmd, **kw):
            if cmd == ["hostname", "-I"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="192.168.1.50 10.0.0.1\n")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert worker_agent.get_host_ip() == "192.168.1.50"

    def test_tailscale_preferred(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", True)

        def fake_run(cmd, **kw):
            if cmd == ["tailscale", "ip", "-4"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="100.64.0.7\n")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert worker_agent.get_host_ip() == "100.64.0.7"

    def test_fallback_unknown(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", False)

        def fake_run(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert worker_agent.get_host_ip() == "unknown"


# ── Heartbeat (formerly register_or_update_host) ─────────────────────

class TestHeartbeat:
    def test_heartbeat_success_without_token(self, monkeypatch):
        _patch_base(monkeypatch, token=None, cost=0.10)
        captured = {}

        def fake_put(url, json, headers, timeout):
            captured.update(url=url, json=json, headers=headers)
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "put", fake_put)

        ok = worker_agent.heartbeat(
            {"gpu_model": "RTX 2060", "total_vram_gb": 6.0, "free_vram_gb": 4.5},
            "127.0.0.1",
        )
        assert ok is True
        assert captured["url"] == "http://localhost:8000/host"
        assert "Authorization" not in captured["headers"]
        assert captured["json"]["host_id"] == "rig-01"
        assert captured["json"]["cost_per_hour"] == 0.10

    def test_heartbeat_includes_bearer_token(self, monkeypatch):
        _patch_base(monkeypatch, token="my-secret")
        captured = {}

        def fake_put(url, json, headers, timeout):
            captured["headers"] = headers
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "put", fake_put)
        worker_agent.heartbeat(
            {"gpu_model": "A100", "total_vram_gb": 80.0, "free_vram_gb": 78.0},
            "10.0.0.5",
        )
        assert captured["headers"]["Authorization"] == "Bearer my-secret"

    def test_heartbeat_returns_false_on_error(self, monkeypatch):
        _patch_base(monkeypatch, token=None)

        def fake_put(url, json, headers, timeout):
            raise requests.ConnectionError("connection refused")

        monkeypatch.setattr(worker_agent.requests, "put", fake_put)
        ok = worker_agent.heartbeat(
            {"gpu_model": "T4", "total_vram_gb": 16.0, "free_vram_gb": 14.0},
            "1.2.3.4",
        )
        assert ok is False


# ── Version Reporting / Admission ─────────────────────────────────────

class TestReportVersions:
    def test_admitted(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(200, {"admitted": True, "details": {}})

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        admitted, details = worker_agent.report_versions({"docker": "24.0.0"})
        assert admitted is True

    def test_rejected(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(200, {"admitted": False, "details": {"docker": "too old"}})

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        admitted, details = worker_agent.report_versions({"docker": "18.0.0"})
        assert admitted is False
        assert "docker" in details

    def test_http_error(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(500)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        admitted, details = worker_agent.report_versions({"docker": "24.0.0"})
        assert admitted is False
        assert "error" in details


# ── Work Polling ─────────────────────────────────────────────────────

class TestPollForWork:
    def test_returns_jobs(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            return _FakeResp(200, {"jobs": [{"job_id": "j-1", "image": "ubuntu:22.04"}]})

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        jobs = worker_agent.poll_for_work()
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == "j-1"

    def test_returns_empty_on_204(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            return _FakeResp(204)

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        assert worker_agent.poll_for_work() == []

    def test_returns_empty_on_error(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            raise requests.ConnectionError("timeout")

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        assert worker_agent.poll_for_work() == []


# ── Preemption Check ─────────────────────────────────────────────────

class TestCheckPreemption:
    def test_returns_job_ids(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            return _FakeResp(200, {"preempt_jobs": ["j-1", "j-2"]})

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        ids = worker_agent.check_preemption()
        assert ids == ["j-1", "j-2"]

    def test_returns_empty_on_failure(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            return _FakeResp(500)

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        assert worker_agent.check_preemption() == []


# ── Lease Management ─────────────────────────────────────────────────

class TestLeaseManagement:
    def test_claim_lease_success(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(200, {"lease_id": "L-123", "expires_at": time.time() + 300})

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        result = worker_agent.claim_lease("j-1")
        assert result is not None
        assert result["lease_id"] == "L-123"

    def test_claim_lease_failure(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(409)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        assert worker_agent.claim_lease("j-1") is None

    def test_renew_lease_success(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(200, {"lease_id": "L-123", "renewed": True})

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        result = worker_agent.renew_lease("j-1")
        assert result is not None

    def test_renew_lease_expired(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            return _FakeResp(410)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        assert worker_agent.renew_lease("j-1") is None

    def test_release_lease(self, monkeypatch):
        """Release is best-effort — should not raise."""
        _patch_base(monkeypatch)
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.release_lease("j-1", reason="preempted")
        assert captured["json"]["reason"] == "preempted"


# ── Job Status Reporting ─────────────────────────────────────────────

class TestReportJobStatus:
    def test_success(self, monkeypatch):
        _patch_base(monkeypatch)
        captured = {}

        def fake_patch(url, json, headers, timeout):
            captured.update(url=url, json=json)
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "patch", fake_patch)
        ok = worker_agent.report_job_status("j-42", "completed")
        assert ok is True
        assert "/job/j-42" in captured["url"]
        assert captured["json"]["status"] == "completed"

    def test_with_host_id(self, monkeypatch):
        _patch_base(monkeypatch)
        captured = {}

        def fake_patch(url, json, headers, timeout):
            captured["json"] = json
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "patch", fake_patch)
        worker_agent.report_job_status("j-42", "running", host_id="rig-01")
        assert captured["json"]["host_id"] == "rig-01"

    def test_failure(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_patch(url, json, headers, timeout):
            return _FakeResp(500)

        monkeypatch.setattr(worker_agent.requests, "patch", fake_patch)
        assert worker_agent.report_job_status("j-99", "failed") is False


# ── Mining Alert ─────────────────────────────────────────────────────

class TestReportMiningAlert:
    def test_sends_alert(self, monkeypatch):
        _patch_base(monkeypatch)
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.report_mining_alert(0, 0.95, "high_util_low_mem")
        assert captured["json"]["gpu_index"] == 0
        assert captured["json"]["confidence"] == 0.95
        assert captured["json"]["host_id"] == "rig-01"

    def test_silent_on_network_error(self, monkeypatch):
        """Best-effort: should not raise."""
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            raise requests.ConnectionError("network error")

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.report_mining_alert(0, 0.80, "eth_signature")


# ── Benchmark Reporting ──────────────────────────────────────────────

class TestReportBenchmark:
    def test_sends_score(self, monkeypatch):
        _patch_base(monkeypatch)
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.report_benchmark({"tflops": 50.0}, "RTX 4090")
        assert captured["json"]["tflops"] == 50.0
        assert captured["json"]["score"] == 5.0  # XCU = TFLOPS / 10
        assert captured["json"]["gpu_model"] == "RTX 4090"


# ── Telemetry Push ───────────────────────────────────────────────────

class TestReportTelemetry:
    def test_sends_metrics(self, monkeypatch):
        _patch_base(monkeypatch)
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _FakeResp(200)

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.report_telemetry({"utilization": 95, "temp": 72})
        assert captured["json"]["host_id"] == "rig-01"
        assert captured["json"]["metrics"]["utilization"] == 95
        assert "timestamp" in captured["json"]

    def test_silent_on_error(self, monkeypatch):
        """Best-effort: must not raise."""
        _patch_base(monkeypatch)

        def fake_post(url, json, headers, timeout):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.report_telemetry({"utilization": 0})


# ── Verification Report ──────────────────────────────────────────────

class TestReportVerification:
    def test_sends_report(self, monkeypatch):
        _patch_base(monkeypatch)
        captured = {}

        def fake_post(url, json, headers, timeout):
            captured["json"] = json
            return _FakeResp(200, {"state": "verified", "score": 85})

        monkeypatch.setattr(worker_agent.requests, "post", fake_post)
        worker_agent.report_verification({"checks": [{"name": "vram", "passed": True}]})
        assert captured["json"]["host_id"] == "rig-01"
        assert captured["json"]["report"]["checks"][0]["passed"] is True


# ── Image Cache ──────────────────────────────────────────────────────

class TestImageCache:
    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Reset cache index before each test."""
        with worker_agent._image_cache_lock:
            worker_agent._image_cache_index.clear()
        yield
        with worker_agent._image_cache_lock:
            worker_agent._image_cache_index.clear()

    def test_cache_track_pull_new(self):
        worker_agent.cache_track_pull("ubuntu:22.04", size_mb=80)
        assert "ubuntu:22.04" in worker_agent._image_cache_index
        entry = worker_agent._image_cache_index["ubuntu:22.04"]
        assert entry["pull_count"] == 1
        assert entry["size_mb"] == 80

    def test_cache_track_pull_existing(self):
        worker_agent.cache_track_pull("alpine:3.18", size_mb=5)
        worker_agent.cache_track_pull("alpine:3.18", size_mb=5)
        assert worker_agent._image_cache_index["alpine:3.18"]["pull_count"] == 2

    def test_total_cache_size_mb(self):
        worker_agent.cache_track_pull("a:1", size_mb=100)
        worker_agent.cache_track_pull("b:1", size_mb=200)
        assert worker_agent._total_cache_size_mb() == 300

    def test_cache_evict_below_limit_no_op(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "IMAGE_CACHE_MAX_GB", 1.0)
        worker_agent.cache_track_pull("tiny:1", size_mb=10)
        evicted = worker_agent.cache_evict_lru()
        assert evicted == 0

    def test_cache_evict_removes_oldest(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "IMAGE_CACHE_MAX_GB", 0.5)   # 512 MB
        monkeypatch.setattr(worker_agent, "IMAGE_CACHE_EVICT_LOW_GB", 0.3)  # 307 MB

        # Simulate 3 images totalling 600 MB (over 512 limit)
        worker_agent._image_cache_index["old:1"] = {
            "last_used": time.time() - 3600, "size_mb": 200, "pull_count": 1,
        }
        worker_agent._image_cache_index["mid:1"] = {
            "last_used": time.time() - 1800, "size_mb": 200, "pull_count": 1,
        }
        worker_agent._image_cache_index["new:1"] = {
            "last_used": time.time(), "size_mb": 200, "pull_count": 1,
        }

        docker_rmi_calls = []

        def fake_run(cmd, **kw):
            if cmd[0] == "docker" and cmd[1] == "ps":
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if cmd[0] == "docker" and cmd[1] == "rmi":
                docker_rmi_calls.append(cmd[2])
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        evicted = worker_agent.cache_evict_lru()
        assert evicted >= 1
        assert "old:1" in docker_rmi_calls  # oldest should be evicted first

    def test_parse_docker_size(self):
        assert worker_agent._parse_docker_size("2.5GB") == 2.5 * 1024
        assert worker_agent._parse_docker_size("850MB") == 850
        assert worker_agent._parse_docker_size("512KB") == 512 / 1024
        assert worker_agent._parse_docker_size("bad") == 0


# ── Fetch Popular Images ────────────────────────────────────────────

class TestFetchPopularImages:
    def test_returns_images(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            return _FakeResp(200, {"images": ["pytorch/pytorch:latest", "ubuntu:22.04"]})

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        imgs = worker_agent.fetch_popular_images()
        assert "pytorch/pytorch:latest" in imgs

    def test_returns_empty_on_failure(self, monkeypatch):
        _patch_base(monkeypatch)

        def fake_get(url, headers, timeout):
            raise requests.ConnectionError("network error")

        monkeypatch.setattr(worker_agent.requests, "get", fake_get)
        assert worker_agent.fetch_popular_images() == []


# ── Tailscale / Headscale Setup ──────────────────────────────────────

class TestSetupTailscale:
    def test_skipped_when_disabled(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", False)
        assert worker_agent.setup_tailscale() is False

    def test_already_running(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", True)
        monkeypatch.setattr(worker_agent, "HOST_ID", "rig-01")

        def fake_run(cmd, **kw):
            if "status" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout=json.dumps({
                        "BackendState": "Running",
                        "TailscaleIPs": ["100.64.0.10"],
                    }),
                )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert worker_agent.setup_tailscale() is True

    def test_brings_up_with_authkey(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", True)
        monkeypatch.setattr(worker_agent, "HOST_ID", "rig-01")
        monkeypatch.setattr(worker_agent, "TAILSCALE_AUTHKEY", "tskey-xxx")
        monkeypatch.setattr(worker_agent, "HEADSCALE_URL", "https://hs.example.com")
        captured = {}

        def fake_run(cmd, **kw):
            if "status" in cmd:
                return subprocess.CompletedProcess(
                    cmd, 0,
                    stdout=json.dumps({"BackendState": "NeedsLogin"}),
                )
            if "up" in cmd:
                captured["cmd"] = cmd
                return subprocess.CompletedProcess(cmd, 0)
            return subprocess.CompletedProcess(cmd, 1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert worker_agent.setup_tailscale() is True
        assert "--authkey" in captured["cmd"]
        assert "--login-server" in captured["cmd"]

    def test_failure_returns_false(self, monkeypatch):
        monkeypatch.setattr(worker_agent, "TAILSCALE_ENABLED", True)
        monkeypatch.setattr(worker_agent, "HOST_ID", "rig-01")
        monkeypatch.setattr(worker_agent, "TAILSCALE_AUTHKEY", "")
        monkeypatch.setattr(worker_agent, "HEADSCALE_URL", "")

        def fake_run(cmd, **kw):
            if "status" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout='{"BackendState":"Stopped"}')
            if "up" in cmd:
                return subprocess.CompletedProcess(cmd, 1, stderr="auth failed")
            return subprocess.CompletedProcess(cmd, 1)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert worker_agent.setup_tailscale() is False


# ── Signal Handler ───────────────────────────────────────────────────

class TestSignalHandler:
    def test_sets_shutdown_event(self):
        worker_agent._shutdown.clear()
        worker_agent._signal_handler(signal.SIGTERM, None)
        assert worker_agent._shutdown.is_set()
        worker_agent._shutdown.clear()  # reset
