"""Phase 5 — worker-agent v2 client: hard lease gate, status mirror,
fence-loss behavior. Mocks the HTTP layer; asserts the agent's decisions.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ.setdefault("XCELSIOR_HOST_ID", "test-host-v2")
os.environ.setdefault("XCELSIOR_API_URL", "http://localhost:9500")

import worker_agent  # noqa: E402


def _resp(status_code=200, body=None):
    return SimpleNamespace(status_code=status_code, json=lambda: body or {})


def _start_cmd():
    return {
        "command_id": "cmd-1",
        "command": "start_attempt",
        "job_id": "job-1",
        "attempt_id": "att-1",
        "fencing_token": 7,
        "args": {
            "lease_id": "lease-1",
            "attempt_id": "att-1",
            "fencing_token": 7,
            "spec": {"job_id": "job-1", "name": "job-1", "image": "x"},
        },
    }


class TestHardLeaseGate:
    def test_rejected_lease_aborts_start(self):
        """§11.2: lease claim 409 → NACK non-retryable, run_job NEVER called."""
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            if "/leases/claim" in url:
                return _resp(409, {"error": {"code": "lease_claim_rejected"}})
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent, "run_job") as run_job:
            worker_agent.handle_start_attempt(_start_cmd())

        run_job.assert_not_called()
        nacks = [p for p in posts if "/nack" in p[0]]
        assert len(nacks) == 1
        assert nacks[0][1]["error_code"] == "lease_claim_rejected"
        assert nacks[0][1]["retryable"] is False
        assert not any("/ack" in p[0] and "/nack" not in p[0] for p in posts)

    def test_granted_lease_starts_job_and_acks(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            if "/leases/claim" in url:
                return _resp(200, {"ok": True, "renewal_ttl_sec": 300})
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent, "run_job") as run_job:
            worker_agent.handle_start_attempt(_start_cmd())

        run_job.assert_called_once()
        started_job = run_job.call_args[0][0]
        assert started_job["job_id"] == "job-1"
        acks = [p for p in posts if p[0].endswith("/cmd-1/ack")]
        assert len(acks) == 1 and acks[0][1]["result"]["lease_claimed"] is True
        status_reports = [p[1] for p in posts if "/attempts/status" in p[0]]
        assert any(s["status"] == "lease_claimed" for s in status_reports)
        # Registry cleaned up for next test.
        worker_agent._v2_forget_attempt("job-1")

    def test_malformed_command_nacked(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        cmd = _start_cmd()
        cmd["args"].pop("lease_id")
        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent, "run_job") as run_job:
            worker_agent.handle_start_attempt(cmd)
        run_job.assert_not_called()
        nacks = [p for p in posts if "/nack" in p[0]]
        assert len(nacks) == 1
        assert nacks[0][1]["error_code"] == "malformed_start_attempt"


class TestStatusMirror:
    def _register(self, job_id="job-1"):
        auth = {
            "lease_id": "lease-1", "job_id": job_id, "attempt_id": "att-1",
            "fencing_token": 7, "renewal_ttl_sec": 300,
            "stop": threading.Event(),
        }
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts[job_id] = auth
        return auth

    def teardown_method(self):
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts.clear()

    def test_terminal_status_mirrors_and_forgets(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        self._register()
        with mock.patch.object(worker_agent.requests, "post", fake_post):
            worker_agent._v2_on_job_status("job-1", "completed")
        reports = [p[1] for p in posts if "/attempts/status" in p[0]]
        assert len(reports) == 1 and reports[0]["status"] == "succeeded"
        with worker_agent._v2_attempts_lock:
            assert "job-1" not in worker_agent._v2_attempts

    def test_failed_status_carries_failure_code(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        self._register()
        with mock.patch.object(worker_agent.requests, "post", fake_post):
            worker_agent._v2_on_job_status("job-1", "failed", error_message="OOM")
        reports = [p[1] for p in posts if "/attempts/status" in p[0]]
        assert reports[0]["failure_code"] == "container_failed"
        assert reports[0]["detail"]["error"] == "OOM"

    def test_fenced_report_stops_container(self):
        posts = []
        killed = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            if "/attempts/status" in url:
                return _resp(409, {"error": {"code": "fencing_violation"}})
            return _resp(200, {"ok": True})

        def fake_run(argv, **kw):
            killed.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        self._register()
        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.subprocess, "run", fake_run):
            worker_agent._v2_on_job_status("job-1", "running")
        assert any("kill" in argv and "xcl-job-1" in argv for argv in killed)
        with worker_agent._v2_attempts_lock:
            assert "job-1" not in worker_agent._v2_attempts

    def test_non_v2_job_untouched(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent.requests, "post", fake_post):
            worker_agent._v2_on_job_status("job-not-v2", "running")
        assert posts == []


class TestRenewalFenceLoss:
    def teardown_method(self):
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts.clear()

    def test_renewal_409_kills_container(self):
        killed = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/leases/renew" in url:
                return _resp(409, {"error": {"code": "lease_renew_rejected"}})
            return _resp(200, {"ok": True})

        def fake_run(argv, **kw):
            killed.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        auth = {
            "lease_id": "lease-1", "job_id": "job-1", "attempt_id": "att-1",
            "fencing_token": 7, "renewal_ttl_sec": 30,
            "stop": threading.Event(),
        }
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts["job-1"] = auth

        # Interval floor is 10s — shrink the wait by pre-setting nothing and
        # calling the loop with a stop event that fires after first renew.
        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.subprocess, "run", fake_run), \
             mock.patch.object(
                 auth["stop"], "wait", side_effect=[False, True]
             ):
            worker_agent._v2_renewal_loop(auth)

        assert any("kill" in argv for argv in killed)
        with worker_agent._v2_attempts_lock:
            assert "job-1" not in worker_agent._v2_attempts
