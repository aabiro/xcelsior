"""Phase 5 — worker-agent v2 client: hard lease gate, status mirror,
fence-loss behavior. Mocks the HTTP layer; asserts the agent's decisions.
"""

from __future__ import annotations

import hashlib
import json
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
    spec = {"job_id": "job-1", "name": "job-1", "image": "x"}
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return {
        "command_id": "cmd-1",
        "command": "start_attempt",
        "job_id": "job-1",
        "attempt_id": "att-1",
        "fencing_token": 7,
        "spec_hash": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
        "args": {
            "lease_id": "lease-1",
            "attempt_id": "att-1",
            "fencing_token": 7,
            "spec": spec,
        },
    }


def _stop_cmd():
    return {
        "command_id": "cmd-stop-1",
        "command": "stop_attempt",
        "job_id": "job-1",
        "attempt_id": "att-1",
        "fencing_token": 7,
        "args": {
            "job_id": "job-1",
            "lease_id": "lease-1",
            "attempt_id": "att-1",
            "fencing_token": 7,
            "preserve": True,
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

    def test_mismatched_lease_grant_aborts_start(self):
        def fake_post(url, json=None, headers=None, timeout=None):
            if "/leases/claim" in url:
                return _resp(
                    200,
                    {
                        "lease_id": "lease-OTHER",
                        "job_id": "job-1",
                        "attempt_id": "att-1",
                        "fencing_token": 7,
                    },
                )
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent, "run_job") as run_job:
            worker_agent.handle_start_attempt(_start_cmd())
        run_job.assert_not_called()

    def test_granted_lease_starts_job_and_acks(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            if "/leases/claim" in url:
                return _resp(
                    200,
                    {
                        "ok": True,
                        "lease_id": "lease-1",
                        "job_id": "job-1",
                        "attempt_id": "att-1",
                        "fencing_token": 7,
                        "renewal_ttl_sec": 300,
                    },
                )
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent, "run_job") as run_job:
            worker_agent.handle_start_attempt(_start_cmd())

        run_job.assert_called_once()
        started_job = run_job.call_args[0][0]
        assert started_job["job_id"] == "job-1"
        assert started_job["_v2_container_name"] == "xcl-job-1-att-1"
        assert started_job["_v2_spec_hash"] == _start_cmd()["spec_hash"]
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

    def test_spec_hash_mismatch_aborts_before_lease_claim(self):
        cmd = _start_cmd()
        cmd["args"]["spec"]["image"] = "tampered:image"
        with mock.patch.object(worker_agent, "v2_claim_lease_fenced") as claim, \
             mock.patch.object(worker_agent, "v2_nack_command") as nack, \
             mock.patch.object(worker_agent, "run_job") as run_job:
            worker_agent.handle_start_attempt(cmd)
        claim.assert_not_called()
        run_job.assert_not_called()
        assert nack.call_args.args[1] == "spec_hash_mismatch"
        assert nack.call_args.kwargs["retryable"] is False

    def test_fence_lost_after_lease_claim_aborts_before_run_job(self):
        with mock.patch.object(
            worker_agent,
            "v2_claim_lease_fenced",
            return_value={"renewal_ttl_sec": 300},
        ), mock.patch.object(
            worker_agent, "v2_report_attempt_status", return_value="fenced"
        ), mock.patch.object(worker_agent, "v2_nack_command") as nack, \
             mock.patch.object(worker_agent, "run_job") as run_job, \
             mock.patch.object(worker_agent, "_v2_journal_save"):
            worker_agent.handle_start_attempt(_start_cmd())
        run_job.assert_not_called()
        assert nack.call_args.args[1] == "lease_authority_lost_before_start"
        assert nack.call_args.kwargs["retryable"] is False

    def test_run_job_never_reenters_legacy_lease_path_for_v2(self):
        job = {
            "job_id": "job-1",
            "name": "job-1",
            "image": "image:test",
            "_v2_attempt_id": "att-1",
            "_v2_fencing_token": 7,
            "_v2_container_name": "xcl-job-1-att-1",
        }
        with mock.patch.object(worker_agent, "get_gpu_info", return_value={}), \
             mock.patch.object(worker_agent, "build_platform_env", return_value={}), \
             mock.patch.object(worker_agent, "claim_lease") as legacy_claim, \
             mock.patch.object(worker_agent, "_kill_container") as kill, \
             mock.patch.object(worker_agent, "_remove_container"), \
             mock.patch.object(worker_agent.time, "sleep"), \
             mock.patch.object(
                 worker_agent, "report_job_status", return_value=False
             ) as report:
            worker_agent.run_job(job)
        legacy_claim.assert_not_called()
        kill.assert_called_once_with("xcl-job-1-att-1")
        report.assert_called_once_with("job-1", "starting", host_id=worker_agent.HOST_ID)

    def test_required_gvisor_missing_aborts_before_docker_start(self):
        job = {
            "job_id": "job-1",
            "name": "job-1",
            "image": "image:test",
            "required_runtime": "gvisor",
            "_v2_attempt_id": "att-1",
            "_v2_fencing_token": 7,
            "_v2_container_name": "xcl-job-1-att-1",
        }
        with mock.patch.object(
            worker_agent, "get_gpu_info", return_value={"gpu_model": "RTX 2060"}
        ), mock.patch.object(worker_agent, "build_platform_env", return_value={}), \
             mock.patch.object(worker_agent, "_kill_container"), \
             mock.patch.object(worker_agent, "_remove_container"), \
             mock.patch.object(worker_agent.time, "sleep"), \
             mock.patch.object(worker_agent, "report_job_status", return_value=True) as report, \
             mock.patch.object(worker_agent, "cache_track_pull"), \
             mock.patch.object(worker_agent, "_image_cache_index", {"image:test": {}}), \
             mock.patch.object(worker_agent, "is_gvisor_available", return_value=False), \
             mock.patch.object(worker_agent, "build_secure_docker_args") as build:
            worker_agent.run_job(job)
        build.assert_not_called()
        assert any(
            call.kwargs.get("error_message") == "required gVisor runtime is unavailable"
            for call in report.call_args_list
        )

    def test_required_image_signature_without_verifier_fails_closed(self):
        job = {
            "job_id": "job-1",
            "name": "job-1",
            "image": "image:test",
            "require_image_signature": True,
            "_v2_attempt_id": "att-1",
            "_v2_fencing_token": 7,
            "_v2_container_name": "xcl-job-1-att-1",
        }
        with mock.patch.object(worker_agent, "get_gpu_info", return_value={}), \
             mock.patch.object(worker_agent, "build_platform_env", return_value={}), \
             mock.patch.object(worker_agent, "_kill_container"), \
             mock.patch.object(worker_agent, "_remove_container"), \
             mock.patch.object(worker_agent.time, "sleep"), \
             mock.patch.object(worker_agent, "report_job_status", return_value=True) as report, \
             mock.patch.object(worker_agent.shutil, "which", return_value=None), \
             mock.patch.object(worker_agent, "build_secure_docker_args") as build:
            worker_agent.run_job(job)
        build.assert_not_called()
        assert any(
            call.kwargs.get("error_message")
            == "required image signature verifier/key is unavailable"
            for call in report.call_args_list
        )


class TestFencedStopCommand:
    def setup_method(self):
        auth = {
            "lease_id": "lease-1",
            "job_id": "job-1",
            "attempt_id": "att-1",
            "fencing_token": 7,
            "renewal_ttl_sec": 300,
            "stop": threading.Event(),
        }
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts["job-1"] = auth
            worker_agent._v2_completed_commands.clear()

    def teardown_method(self):
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts.clear()
            worker_agent._v2_completed_commands.clear()

    def test_stop_revalidates_fence_reports_terminal_then_acks(self, tmp_path):
        posts = []
        docker_calls = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        def fake_run(argv, **kwargs):
            docker_calls.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(
            worker_agent, "_V2_JOURNAL_PATH", str(tmp_path / "v2.json")
        ), mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(
                 worker_agent,
                 "_v2_container_state",
                 return_value=("running", -1, "att-1"),
             ), mock.patch.object(worker_agent.subprocess, "run", fake_run):
            worker_agent.handle_stop_attempt(_stop_cmd())

        assert any(call[:2] == ["docker", "stop"] for call in docker_calls)
        assert any("/leases/renew" in url for url, _ in posts)
        reports = [body for url, body in posts if "/attempts/status" in url]
        assert reports and reports[0]["status"] == "stopped"
        assert any(url.endswith("/cmd-stop-1/ack") for url, _ in posts)
        with worker_agent._v2_attempts_lock:
            assert "job-1" not in worker_agent._v2_attempts
        assert worker_agent._v2_completed_command_result("cmd-stop-1") == {
            "stopped": True,
            "preserved": True,
            "intent": None,
        }

    def test_stop_preserve_false_removes_container(self, tmp_path):
        """Terminate/cancel (preserve=False) must docker rm after stop."""
        posts = []
        docker_calls = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        def fake_run(argv, **kwargs):
            docker_calls.append(list(argv))
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        cmd = _stop_cmd()
        cmd["args"] = {**cmd["args"], "preserve": False, "intent": "terminate"}
        with mock.patch.object(
            worker_agent, "_V2_JOURNAL_PATH", str(tmp_path / "v2.json")
        ), mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(
                 worker_agent,
                 "_v2_container_state",
                 return_value=("running", -1, "att-1"),
             ), mock.patch.object(worker_agent.subprocess, "run", fake_run):
            worker_agent.handle_stop_attempt(cmd)

        assert any(c[:2] == ["docker", "stop"] for c in docker_calls)
        assert any(c[:3] == ["docker", "rm", "-f"] for c in docker_calls)
        assert worker_agent._v2_completed_command_result("cmd-stop-1") == {
            "stopped": True,
            "preserved": False,
            "intent": "terminate",
        }

    def test_stale_command_never_touches_docker(self):
        cmd = _stop_cmd()
        cmd["fencing_token"] = 8
        nacks = []
        with mock.patch.object(worker_agent, "v2_nack_command") as nack, \
             mock.patch.object(worker_agent.subprocess, "run") as docker_run:
            worker_agent.handle_stop_attempt(cmd)
            nacks.extend(nack.call_args_list)
        docker_run.assert_not_called()
        assert nacks and nacks[0].args[1] == "authority_not_local"

    def test_stop_transport_error_is_retryable_and_keeps_authority(self):
        with mock.patch.object(
            worker_agent, "_v2_post", return_value=(503, {"error": "down"})
        ), mock.patch.object(worker_agent, "v2_nack_command") as nack, \
             mock.patch.object(worker_agent.subprocess, "run") as docker_run:
            worker_agent.handle_stop_attempt(_stop_cmd())
        docker_run.assert_not_called()
        assert nack.call_args.args[1] == "stop_authority_unavailable"
        assert nack.call_args.kwargs["retryable"] is True
        with worker_agent._v2_attempts_lock:
            assert "job-1" in worker_agent._v2_attempts

    def test_completed_stop_replays_ack_without_docker(self, tmp_path):
        result = {"stopped": True, "preserved": True}
        with mock.patch.object(
            worker_agent, "_V2_JOURNAL_PATH", str(tmp_path / "v2.json")
        ):
            worker_agent._v2_mark_command_complete("cmd-stop-1", result)
        with mock.patch.object(
            worker_agent, "v2_claim_commands", return_value=[_stop_cmd()]
        ), mock.patch.object(worker_agent, "v2_ack_command", return_value=True) as ack, \
             mock.patch.object(worker_agent, "handle_stop_attempt") as stop:
            worker_agent._v2_enabled = True
            try:
                assert worker_agent.drain_v2_commands() == 1
            finally:
                worker_agent._v2_enabled = False
        ack.assert_called_once_with("cmd-stop-1", result)
        stop.assert_not_called()


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
            if "inspect" in argv:
                return SimpleNamespace(
                    returncode=0, stdout="true 0 att-1", stderr=""
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        self._register()
        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.subprocess, "run", fake_run):
            worker_agent._v2_on_job_status("job-1", "running")
        assert any("kill" in argv and "xcl-job-1" in argv for argv in killed)
        with worker_agent._v2_attempts_lock:
            assert "job-1" not in worker_agent._v2_attempts

    def test_fence_loss_never_kills_same_name_with_wrong_label(self):
        docker_calls = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/attempts/status" in url:
                return _resp(409, {"error": {"code": "fencing_violation"}})
            return _resp(200, {"ok": True})

        def fake_run(argv, **kwargs):
            docker_calls.append(argv)
            if "inspect" in argv:
                return SimpleNamespace(
                    returncode=0, stdout="true 0 att-newer", stderr=""
                )
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        self._register()
        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.subprocess, "run", fake_run):
            worker_agent._v2_on_job_status("job-1", "running")
        assert not any("kill" in argv for argv in docker_calls)

    def test_non_v2_job_untouched(self):
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent.requests, "post", fake_post):
            worker_agent._v2_on_job_status("job-not-v2", "running")
        assert posts == []

    def test_v2_report_error_never_falls_back_to_legacy_patch(self):
        def fake_post(url, json=None, headers=None, timeout=None):
            if "/attempts/status" in url:
                return _resp(503, {"error": {"code": "unavailable"}})
            return _resp(200, {"ok": True})

        self._register()
        with mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.requests, "patch") as legacy_patch:
            assert worker_agent.report_job_status("job-1", "failed") is False
        legacy_patch.assert_not_called()
        with worker_agent._v2_attempts_lock:
            assert "job-1" in worker_agent._v2_attempts


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
            if "inspect" in argv:
                return SimpleNamespace(
                    returncode=0, stdout="true 0 att-1", stderr=""
                )
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


class TestJournalAndAdoption:
    def setup_method(self):
        worker_agent._v2_enabled = True

    def teardown_method(self):
        worker_agent._v2_enabled = False
        with worker_agent._v2_attempts_lock:
            worker_agent._v2_attempts.clear()
            worker_agent._v2_completed_commands.clear()

    def _journal(self, tmp_path, records):
        path = tmp_path / "v2_attempts.json"
        path.write_text(__import__("json").dumps(records))
        return str(path)

    def _rec(self, job_id="job-1"):
        return {
            job_id: {
                "lease_id": "lease-1", "job_id": job_id, "attempt_id": "att-1",
                "fencing_token": 7, "renewal_ttl_sec": 300,
            }
        }

    def test_journal_roundtrip(self, tmp_path):
        path = str(tmp_path / "v2_attempts.json")
        auth = {
            "lease_id": "lease-1", "job_id": "job-1", "attempt_id": "att-1",
            "fencing_token": 7, "renewal_ttl_sec": 120,
            "stop": threading.Event(),
        }
        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path):
            with worker_agent._v2_attempts_lock:
                worker_agent._v2_attempts["job-1"] = auth
            worker_agent._v2_journal_save()
            loaded = worker_agent._v2_journal_load()
        assert loaded["job-1"]["attempt_id"] == "att-1"
        assert loaded["job-1"]["fencing_token"] == 7
        assert "stop" not in loaded["job-1"]

    def test_adopts_running_container_with_valid_lease(self, tmp_path):
        path = self._journal(tmp_path, self._rec())
        renews = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/leases/renew" in url:
                renews.append(json)
                return _resp(200, {"ok": True})
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("running", -1, "att-1"),
             ):
            adopted = worker_agent.v2_adopt_attempts()

        assert adopted == 1
        assert renews and renews[0]["attempt_id"] == "att-1"
        with worker_agent._v2_attempts_lock:
            assert "job-1" in worker_agent._v2_attempts

    def test_transport_error_enters_bounded_pending_adoption(self, tmp_path):
        path = self._journal(tmp_path, self._rec())
        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(
                 worker_agent, "_v2_post", return_value=(503, {"error": "down"})
             ), mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("running", -1, "att-1"),
             ), mock.patch.object(worker_agent.threading, "Thread") as thread:
            adopted = worker_agent.v2_adopt_attempts()
        assert adopted == 0
        with worker_agent._v2_attempts_lock:
            assert "job-1" in worker_agent._v2_attempts
        assert thread.call_args.kwargs["target"] is worker_agent._v2_pending_adoption_loop

    def test_fenced_adoption_kills_container(self, tmp_path):
        path = self._journal(tmp_path, self._rec())
        killed = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/leases/renew" in url:
                return _resp(409, {"error": {"code": "lease_renew_rejected"}})
            return _resp(200, {"ok": True})

        def fake_run(argv, **kw):
            killed.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.subprocess, "run", fake_run), \
             mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("running", -1, "att-1"),
             ):
            adopted = worker_agent.v2_adopt_attempts()

        assert adopted == 0
        assert any("kill" in argv for argv in killed)
        with worker_agent._v2_attempts_lock:
            assert "job-1" not in worker_agent._v2_attempts

    def test_exited_container_reports_terminal(self, tmp_path):
        path = self._journal(tmp_path, self._rec())
        reports = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/attempts/status" in url:
                reports.append(json)
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("exited", 0, "att-1"),
             ):
            worker_agent.v2_adopt_attempts()
        assert reports and reports[0]["status"] == "succeeded"

    def test_exited_pending_stop_recovers_stop_and_ack(self, tmp_path):
        records = self._rec()
        records["job-1"]["pending_command"] = {
            "command_id": "cmd-stop-recover",
            "terminal_status": "stopped",
            "result": {"stopped": True, "preserved": True},
        }
        path = self._journal(tmp_path, records)
        posts = []

        def fake_post(url, json=None, headers=None, timeout=None):
            posts.append((url, json))
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("exited", 0, "att-1"),
             ):
            worker_agent.v2_adopt_attempts()

        reports = [body for url, body in posts if "/attempts/status" in url]
        assert reports and reports[0]["status"] == "stopped"
        assert any(url.endswith("/cmd-stop-recover/ack") for url, _ in posts)
        assert worker_agent._v2_completed_command_result("cmd-stop-recover") == {
            "stopped": True,
            "preserved": True,
        }

    def test_missing_container_reports_failure(self, tmp_path):
        path = self._journal(tmp_path, self._rec())
        reports = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/attempts/status" in url:
                reports.append(json)
            return _resp(200, {"ok": True})

        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("missing", -1, ""),
             ):
            worker_agent.v2_adopt_attempts()
        assert reports and reports[0]["status"] == "failed"
        assert reports[0]["failure_code"] == "container_missing"

    def test_label_mismatch_never_killed(self, tmp_path):
        """A container from a NEWER attempt must not be touched."""
        path = self._journal(tmp_path, self._rec())
        killed = []
        reports = []

        def fake_post(url, json=None, headers=None, timeout=None):
            if "/attempts/status" in url:
                reports.append(json)
            return _resp(200, {"ok": True})

        def fake_run(argv, **kw):
            killed.append(argv)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with mock.patch.object(worker_agent, "_V2_JOURNAL_PATH", path), \
             mock.patch.object(worker_agent.requests, "post", fake_post), \
             mock.patch.object(worker_agent.subprocess, "run", fake_run), \
             mock.patch.object(
                 worker_agent, "_v2_container_state",
                 return_value=("running", -1, "att-NEWER"),
             ):
            adopted = worker_agent.v2_adopt_attempts()
        assert adopted == 0 and killed == [] and reports == []
