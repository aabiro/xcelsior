"""Worker-agent defects found in review, each pinned by a test that failed before its fix.

Docker, the API and the clock are scripted; nothing here touches a real
container, GPU or control plane.
"""

import json
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest import mock

import pytest

import worker_agent as worker


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "HOST_ID", "hardening-host")
    monkeypatch.setattr(worker, "WORKER_SESSION_ID", "hardening-session")
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_v2_cleanup_pending", {})
    monkeypatch.setattr(worker, "_v2_pending_fence_kills", {})
    monkeypatch.setattr(worker, "_v2_starts_in_progress", set())
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_adopted_containers", set())
    # raising=False: these tests must also run (and fail) against the code
    # that predates the attributes they reset.
    monkeypatch.setattr(worker, "_v2_journal_loaded", True, raising=False)
    monkeypatch.setattr(worker, "_v2_enabled", False)
    monkeypatch.setattr(worker, "_protocol_negotiated", False)
    monkeypatch.setattr(worker, "LogForwarder", mock.Mock())
    monkeypatch.setattr(worker, "_DEREGISTER_ON_EXIT", False)
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json"))
    monkeypatch.setattr(worker, "_V2_CLAIM_RETRY_DELAYS_SEC", (0, 0, 0))
    monkeypatch.setattr(worker, "_api_request", mock.Mock(side_effect=AssertionError("legacy")))
    monkeypatch.setattr(
        worker, "_v2_post", mock.Mock(side_effect=AssertionError("unexpected API call"))
    )
    return worker


class Docker:
    """Scripted docker CLI: ``inspect`` answers from ``states`` (the last
    answer repeats), every other subcommand succeeds and is recorded."""

    def __init__(self, *states):
        self.states = list(states)
        self.calls = []

    def __call__(self, argv, **_kwargs):
        argv = list(argv)
        self.calls.append(argv)
        if argv[:2] == ["docker", "inspect"]:
            state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
            if state == "missing":
                return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object")
            if state == "error":
                return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")
            return SimpleNamespace(returncode=0, stdout=state, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def ran(self, subcommand):
        return [argv for argv in self.calls if argv[1:2] == [subcommand]]


def record(job_id="job-1", attempt_id="att-1", **extra):
    return {
        "lease_id": "lease-1",
        "job_id": job_id,
        "attempt_id": attempt_id,
        "fencing_token": 7,
        "renewal_ttl_sec": 300,
        "container_name": f"xcl-{job_id}-{attempt_id}",
        **extra,
    }


def write_journal(agent, data):
    with open(agent._V2_JOURNAL_PATH, "w") as fh:
        json.dump(data, fh)


def journal(agent):
    with open(agent._V2_JOURNAL_PATH) as fh:
        return json.load(fh)


def live_auth(**extra):
    return {**record(), "stop": threading.Event(), **extra}


class _OneWait(threading.Event):
    """A shutdown event whose first wait ends the loop under test."""

    def wait(self, timeout=None):
        self.waited = timeout
        self.set()
        return True


# ── 1. A save before the journal is read must not erase it ──────────────


def _unsettled_journal(agent):
    write_journal(
        agent,
        {
            "job-1": record(
                pending_terminal_report={"status": "failed", "failure_code": "container_failed"}
            ),
            agent._V2_COMPLETED_COMMANDS_KEY: {
                "stop-1": {"result": {"stopped": True}, "completed_at": time.time()}
            },
            agent._V2_CLEANUP_KEY: {
                "att-0": {
                    "job_id": "job-0",
                    "attempt_id": "att-0",
                    "lease_id": "lease-0",
                    "fencing_token": 6,
                    "container_name": "xcl-job-0-att-0",
                    "resources": {"host_id": "hardening-host", "nfs_mounts": ["/mnt/x"]},
                }
            },
            agent._V2_FENCE_KILLS_KEY: {
                "xcl-job-9-att-9": {
                    "job_id": "job-9",
                    "container_name": "xcl-job-9-att-9",
                    "expected_attempt_id": "att-9",
                    "why": "renewal fenced",
                    "queued_at": time.time(),
                }
            },
        },
    )


def test_shutdown_before_the_journal_is_read_keeps_every_recovery_record(agent, monkeypatch):
    _unsettled_journal(agent)
    monkeypatch.setattr(agent, "_v2_journal_loaded", False, raising=False)
    monkeypatch.setattr(agent, "_DEREGISTER_ON_EXIT", True)
    mounts = mock.Mock()
    monkeypatch.setattr(agent, "cleanup_orphaned_volume_mounts", mounts)

    agent.graceful_shutdown()  # e.g. SIGTERM during startup, API still unreachable

    saved = journal(agent)
    assert saved["job-1"]["pending_terminal_report"]["failure_code"] == "container_failed"
    assert "stop-1" in saved[agent._V2_COMPLETED_COMMANDS_KEY]
    assert "att-0" in saved[agent._V2_CLEANUP_KEY]
    assert "xcl-job-9-att-9" in saved[agent._V2_FENCE_KILLS_KEY]
    mounts.assert_not_called()  # the unread work counts as surviving work
    agent._api_request.assert_not_called()  # and never decommissions the host


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file")
def test_an_unreadable_journal_is_never_overwritten(agent, monkeypatch):
    _unsettled_journal(agent)
    before = open(agent._V2_JOURNAL_PATH).read()
    monkeypatch.setattr(agent, "_v2_journal_loaded", False, raising=False)
    os.chmod(agent._V2_JOURNAL_PATH, 0)
    try:
        assert agent._v2_journal_save() is False
    finally:
        os.chmod(agent._V2_JOURNAL_PATH, 0o600)
    assert open(agent._V2_JOURNAL_PATH).read() == before


# ── 2. A local lease deadline holds before protocol negotiation ─────────


def _restore_and_run_recovery(agent):
    agent._v2_journal_loaded = False  # a fresh process (the fixture restores it)
    with mock.patch.object(agent.threading, "Thread") as thread:
        agent._v2_start_local_recovery()
    thread.call_args.kwargs["target"]()


def test_restart_during_outage_fences_a_survivor_at_its_lease_deadline(agent, monkeypatch):
    # Last renewed 290 s ago with a 300 s TTL: the server may expire the lease
    # at +300 s and requeue it at +330 s. Nothing here can renew it.
    write_journal(agent, {"job-1": record(last_renewed_at=time.time() - 290)})
    docker = Docker("running 0 att-1")
    monkeypatch.setattr(agent.subprocess, "run", docker)

    _restore_and_run_recovery(agent)

    assert docker.ran("kill") == [["docker", "kill", "xcl-job-1-att-1"]]
    assert "job-1" not in agent._v2_attempts
    assert "job-1" not in journal(agent)
    agent._v2_post.assert_not_called()


def test_survivor_inside_its_deadline_waits_for_adoption_of_the_same_record(agent, monkeypatch):
    write_journal(agent, {"job-1": record(last_renewed_at=time.time() - 10)})
    docker = Docker("running 0 att-1")
    monkeypatch.setattr(agent.subprocess, "run", docker)
    monkeypatch.setattr(agent, "_shutdown", _OneWait())

    _restore_and_run_recovery(agent)

    assert docker.ran("kill") == []
    restored = agent._v2_attempts["job-1"]
    assert 0 < agent._shutdown.waited <= 5
    assert journal(agent)["job-1"]["attempt_id"] == "att-1"

    monkeypatch.setattr(agent, "_shutdown", threading.Event())
    monkeypatch.setattr(agent, "_v2_enabled", True)
    monkeypatch.setattr(agent, "_v2_post", mock.Mock(return_value=(200, {})))
    with mock.patch.object(agent.threading, "Thread"):
        assert agent.v2_adopt_attempts() == 1
    assert agent._v2_attempts["job-1"] is restored
    assert not restored.get("_awaiting_adoption")


def test_pending_adoption_is_bounded_by_the_journaled_lease_not_the_restart(agent, monkeypatch):
    auth = live_auth(last_renewed_at=time.time() - 290)
    agent._v2_attempts["job-1"] = auth
    monkeypatch.setattr(agent, "_v2_post", mock.Mock(return_value=(503, {})))
    docker = Docker("running 0 att-1")
    monkeypatch.setattr(agent.subprocess, "run", docker)
    monkeypatch.setattr(auth["stop"], "wait", lambda timeout=None: agent._shutdown.set())

    agent._v2_pending_adoption_loop(auth)

    assert docker.ran("kill") == [["docker", "kill", "xcl-job-1-att-1"]]
    assert "job-1" not in agent._v2_attempts


def test_renewal_loop_wakes_for_the_deadline_not_the_next_interval(agent, monkeypatch):
    auth = live_auth(last_renewed_at=time.time() - 200)  # deadline in ~85 s
    agent._v2_attempts["job-1"] = auth
    waits = []
    monkeypatch.setattr(auth["stop"], "wait", lambda timeout=None: waits.append(timeout) or True)
    agent._v2_renewal_loop(auth)
    assert waits and waits[0] <= 86  # not the 100 s renewal interval


def test_renewal_loop_fences_once_the_recorded_renewal_is_too_old(agent, monkeypatch):
    auth = live_auth(last_renewed_at=time.time() - 290)
    agent._v2_attempts["job-1"] = auth
    monkeypatch.setattr(agent, "_v2_post", mock.Mock(return_value=(503, {})))
    docker = Docker("running 0 att-1")
    monkeypatch.setattr(agent.subprocess, "run", docker)
    waits = iter([False, True])
    monkeypatch.setattr(auth["stop"], "wait", lambda timeout=None: next(waits))

    agent._v2_renewal_loop(auth)

    assert docker.ran("kill") == [["docker", "kill", "xcl-job-1-att-1"]]
    assert "job-1" not in agent._v2_attempts


def test_a_successful_renewal_is_journaled_for_the_next_restart(agent, monkeypatch):
    auth = live_auth(last_renewed_at=time.time() - 250)
    agent._v2_attempts["job-1"] = auth
    agent._v2_journal_save()
    monkeypatch.setattr(agent, "_v2_post", mock.Mock(return_value=(200, {})))
    before = time.time()
    assert agent._v2_renew(auth)[0] == 200
    assert before <= journal(agent)["job-1"]["last_renewed_at"] <= time.time()


# ── 3. The mining kill must not deadlock on _active_lock ───────────────


def test_sustained_mining_kill_settles_the_attempt_without_deadlock(agent, monkeypatch):
    # The loop used to hold the non-reentrant _active_lock while the terminal
    # report forgot the attempt, which takes that lock again: the thread froze.
    monkeypatch.setattr(agent, "_active_lock", threading.Lock())
    auth = live_auth()
    agent._v2_attempts["job-1"] = auth
    agent._active_containers["job-1"] = auth["container_name"]
    agent._adopted_containers.add("job-1")
    agent._v2_journal_save()
    monkeypatch.setattr(agent, "_v2_post", mock.Mock(return_value=(200, {})))
    for name in ("_kill_container", "report_mining_alert"):
        monkeypatch.setattr(agent, name, mock.Mock())
    monkeypatch.setattr(agent, "_job_containers_on_gpu", lambda idx, active: dict(active))
    monkeypatch.setattr(agent, "MINING_CHECK_INTERVAL", 0)
    passes = iter(range(100))

    def telemetry():
        if next(passes) >= 5:
            agent._shutdown.set()
            return []
        return [{"index": 0, "utilization": 100, "memory_util": 19, "pcie_tx_mb_s": 0.5}]

    monkeypatch.setattr(agent, "get_gpu_telemetry", telemetry)
    loop = threading.Thread(target=agent.mining_detection_loop, daemon=True)
    loop.start()
    loop.join(10)
    assert not loop.is_alive(), "mining loop deadlocked on _active_lock"
    agent._kill_container.assert_called_once_with(auth["container_name"])
    assert agent._v2_post.call_args.args[1]["failure_code"] == "mining_detected"
    assert "job-1" not in agent._v2_attempts
    assert "job-1" not in agent._active_containers


# ── 4. The orphan-mount sweep spares mounts an attempt journals ─────────


class _ObservedLock:
    def __init__(self):
        self._lock = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, *exc):
        self.depth -= 1
        self._lock.release()


def test_orphan_sweep_spares_mounts_owned_by_live_and_cleanup_attempts(agent, monkeypatch):
    base = "/mnt/xcelsior-volumes"
    # A volume mounted for an attempt still pulling its image (no container yet)
    agent._v2_attempts["job-1"] = live_auth(
        resources={"host_id": "hardening-host", "nfs_mounts": [f"{base}/vol-setup"]}
    )
    agent._v2_cleanup_pending["att-0"] = {
        "job_id": "job-0",
        "attempt_id": "att-0",
        "resources": {"host_id": "hardening-host", "nfs_mounts": [f"{base}/vol-cleanup"]},
    }
    real_isdir, real_listdir = os.path.isdir, os.listdir
    monkeypatch.setattr(agent.os.path, "isdir", lambda p: p == base or real_isdir(p))
    monkeypatch.setattr(
        agent.os,
        "listdir",
        lambda p: ["vol-setup", "vol-cleanup", "vol-orphan"] if p == base else real_listdir(p),
    )
    docker = Docker("missing")  # `docker ps` lists no running container
    monkeypatch.setattr(agent.subprocess, "run", docker)
    lock = _ObservedLock()
    monkeypatch.setattr(agent, "_v2_resources_lock", lock)
    unmounted = []
    monkeypatch.setattr(
        agent, "_unmount_nfs", lambda path: unmounted.append((path, lock.depth > 0))
    )

    agent.cleanup_orphaned_volume_mounts()

    assert unmounted == [(f"{base}/vol-orphan", True)]


# ── run_job harness: everything but the decisions under test is isolated ─


class Api:
    """Scripted /agent/v2 endpoints; records every status report."""

    def __init__(self, statuses=None, renew=200):
        self.statuses = dict(statuses or {})  # status -> list of HTTP codes
        self.renew = renew
        self.reports = []

    def __call__(self, path, body, timeout=10):
        if path == "/agent/v2/leases/renew":
            return self.renew, {}
        assert path == "/agent/v2/attempts/status", path
        self.reports.append(dict(body))
        codes = self.statuses.get(body["status"]) or [200]
        return (codes.pop(0) if len(codes) > 1 else codes[0]), {}

    def sent(self):
        return [r["status"] for r in self.reports]


@pytest.fixture
def runnable(agent, monkeypatch):
    import egress_isolation

    for name, value in {
        "get_gpu_info": mock.Mock(return_value={"gpu_model": "RTX 2060"}),
        "build_platform_env": mock.Mock(return_value={}),
        "cache_image_available": mock.Mock(return_value=True),
        "cache_track_pull": mock.Mock(),
        "_push_log_lines": mock.Mock(),
        "build_secure_docker_args": mock.Mock(return_value=["docker", "run", "-d", "image"]),
        "build_egress_iptables_rules": mock.Mock(return_value=[]),
        "_inject_ssh_keys": mock.Mock(),
        "_run_provisioning_hooks": mock.Mock(),
        "_run_auto_launch": mock.Mock(),
        "_report_http_ports": mock.Mock(),
        "_kill_container": mock.Mock(),
        "_remove_container": mock.Mock(),
        "_monitor_container": mock.Mock(),
        "_monitor_interactive": mock.Mock(),
        "get_host_ip": mock.Mock(return_value="192.0.2.10"),
        "PREFER_GVISOR": False,
    }.items():
        monkeypatch.setattr(agent, name, value)
    monkeypatch.setattr(egress_isolation, "launch_args", lambda job, env: ([], env))
    monkeypatch.setattr(agent.time, "sleep", lambda _: None)
    monkeypatch.setattr(agent.subprocess, "run", Docker("running 0 att-1"))
    monkeypatch.setattr(agent, "_v2_container_state", lambda *a: ("missing", -1, ""))
    return agent


def job_for(auth, **spec):
    return {
        "job_id": auth["job_id"],
        "name": "hardening",
        "image": "image:1",
        "_v2_attempt_id": auth["attempt_id"],
        "_v2_fencing_token": auth["fencing_token"],
        "_v2_container_name": auth["container_name"],
        **spec,
    }


# ── 5. Serverless readiness failure settles; a restart preserves warmup ─


def test_unhealthy_serverless_worker_is_failed_not_left_holding_its_gpus(runnable, monkeypatch):
    agent = runnable
    auth = live_auth(container_started=False)
    agent._v2_attempts["job-1"] = auth
    agent._v2_journal_save()
    api = Api()
    monkeypatch.setattr(agent, "_v2_post", api)
    monkeypatch.setattr(agent, "_wait_serverless_health", mock.Mock(return_value=False))
    removal = []

    def kill(name):
        if api.sent()[-1:] == ["starting"]:  # the setup guard has already run
            removal.append(journal(agent)["job-1"].get("pending_terminal_report"))

    agent._kill_container.side_effect = kill

    agent.run_job(job_for(auth, job_type="serverless_worker", http_port=8080))

    assert removal == [{"status": "failed", "failure_code": "serverless_health_failed"}]
    failed = api.reports[-1]
    assert failed["status"] == "failed"
    assert failed["failure_code"] == "serverless_health_failed"
    assert "health check failed" in failed["detail"]["error"]
    assert "job-1" not in agent._v2_attempts  # settled: lease and GPUs released
    agent._monitor_container.assert_not_called()


def test_agent_restart_during_warmup_preserves_the_worker_for_adoption(runnable, monkeypatch):
    agent = runnable
    auth = live_auth(container_started=False)
    agent._v2_attempts["job-1"] = auth
    agent._v2_journal_save()
    api = Api()
    monkeypatch.setattr(agent, "_v2_post", api)
    cleanup = mock.Mock()
    monkeypatch.setattr(agent, "_v2_cleanup_resources", cleanup)

    def restarting(*args, **kwargs):
        agent._shutdown.set()
        return False

    monkeypatch.setattr(agent, "_wait_serverless_health", restarting)

    agent.run_job(job_for(auth, job_type="serverless_worker", http_port=8080))

    assert agent._kill_container.call_count == 1  # only the pre-setup name guard
    assert "failed" not in api.sent()
    assert journal(agent)["job-1"]["serverless_health"]["path"] == "/health"
    cleanup.assert_not_called()  # mounts and workspace stay with the container


def _adopted_warming(agent, monkeypatch, *, healthy, deadline):
    auth = live_auth(serverless_health={"port": 58080, "path": "/health", "deadline": deadline})
    agent._v2_attempts["job-1"] = auth
    agent._v2_journal_save()
    api = Api()
    monkeypatch.setattr(agent, "_v2_post", api)
    inspections = iter(["running 0 att-1"] * 3 + ["exited 137 att-1"] * 3)

    def docker(argv, **kwargs):
        if argv[:2] == ["docker", "inspect"]:
            try:
                state = next(inspections)
            except StopIteration:
                agent._shutdown.set()
                state = "exited 137 att-1"
            return SimpleNamespace(returncode=0, stdout=state, stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(agent.subprocess, "run", docker)
    monkeypatch.setattr(agent.time, "sleep", lambda _: None)
    monkeypatch.setattr(agent, "_remove_container", mock.Mock())
    monkeypatch.setattr(
        agent.requests,
        "get",
        mock.Mock(return_value=SimpleNamespace(status_code=200 if healthy else 503)),
    )
    agent._v2_monitor_container(auth)
    return auth, api


def test_adopted_warming_worker_resumes_its_readiness_wait(agent, monkeypatch):
    auth, api = _adopted_warming(agent, monkeypatch, healthy=True, deadline=time.time() + 60)
    assert api.sent()[0] == "running"
    assert "serverless_health" not in auth


def test_adopted_worker_past_its_readiness_deadline_is_failed(agent, monkeypatch):
    auth, api = _adopted_warming(agent, monkeypatch, healthy=False, deadline=time.time() - 1)
    assert api.sent() == ["failed"]
    assert api.reports[0]["failure_code"] == "serverless_health_failed"
    assert "job-1" not in agent._v2_attempts
