"""A lost claim response is replayed to its session; a failed fence kill is retried.

Before: a claim that committed but whose response was lost was refused on
retry ("lease is active"), so the reserved GPU sat idle until the lease
expired; and a fence-loss ``docker kill`` that failed (or an inspect that
failed) was logged once and forgotten, leaving the container running with no
authority and nothing left that would ever stop it.
"""

import json
import os
import subprocess
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import worker_agent as worker
from control_plane.attempts import report_attempt_status
from control_plane.db import run_transaction
from control_plane.leases import LeaseClaimRejected
from control_plane.serverless_credentials import authenticate_callback
from tests.test_control_plane_leases import _claim_grant, _pool, _reserved, fleet  # noqa: F401
from tests.test_serverless_native_lifecycle import native  # noqa: F401


class TestSameSessionClaimReplay:
    def test_committed_claim_is_replayed_to_its_own_session(self, fleet):
        resv = _reserved(fleet)
        first = _claim_grant(resv, session="lost-response")
        with _pool.connection() as conn:
            claimed_at = conn.execute(
                "SELECT claimed_at FROM placement_leases WHERE lease_id=%s", (resv.lease_id,)
            ).fetchone()[0]
        again = _claim_grant(resv, session="lost-response")
        assert (again.lease_id, again.attempt_id, again.fencing_token) == (
            first.lease_id,
            first.attempt_id,
            first.fencing_token,
        )
        assert again.gpu_device_uuids == first.gpu_device_uuids
        with _pool.connection() as conn:
            lease = conn.execute(
                "SELECT status, claimed_at FROM placement_leases WHERE lease_id=%s",
                (resv.lease_id,),
            ).fetchone()
            attempt = conn.execute(
                "SELECT status FROM job_attempts WHERE attempt_id=%s", (resv.attempt_id,)
            ).fetchone()
        assert lease == ("active", claimed_at)
        assert attempt[0] == "lease_claimed"

    def test_another_session_is_still_refused(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv, session="lost-response")
        with pytest.raises(LeaseClaimRejected, match="not claimable"):
            _claim_grant(resv, session="someone-else")

    def test_no_replay_once_the_attempt_has_progressed(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv, session="lost-response")
        run_transaction(
            lambda conn: report_attempt_status(
                conn,
                job_id=resv.job_id,
                attempt_id=resv.attempt_id,
                host_id=resv.host_id,
                fencing_token=resv.fencing_token,
                status="starting",
            )
        )
        with pytest.raises(LeaseClaimRejected, match="not claimable"):
            _claim_grant(resv, session="lost-response")

    def test_no_replay_of_an_expired_lease(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv, session="lost-response")
        with _pool.connection() as conn:
            conn.execute(
                "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 second' "
                "WHERE lease_id=%s",
                (resv.lease_id,),
            )
        with pytest.raises(LeaseClaimRejected):
            _claim_grant(resv, session="lost-response")

    def test_empty_session_never_replays(self, fleet):
        resv = _reserved(fleet)
        _claim_grant(resv, session="")
        with pytest.raises(LeaseClaimRejected):
            _claim_grant(resv, session="")


def _claim_over_http(client, r, session):
    body = {
        "lease_id": r.lease_id,
        "job_id": r.job_id,
        "attempt_id": r.attempt_id,
        "host_id": r.host_id,
        "fencing_token": r.fencing_token,
        "worker_session_id": session,
    }
    response = client.post(
        "/agent/v2/leases/claim",
        json=body,
        headers={"Authorization": f"Bearer {os.environ['XCELSIOR_API_TOKEN']}"},
    )
    return response.status_code, response.json()


def test_replayed_serverless_claim_rotates_the_callback_credential(native):  # noqa: F811
    import api

    r = native.reserve()
    with TestClient(api.app) as client:
        code, first = _claim_over_http(client, r, "lost-response")
        assert code == 200 and first["callback_token"]
        code, again = _claim_over_http(client, r, "lost-response")
        assert code == 200 and again["callback_token"]
        assert again["callback_token"] != first["callback_token"]
        assert again["fencing_token"] == first["fencing_token"]
        code, _ = _claim_over_http(client, r, "other-session")
        assert code == 409
    with native.pool.connection() as conn:
        assert not authenticate_callback(conn, first["callback_token"], native.wid)
        assert authenticate_callback(conn, again["callback_token"], native.wid)
        assert (
            conn.execute(
                "SELECT count(*) FROM serverless_attempt_credentials WHERE attempt_id=%s",
                (r.attempt_id,),
            ).fetchone()[0]
            == 1
        )
        conn.execute(
            "UPDATE serverless_attempt_credentials SET revoked_at=clock_timestamp() "
            "WHERE attempt_id=%s",
            (r.attempt_id,),
        )
    with TestClient(api.app) as client:
        code, refused = _claim_over_http(client, r, "lost-response")
    assert code == 409, refused
    with native.pool.connection() as conn:
        assert not authenticate_callback(conn, again["callback_token"], native.wid)


# ── Worker side ─────────────────────────────────────────────────────


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "HOST_ID", "fence-kill-host")
    monkeypatch.setattr(worker, "WORKER_SESSION_ID", "fence-kill-session")
    monkeypatch.setattr(worker, "_shutdown", threading.Event())
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_v2_completed_commands", {})
    monkeypatch.setattr(worker, "_v2_cleanup_pending", {})
    monkeypatch.setattr(worker, "_v2_pending_fence_kills", {})
    monkeypatch.setattr(worker, "_v2_starts_in_progress", set())
    monkeypatch.setattr(worker, "_V2_CLAIM_RETRY_DELAYS_SEC", (0, 0, 0))
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json"))
    return worker


class Docker:
    """Scripted docker CLI: inspect answers from ``states``, kill from ``kills``."""

    def __init__(self, states, kills):
        self.states = list(states)
        self.kills = list(kills)
        self.calls = []

    def __call__(self, argv, **_kwargs):
        self.calls.append(argv[1])
        if argv[1] == "inspect":
            state = self.states.pop(0)
            if state == "error":
                return SimpleNamespace(returncode=1, stdout="", stderr="daemon unavailable")
            if state == "missing":
                return SimpleNamespace(returncode=1, stdout="", stderr="Error: No such object")
            return SimpleNamespace(returncode=0, stdout=state, stderr="")
        if argv[1] == "kill":
            outcome = self.kills.pop(0)
            if outcome == "timeout":
                raise subprocess.TimeoutExpired(argv, 30)
            if outcome == "ok":
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=1, stdout="", stderr=outcome)
        raise AssertionError(argv)


def _journal(agent):
    with open(agent._V2_JOURNAL_PATH) as fh:
        return json.load(fh)


def test_failed_fence_kill_is_journaled_and_retried_until_stopped(agent, monkeypatch):
    docker = Docker(
        states=["running 0 att-1", "running 0 att-1", "exited 137 att-1"],
        kills=["Error response from daemon: connection reset", "ok"],
    )
    monkeypatch.setattr(agent.subprocess, "run", docker)
    agent._v2_stop_container("job-1", "renewal fenced", "xcl-job-1-att", "att-1")
    assert "xcl-job-1-att" in agent._v2_pending_fence_kills
    assert _journal(agent)[agent._V2_FENCE_KILLS_KEY]["xcl-job-1-att"]["expected_attempt_id"] == (
        "att-1"
    )
    agent._v2_retry_fence_kills()
    assert agent._v2_pending_fence_kills == {}
    assert agent._V2_FENCE_KILLS_KEY not in _journal(agent)
    assert docker.calls == ["inspect", "kill", "inspect", "kill"]


def test_inspect_failure_is_retried_instead_of_refusing_the_stop(agent, monkeypatch):
    docker = Docker(states=["error", "running 0 att-1"], kills=["ok"])
    monkeypatch.setattr(agent.subprocess, "run", docker)
    agent._v2_stop_container("job-1", "renewal fenced", "xcl-job-1-att", "att-1")
    assert docker.calls == ["inspect"]
    assert "xcl-job-1-att" in agent._v2_pending_fence_kills
    agent._v2_retry_fence_kills()
    assert docker.calls == ["inspect", "inspect", "kill"]
    assert agent._v2_pending_fence_kills == {}


def test_pending_fence_kill_survives_a_restart(agent, monkeypatch):
    docker = Docker(states=["running 0 att-1", "running 0 att-1"], kills=["timeout", "ok"])
    monkeypatch.setattr(agent.subprocess, "run", docker)
    agent._v2_stop_container("job-1", "renewal fenced", "xcl-job-1-att", "att-1")
    monkeypatch.setattr(agent, "_v2_pending_fence_kills", {})
    monkeypatch.setattr(agent, "_v2_enabled", True)
    assert agent.v2_adopt_attempts() == 0
    assert docker.calls == ["inspect", "kill", "inspect", "kill"]
    assert agent._v2_pending_fence_kills == {}


def test_a_different_attempts_container_is_still_never_killed(agent, monkeypatch):
    docker = Docker(states=["running 0 att-2"], kills=[])
    monkeypatch.setattr(agent.subprocess, "run", docker)
    agent._v2_stop_container("job-1", "renewal fenced", "xcl-job-1-att", "att-1")
    assert docker.calls == ["inspect"]
    assert agent._v2_pending_fence_kills == {}


def test_already_stopped_container_settles_without_retry(agent, monkeypatch):
    docker = Docker(
        states=["running 0 att-1"],
        kills=["Error response from daemon: Cannot kill container: x: is not running"],
    )
    monkeypatch.setattr(agent.subprocess, "run", docker)
    agent._v2_stop_container("job-1", "renewal fenced", "xcl-job-1-att", "att-1")
    assert agent._v2_pending_fence_kills == {}


def _auth():
    return {
        "lease_id": "lease-1",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "fencing_token": 7,
    }


def _grant():
    return {
        "lease_id": "lease-1",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "fencing_token": 7,
    }


@pytest.mark.parametrize("unknown", [0, 429, 502, 503])
def test_claim_with_unknown_outcome_is_retried(agent, monkeypatch, unknown):
    replies = [(unknown, {}), (200, _grant())]
    post = mock.Mock(side_effect=lambda path, body: replies.pop(0))
    monkeypatch.setattr(agent, "_v2_post", post)
    assert agent.v2_claim_lease_fenced(_auth()) == _grant()
    assert post.call_count == 2
    assert post.call_args_list[0].args == post.call_args_list[1].args


@pytest.mark.parametrize("definitive", [404, 409, 410])
def test_definitive_claim_rejection_is_not_retried(agent, monkeypatch, definitive):
    post = mock.Mock(return_value=(definitive, {"detail": "lease is released"}))
    monkeypatch.setattr(agent, "_v2_post", post)
    assert agent.v2_claim_lease_fenced(_auth()) is None
    assert post.call_count == 1


def test_claim_retries_are_bounded(agent, monkeypatch):
    post = mock.Mock(return_value=(0, {}))
    monkeypatch.setattr(agent, "_v2_post", post)
    assert agent.v2_claim_lease_fenced(_auth()) is None
    assert post.call_count == 1 + len(agent._V2_CLAIM_RETRY_DELAYS_SEC)


def test_redelivered_start_during_a_start_is_ignored(agent, monkeypatch):
    inner = mock.Mock()
    monkeypatch.setattr(agent, "_handle_start_attempt", inner)
    agent._v2_starts_in_progress.add("attempt-1")
    agent.handle_start_attempt({"command_id": "c-2", "attempt_id": "attempt-1"})
    inner.assert_not_called()
    agent._v2_starts_in_progress.clear()
    agent.handle_start_attempt({"command_id": "c-3", "attempt_id": "attempt-1"})
    inner.assert_called_once()
    assert agent._v2_starts_in_progress == set()
