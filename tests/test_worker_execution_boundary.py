"""No legacy execution after failed negotiation, failed claim, or fenced ownership."""

from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import worker_agent as worker


@pytest.fixture
def isolated_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_adopted_containers", set())
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json"))
    monkeypatch.setattr(worker, "_v2_enabled", False)
    monkeypatch.setattr(worker, "_protocol_negotiated", True, raising=False)
    monkeypatch.setattr(worker, "_legacy_execution_allowed", True, raising=False)
    monkeypatch.setattr(worker, "get_gpu_info", mock.Mock(return_value={}))
    monkeypatch.setattr(worker, "build_platform_env", mock.Mock(return_value={}))
    monkeypatch.setattr(worker.time, "sleep", mock.Mock())
    monkeypatch.setattr(worker, "claim_lease", mock.Mock(return_value=None))
    monkeypatch.setattr(worker, "report_job_status", mock.Mock(return_value=False))
    monkeypatch.setattr(worker, "_kill_container", mock.Mock())
    monkeypatch.setattr(worker, "_remove_container", mock.Mock())
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        mock.Mock(return_value=SimpleNamespace(returncode=1, stdout="", stderr="isolated test")),
    )
    return {"job_id": "boundary-job", "image": "test/image:1"}


def test_failed_legacy_claim_cannot_touch_container_or_report_starting(isolated_worker):
    worker.run_job(isolated_worker)
    worker.claim_lease.assert_called_once_with(isolated_worker["job_id"])
    worker._kill_container.assert_not_called()
    worker._remove_container.assert_not_called()
    worker.report_job_status.assert_not_called()


@pytest.mark.parametrize("authority", [None, {"attempt_id": "replacement", "fencing_token": 8}])
def test_v2_marker_requires_matching_local_authority_before_cleanup(
    isolated_worker, monkeypatch, authority
):
    job = dict(
        isolated_worker,
        _v2_attempt_id="attempt-1",
        _v2_fencing_token=7,
        _v2_container_name="xcl-boundary-job-attempt-",
    )
    if authority is not None:
        worker._v2_attempts[job["job_id"]] = authority
    worker.run_job(job)
    worker.claim_lease.assert_not_called()
    worker._kill_container.assert_not_called()
    worker._remove_container.assert_not_called()
    worker.report_job_status.assert_not_called()


def test_legacy_job_cannot_reenter_an_existing_fenced_attempt(isolated_worker):
    worker._v2_attempts[isolated_worker["job_id"]] = {"attempt_id": "attempt-1"}
    worker.run_job(isolated_worker)
    worker.claim_lease.assert_not_called()
    worker._kill_container.assert_not_called()
    worker._remove_container.assert_not_called()
    worker.report_job_status.assert_not_called()


def test_adoption_requires_lease_before_tracking_or_monitoring(isolated_worker, monkeypatch):
    monkeypatch.setattr(
        worker.subprocess,
        "run",
        mock.Mock(
            side_effect=[
                SimpleNamespace(returncode=0, stdout="xcl-boundary-job", stderr=""),
                SimpleNamespace(returncode=0, stdout="{}", stderr=""),
            ]
        ),
    )
    with (
        mock.patch.object(worker, "LogForwarder") as logs,
        mock.patch.object(worker.threading, "Thread") as thread,
    ):
        worker.adopt_running_containers()
    assert worker._active_containers == {} and worker._adopted_containers == set()
    logs.assert_not_called()
    thread.assert_not_called()


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status_code=503, json=lambda: {"detail": "unavailable"}),
        SimpleNamespace(status_code=401, json=lambda: {"detail": "unauthorized"}),
        requests.ConnectionError("offline"),
    ],
)
def test_negotiation_failure_never_falls_back_to_legacy_poll(
    isolated_worker, monkeypatch, response
):
    api = mock.Mock()
    if isinstance(response, Exception):
        api.side_effect = response
    else:
        api.return_value = response
    monkeypatch.setattr(worker, "_api_request", api)
    worker.negotiate_protocol_v2()
    assert worker.poll_for_work() == []
    assert api.call_count == 1


@pytest.mark.parametrize("legacy_allowed", [False, True, None, "false"])
def test_negotiation_requires_explicit_boolean_permission(
    isolated_worker, monkeypatch, legacy_allowed
):
    body = {"protocol": 2, "v2": True, "features": []}
    if legacy_allowed is not None:
        body["legacy_execution_allowed"] = legacy_allowed
    api = mock.Mock(return_value=SimpleNamespace(status_code=200, json=lambda: body))
    monkeypatch.setattr(worker, "_api_request", api)
    assert worker.negotiate_protocol_v2()
    api.reset_mock()
    worker.poll_for_work()
    assert api.call_count == (1 if legacy_allowed is True else 0)


def test_disabled_legacy_adoption_never_inspects_or_claims(isolated_worker, monkeypatch):
    monkeypatch.setattr(worker, "_legacy_execution_allowed", False, raising=False)
    with mock.patch.object(worker.subprocess, "run") as docker:
        worker.adopt_running_containers()
    docker.assert_not_called()
    worker.claim_lease.assert_not_called()


def test_disabled_legacy_execution_cannot_start_work(isolated_worker, monkeypatch):
    monkeypatch.setattr(worker, "_legacy_execution_allowed", False, raising=False)
    worker.run_job(isolated_worker)
    worker.claim_lease.assert_not_called()
    worker._kill_container.assert_not_called()
    worker._remove_container.assert_not_called()


def test_uncertain_startup_retries_then_adopts_once(isolated_worker, monkeypatch):
    monkeypatch.setattr(worker, "_protocol_negotiated", False)
    bodies = [
        SimpleNamespace(status_code=503, json=lambda: {}),
        SimpleNamespace(
            status_code=200,
            json=lambda: {"protocol": 2, "v2": True, "legacy_execution_allowed": False},
        ),
    ]
    api = mock.Mock(side_effect=bodies)
    monkeypatch.setattr(worker, "_api_request", api)
    with (
        mock.patch.object(worker, "v2_adopt_attempts") as fenced,
        mock.patch.object(worker, "adopt_running_containers") as legacy,
    ):
        worker._ensure_work_protocol()
        fenced.assert_not_called()
        worker._ensure_work_protocol()
        worker._ensure_work_protocol()
        fenced.assert_called_once()
        legacy.assert_called_once()
    assert api.call_count == 2
