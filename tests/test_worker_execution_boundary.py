"""Only a complete native handshake and exact local authority can execute work."""

from types import SimpleNamespace
from unittest import mock

import pytest
import requests

import worker_agent as worker

FEATURES = [
    "commands.host_claim_ack",
    "commands.session_fencing",
    "attempts.runtime_metadata",
    "leases.gpu_device_binding",
    "leases.start_authority",
]


@pytest.fixture
def isolated_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(worker, "_v2_attempts", {})
    monkeypatch.setattr(worker, "_active_containers", {})
    monkeypatch.setattr(worker, "_adopted_containers", set())
    monkeypatch.setattr(worker, "_V2_JOURNAL_PATH", str(tmp_path / "attempts.json"))
    monkeypatch.setattr(worker, "_v2_enabled", False)
    monkeypatch.setattr(worker, "_protocol_negotiated", False)
    monkeypatch.setattr(worker, "get_gpu_info", mock.Mock(return_value={}))
    monkeypatch.setattr(worker, "build_platform_env", mock.Mock(return_value={}))
    monkeypatch.setattr(worker.time, "sleep", mock.Mock())
    monkeypatch.setattr(worker, "report_job_status", mock.Mock(return_value=False))
    monkeypatch.setattr(worker, "_kill_container", mock.Mock())
    monkeypatch.setattr(worker, "_remove_container", mock.Mock())
    monkeypatch.setattr(
        worker.subprocess, "run", mock.Mock(side_effect=AssertionError("unexpected Docker call"))
    )
    return {"job_id": "boundary-job", "image": "test/image:1"}


def _assert_no_execution():
    worker._kill_container.assert_not_called()
    worker._remove_container.assert_not_called()
    worker.report_job_status.assert_not_called()
    worker.subprocess.run.assert_not_called()


@pytest.mark.parametrize("negotiated", [True, False])
def test_unfenced_job_cannot_touch_container_or_report_starting(
    isolated_worker, monkeypatch, negotiated
):
    monkeypatch.setattr(worker, "_protocol_negotiated", negotiated)
    worker.run_job(isolated_worker)
    _assert_no_execution()


@pytest.mark.parametrize("authority", [None, {"attempt_id": "replacement", "fencing_token": 8}])
def test_attempt_marker_requires_matching_local_authority_before_cleanup(
    isolated_worker, authority
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
    _assert_no_execution()


def test_unfenced_job_cannot_reenter_an_existing_fenced_attempt(isolated_worker):
    worker._v2_attempts[isolated_worker["job_id"]] = {"attempt_id": "attempt-1"}
    worker.run_job(isolated_worker)
    _assert_no_execution()


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(status_code=503, json=lambda: {}),
        SimpleNamespace(status_code=401, json=lambda: {}),
        requests.ConnectionError("offline"),
    ],
)
def test_negotiation_failure_never_executes_unfenced_work(isolated_worker, monkeypatch, response):
    api = mock.Mock()
    if isinstance(response, Exception):
        api.side_effect = response
    else:
        api.return_value = response
    monkeypatch.setattr(worker, "_api_request", api)
    assert not worker.negotiate_protocol_v2()
    assert not worker._protocol_negotiated
    worker.run_job(isolated_worker)
    _assert_no_execution()
    assert api.call_count == 1


@pytest.mark.parametrize("legacy_allowed", [False, True, None, "false"])
def test_complete_native_contract_ignores_old_permission_flag(
    isolated_worker, monkeypatch, legacy_allowed
):
    body = {
        "protocol": 2,
        "v2": True,
        "features": FEATURES,
        "legacy_execution_allowed": legacy_allowed,
    }
    api = mock.Mock(return_value=SimpleNamespace(status_code=200, json=lambda: body))
    monkeypatch.setattr(worker, "_api_request", api)
    assert worker.negotiate_protocol_v2()
    worker.run_job(isolated_worker)
    _assert_no_execution()
    assert api.call_count == 1


@pytest.mark.parametrize(
    "body",
    [
        {"v2": False, "protocol": 1, "features": FEATURES},
        {"v2": True, "protocol": 2, "features": []},
        {"v2": True, "protocol": 2, "features": FEATURES[:-1]},
        {"v2": True, "protocol": "2", "features": FEATURES},
        {"v2": "true", "protocol": 2, "features": FEATURES},
        {"v2": True, "protocol": 2, "features": {"commands.host_claim_ack": True}},
        {"v2": True, "protocol": 2, "features": FEATURES + [{}]},
        [],
        None,
    ],
)
def test_incomplete_native_contract_cannot_enable_execution(isolated_worker, monkeypatch, body):
    if isinstance(body, dict):
        body = dict(body, legacy_execution_allowed=True)
    monkeypatch.setattr(worker, "_v2_enabled", True)
    monkeypatch.setattr(worker, "_protocol_negotiated", True)
    monkeypatch.setattr(
        worker,
        "_api_request",
        mock.Mock(return_value=SimpleNamespace(status_code=200, json=lambda: body)),
    )
    assert not worker.negotiate_protocol_v2()
    assert not worker._protocol_negotiated
    assert not worker._v2_enabled


def test_uncertain_startup_retries_then_recovers_journal_once(isolated_worker, monkeypatch):
    api = mock.Mock(
        side_effect=[
            SimpleNamespace(status_code=503, json=lambda: {}),
            SimpleNamespace(
                status_code=200, json=lambda: {"protocol": 2, "v2": True, "features": FEATURES}
            ),
        ]
    )
    monkeypatch.setattr(worker, "_api_request", api)
    with mock.patch.object(worker, "v2_adopt_attempts") as recover:
        worker._ensure_work_protocol()
        recover.assert_not_called()
        worker._ensure_work_protocol()
        worker._ensure_work_protocol()
        recover.assert_called_once()
    assert api.call_count == 2


@pytest.mark.parametrize("legacy_allowed", [True, False])
def test_startup_with_no_fenced_journal_never_discovers_named_containers(
    isolated_worker, monkeypatch, legacy_allowed
):
    body = {
        "protocol": 2,
        "v2": True,
        "features": FEATURES,
        "legacy_execution_allowed": legacy_allowed,
    }
    monkeypatch.setattr(
        worker,
        "_api_request",
        mock.Mock(return_value=SimpleNamespace(status_code=200, json=lambda: body)),
    )
    worker._ensure_work_protocol()
    assert worker._protocol_negotiated
    assert worker._v2_attempts == {}
    worker.subprocess.run.assert_not_called()


def test_replaced_runner_cannot_borrow_new_attempt_for_startup_report(monkeypatch):
    old = {
        "job_id": "runner-race",
        "attempt_id": "old",
        "lease_id": "old-lease",
        "fencing_token": 7,
        "container_name": "xcl-runner-old",
    }
    replacement = dict(
        old,
        attempt_id="new",
        lease_id="new-lease",
        fencing_token=8,
        container_name="xcl-runner-new",
    )
    monkeypatch.setattr(worker, "_v2_attempts", {old["job_id"]: old})
    monkeypatch.setattr(worker, "_active_containers", {})

    def replace_authority():
        worker._v2_attempts[old["job_id"]] = replacement
        return {}

    monkeypatch.setattr(worker, "get_gpu_info", replace_authority)
    monkeypatch.setattr(worker, "build_platform_env", lambda *args: {})
    monkeypatch.setattr(worker.time, "sleep", lambda _: None)
    monkeypatch.setattr(worker, "_kill_container", mock.Mock())
    monkeypatch.setattr(worker, "_remove_container", mock.Mock())
    post = mock.Mock(side_effect=AssertionError("old runner borrowed replacement authority"))
    monkeypatch.setattr(worker, "_v2_post", post)
    worker.run_job(
        {
            "job_id": old["job_id"],
            "image": "test/image:1",
            "_v2_attempt_id": "old",
            "_v2_fencing_token": 7,
            "_v2_container_name": old["container_name"],
        }
    )
    post.assert_not_called()
    assert worker._v2_attempts[old["job_id"]] is replacement
