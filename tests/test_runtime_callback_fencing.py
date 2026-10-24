"""Callbacks from stale attempts cannot publish ports or interactive credentials."""

import hashlib
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import worker_agent as worker
from db import _get_pg_pool
from tests.test_auto_launch import api_client, api_seeded_job  # noqa: F401
from tests.test_worker_resource_recovery import resources  # noqa: F401

TOKEN = "b" * 32


def body(instance):
    return {
        "host_id": instance["host_id"],
        **instance["authority"],
        "ports": {"jupyter": 8888},
        "token": TOKEN,
        "token_sha": hashlib.sha256(TOKEN.encode()).hexdigest()[:8],
    }


@pytest.mark.parametrize("field", ["attempt_id", "fencing_token"])
def test_callback_rejects_stale_attempt_authority(api_client, api_seeded_job, field):
    data = body(api_seeded_job)
    data[field] = str(uuid.uuid4()) if field == "attempt_id" else data[field] + 1
    for kind in ("auto-launch", "http-ports"):
        if kind == "http-ports":
            data["ports"] = {"8888": 55123}
        response = api_client.post(
            f"/instances/{api_seeded_job['job_id']}/{kind}/report", json=data
        )
        assert response.status_code == 409
    with _get_pg_pool().connection() as conn:
        assert conn.execute(
            "SELECT payload->'http_ports',payload->'auto_launch_ports' FROM jobs WHERE job_id=%s",
            (api_seeded_job["job_id"],),
        ).fetchone() == (None, None)


def test_callback_requires_fresh_lease_and_hides_expired_credentials(api_client, api_seeded_job):
    url = f"/instances/{api_seeded_job['job_id']}"
    assert (
        api_client.post(url + "/auto-launch/report", json=body(api_seeded_job)).status_code == 200
    )
    assert (
        api_client.post(
            url + "/http-ports/report", json={**body(api_seeded_job), "ports": {"8888": 55123}}
        ).status_code
        == 200
    )
    assert api_client.get(url + "/auto-launch").json()["ready"]
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 second' WHERE attempt_id=%s",
            (api_seeded_job["authority"]["attempt_id"],),
        )
    assert (
        api_client.post(url + "/auto-launch/report", json=body(api_seeded_job)).status_code == 409
    )
    assert api_client.get(url + "/auto-launch").json() == {"ready": False, "services": {}}


def test_credential_digest_and_admitted_ports_are_enforced(api_client, api_seeded_job):
    url = f"/instances/{api_seeded_job['job_id']}"
    for bad in ({"token_sha": "00000000"}, {"ports": {"jupyter": 9999}}):
        response = api_client.post(
            url + "/auto-launch/report", json={**body(api_seeded_job), **bad}
        )
        assert response.status_code == 422
    response = api_client.post(
        url + "/http-ports/report", json={**body(api_seeded_job), "ports": {"9999": 55123}}
    )
    assert response.status_code == 422


def test_callback_event_is_durable_idempotent_and_contains_no_credential(
    api_client, api_seeded_job
):
    url = f"/instances/{api_seeded_job['job_id']}/auto-launch/report"
    for _ in range(2):
        assert api_client.post(url, json=body(api_seeded_job)).status_code == 200
    with _get_pg_pool().connection() as conn:
        events = conn.execute(
            "SELECT payload FROM outbox_events WHERE aggregate_id=%s AND event_type='job.v1.auto_launch_ports_ready'",
            (api_seeded_job["job_id"],),
        ).fetchall()
    assert len(events) == 1
    assert TOKEN not in json.dumps(events)
    assert "token" not in events[0][0]


def test_runtime_report_survives_restart_and_retries_with_original_fence(resources, monkeypatch):
    request = mock.Mock(side_effect=TimeoutError("response lost"))
    monkeypatch.setattr(worker, "_api_request", request)
    monkeypatch.setattr(worker, "_api_headers", lambda: {})
    resources["auto_launch_token"] = TOKEN
    assert not worker._v2_runtime_report(
        resources, "auto-launch", {"ports": {"jupyter": 8888}, "token": TOKEN}
    )
    saved = json.loads(Path(worker._V2_JOURNAL_PATH).read_text())["job"]
    assert saved["pending_runtime_reports"]["auto-launch"]["fencing_token"] == 7
    assert Path(worker._V2_JOURNAL_PATH).stat().st_mode & 0o777 == 0o600
    worker._v2_attempts.clear()
    worker._v2_container_state.return_value = ("running", 0, "attempt")
    with mock.patch.object(worker.threading, "Thread"):
        assert worker.v2_adopt_attempts() == 1
    adopted = worker._v2_attempts["job"]
    assert adopted["auto_launch_token"] == TOKEN
    request.side_effect = None
    request.return_value = SimpleNamespace(status_code=200)
    worker._v2_retry_runtime_reports(adopted)
    assert request.call_args.kwargs["json"]["fencing_token"] == 7
    assert request.call_args.kwargs["json"]["token"] == TOKEN
    assert not json.loads(Path(worker._V2_JOURNAL_PATH).read_text())["job"][
        "pending_runtime_reports"
    ]


def test_stale_thread_cannot_borrow_new_authority_or_send_unjournaled_report(
    resources, monkeypatch
):
    request = mock.Mock()
    monkeypatch.setattr(worker, "_api_request", request)
    worker._v2_attempts["job"] = dict(resources, fencing_token=8)
    assert not worker._v2_runtime_report(resources, "http-ports", {"ports": {"8888": 55123}})
    request.assert_not_called()
    worker._v2_attempts["job"] = resources
    monkeypatch.setattr(worker, "_v2_journal_save", lambda: False)
    assert not worker._v2_runtime_report(resources, "http-ports", {"ports": {"8888": 55123}})
    request.assert_not_called()
