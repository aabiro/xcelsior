"""The last of the handlers `measure_route_execution.py` found unentered.

Image sweeps, instance termination, the `/v1/inference` pair, launch-plan read
and revoke, the agent sweep-fingerprint callback, and the SSE stream.

Two of these guard the platform's largest spend. `POST /api/v1/image-sweeps`
says so in its own docstring — "a sweep is the largest single spend on the
platform and had the weakest gate", since it can launch up to sixty-four
instances where the single-instance path refuses to launch one without an
approved plan. That gate had no test calling it.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def funded():
    email = f"remaining-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Remaining"},
    )
    assert reg.status_code == 200, reg.text[:300]
    body = reg.json()
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    customer_id = (body.get("user") or {}).get("customer_id")
    if customer_id:
        client.post(
            f"/api/billing/wallet/{customer_id}/deposit",
            json={"amount_cad": 50.0},
            headers=headers,
        )
    return headers


# ── Image sweeps ──────────────────────────────────────────────────────────


def test_creating_a_sweep_quotes_rather_than_launches(funded) -> None:
    """"Nothing launches here" — it must produce a plan awaiting approval."""
    r = client.post(
        "/api/v1/image-sweeps",
        json={"image_id": f"img-{uuid.uuid4().hex[:10]}", "count": 2, "name": "probe-sweep"},
        headers=funded,
    )
    assert r.status_code < 500, f"sweep creation faulted: {r.status_code} {r.text[:300]}"
    if r.status_code == 200:
        body = r.json()
        assert "approval_url" in body, (
            f"a sweep came back without an approval route: {body}"
        )
        assert body.get("approval_state") in ("pending", "approved"), body


def test_a_sweep_larger_than_the_cap_is_refused(funded) -> None:
    """`count` is `le=64`. The cap is the difference between a sweep and a bill."""
    r = client.post(
        "/api/v1/image-sweeps",
        json={"image_id": f"img-{uuid.uuid4().hex[:10]}", "count": 65},
        headers=funded,
    )
    assert r.status_code == 422, r.text[:300]


def test_a_zero_member_sweep_is_refused(funded) -> None:
    r = client.post(
        "/api/v1/image-sweeps",
        json={"image_id": f"img-{uuid.uuid4().hex[:10]}", "count": 0},
        headers=funded,
    )
    assert r.status_code == 422, r.text[:300]


def test_executing_an_unapproved_sweep_plan_is_refused(funded) -> None:
    """Gate P7: execution requires an approved plan, not merely a plan id."""
    r = client.post(
        f"/api/v1/image-sweep-plans/{uuid.uuid4()}/execute", headers=funded
    )
    assert r.status_code < 500, f"sweep execute faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404, 409), r.text[:200]


def test_reading_an_unknown_sweep(funded) -> None:
    r = client.get(f"/api/v1/image-sweeps/{uuid.uuid4()}", headers=funded)
    assert r.status_code < 500, f"sweep read faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (403, 404), r.text[:200]


def test_reporting_a_sweep_fingerprint_for_an_unknown_sweep() -> None:
    """Agent-authenticated, so it is called *without* a user bearer token.

    Written first with the admin header, which never reached the handler: a
    platform bearer on an `/agent/`-authenticated route is treated as a retired
    API key and refused with 410 by `_require_agent_auth` before the body is
    ever seen. In the test environment an unauthenticated agent call is
    accepted (conftest sets `XCELSIOR_ENV=test`), which is how the reporter is
    simulated here.
    """
    r = client.post(
        f"/api/v1/image-sweeps/{uuid.uuid4()}/members/0/fingerprint",
        # `hash` is min_length=16 and `manifest` is required. Written first as
        # {"fingerprint": {...}}, which is a 422 — FastAPI refusing before the
        # handler ran, and a test that passed while covering nothing. The
        # coverage measurement is what caught it, for the third time.
        json={"hash": "a" * 64, "manifest": {"driver": "550.54", "cuda": "12.4"}},
    )
    assert r.status_code != 410, (
        "the request was refused as a retired API key, so the handler never ran"
    )
    assert r.status_code != 422, f"the body did not satisfy the model: {r.text[:300]}"
    assert r.status_code < 500, f"fingerprint report faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404, 409), r.text[:200]


# ── Instance termination ──────────────────────────────────────────────────


def test_terminating_an_unknown_instance(funded) -> None:
    """Irreversible by design, so an unknown id must refuse rather than fault."""
    r = client.post(f"/instances/job-{uuid.uuid4().hex[:10]}/terminate", headers=funded)
    assert r.status_code < 500, f"terminate faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404, 409), r.text[:200]


def test_terminating_another_tenants_instance_is_not_found(funded) -> None:
    """A cross-tenant id must look absent rather than forbidden — no existence oracle."""
    from scheduler import submit_job
    from db import _get_pg_pool

    job = submit_job(
        name=f"terminate-probe-{uuid.uuid4().hex[:8]}",
        vram_needed_gb=1,
        image="xcelsior/probe:latest",
        owner="someone-else-entirely",
    )
    job_id = job.get("job_id") or job.get("id")
    try:
        r = client.post(f"/instances/{job_id}/terminate", headers=funded)
        assert r.status_code in (403, 404), (
            f"another tenant's instance answered {r.status_code} to a terminate"
        )
    finally:
        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


# ── /v1/inference ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ["/v1/inference", "/v1/inference/async"])
def test_v1_inference_reaches_its_handler(funded, path) -> None:
    r = client.post(
        path,
        json={"model": "distilbert-base-uncased", "inputs": "hello", "max_tokens": 16},
        headers=funded,
    )
    assert r.status_code != 422, f"{path} did not accept a well-formed body: {r.text[:300]}"
    assert r.status_code < 500, f"{path} faulted: {r.status_code} {r.text[:300]}"


@pytest.mark.parametrize("path", ["/v1/inference", "/v1/inference/async"])
def test_v1_inference_enforces_its_token_ceiling(funded, path) -> None:
    """`max_tokens` is `le=8192`; the ceiling is a cost control."""
    r = client.post(
        path,
        json={"model": "distilbert-base-uncased", "inputs": "hello", "max_tokens": 999999},
        headers=funded,
    )
    assert r.status_code == 422, r.text[:300]


def test_async_inference_accepts_stream_without_honouring_it(funded) -> None:
    """Documented rather than changed.

    `V1InferenceRequest` is shared with the synchronous route, where `stream`
    selects SSE. The async route returns a job id to poll, so the field has no
    meaning there and is ignored. That is a shared-model artifact, not a
    dropped option like `gpu_model` was on `/api/inference` — a client setting
    `stream` uniformly across both should not start getting 422s.
    """
    r = client.post(
        "/v1/inference/async",
        json={"model": "distilbert-base-uncased", "inputs": "hi", "stream": True},
        headers=funded,
    )
    assert r.status_code != 422, r.text[:300]
    assert r.status_code < 500, r.text[:300]


# ── Launch plans ──────────────────────────────────────────────────────────


def test_reading_an_unknown_launch_plan(funded) -> None:
    """A foreign plan id must be not-found, never forbidden."""
    r = client.get(f"/api/v1/launch-plans/{uuid.uuid4()}", headers=funded)
    assert r.status_code < 500, f"plan read faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code == 404, (
        f"an unowned plan answered {r.status_code}; anything but 404 tells the "
        "caller the id exists"
    )


def test_revoking_an_unknown_launch_plan(funded) -> None:
    r = client.post(f"/api/v1/launch-plans/{uuid.uuid4()}/revoke", json={}, headers=funded)
    assert r.status_code < 500, f"plan revoke faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (403, 404, 409), r.text[:200]


def test_a_malformed_plan_id_is_a_client_error(funded) -> None:
    r = client.get("/api/v1/launch-plans/not-a-uuid", headers=funded)
    assert r.status_code < 500, f"a malformed plan id faulted: {r.text[:300]}"
    assert 400 <= r.status_code < 500, r.text[:200]
