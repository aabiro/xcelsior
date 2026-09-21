"""The serverless handlers the suite never reached.

The largest of the clusters `scripts/measure_route_execution.py` turned up:
around twenty handlers never entered by any test, every one scored "covered"
because `/api/v2/serverless/` appears throughout `tests/`. The existing
serverless files are numerous and thorough about the *engine* — autoscaling,
billing, idempotency, limits — and none of them creates an endpoint over HTTP
and then calls the dashboard and OpenAI-compatible routes that hang off it.

So a real endpoint is created here and the rest are driven against it. That
matters more than usual for this surface: most of these routes take
`endpoint_id` from the path and are the ones a dashboard polls, so "does it
answer at all for an endpoint that exists" is precisely the question nothing
was asking.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def persistent_auth(monkeypatch):
    import api as api_mod
    import routes._deps as deps
    import routes.auth as auth

    monkeypatch.setattr(deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(auth, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(api_mod, "_USE_PERSISTENT_AUTH", True)
    api_mod._RATE_BUCKETS.clear()
    deps._AUTH_RATE_BUCKETS.clear()


@pytest.fixture
def funded():
    email = f"sls-routes-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Serverless Routes"},
    )
    assert reg.status_code == 200, reg.text[:300]
    body = reg.json()
    headers = {"Authorization": f"Bearer {body['access_token']}"}
    customer_id = (body.get("user") or {}).get("customer_id")
    assert customer_id, f"registration returned no customer: {body}"
    dep = client.post(
        f"/api/billing/wallet/{customer_id}/deposit",
        json={"amount_cad": 50.0},
        headers=headers,
    )
    assert dep.status_code == 200, dep.text[:300]
    return headers


@pytest.fixture
def endpoint(funded):
    """A real custom-mode endpoint. Preset mode would demand a `model_ref`."""
    r = client.post(
        "/api/v2/serverless/endpoints",
        json={
            "name": f"probe-{uuid.uuid4().hex[:6]}",
            "mode": "custom",
            "source_type": "image",
            "image_ref": "xcelsior/echo:1.0",
            "docker_image": "xcelsior/echo:1.0",
            "gpu_type": "RTX4090",
            "min_workers": 0,
            "max_workers": 1,
        },
        headers=funded,
    )
    assert r.status_code == 200, f"could not create an endpoint: {r.text[:400]}"
    ep = r.json()["endpoint"]
    yield ep, funded
    client.delete(f"/api/v2/inference/endpoints/{ep['endpoint_id']}", headers=funded)


# ── Dashboard reads ───────────────────────────────────────────────────────


@pytest.mark.parametrize("suffix", ["metrics", "workers", "keys"])
def test_dashboard_reads_answer_for_a_real_endpoint(endpoint, suffix) -> None:
    ep, headers = endpoint
    r = client.get(f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/{suffix}", headers=headers)
    assert r.status_code == 200, f"{suffix} returned {r.status_code}: {r.text[:300]}"


def test_listing_keys_does_not_return_the_secrets(endpoint) -> None:
    """A key listing is a management view, not a credential handout."""
    ep, headers = endpoint
    r = client.get(f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/keys", headers=headers)
    assert r.status_code == 200, r.text[:300]
    body = r.text
    for giveaway in ("secret_key", "private_key", "plaintext"):
        assert giveaway not in body, f"{giveaway} appeared in the key listing"


@pytest.mark.parametrize("suffix", ["metrics", "workers", "keys"])
def test_dashboard_reads_refuse_another_owners_endpoint(endpoint, suffix) -> None:
    """`endpoint_id` is caller-supplied; ownership is the only thing checking it."""
    ep, _ = endpoint
    other = f"sls-other-{uuid.uuid4().hex[:10]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": other, "password": "StrongPass123!", "name": "Other"},
    )
    other_headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
    r = client.get(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/{suffix}", headers=other_headers
    )
    assert r.status_code in (403, 404), (
        f"another account read {suffix} for an endpoint it does not own ({r.status_code})"
    )


# ── Jobs ──────────────────────────────────────────────────────────────────


def test_dashboard_job_status_for_an_unknown_job(endpoint) -> None:
    ep, headers = endpoint
    r = client.get(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/jobs/job-{uuid.uuid4().hex[:8]}",
        headers=headers,
    )
    assert r.status_code == 404, r.text[:300]


def test_dashboard_cancel_for_an_unknown_job(endpoint) -> None:
    ep, headers = endpoint
    r = client.post(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/jobs/job-{uuid.uuid4().hex[:8]}/cancel",
        headers=headers,
    )
    assert r.status_code == 404, r.text[:300]


def test_public_job_status_for_an_unknown_job(endpoint) -> None:
    """The `/v1/serverless/...` surface, which SDK clients poll."""
    ep, headers = endpoint
    r = client.get(
        f"/v1/serverless/{ep['endpoint_id']}/status/job-{uuid.uuid4().hex[:8]}", headers=headers
    )
    assert r.status_code < 500, f"status faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (401, 403, 404), r.text[:200]


def test_public_cancel_for_an_unknown_job(endpoint) -> None:
    ep, headers = endpoint
    r = client.post(
        f"/v1/serverless/{ep['endpoint_id']}/cancel/job-{uuid.uuid4().hex[:8]}", headers=headers
    )
    assert r.status_code < 500, f"cancel faulted: {r.status_code} {r.text[:300]}"


def test_worker_job_fetch_for_an_unknown_pair(funded) -> None:
    r = client.get(
        f"/api/v2/serverless/workers/worker-{uuid.uuid4().hex[:8]}/jobs/job-{uuid.uuid4().hex[:8]}",
        headers=funded,
    )
    assert r.status_code < 500, f"worker job fetch faulted: {r.status_code} {r.text[:300]}"


# ── Batches ───────────────────────────────────────────────────────────────


def test_creating_a_batch(endpoint) -> None:
    """`requests` has `min_length=1`; an empty body never reaches the handler."""
    ep, headers = endpoint
    r = client.post(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/batches",
        json={"requests": [{"input": {"prompt": "hello"}}], "completion_window": "24h"},
        headers=headers,
    )
    assert r.status_code != 422, f"the body did not satisfy the model: {r.text[:300]}"
    assert r.status_code < 500, f"batch creation faulted: {r.status_code} {r.text[:300]}"


def test_an_empty_batch_is_refused(endpoint) -> None:
    ep, headers = endpoint
    r = client.post(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/batches",
        json={"requests": []},
        headers=headers,
    )
    assert r.status_code == 422, r.text[:300]


def test_getting_an_unknown_batch(funded) -> None:
    r = client.get(f"/api/v2/serverless/batches/{uuid.uuid4()}", headers=funded)
    assert r.status_code < 500, f"batch fetch faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (403, 404), r.text[:200]


# ── Test-invoke surface ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "suffix, body",
    [
        ("test/runsync", {"input": {"prompt": "hello"}}),
        (
            "test/openai/v1/chat/completions",
            {"model": "probe", "messages": [{"role": "user", "content": "hi"}]},
        ),
    ],
)
def test_the_dashboard_test_invokers_do_not_fault(endpoint, suffix, body) -> None:
    """No worker is attached, so these refuse — the point is *how*.

    429 (the test-invoke rate limit) and 5xx are different answers: the first
    is the handler applying its own policy, the second is it falling over.
    """
    ep, headers = endpoint
    r = client.post(
        f"/api/v2/serverless/endpoints/{ep['endpoint_id']}/{suffix}", json=body, headers=headers
    )
    assert r.status_code < 500, f"{suffix} faulted: {r.status_code} {r.text[:300]}"


# ── OpenAI-compatible endpoint management ─────────────────────────────────


def test_inference_compat_get_and_health_and_usage(endpoint) -> None:
    ep, headers = endpoint
    eid = ep["endpoint_id"]
    for path in (
        f"/api/v2/inference/endpoints/{eid}",
        f"/api/v2/inference/endpoints/{eid}/health",
        f"/api/v2/inference/endpoints/{eid}/usage",
    ):
        r = client.get(path, headers=headers)
        assert r.status_code == 200, f"{path} returned {r.status_code}: {r.text[:300]}"


def test_inference_compat_delete_removes_the_endpoint(funded) -> None:
    r = client.post(
        "/api/v2/serverless/endpoints",
        json={
            "name": f"probe-del-{uuid.uuid4().hex[:6]}",
            "mode": "custom",
            "source_type": "image",
            "image_ref": "xcelsior/echo:1.0",
            "docker_image": "xcelsior/echo:1.0",
            "gpu_type": "RTX4090",
            "min_workers": 0,
            "max_workers": 1,
        },
        headers=funded,
    )
    assert r.status_code == 200, r.text[:300]
    eid = r.json()["endpoint"]["endpoint_id"]

    delete = client.delete(f"/api/v2/inference/endpoints/{eid}", headers=funded)
    assert delete.status_code == 200, delete.text[:300]

    after = client.get(f"/api/v2/inference/endpoints/{eid}", headers=funded)
    assert after.status_code in (404, 410), (
        f"the endpoint still reads back after deletion ({after.status_code})"
    )


def test_inference_compat_routes_reject_an_unknown_endpoint(funded) -> None:
    eid = f"sep-{uuid.uuid4().hex[:12]}"
    for path in (
        f"/api/v2/inference/endpoints/{eid}",
        f"/api/v2/inference/endpoints/{eid}/health",
        f"/api/v2/inference/endpoints/{eid}/usage",
    ):
        r = client.get(path, headers=funded)
        assert r.status_code < 500, f"{path} faulted: {r.status_code} {r.text[:300]}"
        assert r.status_code in (403, 404), f"{path} returned {r.status_code}"


# ── GitHub source resolution ──────────────────────────────────────────────


def test_github_resolve_reaches_its_handler(funded) -> None:
    r = client.post(
        "/api/v2/serverless/github/resolve",
        json={"source_ref": "octocat/Hello-World", "source_ref_branch": "main"},
        headers=funded,
    )
    assert r.status_code != 422, f"the body did not satisfy the model: {r.text[:300]}"
    assert r.status_code < 500, f"github resolve faulted: {r.status_code} {r.text[:300]}"
