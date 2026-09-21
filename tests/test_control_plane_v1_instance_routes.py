"""The v1 instance routes nothing had ever called.

`scripts/measure_route_execution.py` found five `control_plane_v1` handlers
never entered by any test, all scored "covered" because the
`/api/v1/instances/` and `/api/v1/hosts/` prefixes appear elsewhere in
`tests/`:

    GET  /api/v1/instances/{job_id}
    GET  /api/v1/instances/{job_id}/active-lease
    GET  /api/v1/instances/{job_id}/events
    POST /api/v1/control-plane/commands/{command_id}/retry
    POST /api/v1/hosts/{host_id}/undrain

The first is the one that mattered most to write: it strips seven credential
fields out of the job record before returning it, including `registry_password`
and `ssh_private_key`. That redaction had no test. A field added to a job
payload later — or one of these renamed — would leak silently, and the response
would look entirely normal.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app

client = TestClient(app)

#: Exactly the set `api_v1_instance` removes. Duplicated here on purpose: if
#: the route's list shrinks, this must fail rather than follow it.
REDACTED_FIELDS = (
    "init_script",
    "environment",
    "env",
    "registry_password",
    "ssh_private_key",
    "nfs_server",
    "nfs_path",
)

SECRET = "sk-live-this-must-not-be-returned"


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def job_with_secrets():
    """A real job carrying every field the route is supposed to strip."""
    from scheduler import _set_job_fields, get_job, submit_job

    job = submit_job(
        name=f"cpv1-probe-{uuid.uuid4().hex[:8]}",
        vram_needed_gb=2,
        image="xcelsior/probe:latest",
        owner="cpv1-probe-owner",
    )
    job_id = job.get("job_id") or job.get("id")
    assert job_id, f"submit_job returned no id: {job}"

    # Plant a value in every field the route claims to strip, so the redaction
    # is tested against something that would actually be visible if it broke.
    _set_job_fields(job_id, **{field: SECRET for field in REDACTED_FIELDS})

    stored = get_job(job_id)
    assert stored, "the probe job vanished before the test ran"
    planted = [f for f in REDACTED_FIELDS if f in stored]
    assert planted, (
        "none of the credential fields could be planted, so a redaction test "
        "against this job would pass without redacting anything"
    )

    yield job_id, stored

    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


# ── Instance detail ───────────────────────────────────────────────────────


def test_instance_detail_is_returned(job_with_secrets) -> None:
    job_id, _ = job_with_secrets
    r = client.get(f"/api/v1/instances/{job_id}", headers=_admin_headers())
    assert r.status_code == 200, r.text[:300]
    assert r.json()["instance"], "no instance in the response"


def test_instance_detail_redacts_every_credential_field(job_with_secrets) -> None:
    """The redaction that had no test.

    Asserted on the serialised response rather than field by field, so a
    credential surviving inside a nested structure fails too.
    """
    job_id, stored = job_with_secrets
    r = client.get(f"/api/v1/instances/{job_id}", headers=_admin_headers())
    assert r.status_code == 200, r.text[:300]

    present = [f for f in REDACTED_FIELDS if f in (stored or {})]
    if present:
        instance = r.json()["instance"]
        leaked = [f for f in present if f in instance]
        assert not leaked, f"the response still carries {leaked}"
        assert SECRET not in r.text, (
            f"a redacted value survived somewhere in the body: {r.text[:400]}"
        )


def test_an_unknown_instance_is_not_found() -> None:
    r = client.get(f"/api/v1/instances/job-{uuid.uuid4().hex}", headers=_admin_headers())
    assert r.status_code == 404, r.text[:200]




# ── Lease and events ──────────────────────────────────────────────────────


def test_active_lease_answers_for_a_job_with_no_lease(job_with_secrets) -> None:
    job_id, _ = job_with_secrets
    r = client.get(f"/api/v1/instances/{job_id}/active-lease", headers=_admin_headers())
    assert r.status_code < 500, f"active-lease faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (200, 404), r.text[:200]


def test_active_lease_never_returns_credentials(job_with_secrets) -> None:
    """The docstring promises "a tenant-safe host alias and no credentials"."""
    job_id, _ = job_with_secrets
    r = client.get(f"/api/v1/instances/{job_id}/active-lease", headers=_admin_headers())
    if r.status_code == 200:
        assert SECRET not in r.text
        for field in ("ssh_private_key", "registry_password"):
            assert field not in r.text, f"{field} appeared in the lease view"


def test_events_page_is_returned(job_with_secrets) -> None:
    job_id, _ = job_with_secrets
    r = client.get(f"/api/v1/instances/{job_id}/events", headers=_admin_headers())
    assert r.status_code < 500, f"events faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (200, 404), r.text[:200]


def test_events_rejects_a_nonsense_cursor(job_with_secrets) -> None:
    """An opaque cursor is caller-supplied; garbage must not fault."""
    job_id, _ = job_with_secrets
    r = client.get(
        f"/api/v1/instances/{job_id}/events",
        params={"cursor": "not-a-real-cursor"},
        headers=_admin_headers(),
    )
    assert r.status_code < 500, f"a bad cursor faulted: {r.status_code} {r.text[:300]}"


# ── Agent command retry ───────────────────────────────────────────────────


def test_retrying_an_unknown_command_is_not_found() -> None:
    r = client.post(
        f"/api/v1/control-plane/commands/{uuid.uuid4()}/retry",
        json={},
        headers=_admin_headers(),
    )
    assert r.status_code < 500, f"retry faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code in (400, 403, 404, 409), r.text[:200]


def test_a_malformed_command_id_is_rejected_by_the_type() -> None:
    """`command_id: UUID` — the annotation keeps this out of the database.

    Worth pinning: the sibling routes that take `str` produced 500s for exactly
    this input until the `psycopg.DataError` floor was added.
    """
    r = client.post(
        "/api/v1/control-plane/commands/not-a-uuid/retry", json={}, headers=_admin_headers()
    )
    assert r.status_code == 422, r.text[:300]




# ── Undrain ───────────────────────────────────────────────────────────────


def test_undraining_an_unknown_host_is_not_found() -> None:
    r = client.post(
        f"/api/v1/hosts/no-such-host-{uuid.uuid4().hex[:8]}/undrain",
        json={},
        headers=_admin_headers(),
    )
    assert r.status_code < 500, f"undrain faulted: {r.status_code} {r.text[:300]}"
    assert r.status_code == 404, r.text[:200]


# Authentication is deliberately not asserted here. The test environment sets
# `AUTH_REQUIRED = False` (see tests/conftest.py), so `_require_auth` hands back
# a synthetic admin and an anonymous request is indistinguishable from an
# authorised one. A test written against that would be asserting the fixture,
# not the route — `tests/test_oauth_operator_scope_refusal.py` covers the scope
# refusals with real principals.
