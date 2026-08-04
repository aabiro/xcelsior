"""The fleet leak, asserted over HTTP against the real route.

`tests/test_host_visibility_scope.py` covers `visible_hosts` directly, which is
the right place for the ownership rules. It cannot, however, demonstrate the
*defect*: without the fix that helper does not exist, so running those tests
against unfixed code produces an import error rather than a disclosure. An
import error is not evidence — it proves the symbol is new, not that the old
behaviour was wrong.

This file goes through `GET /hosts` and asserts on the rows returned, so it
fails against unfixed code for exactly the reason the fix exists: one provider
could see another provider's host.

Deliberately uses an ordinary browser-session credential rather than an OAuth
client. `_require_scope` is a no-op for interactive sessions, so this also pins
the case the unit tests call out as the easy mistake — if visibility keyed on
"caller has no scopes", every logged-in user would get the whole fleet, and this
test would still pass while the platform leaked.
"""

from __future__ import annotations

import json
import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from api import app  # noqa: E402

client = TestClient(app)


def _register(label: str) -> tuple[str, dict]:
    """Return (user_id, auth headers) for a fresh non-admin account."""
    email = f"fleet-{label}-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": f"Fleet {label}"},
    )
    assert reg.status_code in (200, 201), reg.text
    user_id = reg.json()["user"]["user_id"]
    login = client.post(
        "/api/auth/login", json={"email": email, "password": "StrongPass123!"}
    )
    assert login.status_code == 200, login.text
    return user_id, {"Authorization": f"Bearer {login.json()['access_token']}"}


def _insert_host(owner_user_id: str) -> str:
    """A host row complete enough for every consumer, not just this file.

    `ip` is not optional. `POST /hosts/check` reaches `host["ip"]` directly, so
    a row without it raises `KeyError` in an unrelated test that happens to run
    later — which is what a partially-populated fixture row costs when the
    database is shared across the suite.
    """
    host_id = f"h-fleet-{uuid.uuid4().hex[:10]}"
    from db import _get_pg_pool

    payload = {
        "host_id": host_id,
        "owner": owner_user_id,
        "ip": "10.255.0.1",
        "gpu_model": "RTX 4090",
        "num_gpus": 1,
        "total_vram_gb": 24,
        "vram_total_gb": 24.0,
    }
    with _get_pg_pool().connection() as conn:
        conn.execute(
            "INSERT INTO hosts (host_id, status, registered_at, payload, admission_state) "
            "VALUES (%s, 'active', %s, %s, 'admitted')",
            (host_id, time.time(), json.dumps(payload)),
        )
        conn.commit()
    return host_id


@pytest.fixture(scope="module")
def two_providers():
    """Two providers with one host each, removed again at teardown.

    The rows are deleted rather than left behind: the suite shares one database,
    and a stray `active`/`admitted` host is visible to every later test that
    lists or checks hosts.
    """
    a_id, a_headers = _register("a")
    b_id, b_headers = _register("b")
    providers = {
        "a": {"user_id": a_id, "headers": a_headers, "host": _insert_host(a_id)},
        "b": {"user_id": b_id, "headers": b_headers, "host": _insert_host(b_id)},
    }

    yield providers

    from db import _get_pg_pool

    with _get_pg_pool().connection() as conn:
        conn.execute(
            "DELETE FROM hosts WHERE host_id = ANY(%s)",
            ([providers["a"]["host"], providers["b"]["host"]],),
        )
        conn.commit()


def test_a_provider_does_not_see_another_providers_host(two_providers):
    """The disclosure, over HTTP.

    Against unfixed code `GET /hosts` returns every row, so provider A's
    response contains provider B's host and this fails — which is the point.
    """
    a, b = two_providers["a"], two_providers["b"]
    r = client.get("/hosts", headers=a["headers"])
    assert r.status_code == 200, r.text
    returned = {h.get("host_id") for h in r.json().get("hosts", [])}

    assert b["host"] not in returned, (
        f"provider A's listing included provider B's host {b['host']} — the "
        f"fleet is readable by anyone holding hosts:read"
    )


def test_a_provider_still_sees_their_own_host(two_providers):
    """The calibration control.

    Returning an empty list would satisfy the assertion above and look like a
    working filter while breaking every provider's dashboard.
    """
    a = two_providers["a"]
    r = client.get("/hosts", headers=a["headers"])
    assert r.status_code == 200, r.text
    returned = {h.get("host_id") for h in r.json().get("hosts", [])}

    assert a["host"] in returned, (
        f"provider A cannot see their own host {a['host']}; the filter is too "
        f"narrow and provider onboarding is broken"
    )


def test_a_single_host_lookup_does_not_leak_by_id(two_providers):
    """Filtering the list alone leaves `GET /host/{id}` open.

    404 rather than 403: a 403 confirms the host exists, which is most of what
    enumeration wanted.
    """
    a, b = two_providers["a"], two_providers["b"]
    r = client.get(f"/host/{b['host']}", headers=a["headers"])
    assert r.status_code == 404, (
        f"provider A read provider B's host directly by id and got "
        f"{r.status_code}: {r.text[:200]}"
    )


def test_a_single_host_lookup_still_works_for_the_owner(two_providers):
    """Control for the route above."""
    a = two_providers["a"]
    r = client.get(f"/host/{a['host']}", headers=a["headers"])
    assert r.status_code == 200, (
        f"provider A cannot read their own host by id: {r.text[:200]}"
    )
