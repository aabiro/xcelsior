"""One host's verification status: authenticated, explicit, and one shape.

`GET /api/verify/{host_id}/status` had three faults that compounded.

**It was anonymous and returned more than the listing beside it.** The body was
`v.__dict__` — the whole record: `deverify_reason`, `gpu_fingerprint`,
`failure_count`, and `checks`, which carry each check's expected-versus-actual
values from the hardware report. `/api/verified-hosts` was gated for exposing a
*curated subset* of exactly that, so leaving this open made that gate
bypassable one host at a time. Host ids are free: `/marketplace/search` and
`/compute-scores` are both public.

**`__dict__` is opt-out exposure.** A field added to `HostVerification` later
would be published without anyone deciding to.

**The two branches returned different shapes.** No record gave `status` at the
top level; an existing record gave `verification` and no `status`. The host
detail page reads `verification.status` for its badge, so the badge read
`undefined` for precisely the hosts that were verified — and
`fetchVerificationStatus` typed the response as `{ok, host_id, status}`, which
describes only the empty branch, so TypeScript enforced the broken shape rather
than catching it.
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

import routes._deps as deps
from api import app

client = TestClient(app)

#: Fields on HostVerification that must never appear in this response.
SENSITIVE = ("deverify_reason", "gpu_fingerprint", "failure_count", "checks")


@pytest.fixture
def signed_in():
    email = f"verifstatus-{uuid.uuid4().hex[:8]}@xcelsior.ca"
    r = client.post(
        "/api/auth/register",
        json={"email": email, "password": "StrongPass123!", "name": "Verif Probe"},
    )
    token = r.json().get("access_token")
    assert token, f"could not register: {r.status_code} {r.text[:200]}"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def verified_host():
    """A host with a real verification record, cleaned up afterwards."""
    from verification import get_verification_engine

    host_id = f"verif-probe-{uuid.uuid4().hex[:10]}"
    engine = get_verification_engine()
    engine.run_verification(
        host_id,
        {
            "gpu_model": "RTX 4090",
            "claimed_gpu_model": "RTX 4090",
            "total_vram_gb": 24,
            "claimed_vram_gb": 24,
            "cuda_version": "12.4",
            "compute_capability": 8.9,
            "pcie_bandwidth_gbps": 25.0,
            "gpu_temp_celsius": 55,
            "packet_loss_pct": 0.1,
            "jitter_ms": 5.0,
            "throughput_mbps": 900.0,
        },
    )
    yield host_id
    try:
        from db import _get_pg_pool

        with _get_pg_pool().connection() as conn:
            conn.execute("DELETE FROM host_verifications WHERE host_id = %s", (host_id,))
    except Exception:
        pass


def test_an_anonymous_caller_is_refused(monkeypatch, verified_host) -> None:
    """`AUTH_REQUIRED` is pinned on — the test env disables it, so without this
    the endpoint answers 200 to everyone here whether or not a guard exists."""
    monkeypatch.setattr(deps, "AUTH_REQUIRED", True)
    r = client.get(f"/api/verify/{verified_host}/status")
    assert r.status_code in (401, 403), (
        f"anonymous read returned {r.status_code}; /api/verified-hosts is gated "
        "and this walks around it one host at a time"
    )
    for field in SENSITIVE:
        assert field not in r.text


def test_status_is_present_whether_or_not_a_record_exists(signed_in, verified_host) -> None:
    """The badge bug: `status` used to be missing on exactly the verified hosts."""
    known = client.get(f"/api/verify/{verified_host}/status", headers=signed_in)
    assert known.status_code == 200, known.text[:300]
    assert "status" in known.json(), (
        f"a host WITH a record returned no top-level status: {known.json()}"
    )
    assert known.json()["status"], "status is present but empty"

    unknown = client.get(
        f"/api/verify/no-such-host-{uuid.uuid4().hex[:8]}/status", headers=signed_in
    )
    assert unknown.status_code == 200, unknown.text[:300]
    assert unknown.json()["status"] == "unverified"


def test_the_two_branches_agree_on_shape(signed_in, verified_host) -> None:
    """A caller should not have to know which branch it got."""
    known = client.get(f"/api/verify/{verified_host}/status", headers=signed_in).json()
    unknown = client.get(
        f"/api/verify/no-such-host-{uuid.uuid4().hex[:8]}/status", headers=signed_in
    ).json()
    for key in ("ok", "host_id", "status"):
        assert key in known and key in unknown, f"{key} missing from one branch"
    assert "verification" not in known, (
        "the nested `verification` object is back; the frontend reads a top-level "
        "`status` and would go blank again"
    )


def test_the_record_internals_are_not_published(signed_in, verified_host) -> None:
    """`__dict__` published whatever the dataclass happened to hold."""
    r = client.get(f"/api/verify/{verified_host}/status", headers=signed_in)
    assert r.status_code == 200, r.text[:300]
    for field in SENSITIVE:
        assert field not in r.text, (
            f"{field} is in the response; this route is back to publishing the "
            "whole record, and a field added to HostVerification later would go "
            "out with it"
        )


def test_the_response_is_an_explicit_shape_not_a_dump() -> None:
    """Structural: opt-in, so adding a dataclass field cannot publish it."""
    import ast
    import inspect
    import textwrap

    from routes.verification import api_verification_status

    # Parse and drop the docstring before looking. The first version scanned
    # the raw source and failed on this handler's *own* explanation of the bug
    # — prose describing a defect reads exactly like the defect to a text scan,
    # which is a trap this repository has fallen into before.
    tree = ast.parse(textwrap.dedent(inspect.getsource(api_verification_status)))
    fn = tree.body[0]
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]

    dumps = [
        node
        for stmt in body
        for node in ast.walk(stmt)
        if isinstance(node, ast.Attribute) and node.attr == "__dict__"
    ]
    assert not dumps, (
        "the handler dumps the record again; every future field on "
        "HostVerification would be published without anyone deciding to"
    )

    calls = {
        getattr(n.func, "id", None) or getattr(n.func, "attr", None)
        for stmt in body
        for n in ast.walk(stmt)
        if isinstance(n, ast.Call)
    }
    assert "_require_auth" in calls, "the handler no longer authenticates"
