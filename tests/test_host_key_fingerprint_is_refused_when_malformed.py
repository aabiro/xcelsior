"""A1's refusal test: the fingerprint is served to users, so it is validated.

From `docs/host-key-fingerprint-plan.md` A1: *"a worker reporting a malformed
fingerprint (`"yes"`, `"SHA256:"`, 4KB of junk) is rejected at the API boundary
and stored as null. The column is served to users and must never carry
attacker-controlled text."*

The three named inputs are asserted by name, because each fails differently:

* `"yes"` — short, harmless-looking, and would render as a value a user compares
  against and never matches, which teaches them to ignore the warning.
* `"SHA256:"` — carries the authoritative-looking prefix and verifies nothing.
* 4 KB of junk — attacker-controlled text on a page, and a row that is mostly
  payload.

**Empty is not malformed.** A non-interactive launch, an older worker, or a
proxy-terminated host where the container legitimately holds no keys all report
nothing, and that is a normal state rather than an error.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from host_key_fingerprint import (  # noqa: E402
    MAX_LENGTH,
    fingerprint_is_valid,
    parse_host_key_fingerprint,
)

#: A real one, from `ssh-keygen -lf` on a live container during the A0 sweep.
REAL = "SHA256:AyDM+WgYhgfT0EIxyrzqQR5ZUL1uwxiUJk5Ck37FWHU"


def test_a_real_fingerprint_survives():
    """Calibration. If everything were rejected the refusals prove nothing."""
    assert parse_host_key_fingerprint(REAL) == REAL
    assert fingerprint_is_valid(REAL)


@pytest.mark.parametrize(
    "value, why",
    [
        ("yes", "the plan's first named input — short and plausible"),
        ("SHA256:", "the plan's second — the prefix alone, verifying nothing"),
        ("x" * 4096, "the plan's third — 4KB of junk"),
        ("SHA256:" + "A" * 42, "one character short of a SHA-256 digest"),
        ("SHA256:" + "A" * 44, "one character long"),
        ("MD5:aa:bb:cc", "a different algorithm entirely"),
        ("SHA256:AyDM+WgYhgfT0EIxyrzqQR5ZUL1uwxiUJk5Ck37FWH=", "padded base64"),
        ("SHA256:AyDM WgYhgfT0EIxyrzqQR5ZUL1uwxiUJk5Ck37FWHU", "embedded space"),
        ("<script>alert(1)</script>", "the reason this is validated at all"),
        (12345, "not a string"),
        (True, "a bool is not a fingerprint, and bool is an int subclass"),
        ({"SHA256": "x"}, "a structure"),
    ],
)
def test_malformed_input_becomes_none(value, why):
    assert parse_host_key_fingerprint(value) is None, why
    assert not fingerprint_is_valid(value)


def test_a_valid_fingerprint_buried_in_junk_is_refused():
    """The anchoring, which is the difference between validating and scraping.

    An unanchored search would pull the real value out of the middle of a hostile
    payload and store its neighbour — accepting the attacker's framing while
    looking like it validated something.
    """
    assert parse_host_key_fingerprint(f"junk {REAL} more junk") is None
    assert parse_host_key_fingerprint(f"{REAL}\n<script>") is None


def test_absent_is_unknown_not_invalid():
    """An agent reporting nothing is a normal state, not an error."""
    for absent in (None, "", "   "):
        assert parse_host_key_fingerprint(absent) is None


def test_surrounding_whitespace_is_tolerated():
    """`ssh-keygen` output trimmed by a caller is still the same fingerprint."""
    assert parse_host_key_fingerprint(f"  {REAL}\n") == REAL


def test_the_length_bound_precedes_the_matcher():
    """A pathological input is never handed to the regex at all."""
    assert MAX_LENGTH < 4096
    assert parse_host_key_fingerprint("SHA256:" + "A" * 5000) is None


# ── Through the route, which is where the plan says the boundary is ───


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import api as api_mod
    import routes.agent as agent_routes

    monkeypatch.setattr(agent_routes, "_require_agent_auth", lambda request: None)
    return TestClient(api_mod.app)


def _report(client, job_id: str, fingerprint):
    body = {"ok": True, "sshd_present": True, "sshd_started": True, "key_count": 1}
    if fingerprint is not None:
        body["host_key_fingerprint"] = fingerprint
    return client.post(f"/agent/ssh-status/{job_id}", json=body)


def test_the_route_stores_a_valid_fingerprint(client):
    import uuid

    from control_plane.db import control_plane_transaction as tx

    job_id = f"hk-{uuid.uuid4().hex[:10]}"
    with tx() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload) "
            "VALUES (%s, 'running', 0, EXTRACT(EPOCH FROM NOW()), '{}'::jsonb)",
            (job_id,),
        )
    try:
        assert _report(client, job_id, REAL).status_code == 200
        with tx() as conn:
            stored = conn.execute(
                "SELECT payload->'ssh_status'->>'host_key_fingerprint' "
                "  FROM jobs WHERE job_id = %s",
                (job_id,),
            ).fetchone()[0]
        assert stored == REAL
    finally:
        with tx() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


@pytest.mark.parametrize("bad", ["yes", "SHA256:", "x" * 4096])
def test_the_route_stores_null_for_the_inputs_the_plan_names(client, bad):
    """Rejected at the boundary and stored as null — not stored and escaped later."""
    import uuid

    from control_plane.db import control_plane_transaction as tx

    job_id = f"hk-{uuid.uuid4().hex[:10]}"
    with tx() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload) "
            "VALUES (%s, 'running', 0, EXTRACT(EPOCH FROM NOW()), '{}'::jsonb)",
            (job_id,),
        )
    try:
        response = _report(client, job_id, bad)
        assert response.status_code == 200, (
            "a malformed fingerprint must not fail the callback — SSH status is "
            "fire-and-forget and rejecting the whole report would lose the "
            "sshd_present/sshd_started signal the dashboard needs"
        )
        with tx() as conn:
            stored = conn.execute(
                "SELECT payload->'ssh_status'->>'host_key_fingerprint' "
                "  FROM jobs WHERE job_id = %s",
                (job_id,),
            ).fetchone()[0]
        assert stored is None, f"{bad[:40]!r} reached storage"
    finally:
        with tx() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


# ── A2: the fingerprint cannot outlive its container ──────────────────


def _make_job(conn, job_id: str, host_id: str, fingerprint=None, status="running"):
    """The payload is the source of truth; the column is its projection.

    `upsert_job` rebuilds the column from the payload on every write, so a
    fixture that set only the column would be nulled by the first unrelated
    update — which is how the "same host keeps it" test caught the defect.
    """
    import json as _j

    payload = {"job_id": job_id, "host_id": host_id, "status": status}
    if fingerprint is not None:
        payload["host_key_fingerprint"] = fingerprint
    conn.execute(
        "INSERT INTO jobs (job_id, status, priority, submitted_at, host_id, "
        "                  host_key_fingerprint, payload) "
        "VALUES (%s, %s, 0, EXTRACT(EPOCH FROM NOW()), %s, %s, %s::jsonb)",
        (job_id, status, host_id, fingerprint, _j.dumps(payload)),
    )


def _stored_fingerprint(conn, job_id: str):
    return conn.execute(
        "SELECT host_key_fingerprint FROM jobs WHERE job_id = %s", (job_id,)
    ).fetchone()[0]


def test_an_automatic_failover_clears_the_fingerprint():
    """**The case `_clear_job_output` skips, and the reason it was the wrong hook.**

    That hook is gated on `user_initiated`; automatic failover does not pass it,
    deliberately, because a failover that erased its own logs would destroy the
    evidence for the retry it just performed. Correct for logs, fatal for a
    fingerprint: failover is the primary way a job changes host, so a fingerprint
    cleared *there* would survive onto the new host and verify against the wrong
    one.

    Testing only the user-initiated door would test the path that was never
    broken.
    """
    import uuid

    from control_plane.db import control_plane_transaction as tx
    from scheduler import update_job_status

    job_id = f"hk-{uuid.uuid4().hex[:10]}"
    with tx() as conn:
        # `restarting` is what the CRIU migration and failover paths pass
        # through; `running -> running` is not a legal transition and the
        # update would have been refused before reaching the clear.
        _make_job(conn, job_id, "host-old", REAL, status="restarting")
        assert _stored_fingerprint(conn, job_id) == REAL, "fixture did not store it"

    try:
        # No `user_initiated` anywhere in sight — this is the failover path.
        update_job_status(job_id, "running", host_id="host-new")
        with tx() as conn:
            assert _stored_fingerprint(conn, job_id) is None, (
                "the fingerprint survived a host change and would now verify "
                "against the wrong host"
            )
    finally:
        with tx() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def test_a_status_change_on_the_same_host_keeps_the_fingerprint():
    """The other direction, or the clear is indiscriminate.

    A job that reports running twice on the same host has the same container and
    the same key. Nulling on every status write would make the value useless.
    """
    import uuid

    from control_plane.db import control_plane_transaction as tx
    from scheduler import update_job_status

    job_id = f"hk-{uuid.uuid4().hex[:10]}"
    with tx() as conn:
        _make_job(conn, job_id, "host-same", REAL)
    try:
        update_job_status(job_id, "completed", host_id="host-same")
        with tx() as conn:
            assert _stored_fingerprint(conn, job_id) == REAL, (
                "the fingerprint was cleared without the host changing"
            )
    finally:
        with tx() as conn:
            conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def test_the_column_and_host_id_move_in_one_statement():
    """Structural, not procedural — no hook anyone has to remember to call.

    `DatabaseOps.upsert_job` writes both columns in a single INSERT … ON
    CONFLICT, so there is no window in which the row carries a new host and an
    old fingerprint.
    """
    import inspect

    from db import DatabaseOps

    source = inspect.getsource(DatabaseOps.upsert_job)
    assert "host_key_fingerprint = EXCLUDED.host_key_fingerprint" in source
    assert "host_id = EXCLUDED.host_id" in source


# ── A3: the fingerprint travels with its port, or not at all ──────────


def _enriched(job: dict) -> dict:
    """Run a job dict through the shared enrichment the detail route uses."""
    from routes.instances import _enrich_instance

    j = dict(job)
    _enrich_instance(j, {})
    return j


def test_a_fingerprint_is_served_with_its_port():
    j = _enriched(
        {
            "job_id": "j1",
            "status": "running",
            "interactive": True,
            "public_ssh_port": 10022,
            "host_key_fingerprint": REAL,
        }
    )
    assert j["ssh_port"] == 10022
    assert j["host_key_fingerprint"] == REAL


def test_a_fingerprint_without_a_port_is_withheld():
    """**A fingerprint beside the wrong port reads to the user as an attack.**

    They run `ssh-keyscan -p PORT`, get a different key, and correctly conclude
    they are being intercepted — a security incident manufactured out of a
    plumbing bug, whose correct response is indistinguishable from the response
    to a real one. No port, no fingerprint.
    """
    j = _enriched(
        {
            "job_id": "j2",
            "status": "completed",
            "interactive": False,
            "host_key_fingerprint": REAL,
        }
    )
    assert not j.get("ssh_port")
    assert j["host_key_fingerprint"] is None


def test_the_served_value_is_revalidated_on_the_way_out():
    """A1 guards the write path; A3 guards what is served.

    Two boundaries for a value whose entire purpose is to be trusted. The
    alternative is assuming nothing ever wrote that payload key by another
    route, and not assuming it costs one anchored regex.
    """
    j = _enriched(
        {
            "job_id": "j3",
            "status": "running",
            "interactive": True,
            "public_ssh_port": 10022,
            "host_key_fingerprint": "<script>alert(1)</script>",
        }
    )
    assert j["ssh_port"] == 10022
    assert j["host_key_fingerprint"] is None, "unvalidated text reached a served field"


def test_an_absent_fingerprint_stays_absent():
    j = _enriched(
        {"job_id": "j4", "status": "running", "interactive": True, "public_ssh_port": 10022}
    )
    assert j["host_key_fingerprint"] is None
