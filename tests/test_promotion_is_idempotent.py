"""Gate P3, clause 1: "a repeated call produces one volume, not two".

`docs/artifact-promotion-plan.md` §4 states how it is proven: *"call twice,
assert one `volume_promotions` row and one volume, second reports `replayed`"*.
That is what this does — against the real route and a real database, because the
mechanism is a unique constraint and `ON CONFLICT DO NOTHING`, neither of which
exists in a fake.

## Why the caller does not have to supply a key

The default idempotency key is the manifest digest (§3.2), so a retry after a
timeout converges without the caller having invented anything. That matters more
than it sounds: the moment a retry is *correct* is exactly the moment the caller
has no idea whether the first attempt landed, and a design that requires them to
have planned ahead will be retried wrongly.

The digest also makes the converse true — a job that has produced new artifacts
since hashes differently, so it is a *new* promotion rather than a replay
against a stale file list.

## Why `replayed` is reported rather than the call silently succeeding

`charge_saved_card` surfaces the same distinction for the same reason: a caller
that retried needs to know whether its first attempt did the work. Reporting
plain success twice is indistinguishable from having copied twice.
"""

from __future__ import annotations

import os
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _has = _c.execute("SELECT to_regclass('volume_promotions')").fetchone()[0] is not None
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 102")


@pytest.fixture
def promotable(monkeypatch):
    """A job with artifacts, a volume that accepts writes, and a live client."""
    from fastapi.testclient import TestClient

    import api as api_mod
    import routes.volumes as vol
    from routes import _deps

    tag = uuid.uuid4().hex[:10]
    tenant = f"tenant-{tag}"
    job_id = f"job-{tag}"
    volume_id = f"vol-{tag}"

    with pg_transaction() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, status, priority, submitted_at, payload, owner_id) "
            "VALUES (%s, 'completed', 0, extract(epoch from now()), '{}'::jsonb, %s)",
            (job_id, tenant),
        )
        for i in range(2):
            aid = str(uuid.uuid4())
            conn.execute(
                """INSERT INTO storage.artifacts
                     (artifact_id, tenant_id, job_id, artifact_type, logical_name, state,
                      primary_provider, primary_bucket, object_key, content_type,
                      retention_class, size_bytes, sha256, legal_hold)
                   VALUES (%s, %s, %s, 'checkpoint', %s, 'available', 'local', 'test',
                           %s, 'application/octet-stream', 'standard', 100, %s, false)""",
                (aid, tenant, job_id, f"shard-{i}.pt", f"k/{aid}", f"{i}" * 64),
            )

    principal = {
        "email": "demo@xcelsior.ca", "user_id": f"user-{tag}", "role": "user",
        "auth_type": "oauth_access_token", "session_type": "browser",
        "client_id": "xcelsior-web", "scopes": ["volumes:write", "volumes:read"],
        "customer_id": tenant,
    }
    monkeypatch.setattr(_deps, "_get_current_user", lambda request: dict(principal))
    monkeypatch.setattr(vol, "_get_current_user", lambda request: dict(principal))
    monkeypatch.setattr(vol, "_effective_billing_customer_id", lambda user: tenant)
    monkeypatch.setattr(vol, "_require_volume_write", lambda user, v: None)
    monkeypatch.setattr(vol, "_require_volume_read", lambda user, v: None)

    class _VE:
        def get_volume(self, vid):
            return {"volume_id": vid, "owner_id": tenant, "host_id": ""}

    monkeypatch.setattr(vol, "get_volume_engine", lambda: _VE())

    yield {
        "client": TestClient(api_mod.app),
        "tenant": tenant, "job_id": job_id, "volume_id": volume_id,
    }

    with pg_transaction() as conn:
        conn.execute("DELETE FROM volume_promotions WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM storage.artifacts WHERE tenant_id = %s", (tenant,))
        conn.execute("DELETE FROM jobs WHERE job_id = %s", (job_id,))


def _rows(tenant: str) -> int:
    with pg_transaction() as conn:
        return int(
            conn.execute(
                "SELECT count(*) FROM volume_promotions WHERE tenant_id = %s", (tenant,)
            ).fetchone()[0]
        )


def _promote(p, **body):
    return p["client"].post(
        f"/api/v2/volumes/{p['volume_id']}/promotions",
        json={"job_id": p["job_id"], **body},
    )


def test_calling_twice_produces_one_promotion(promotable):
    """The clause, proven the way §4 says to prove it."""
    first = _promote(promotable)
    assert first.status_code == 200, first.text
    second = _promote(promotable)
    assert second.status_code == 200, second.text

    assert _rows(promotable["tenant"]) == 1, (
        "a repeated call created a second promotion — the retry a caller makes "
        "after a timeout would copy the same bytes twice"
    )
    assert first.json()["promotion_id"] == second.json()["promotion_id"]


def test_the_second_call_says_it_was_a_replay(promotable):
    """Plain success twice is indistinguishable from having copied twice."""
    assert _promote(promotable).json()["replayed"] is False
    assert _promote(promotable).json()["replayed"] is True


def test_no_key_is_needed_for_a_retry_to_converge(promotable):
    """§3.2: the digest is the default key.

    The moment a retry is correct is the moment the caller does not know whether
    the first attempt landed — requiring them to have planned ahead means the
    retry happens wrongly or not at all.
    """
    a = _promote(promotable)
    b = _promote(promotable)
    assert a.json()["promotion_id"] == b.json()["promotion_id"]
    assert _rows(promotable["tenant"]) == 1


def test_a_different_explicit_key_is_a_different_promotion(promotable):
    """Calibration: if everything collapsed to one row, the tests above pass
    while the idempotency does nothing."""
    _promote(promotable)
    _promote(promotable, idempotency_key="a-deliberately-separate-run")
    assert _rows(promotable["tenant"]) == 2


def test_a_job_with_no_artifacts_is_refused_rather_than_promoted_empty(promotable):
    """An empty promotion would reach `succeeded` having copied nothing, which
    reads to a model as "your weights are saved"."""
    with pg_transaction() as conn:
        conn.execute(
            "DELETE FROM storage.artifacts WHERE tenant_id = %s", (promotable["tenant"],)
        )
    r = _promote(promotable)
    assert r.status_code == 409
    assert _rows(promotable["tenant"]) == 0


def test_the_hold_is_taken_exactly_once_across_a_replay(promotable):
    """A replay must not re-hold, and must not release what the first call took.

    Both directions matter: a second hold is harmless but a second *release*
    later would free artifacts the running copy still needs.
    """
    _promote(promotable)
    with pg_transaction() as conn:
        held_after_first = int(conn.execute(
            "SELECT count(*) FROM storage.artifacts "
            " WHERE tenant_id = %s AND legal_hold = true", (promotable["tenant"],)
        ).fetchone()[0])
    _promote(promotable)
    with pg_transaction() as conn:
        held_after_replay = int(conn.execute(
            "SELECT count(*) FROM storage.artifacts "
            " WHERE tenant_id = %s AND legal_hold = true", (promotable["tenant"],)
        ).fetchone()[0])

    assert held_after_first == 2, "the first call did not hold the manifest"
    assert held_after_replay == 2, "a replay changed the hold"
