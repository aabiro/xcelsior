"""Per-host agent tokens and field-wide bearer rotation (blueprint §19.2).

The property under test is the one that matters operationally: a
credential lifted from one GPU host must not grant authority over the
fleet, and rotating the whole field must never lock a host out.

Every assertion here runs against real PostgreSQL and, where the surface
is HTTP, through the real FastAPI routes — including the auth gate in
``routes/agent.py`` that decides which credential a worker may present.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")

try:
    from db import _get_pg_pool

    _pool = _get_pg_pool()
    with _pool.connection() as _c:
        _c.execute("SELECT 1").fetchone()
        _has_065 = _c.execute("SELECT to_regclass('host_agent_tokens')").fetchone()[0] is not None
except Exception as _exc:  # pragma: no cover - skip path
    pytestmark = pytest.mark.skip(f"no pg pool available: {_exc}")
    _pool = None
    _has_065 = False
else:
    if not _has_065:  # pragma: no cover - skip path
        pytestmark = pytest.mark.skip("test database not migrated to >= 065")

from fastapi.testclient import TestClient  # noqa: E402
from psycopg.errors import UniqueViolation  # noqa: E402

from api import app  # noqa: E402
from control_plane import agent_tokens  # noqa: E402

client = TestClient(app)


def _shared_bearer() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def hosts():
    """Two admitted hosts, torn down with all their credentials."""
    marker = uuid.uuid4().hex[:8]
    ids = [f"tok-{marker}-a", f"tok-{marker}-b"]
    with _pool.connection() as conn:
        for host_id in ids:
            conn.execute(
                "INSERT INTO hosts (host_id, status, registered_at, payload) "
                "VALUES (%s, 'active', %s, %s)",
                (
                    host_id,
                    time.time(),
                    json.dumps(
                        {
                            "host_id": host_id,
                            "gpu_model": "RTX-TOK",
                            "gpu_count": 1,
                            "admitted": True,
                            "last_seen": time.time(),
                        }
                    ),
                ),
            )
        conn.commit()
    try:
        yield ids
    finally:
        with _pool.connection() as conn:
            conn.execute("DELETE FROM host_agent_tokens WHERE host_id = ANY(%s)", (ids,))
            conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", (ids,))
            conn.commit()


def _issue(host_id: str, **kwargs):
    with _pool.connection() as conn:
        issued = agent_tokens.issue_token(conn, host_id, **kwargs)
        conn.commit()
    return issued


def _verify(secret: str, **kwargs):
    with _pool.connection() as conn:
        try:
            result = agent_tokens.verify_token(conn, secret, **kwargs)
            conn.commit()
            return result
        except Exception:
            conn.rollback()
            raise


# ── Storage and issuance ──────────────────────────────────────────────


def test_secret_is_never_stored_in_cleartext(hosts):
    host_id = hosts[0]
    issued = _issue(host_id)

    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT token_hash, token_prefix FROM host_agent_tokens WHERE token_id = %s",
            (issued.token_id,),
        ).fetchone()
        # No column anywhere in the row holds the plaintext.
        full = conn.execute(
            "SELECT * FROM host_agent_tokens WHERE token_id = %s",
            (issued.token_id,),
        ).fetchone()

    assert row[0] == agent_tokens.hash_token(issued.secret)
    assert row[0] != issued.secret
    assert issued.secret.startswith(agent_tokens.TOKEN_PREFIX)
    assert row[1] == issued.secret[:12]
    assert not any(
        isinstance(value, str) and issued.secret in value for value in full
    ), "plaintext secret found in a stored column"


def test_issuing_requires_a_registered_host():
    with _pool.connection() as conn:
        with pytest.raises(agent_tokens.UnknownHost):
            agent_tokens.issue_token(conn, f"ghost-{uuid.uuid4().hex[:8]}")
        conn.rollback()


def test_only_one_active_token_per_host_is_possible(hosts):
    """The partial unique index is the last line of defence, so prove it."""
    host_id = hosts[0]
    _issue(host_id)

    with _pool.connection() as conn:
        with pytest.raises(UniqueViolation):
            conn.execute(
                """
                INSERT INTO host_agent_tokens
                    (host_id, token_prefix, token_hash, status, expires_at)
                VALUES (%s, 'xat_dupdup', %s, 'active',
                        clock_timestamp() + interval '1 day')
                """,
                (host_id, "deadbeef" * 8),
            )
        conn.rollback()


def test_issuing_again_supersedes_rather_than_revokes(hosts):
    """A replacement must not strand a worker mid-poll."""
    host_id = hosts[0]
    first = _issue(host_id)
    second = _issue(host_id)

    with _pool.connection() as conn:
        rows = dict(
            conn.execute(
                "SELECT token_id::text, status FROM host_agent_tokens WHERE host_id = %s",
                (host_id,),
            ).fetchall()
        )
    assert rows[first.token_id] == "superseded"
    assert rows[second.token_id] == "active"

    # Both still authenticate while the grace window is open.
    assert _verify(first.secret).host_id == host_id
    assert _verify(second.secret).host_id == host_id
    assert _verify(first.secret).is_superseded is True


# ── Scoping: the reason per-host tokens exist ─────────────────────────


def test_a_stolen_token_cannot_be_replayed_against_another_host(hosts):
    host_a, host_b = hosts
    issued = _issue(host_a)

    assert _verify(issued.secret, required_host_id=host_a).host_id == host_a
    with pytest.raises(agent_tokens.TokenRejected) as excinfo:
        _verify(issued.secret, required_host_id=host_b)
    assert excinfo.value.reason == "host_mismatch"


def test_unknown_and_malformed_credentials_are_rejected(hosts):
    with pytest.raises(agent_tokens.TokenRejected) as unknown:
        _verify(agent_tokens.TOKEN_PREFIX + "not-a-real-token")
    assert unknown.value.reason == "unknown_token"

    with pytest.raises(agent_tokens.TokenRejected) as shaped:
        _verify("some-platform-bearer")
    assert shaped.value.reason == "not_a_host_token"


# ── Rotation and revocation ───────────────────────────────────────────


def test_rotation_keeps_the_old_token_valid_for_its_grace_window(hosts):
    host_id = hosts[0]
    original = _issue(host_id)

    with _pool.connection() as conn:
        rotated = agent_tokens.rotate_token(conn, original.secret, grace_minutes=60)
        conn.commit()

    assert rotated.host_id == host_id
    assert rotated.secret != original.secret
    # Overlap: a worker that has not yet adopted the new token still works.
    assert _verify(original.secret).is_superseded is True
    assert _verify(rotated.secret).is_superseded is False


def test_rotation_grace_expiry_closes_the_old_token(hosts):
    """After the grace window the superseded credential stops working."""
    host_id = hosts[0]
    original = _issue(host_id)

    with _pool.connection() as conn:
        agent_tokens.rotate_token(conn, original.secret, grace_minutes=0)
        # grace_minutes=0 stamps expiry at "now"; push it unambiguously
        # into the past so the assertion is not a clock race.
        conn.execute(
            """
            UPDATE host_agent_tokens
               SET expires_at = clock_timestamp() - interval '1 second'
             WHERE host_id = %s AND status = 'superseded'
            """,
            (host_id,),
        )
        conn.commit()

    with pytest.raises(agent_tokens.TokenRejected) as excinfo:
        _verify(original.secret)
    assert excinfo.value.reason == "expired"


def test_rotation_chain_never_leaves_a_host_without_a_credential(hosts):
    """Rotate repeatedly; a valid token exists at every step."""
    host_id = hosts[0]
    current = _issue(host_id)
    for _ in range(5):
        with _pool.connection() as conn:
            nxt = agent_tokens.rotate_token(conn, current.secret)
            conn.commit()
        assert _verify(nxt.secret, required_host_id=host_id).host_id == host_id
        current = nxt

    with _pool.connection() as conn:
        active = conn.execute(
            "SELECT count(*) FROM host_agent_tokens " "WHERE host_id = %s AND status = 'active'",
            (host_id,),
        ).fetchone()[0]
    assert active == 1


def test_revocation_is_immediate_and_has_no_grace(hosts):
    host_id = hosts[0]
    issued = _issue(host_id)
    assert _verify(issued.secret).host_id == host_id

    with _pool.connection() as conn:
        revoked = agent_tokens.revoke_token(conn, host_id=host_id, reason="compromise")
        conn.commit()
    assert revoked == 1

    with pytest.raises(agent_tokens.TokenRejected) as excinfo:
        _verify(issued.secret)
    assert excinfo.value.reason == "revoked"


def test_revoking_a_host_kills_superseded_tokens_too(hosts):
    """Compromise response must not leave the grace-window token alive."""
    host_id = hosts[0]
    first = _issue(host_id)
    second = _issue(host_id)

    with _pool.connection() as conn:
        assert agent_tokens.revoke_token(conn, host_id=host_id) == 2
        conn.commit()

    for secret in (first.secret, second.secret):
        with pytest.raises(agent_tokens.TokenRejected) as excinfo:
            _verify(secret)
        assert excinfo.value.reason == "revoked"


def test_expiry_sweep_settles_past_deadline_tokens(hosts):
    host_id = hosts[0]
    issued = _issue(host_id)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE host_agent_tokens SET expires_at = clock_timestamp() - interval '1 hour' "
            "WHERE token_id = %s",
            (issued.token_id,),
        )
        conn.commit()

    # Verification already refuses it before any sweep runs.
    with pytest.raises(agent_tokens.TokenRejected) as excinfo:
        _verify(issued.secret)
    assert excinfo.value.reason == "expired"

    with _pool.connection() as conn:
        swept = agent_tokens.expire_stale_tokens(conn)
        conn.commit()
    assert swept >= 1

    with _pool.connection() as conn:
        status = conn.execute(
            "SELECT status FROM host_agent_tokens WHERE token_id = %s",
            (issued.token_id,),
        ).fetchone()[0]
    assert status == "expired"

    # The freed slot lets a new token be issued without index conflict.
    fresh = _issue(host_id)
    assert _verify(fresh.secret).host_id == host_id


def test_last_used_is_stamped_and_throttled(hosts):
    host_id = hosts[0]
    issued = _issue(host_id)

    _verify(issued.secret)
    with _pool.connection() as conn:
        first_seen = conn.execute(
            "SELECT last_used_at FROM host_agent_tokens WHERE token_id = %s",
            (issued.token_id,),
        ).fetchone()[0]
    assert first_seen is not None

    # A busy poll loop must not turn every request into a write.
    _verify(issued.secret)
    with _pool.connection() as conn:
        second_seen = conn.execute(
            "SELECT last_used_at FROM host_agent_tokens WHERE token_id = %s",
            (issued.token_id,),
        ).fetchone()[0]
    assert second_seen == first_seen


def test_rotation_coverage_reports_hosts_missing_credentials(hosts):
    host_a, host_b = hosts
    _issue(host_a)
    with _pool.connection() as conn:
        coverage = agent_tokens.rotation_coverage(conn, host_ids=hosts)
    assert coverage["hosts"] == 2
    assert coverage["with_active_token"] == 1
    assert coverage["missing"] == [host_b]
    assert coverage["ready"] is False

    _issue(host_b)
    with _pool.connection() as conn:
        coverage = agent_tokens.rotation_coverage(conn, host_ids=hosts)
    assert coverage["ready"] is True
    assert coverage["missing"] == []


# ── Mode flag ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "allow"),
        ("allow", "allow"),
        ("require", "require"),
        ("off", "off"),
        ("true", "allow"),
        ("false", "off"),
        # An operator typo must not silently enable an auth path.
        ("requre", "off"),
        ("", "allow"),
    ],
)
def test_mode_flag_resolution_fails_closed(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("XCELSIOR_AGENT_HOST_TOKENS", raising=False)
    else:
        monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", value)
    assert agent_tokens.host_token_mode() == expected


# ── HTTP: the real auth gate ──────────────────────────────────────────


def _telemetry_body(host_id: str) -> dict:
    return {
        "host_id": host_id,
        "gpus": [],
        "cpu_percent": 1.0,
        "mem_percent": 1.0,
    }


def test_host_token_authenticates_a_real_agent_route(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_id = hosts[0]
    issued = _issue(host_id)

    resp = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    assert resp.status_code == 200, resp.text


def test_host_token_cannot_act_for_another_host_over_http(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_a, host_b = hosts
    issued = _issue(host_a)

    resp = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_b),
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    assert resp.status_code == 403, resp.text
    assert "host_mismatch" in resp.text


def test_revoked_token_is_refused_over_http(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_id = hosts[0]
    issued = _issue(host_id)
    with _pool.connection() as conn:
        agent_tokens.revoke_token(conn, host_id=host_id)
        conn.commit()

    resp = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    assert resp.status_code == 401, resp.text
    assert "revoked" in resp.text


def test_presented_host_token_never_degrades_to_shared_bearer(monkeypatch, hosts):
    """A bad ``xat_`` credential must fail, not fall through to the fleet token."""
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_id = hosts[0]

    resp = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {agent_tokens.TOKEN_PREFIX}bogus"},
    )
    assert resp.status_code == 401, resp.text


def test_require_mode_refuses_the_shared_fleet_bearer(monkeypatch, hosts):
    """This is what 'field-wide bearer rotation complete' actually means."""
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "require")
    host_id = hosts[0]

    denied = client.post(
        "/agent/telemetry", json=_telemetry_body(host_id), headers=_shared_bearer()
    )
    assert denied.status_code == 403, denied.text
    assert "Per-host agent token required" in denied.text

    issued = _issue(host_id)
    allowed = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    assert allowed.status_code == 200, allowed.text


def test_require_mode_also_covers_host_agnostic_agent_routes(monkeypatch, hosts):
    """Rotation is field-wide: no /agent/* surface keeps the fleet bearer."""
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "require")
    denied = client.get("/agent/popular-images", headers=_shared_bearer())
    assert denied.status_code == 403, denied.text


def test_off_mode_never_consults_the_token_store(monkeypatch, hosts):
    """``off`` must be inert — not merely "usually rejects".

    Proven by making any read of the token store an error: if the auth
    gate touched it, the request would fail loudly instead of taking the
    pre-existing path.
    """
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "off")
    host_id = hosts[0]
    issued = _issue(host_id)

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("token store consulted while XCELSIOR_AGENT_HOST_TOKENS=off")

    monkeypatch.setattr(agent_tokens, "verify_token", _explode)

    resp = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    # Whatever the ambient auth policy decides, the host-token path is
    # not in the decision at all.
    assert resp.status_code != 500, resp.text
    assert agent_tokens.host_tokens_enabled() is False
    assert agent_tokens.host_tokens_required() is False


def test_rotate_endpoint_requires_the_current_host_token(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_id = hosts[0]
    issued = _issue(host_id)

    # The shared bearer cannot rotate a host credential.
    refused = client.post("/agent/v2/tokens/rotate", headers=_shared_bearer())
    assert refused.status_code == 401, refused.text

    rotated = client.post(
        "/agent/v2/tokens/rotate",
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    assert rotated.status_code == 200, rotated.text
    body = rotated.json()
    assert body["host_id"] == host_id
    assert body["token"].startswith(agent_tokens.TOKEN_PREFIX)
    assert body["token"] != issued.secret

    # The new credential works on a real agent route immediately.
    used = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {body['token']}"},
    )
    assert used.status_code == 200, used.text


def test_rotate_endpoint_is_absent_when_tokens_are_disabled(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "off")
    resp = client.post("/agent/v2/tokens/rotate", headers=_shared_bearer())
    assert resp.status_code == 404, resp.text


def test_revoked_token_cannot_rotate_itself_back(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_id = hosts[0]
    issued = _issue(host_id)
    with _pool.connection() as conn:
        agent_tokens.revoke_token(conn, host_id=host_id)
        conn.commit()

    resp = client.post(
        "/agent/v2/tokens/rotate",
        headers={"Authorization": f"Bearer {issued.secret}"},
    )
    assert resp.status_code == 401, resp.text
    assert "revoked" in resp.text


# ── HTTP: admin surface ───────────────────────────────────────────────


def test_admin_can_issue_list_and_revoke(monkeypatch, hosts):
    monkeypatch.setenv("XCELSIOR_AGENT_HOST_TOKENS", "allow")
    host_id = hosts[0]

    issued = client.post(f"/api/admin/hosts/{host_id}/agent-tokens", headers=_shared_bearer())
    assert issued.status_code == 200, issued.text
    secret = issued.json()["token"]
    assert secret.startswith(agent_tokens.TOKEN_PREFIX)

    listed = client.get(f"/api/admin/hosts/{host_id}/agent-tokens", headers=_shared_bearer())
    assert listed.status_code == 200, listed.text
    tokens = listed.json()["tokens"]
    assert len(tokens) == 1
    assert tokens[0]["status"] == "active"
    # The operator view must never leak the credential.
    assert secret not in listed.text
    assert tokens[0]["token_prefix"] == secret[:12]

    revoked = client.post(
        f"/api/admin/hosts/{host_id}/agent-tokens/revoke", headers=_shared_bearer()
    )
    assert revoked.status_code == 200, revoked.text
    assert revoked.json()["revoked"] == 1

    dead = client.post(
        "/agent/telemetry",
        json=_telemetry_body(host_id),
        headers={"Authorization": f"Bearer {secret}"},
    )
    assert dead.status_code == 401


def test_admin_issue_rejects_unknown_host():
    resp = client.post(
        f"/api/admin/hosts/ghost-{uuid.uuid4().hex[:6]}/agent-tokens",
        headers=_shared_bearer(),
    )
    assert resp.status_code == 404, resp.text


def test_admin_coverage_endpoint_reports_readiness(monkeypatch, hosts):
    resp = client.get("/api/admin/agent-tokens/coverage", headers=_shared_bearer())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ok"] is True
    assert "mode" in body and "missing" in body and "ready" in body
    # Fixture hosts have no tokens yet, so they must appear as missing.
    assert set(hosts).issubset(set(body["missing"]))


def test_token_admin_routes_are_admin_gated():
    """Every credential route must pass through ``_require_admin``.

    (The suite runs with ``AUTH_REQUIRED`` off, so an anonymous HTTP probe
    proves nothing here — the guard itself is what must be present.)
    """
    import inspect

    import routes.admin as admin_routes

    for func in (
        admin_routes.api_admin_agent_token_coverage,
        admin_routes.api_admin_list_host_agent_tokens,
        admin_routes.api_admin_issue_host_agent_token,
        admin_routes.api_admin_revoke_host_agent_tokens,
    ):
        assert "_require_admin(request)" in inspect.getsource(func), func.__name__


# ── Worker-side credential handling ───────────────────────────────────


@pytest.fixture
def worker_token_file(tmp_path, monkeypatch):
    import worker_agent

    path = tmp_path / "agent-token.json"
    monkeypatch.setattr(worker_agent, "AGENT_TOKEN_FILE", str(path))
    worker_agent._clear_host_token_cache()
    yield worker_agent, path
    worker_agent._clear_host_token_cache()


def test_worker_prefers_the_host_token_over_the_fleet_bearer(worker_token_file):
    worker_agent, path = worker_token_file
    assert worker_agent._api_headers().get("Authorization") != "Bearer xat_local"

    worker_agent._write_host_token({"token": "xat_local", "host_id": worker_agent.HOST_ID})
    assert worker_agent._api_headers()["Authorization"] == "Bearer xat_local"


def test_worker_ignores_a_token_issued_for_a_different_host(worker_token_file):
    """A misplaced credential must not strand this host."""
    worker_agent, _ = worker_token_file
    worker_agent._write_host_token({"token": "xat_someone_else", "host_id": "not-this-host"})
    auth = worker_agent._api_headers().get("Authorization", "")
    assert "xat_someone_else" not in auth


def test_worker_token_file_is_written_atomically_and_locked_down(worker_token_file):
    worker_agent, path = worker_token_file
    assert worker_agent._write_host_token({"token": "xat_a", "host_id": worker_agent.HOST_ID})
    assert oct(os.stat(path).st_mode & 0o777) == "0o600"
    assert not os.path.exists(f"{path}.tmp")

    # Overwriting keeps exactly one file and the newest value.
    assert worker_agent._write_host_token({"token": "xat_b", "host_id": worker_agent.HOST_ID})
    assert json.loads(path.read_text())["token"] == "xat_b"
    assert worker_agent._host_token_value() == "xat_b"


def test_worker_rotates_only_when_the_token_is_near_expiry(worker_token_file, monkeypatch):
    worker_agent, _ = worker_token_file
    calls: list[str] = []

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {
                "token": "xat_rotated",
                "token_id": "t2",
                "expires_at": "2030-01-01T00:00:00+00:00",
            }

    def _fake_post(url, **kwargs):
        calls.append(url)
        return _Resp()

    monkeypatch.setattr(worker_agent.requests, "post", _fake_post)

    now = time.time()
    # Fresh token: 90-day lifetime, 80 days left — no rotation.
    worker_agent._write_host_token(
        {
            "token": "xat_fresh",
            "host_id": worker_agent.HOST_ID,
            "issued_at_epoch": now - 10 * 86400,
            "expires_at_epoch": now + 80 * 86400,
        }
    )
    worker_agent._last_token_rotation_attempt = 0.0
    assert worker_agent.maybe_rotate_host_token() is False
    assert calls == []

    # Same lifetime, 5 days left — inside the rotation margin.
    worker_agent._write_host_token(
        {
            "token": "xat_old",
            "host_id": worker_agent.HOST_ID,
            "issued_at_epoch": now - 85 * 86400,
            "expires_at_epoch": now + 5 * 86400,
        }
    )
    worker_agent._last_token_rotation_attempt = 0.0
    assert worker_agent.maybe_rotate_host_token() is True
    assert len(calls) == 1
    assert worker_agent._host_token_value() == "xat_rotated"
    # The replacement is what subsequent API calls present.
    assert worker_agent._api_headers()["Authorization"] == "Bearer xat_rotated"


def test_worker_keeps_the_old_token_when_rotation_fails(worker_token_file, monkeypatch):
    """A rejected rotation must not blank out a still-valid credential."""
    worker_agent, _ = worker_token_file

    class _Resp:
        status_code = 500
        text = "boom"

    monkeypatch.setattr(worker_agent.requests, "post", lambda url, **kw: _Resp())

    now = time.time()
    worker_agent._write_host_token(
        {
            "token": "xat_still_good",
            "host_id": worker_agent.HOST_ID,
            "issued_at_epoch": now - 85 * 86400,
            "expires_at_epoch": now + 5 * 86400,
        }
    )
    worker_agent._last_token_rotation_attempt = 0.0
    assert worker_agent.maybe_rotate_host_token() is False
    assert worker_agent._host_token_value() == "xat_still_good"


def test_worker_rotation_is_rate_limited(worker_token_file, monkeypatch):
    """A failing server must not turn the poll loop into a rotation storm."""
    worker_agent, _ = worker_token_file
    calls: list[str] = []
    monkeypatch.setattr(
        worker_agent.requests,
        "post",
        lambda url, **kw: (calls.append(url), _RaiseOnJson())[1],
    )

    class _RaiseOnJson:
        status_code = 500
        text = "nope"

    now = time.time()
    worker_agent._write_host_token(
        {
            "token": "xat_expiring",
            "host_id": worker_agent.HOST_ID,
            "issued_at_epoch": now - 89 * 86400,
            "expires_at_epoch": now + 1 * 86400,
        }
    )
    worker_agent._last_token_rotation_attempt = 0.0
    worker_agent.maybe_rotate_host_token()
    worker_agent.maybe_rotate_host_token()
    worker_agent.maybe_rotate_host_token()
    assert len(calls) == 1


def test_token_expiry_is_a_durable_scheduled_task(hosts):
    """The sweep must survive a restart, and must actually sweep.

    Registered from ``bg_worker.main`` (so every replica converges the row)
    and executed through the real ``scheduled_tasks`` claim path — a
    process-local timer would silently stop sweeping whenever the process
    that happened to own it died.
    """
    import inspect

    import bg_worker
    from control_plane.scheduled_tasks import claim_and_run_tasks, register_task

    assert 'register_task("host_agent_token_expiry"' in inspect.getsource(bg_worker.main)

    def _sweep():
        from control_plane.agent_tokens import expire_stale_tokens
        from control_plane.db import run_transaction

        run_transaction(
            lambda c: expire_stale_tokens(c, limit=500), what="host_agent_token_expiry"
        )

    register_task("host_agent_token_expiry", _sweep, 300)

    with _pool.connection() as conn:
        row = conn.execute(
            "SELECT enabled, interval_seconds FROM scheduled_tasks "
            "WHERE task_name = 'host_agent_token_expiry'"
        ).fetchone()
    assert row is not None, "registration must persist the task row"
    assert row[0] is True

    # Seed a past-deadline token and make the task due right now.
    host_id = hosts[0]
    issued = _issue(host_id)
    with _pool.connection() as conn:
        conn.execute(
            "UPDATE host_agent_tokens SET expires_at = clock_timestamp() - interval '1 hour' "
            "WHERE token_id = %s",
            (issued.token_id,),
        )
        conn.execute(
            "UPDATE scheduled_tasks SET next_run_at = clock_timestamp() - interval '1 minute', "
            "claim_owner = NULL, claim_expires_at = NULL "
            "WHERE task_name = 'host_agent_token_expiry'"
        )
        conn.commit()

    claim_and_run_tasks("pytest-token-sweeper")

    with _pool.connection() as conn:
        status = conn.execute(
            "SELECT status FROM host_agent_tokens WHERE token_id = %s",
            (issued.token_id,),
        ).fetchone()[0]
        task = conn.execute(
            "SELECT last_status FROM scheduled_tasks "
            "WHERE task_name = 'host_agent_token_expiry'"
        ).fetchone()
    assert status == "expired", "the durable task must actually run the sweep"
    assert task is not None and task[0] == "succeeded"
