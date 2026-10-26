"""Streams must lose authority after admission and never broadcast across tenants."""

import asyncio
import json
import time
import uuid

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import abuse_policy
import oauth_service as oauth
from db import UserStore, auth_connection, pg_transaction
from routes import _deps, health, stream_auth
from tests.fixtures.native_runtime import running_instance


class Request:
    def __init__(self, token=""):
        self.headers = {"authorization": "Bearer " + token} if token else {}
        self.cookies = {}
        self.query_params = {}

    async def is_disconnected(self):
        return False


@pytest.fixture(autouse=True)
def require_auth(monkeypatch, _pin_test_auth_env):
    monkeypatch.setattr(_deps, "AUTH_REQUIRED", True)


@pytest.fixture
def principals():
    records = []
    for _ in range(2):
        uid = "stream-test-" + uuid.uuid4().hex
        user = {
            "email": uid + "@example.com",
            "user_id": uid,
            "customer_id": uid,
            "role": "submitter",
        }
        UserStore.create_user(user)
        token = oauth._issue_opaque_access_token(
            user,
            client_id="xcelsior-web",
            scopes=["instances:read", "instances:connect", "events:read"],
            session_token="session-" + uid,
            session_type="browser",
        )["access_token"]
        request = Request(token)
        principal = _deps._get_current_user(request)
        with pg_transaction() as conn:
            conn.execute(
                """INSERT INTO jobs(job_id,status,priority,submitted_at,payload, phase, desired_state, effective_priority, queued_at) VALUES (%s,'running',0,0,%s, 'running', 'running', 0, NULL)""",
                (uid, json.dumps({"job_id": uid, "status": "running", "owner": uid})),
            )
            conn.execute(
                "UPDATE jobs SET owner_id=%s,tenant_id=%s WHERE job_id=%s", (uid, uid, uid)
            )
        records.append((uid, token, request, principal))
    yield records
    with pg_transaction() as conn:
        for uid, token, _, _ in records:
            conn.execute("DELETE FROM jobs WHERE job_id=%s", (uid,))
            conn.execute("DELETE FROM state WHERE namespace=%s", ("security.suspended." + uid,))
    with auth_connection() as conn:
        for uid, _, _, _ in records:
            conn.execute("DELETE FROM users WHERE user_id=%s", (uid,))
    oauth.reset_auth_cache_for_tests()


@pytest.mark.asyncio
async def test_global_stream_rejects_anonymous_at_route_admission():
    with pytest.raises(HTTPException) as error:
        await health.sse_stream(Request())
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_live_feed_filters_other_tenant_and_unknown_fleet_events(principals):
    (uid, _, request, _), (other, _, _, _) = principals
    generator = health._sse_generator(request)
    try:
        assert (await anext(generator)).startswith("retry:")
        assert "connected" in await anext(generator)
        queue = _deps._sse_subscribers[-1]
        await queue.put({"event": "job_log", "data": {"job_id": other, "line": "private-other"}})
        await queue.put({"event": "fleet_secret", "data": {"line": "private-fleet"}})
        await queue.put({"event": "job_log", "data": {"job_id": uid, "line": "authorized"}})
        result = await asyncio.wait_for(anext(generator), 2)
        assert "authorized" in result and "private" not in result
    finally:
        await generator.aclose()


@pytest.mark.asyncio
async def test_replay_uses_the_same_object_authorization(principals, monkeypatch):
    from control_plane import event_stream

    (uid, _, request, _), (other, _, _, _) = principals
    events = [
        event_stream.StreamEvent("cursor-" + job, job, "job_log", {"job_id": job})
        for job in (other, uid)
    ]
    monkeypatch.setattr(event_stream, "resume_after", lambda *a, **k: events)
    generator = health._sse_generator(request, last_event_id="prior")
    try:
        await anext(generator)
        await anext(generator)
        result = await asyncio.wait_for(anext(generator), 2)
        assert uid in result and other not in result
    finally:
        await generator.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["suspend", "revoke_token", "revoke_session"])
async def test_open_sse_stops_after_authority_loss(principals, action):
    uid, token, request, principal = principals[0]
    generator = health._sse_generator(request)
    await anext(generator)
    await anext(generator)
    if action == "suspend":
        abuse_policy.suspend_account(identifier=uid, actor="test", reason="synthetic")
    elif action == "revoke_token":
        oauth.revoke_access_token(token)
    else:
        oauth.revoke_session(principal["session_token"])
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(generator), 3)
    assert _deps._get_current_user(principals[1][2]) is not None


@pytest.mark.parametrize("action", ["suspend", "revoke_token", "revoke_session", "transfer"])
def test_bound_ticket_continuation_rechecks_actual_credential_and_owner(principals, action):
    uid, token, request, principal = principals[0]
    bound = stream_auth.bind_stream_principal(principal, request)
    assert token not in json.dumps(bound)
    assert stream_auth.authorize_job_stream(bound, uid)["user_id"] == uid
    if action == "suspend":
        abuse_policy.suspend_account(identifier=uid, actor="test", reason="synthetic")
    elif action == "revoke_token":
        oauth.revoke_access_token(token)
    elif action == "revoke_session":
        oauth.revoke_session(principal["session_token"])
    else:
        with pg_transaction() as conn:
            conn.execute(
                "UPDATE jobs SET owner_id=%s WHERE job_id=%s",
                (principals[1][0], uid),
            )
    with pytest.raises(HTTPException):
        stream_auth.authorize_job_stream(bound, uid)
    assert stream_auth.authorize_job_stream(
        stream_auth.bind_stream_principal(principals[1][3], principals[1][2]), principals[1][0]
    )


@pytest.mark.parametrize("invalid", [None, "projection_only", "expired_lease"])
def test_terminal_authority_is_the_running_native_attempt(principals, invalid):
    """`status='running'` outlives a lost lease; a shell needs the attempt itself."""
    uid, _, request, principal = principals[0]
    bound = stream_auth.bind_stream_principal(principal, request)
    if invalid == "projection_only":
        # The fixture row says running and has no placement behind it.
        with pytest.raises(HTTPException) as refused:
            stream_auth.authorize_job_stream(bound, uid, terminal=True)
        assert refused.value.status_code == 403
        return
    with running_instance(owner=uid) as instance:
        if invalid == "expired_lease":
            with pg_transaction() as conn:
                conn.execute(
                    "UPDATE placement_leases SET expires_at=clock_timestamp()-interval '1 second'"
                    " WHERE attempt_id=%s",
                    (instance["authority"]["attempt_id"],),
                )
            with pytest.raises(HTTPException) as refused:
                stream_auth.authorize_job_stream(bound, instance["job_id"], terminal=True)
            assert refused.value.status_code == 403
        else:
            assert stream_auth.authorize_job_stream(bound, instance["job_id"], terminal=True)


def test_terminal_socket_never_probes_a_container_without_a_running_attempt(
    principals, monkeypatch
):
    from api import app
    from routes import terminal

    uid, _, request, principal = principals[0]
    probed = []
    monkeypatch.setattr(
        terminal, "_container_probe", lambda ref, *a, **k: probed.append(ref) or {"found": False}
    )
    monkeypatch.setattr(terminal, "_CONTAINER_POLL_BUDGET_SEC", 0.05)
    issued = _deps._issue_ws_ticket(
        principal, request=request, purpose="terminal", target=uid, client_ip="testclient"
    )
    with TestClient(app) as client:
        with client.websocket_connect("/ws/terminal/" + uid + "?ticket=" + issued["ticket"]) as ws:
            message = ws.receive_json()
    assert message["type"] == "error" and message["code"] == 4003
    assert "running attempt" in message["message"]
    assert probed == []


@pytest.mark.asyncio
async def test_quiet_terminal_watch_closes_on_real_suspension(principals, monkeypatch):
    uid, _, request, principal = principals[0]
    bound = stream_auth.bind_stream_principal(principal, request)
    with running_instance(owner=uid) as instance:
        await _watch_until_suspended(bound, uid, instance["job_id"])


async def _watch_until_suspended(bound, uid, job_id):
    stream_auth.authorize_job_stream(bound, job_id, terminal=True)

    class Socket:
        def __init__(self):
            self.closed = None

        async def send_json(self, data):
            self.error = data

        async def close(self, code):
            self.closed = code

    socket = Socket()
    task = asyncio.create_task(
        stream_auth.watch_job_authorization(socket, bound, job_id, terminal=True)
    )
    await asyncio.sleep(0)
    abuse_policy.suspend_account(identifier=uid, actor="test", reason="synthetic")
    await asyncio.wait_for(task, 3)
    assert socket.closed == 4003 and socket.error["code"] == 4003


def test_established_instance_websocket_closes_when_original_token_is_revoked(principals):
    from api import app

    uid, token, request, principal = principals[0]
    issued = _deps._issue_ws_ticket(
        principal, request=request, purpose="instance_stream", target=uid, client_ip="testclient"
    )
    with TestClient(app) as client:
        with client.websocket_connect(
            "/ws/instances/" + uid + "?ticket=" + issued["ticket"]
        ) as socket:
            assert socket.receive_json()["event"] == "instance"
            oauth.revoke_access_token(token)
            message = socket.receive_json()
            assert message["type"] == "error" and message["code"] == 4003
            with pytest.raises(WebSocketDisconnect):
                socket.receive_json()


@pytest.mark.asyncio
async def test_open_stream_fails_closed_when_policy_store_is_unavailable(principals, monkeypatch):
    _, _, request, _ = principals[0]
    generator = health._sse_generator(request)
    await anext(generator)
    await anext(generator)

    def unavailable(*args, **kwargs):
        raise RuntimeError("policy store unavailable")

    monkeypatch.setattr(abuse_policy, "account_suspended", unavailable)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(generator), 3)


@pytest.mark.asyncio
async def test_job_log_revocation_stops_before_the_connected_frame(principals):
    from routes import instances

    uid, token, request, _ = principals[0]
    generator = instances._job_log_generator(request, uid)
    try:
        assert (await anext(generator)).startswith("retry:")
        oauth.revoke_access_token(token)
        with pytest.raises(HTTPException):
            await asyncio.wait_for(anext(generator), 3)
    finally:
        await generator.aclose()


async def test_job_log_stream_reauthorizes_before_replaying_another_chunk(principals, monkeypatch):
    from routes import instances

    uid, token, request, _ = principals[0]
    calls = {"n": 0}
    real_require_auth = instances._require_auth

    def counted_require_auth(req):
        calls["n"] += 1
        return real_require_auth(req)

    monkeypatch.setattr(instances, "_require_auth", counted_require_auth)
    generator = instances._job_log_generator(request, uid)
    try:
        assert (await anext(generator)).startswith("retry:")
        assert "connected" in await anext(generator)
        seen = calls["n"]
        oauth.revoke_access_token(token)
        with pytest.raises(HTTPException):
            _deps._require_auth(request)
        with pytest.raises(HTTPException):
            await asyncio.wait_for(anext(generator), 3)
        assert calls["n"] > seen
    finally:
        await generator.aclose()


def test_websocket_cannot_resurrect_a_permanently_disabled_api_key(monkeypatch):
    monkeypatch.setattr(_deps, "_USE_PERSISTENT_AUTH", True)
    monkeypatch.setattr(UserStore, "get_session", lambda token: None)
    monkeypatch.setattr(UserStore, "get_api_key", lambda token: {"email": "old@example.com"})
    request = Request()
    request.cookies = {_deps._AUTH_COOKIE_NAME: "legacy-disabled-key"}
    assert _deps._validate_ws_auth(request, allow_query_token=False) is None
