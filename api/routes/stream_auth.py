"""Continuation checks for streams; bearer secrets are never added to tickets."""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import HTTPException


def bind_stream_principal(user: dict, request) -> dict:
    """Retain a non-authorizing cache reference for OAuth ticket continuation."""
    from oauth_service import _cache_key, is_oauth_access_token
    from routes._deps import _AUTH_COOKIE_NAME

    principal = dict(user)
    auth = request.headers.get("authorization", "")
    token = auth[7:] if auth.startswith("Bearer ") else ""
    token = token or getattr(request, "cookies", {}).get(_AUTH_COOKIE_NAME, "")
    if is_oauth_access_token(token):
        principal["_stream_access_cache_key"] = _cache_key("access_token", token)
    return principal


def refresh_stream_principal(user: dict) -> dict:
    """Recheck current suspension, OAuth revocation/expiry and user membership.

    The input is a server-issued ticket principal, never caller-supplied JSON.
    A hashed cache reference can locate an access record but cannot authenticate
    a new HTTP request. Authorization-store errors propagate to close the stream.
    """
    from abuse_policy import account_suspended
    from db import UserStore, auth_connection
    from oauth_service import _cache_get_json, get_auth_cache
    from routes._deps import _merge_auth_user

    if account_suspended(user):
        raise HTTPException(403, "Stream authorization ended")
    current = dict(user)
    reference = user.get("_stream_access_cache_key")
    if reference:
        payload = json.loads(get_auth_cache().get(reference) or "null")
        if not isinstance(payload, dict) or any(
            payload.get(k) != user.get(k) for k in ("user_id", "client_id")
        ):
            raise HTTPException(401, "Stream credential revoked")
        current = dict(payload, _stream_access_cache_key=reference)
    if float(current.get("expires_at") or float("inf")) <= time.time():
        raise HTTPException(401, "Stream credential expired")
    session = str(current.get("session_token") or "")
    if session and _cache_get_json("revoked_session", session):
        raise HTTPException(401, "Stream session revoked")
    if current.get("auth_type") == "legacy_session":
        from routes._deps import _USE_PERSISTENT_AUTH, _sessions, _user_lock

        token = str(current.get("token") or "")
        if _USE_PERSISTENT_AUTH:
            live = UserStore.get_session(token)
        else:
            with _user_lock:
                live = dict(_sessions.get(token) or {})
        if not live or float(live.get("expires_at") or 0) <= time.time():
            raise HTTPException(401, "Stream session revoked")
    if current.get("auth_type") == "agent_api_key":
        with auth_connection() as conn:
            row = conn.execute(
                "SELECT user_id, scopes FROM agent_api_keys WHERE key_id=%s AND revoked_at IS NULL",
                (current.get("key_id"),),
            ).fetchone()
        if not row or row["user_id"] != current.get("user_id"):
            raise HTTPException(401, "Stream credential revoked")
        current["scopes"] = str(row["scopes"] or "").split()
    full_user = None
    if current.get("email") and current.get("auth_type") not in (
        "master_token",
        "client_credentials",
    ):
        full_user = UserStore.get_user(current["email"])
        if current.get("auth_type") and not full_user:
            raise HTTPException(401, "Stream user no longer exists")
    return _merge_auth_user(current, full_user)


def require_terminal_attempt(job: dict | None, job_id: str):
    """The running native attempt a shell may enter, or 403.

    A shell is execution inside the container, so it needs the authority
    execution has: the job's current attempt running and holding an active,
    unexpired lease on the job's host. The projected ``status`` keeps saying
    ``running`` after a lease is lost or an attempt is fenced, so it is not
    enough on its own.
    """
    from control_plane.job_targets import resolve_job_command_target

    target = resolve_job_command_target(job_id)
    host_id = str((job or {}).get("host_id") or "")
    if (
        target is None
        or target.class_ != "attempt_owned"
        or not target.host_id
        or target.host_id != host_id
    ):
        raise HTTPException(403, "Instance has no running attempt")
    return target


def authorize_job_stream(user: dict, job_id: str, *, terminal: bool = False) -> dict:
    from scheduler import get_job
    from routes._deps import _require_scope, _require_team_instance_write, _user_owns_job

    current = refresh_stream_principal(user)
    _require_scope(current, "instances:connect" if terminal else "instances:read")
    job = get_job(job_id)
    if not job or not _user_owns_job(current, job):
        raise HTTPException(403, "Stream object access ended")
    if terminal:
        _require_team_instance_write(current)
        if job.get("status") != "running":
            raise HTTPException(403, "Instance is no longer running")
        require_terminal_attempt(job, job_id)
    return current


async def watch_job_authorization(websocket, user: dict, job_id: str, *, terminal: bool = False):
    """Close quiet sockets too; all data-plane loops are cancelled by the caller."""
    while True:
        await asyncio.sleep(2)
        try:
            await asyncio.wait_for(
                asyncio.to_thread(authorize_job_stream, user, job_id, terminal=terminal),
                timeout=5,
            )
        except Exception:
            try:
                await websocket.send_json(
                    {"type": "error", "message": "Stream authorization ended", "code": 4003}
                )
            finally:
                await websocket.close(code=4003)
            return
