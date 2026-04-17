"""Routes: Interactive web terminal.

Architecture
------------
Browser (xterm.js) <- binary WebSocket -> FastAPI (/ws/terminal/{instance_id})
    -> Docker SDK (docker-py) -> container exec attach -> tmux / bare shell

For remote hosts, the Docker SDK connects via ``ssh://user@host`` transport
(paramiko). For local containers, it uses the default Docker socket.

Protocol
--------
Client -> server (text JSON):
    {"type": "input",  "data": "<chars>"}
    {"type": "resize", "cols": N, "rows": N}
    {"type": "ping"}

Server -> client (binary): raw PTY bytes (xterm.js writes directly; no JSON wrap)

Server -> client (text JSON, control only):
    {"type": "status", "message": "...", "retry": true|false}
    {"type": "error",  "message": "...", "code": N}
    {"type": "exit",   "code": N}
    {"type": "pong",   "ts": <float>}

Security
--------
- Browser origin validated before accept().
- WebSocket auth validated before accept() via cookie or short-lived one-time ticket.
- Per-IP connection throttling and per-user concurrent session limits.
- Container refs, host IPs, and shell paths validated against strict allowlists.
- Docker SDK exec always runs inside the container sandbox.
- No Docker socket exposed to the browser.
"""

import asyncio
from collections import defaultdict
import json
import os
import re as _re
import secrets
import subprocess
import threading
import time
from typing import Optional

import docker
from docker.errors import APIError, DockerException, NotFound
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from routes._deps import (
    _check_ws_connect_rate_limit,
    _consume_ws_ticket,
    _get_ws_client_ip,
    _issue_ws_ticket,
    _require_auth,
    _shared_state_update,
    _validate_ws_auth,
    _validate_ws_origin,
    log,
)
from scheduler import (
    SSH_KEY_PATH,
    SSH_USER,
    list_hosts,
    list_jobs,
)

# -- Optional Prometheus metrics (graceful no-op when not installed) -----------

try:
    from prometheus_client import Counter, Gauge  # type: ignore[import]

    _active_sessions: object = Gauge(
        "xcelsior_terminal_sessions_active",
        "Currently active interactive terminal WebSocket sessions",
    )
    _bytes_sent: object = Counter(
        "xcelsior_terminal_bytes_sent_total",
        "Total raw PTY bytes relayed to terminal clients",
    )
    _bytes_recv: object = Counter(
        "xcelsior_terminal_bytes_recv_total",
        "Total raw bytes received from terminal clients",
    )
    _conn_errors: object = Counter(
        "xcelsior_terminal_errors_total",
        "Terminal WebSocket connection errors by reason",
        ["reason"],
    )
    _disconnections: object = Counter(
        "xcelsior_terminal_disconnections_total",
        "Terminal session disconnections by cause",
        ["cause"],
    )
    _PROMETHEUS = True
except ImportError:
    _PROMETHEUS = False

    class _Noop:
        """Null-object that silently absorbs any metric calls."""

        def inc(self, *a: object, **kw: object) -> None: ...
        def dec(self, *a: object, **kw: object) -> None: ...
        def labels(self, *a: object, **kw: object) -> "_Noop": return self

    _active_sessions = _Noop()  # type: ignore[assignment]
    _bytes_sent = _Noop()       # type: ignore[assignment]
    _bytes_recv = _Noop()       # type: ignore[assignment]
    _conn_errors = _Noop()      # type: ignore[assignment]
    _disconnections = _Noop()   # type: ignore[assignment]


router = APIRouter()

# -- Constants -----------------------------------------------------------------

_SESSION_TIMEOUT_SEC: int = 14_400
_IDLE_WARN_THRESHOLD_SEC: int = _SESSION_TIMEOUT_SEC - 300
_RATE_LIMIT_BYTES_PER_SEC: int = 524_288
_EXEC_READ_CHUNK: int = 4096
_CONTAINER_POLL_INTERVAL_SEC: float = 2.0
_CONTAINER_POLL_MAX_ATTEMPTS: int = 15
_STDIN_RECV_TIMEOUT_SEC: float = 10.0
_EXEC_READ_TIMEOUT_SEC: float = 30.0
_EXEC_INSPECT_TIMEOUT_SEC: float = 2.0
_DOCKER_CLIENT_TIMEOUT_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_DOCKER_CLIENT_TIMEOUT_SEC", "10")
)
_DOCKER_CONTAINER_GET_TIMEOUT_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_DOCKER_CONTAINER_GET_TIMEOUT_SEC", "10")
)
_DOCKER_EXEC_CREATE_TIMEOUT_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_DOCKER_EXEC_CREATE_TIMEOUT_SEC", "10")
)
_DOCKER_EXEC_START_TIMEOUT_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_DOCKER_EXEC_START_TIMEOUT_SEC", "10")
)
_REMOTE_DOCKER_TIMEOUT_SEC: float = float(
    os.environ.get("XCELSIOR_REMOTE_DOCKER_TIMEOUT_SEC", "10")
)
_MAX_SESSION_LIFETIME_SEC: int = int(
    os.environ.get("XCELSIOR_TERMINAL_MAX_SESSION_LIFETIME_SEC", "14400")
)
_MAX_INPUT_FRAME_BYTES: int = int(
    os.environ.get("XCELSIOR_TERMINAL_MAX_INPUT_FRAME_BYTES", "65536")
)
_MAX_CONCURRENT_SESSIONS_PER_USER: int = int(
    os.environ.get("XCELSIOR_TERMINAL_MAX_CONCURRENT_SESSIONS_PER_USER", "3")
)
_TERMINAL_SESSION_STATE_NAMESPACE: str = os.environ.get(
    "XCELSIOR_TERMINAL_SESSION_STATE_NAMESPACE",
    "runtime.terminal_session_slots",
)
_TERMINAL_SESSION_LEASE_TTL_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_SESSION_LEASE_TTL_SEC", "120")
)
_TERMINAL_SESSION_LEASE_REFRESH_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_SESSION_LEASE_REFRESH_SEC", "30")
)
_MAX_RESIZE_COLS: int = int(os.environ.get("XCELSIOR_TERMINAL_MAX_RESIZE_COLS", "500"))
_MAX_RESIZE_ROWS: int = int(os.environ.get("XCELSIOR_TERMINAL_MAX_RESIZE_ROWS", "200"))
_MAX_MALFORMED_CONTROL_FRAMES: int = int(
    os.environ.get("XCELSIOR_TERMINAL_MAX_MALFORMED_CONTROL_FRAMES", "5")
)
_REQUIRE_PINNED_HOST_KEYS: bool = os.environ.get(
    "XCELSIOR_TERMINAL_REQUIRE_PINNED_HOST_KEYS",
    "true" if os.environ.get("XCELSIOR_ENV", "dev").lower() in {"prod", "production"} else "false",
).lower() not in {"0", "false", "no", "off"}
_KNOWN_HOSTS_PATH: str = os.path.expanduser(
    os.environ.get("XCELSIOR_TERMINAL_KNOWN_HOSTS_PATH", "~/.ssh/known_hosts")
)

_HOST_IP_RE: _re.Pattern[str] = _re.compile(r"\A[a-zA-Z0-9.\-]+\Z")
_CONTAINER_REF_RE: _re.Pattern[str] = _re.compile(
    r"\A[a-zA-Z0-9][a-zA-Z0-9_.-]{0,254}\Z"
)
_DEFAULT_ALLOWED_SHELLS = (
    "/bin/bash,/bin/sh,/bin/zsh,/usr/bin/bash,/usr/bin/sh,/usr/bin/zsh,/bin/ash"
)
_ALLOWED_SHELL_PATHS: frozenset[str] = frozenset(
    shell.strip()
    for shell in os.environ.get(
        "XCELSIOR_TERMINAL_ALLOWED_SHELLS",
        _DEFAULT_ALLOWED_SHELLS,
    ).split(",")
    if shell.strip()
)

_terminal_session_counts: dict[str, int] = defaultdict(int)
_terminal_session_lock = threading.Lock()


# -- Docker client factory -----------------------------------------------------

def _docker_client(
    host_ip: str | None = None,
    ssh_user: str = SSH_USER,
    ssh_key_path: str = SSH_KEY_PATH,
) -> docker.DockerClient:
    """Return a Docker SDK client for the given host."""
    if not host_ip or host_ip in ("127.0.0.1", "localhost", "0.0.0.0"):
        return docker.from_env()

    _ensure_remote_host_key_pinned(host_ip)
    return docker.DockerClient(
        base_url=f"ssh://{ssh_user}@{host_ip}",
        use_ssh_client=True,
        timeout=_REMOTE_DOCKER_TIMEOUT_SEC,
        environment={"SSH_KEY_PATH": ssh_key_path},
    )


# -- Preflight helpers ---------------------------------------------------------

def _container_exists(container_ref: str, host_ip: str | None = None) -> bool:
    """Return True when *container_ref* exists on the Docker daemon."""
    cl = None
    try:
        cl = _docker_client(host_ip)
        cl.containers.get(container_ref)
        return True
    except (NotFound, DockerException, Exception):
        return False
    finally:
        if cl is not None:
            try:
                cl.close()
            except Exception:
                pass


def _tmux_available(container_ref: str, host_ip: str | None = None) -> bool:
    """Return True when tmux is on PATH inside *container_ref*."""
    cl = None
    try:
        cl = _docker_client(host_ip)
        container = cl.containers.get(container_ref)
        result = container.exec_run("which tmux")
        return result.exit_code == 0
    except (NotFound, DockerException, Exception):
        return False
    finally:
        if cl is not None:
            try:
                cl.close()
            except Exception:
                pass


# -- Legacy subprocess helpers (kept for backward compat in tests) -------------

def _build_ssh_opts(key_path: str) -> list[str]:
    return [
        "-o", "StrictHostKeyChecking=yes",
        "-o", f"UserKnownHostsFile={_KNOWN_HOSTS_PATH}",
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=10",
        "-i", key_path,
    ]


def _terminal_session_key(user: dict) -> str:
    return str(
        user.get("customer_id")
        or user.get("user_id")
        or user.get("email")
        or "anonymous"
    )


def _session_slot_is_shared(session_slot: str | None) -> bool:
    return bool(session_slot and session_slot.startswith("shared:"))


def _purge_shared_terminal_sessions(
    sessions: dict[str, dict] | None,
    now: float,
) -> dict[str, dict]:
    pruned: dict[str, dict] = {}
    for lease_id, payload in (sessions or {}).items():
        expires_at = float((payload or {}).get("expires_at", 0) or 0)
        if expires_at > now:
            pruned[lease_id] = {
                "user_key": str((payload or {}).get("user_key", "") or ""),
                "expires_at": expires_at,
            }
    return pruned


def _acquire_terminal_session_slot(user: dict) -> str | None:
    if _MAX_CONCURRENT_SESSIONS_PER_USER <= 0:
        return "unlimited"

    key = _terminal_session_key(user)
    lease_id = secrets.token_urlsafe(16)
    now = time.time()

    shared_ok, shared_slot = _shared_state_update(
        _TERMINAL_SESSION_STATE_NAMESPACE,
        lambda: {"sessions": {}},
        lambda state: _mutate_shared_terminal_sessions_acquire(state, key, lease_id, now),
    )
    if shared_ok:
        return f"shared:{shared_slot}" if shared_slot else None

    with _terminal_session_lock:
        if _terminal_session_counts[key] >= _MAX_CONCURRENT_SESSIONS_PER_USER:
            return None
        _terminal_session_counts[key] += 1
    return f"local:{lease_id}"


def _refresh_terminal_session_slot(user: dict, session_slot: str | None) -> None:
    if not _session_slot_is_shared(session_slot):
        return

    key = _terminal_session_key(user)
    lease_id = str(session_slot).split(":", 1)[1]
    now = time.time()
    _shared_state_update(
        _TERMINAL_SESSION_STATE_NAMESPACE,
        lambda: {"sessions": {}},
        lambda state: _mutate_shared_terminal_sessions_refresh(state, key, lease_id, now),
    )


def _release_terminal_session_slot(user: dict, session_slot: str | None) -> None:
    if _MAX_CONCURRENT_SESSIONS_PER_USER <= 0 or not session_slot or session_slot == "unlimited":
        return

    if _session_slot_is_shared(session_slot):
        lease_id = str(session_slot).split(":", 1)[1]
        _shared_state_update(
            _TERMINAL_SESSION_STATE_NAMESPACE,
            lambda: {"sessions": {}},
            lambda state: _mutate_shared_terminal_sessions_release(state, lease_id),
        )
        return

    key = _terminal_session_key(user)
    with _terminal_session_lock:
        remaining = _terminal_session_counts.get(key, 0) - 1
        if remaining > 0:
            _terminal_session_counts[key] = remaining
        else:
            _terminal_session_counts.pop(key, None)


def _mutate_shared_terminal_sessions_acquire(
    state: dict,
    user_key: str,
    lease_id: str,
    now: float,
) -> tuple[dict, str | None]:
    sessions = _purge_shared_terminal_sessions(state.get("sessions"), now)
    active_for_user = sum(
        1
        for payload in sessions.values()
        if payload.get("user_key") == user_key
    )
    if active_for_user >= _MAX_CONCURRENT_SESSIONS_PER_USER:
        return {"sessions": sessions, "updated_at": now}, None

    sessions[lease_id] = {
        "user_key": user_key,
        "expires_at": now + max(1.0, _TERMINAL_SESSION_LEASE_TTL_SEC),
    }
    return {"sessions": sessions, "updated_at": now}, lease_id


def _mutate_shared_terminal_sessions_refresh(
    state: dict,
    user_key: str,
    lease_id: str,
    now: float,
) -> tuple[dict, bool]:
    sessions = _purge_shared_terminal_sessions(state.get("sessions"), now)
    payload = sessions.get(lease_id)
    if payload and payload.get("user_key") == user_key:
        payload["expires_at"] = now + max(1.0, _TERMINAL_SESSION_LEASE_TTL_SEC)
        sessions[lease_id] = payload
        return {"sessions": sessions, "updated_at": now}, True
    return {"sessions": sessions, "updated_at": now}, False


def _mutate_shared_terminal_sessions_release(
    state: dict,
    lease_id: str,
) -> tuple[dict, bool]:
    sessions = _purge_shared_terminal_sessions(state.get("sessions"), time.time())
    removed = lease_id in sessions
    sessions.pop(lease_id, None)
    return {"sessions": sessions, "updated_at": time.time()}, removed


def _sanitize_container_ref(container_ref: object) -> str | None:
    raw_value = str(container_ref or "")
    value = raw_value.strip()
    if not value or raw_value != value or not _CONTAINER_REF_RE.fullmatch(value):
        return None
    return value


def _shell_path_allowed(shell: str) -> bool:
    return shell in _ALLOWED_SHELL_PATHS


def _frame_size_exceeded(payload: bytes | str | None) -> bool:
    if payload is None:
        return False
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return len(payload) > _MAX_INPUT_FRAME_BYTES


def _container_identity_matches(
    container: object,
    *,
    instance_id: str,
    expected_name: str,
    expected_container_id: str = "",
) -> bool:
    actual_name = str(getattr(container, "name", "") or "").lstrip("/")
    if actual_name and actual_name != expected_name:
        return False

    actual_id = str(getattr(container, "id", "") or "")
    if expected_container_id and actual_id and not actual_id.startswith(expected_container_id):
        return False

    labels = getattr(container, "labels", None)
    if labels is None:
        attrs = getattr(container, "attrs", {}) or {}
        labels = attrs.get("Config", {}).get("Labels", {}) or {}

    job_label = str((labels or {}).get("xcelsior.job_id", "") or "")
    if job_label:
        return job_label == instance_id

    return bool(actual_name == expected_name or (expected_container_id and actual_id.startswith(expected_container_id)))


def _ensure_remote_host_key_pinned(host_ip: str) -> None:
    if not host_ip or not _REQUIRE_PINNED_HOST_KEYS:
        return
    if not os.path.exists(_KNOWN_HOSTS_PATH):
        raise DockerException(f"Known hosts file not found: {_KNOWN_HOSTS_PATH}")
    try:
        result = subprocess.run(
            ["ssh-keygen", "-F", host_ip, "-f", _KNOWN_HOSTS_PATH],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError as exc:
        raise DockerException("ssh-keygen is required for remote host key verification") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerException(f"Timed out verifying pinned SSH host key for {host_ip}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise DockerException(f"Remote host key for {host_ip} is not pinned in {_KNOWN_HOSTS_PATH}")


def _check_terminal_access(user: dict, instance: dict | None, *, instance_id: str) -> None:
    if not instance:
        raise HTTPException(404, "Instance not found")
    is_admin = user.get("role") == "admin" or bool(user.get("is_admin"))
    if is_admin:
        return
    job_owner = instance.get("owner", "")
    caller_id = user.get("customer_id", user.get("user_id", ""))
    if job_owner and job_owner != caller_id:
        raise HTTPException(403, "Not authorized")


class TerminalTicketIn(BaseModel):
    instance_id: str = Field(min_length=1, max_length=128)


# -- Control-frame helpers -----------------------------------------------------

async def _send_status(ws: WebSocket, message: str, *, retry: bool = False) -> None:
    try:
        await ws.send_json({"type": "status", "message": message, "retry": retry})
    except Exception:
        pass


async def _send_error(ws: WebSocket, message: str, code: int) -> None:
    try:
        await ws.send_json({"type": "error", "message": message, "code": code})
    except Exception:
        pass


@router.post("/api/terminal/ticket")
def api_terminal_ticket(body: TerminalTicketIn, request: Request) -> dict:
    """Issue a short-lived one-time WebSocket ticket for terminal access."""
    user = _require_auth(request)
    instance = next((j for j in list_jobs() if j["job_id"] == body.instance_id), None)
    _check_terminal_access(user, instance, instance_id=body.instance_id)
    ticket = _issue_ws_ticket(
        user,
        request=request,
        purpose="terminal",
        target=body.instance_id,
    )
    return {
        "ok": True,
        "ticket": ticket["ticket"],
        "expires_in": int(max(0, ticket["expires_at"] - time.time())),
    }


# -- WebSocket endpoint --------------------------------------------------------

@router.websocket("/ws/terminal/{instance_id}")
async def ws_terminal(websocket: WebSocket, instance_id: str) -> None:
    """Interactive terminal session for a running GPU instance."""
    client_ip = _get_ws_client_ip(websocket)

    if not _validate_ws_origin(
        websocket,
        require_for_cookie_auth=True,
        allow_query_token=False,
    ):
        _conn_errors.labels(reason="origin_rejected").inc()  # type: ignore[attr-defined]
        log.warning("terminal.ws.origin_rejected ip=%s instance=%s", client_ip, instance_id)
        await websocket.close(code=1008, reason="Invalid origin")
        return

    if not _check_ws_connect_rate_limit(websocket, bucket="terminal"):
        _conn_errors.labels(reason="connect_rate_limited").inc()  # type: ignore[attr-defined]
        log.warning("terminal.ws.rate_limited ip=%s instance=%s", client_ip, instance_id)
        await websocket.close(code=4429, reason="Connection rate limit exceeded")
        return

    ticket = websocket.query_params.get("ticket", "").strip()
    if ticket:
        user = _consume_ws_ticket(
            ticket,
            websocket,
            purpose="terminal",
            target=instance_id,
        )
    else:
        user = _validate_ws_auth(websocket, allow_query_token=False)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    session_slot = _acquire_terminal_session_slot(user)
    if not session_slot:
        _conn_errors.labels(reason="session_limit").inc()  # type: ignore[attr-defined]
        log.warning(
            "terminal.ws.session_limit user=%s ip=%s instance=%s",
            _terminal_session_key(user),
            client_ip,
            instance_id,
        )
        await websocket.close(code=4429, reason="Too many concurrent terminal sessions")
        return

    session_opened = False
    try:
        await websocket.accept()

        instance = next((j for j in list_jobs() if j["job_id"] == instance_id), None)
        try:
            _check_terminal_access(user, instance, instance_id=instance_id)
        except HTTPException as exc:
            ws_code = 4004 if exc.status_code == 404 else 4003
            if exc.status_code == 404:
                _conn_errors.labels(reason="not_found").inc()  # type: ignore[attr-defined]
            elif exc.status_code == 403:
                _conn_errors.labels(reason="unauthorized").inc()  # type: ignore[attr-defined]
            await _send_error(websocket, str(exc.detail), ws_code)
            await websocket.close(code=ws_code)
            return

        if instance.get("status") not in ("running", "starting"):
            await _send_error(
                websocket,
                f"Instance is '{instance.get('status', 'unknown')}', not running",
                4003,
            )
            await websocket.close(code=4003)
            _conn_errors.labels(reason="not_running").inc()  # type: ignore[attr-defined]
            return

        status_poll_max = 90
        status_poll_sec = 2.0
        if instance.get("status") == "starting":
            for s_attempt in range(status_poll_max):
                await _send_status(
                    websocket,
                    f"Image pulling / container starting… ({s_attempt + 1})",
                    retry=True,
                )
                await asyncio.sleep(status_poll_sec)
                instance = next((j for j in list_jobs() if j["job_id"] == instance_id), None)
                if not instance:
                    await _send_error(websocket, "Instance disappeared", 4004)
                    await websocket.close(code=4004)
                    return
                cur = instance.get("status", "")
                if cur == "running":
                    break
                if cur in ("failed", "cancelled", "terminated"):
                    await _send_error(websocket, f"Instance {cur} during startup", 4003)
                    await websocket.close(code=4003)
                    return
            else:
                await _send_error(
                    websocket,
                    "Instance still starting after 3 min — please retry",
                    4410,
                )
                await websocket.close(code=4410)
                return

        host_id = instance.get("host_id", "")
        container_ref = _sanitize_container_ref(
            instance.get("container_name") or f"xcl-{instance_id}"
        )
        if not container_ref:
            await _send_error(websocket, "Invalid container reference", 4003)
            await websocket.close(code=4003)
            _conn_errors.labels(reason="invalid_container_ref").inc()  # type: ignore[attr-defined]
            return

        host_ip = str(instance.get("host_ip") or "").strip()
        if not host_ip and host_id:
            hmap = {h["host_id"]: h for h in list_hosts(active_only=False)}
            host_rec = hmap.get(host_id)
            host_ip = str(host_rec.get("ip", "") if host_rec else "").strip()

        if host_ip and not _HOST_IP_RE.fullmatch(host_ip):
            await _send_error(websocket, "Invalid host address", 4003)
            await websocket.close(code=4003)
            return

        shell = str(instance.get("shell") or "/bin/bash").strip()
        if not _shell_path_allowed(shell):
            await _send_error(websocket, "Unsupported shell path", 4003)
            await websocket.close(code=4003)
            _conn_errors.labels(reason="invalid_shell").inc()  # type: ignore[attr-defined]
            return

        is_remote = bool(host_ip) and host_ip not in ("127.0.0.1", "localhost", "0.0.0.0")
        loop = asyncio.get_running_loop()
        if is_remote:
            try:
                _ensure_remote_host_key_pinned(host_ip)
            except DockerException as exc:
                await _send_error(websocket, str(exc), 4003)
                await websocket.close(code=4003)
                _conn_errors.labels(reason="ssh_host_key_unpinned").inc()  # type: ignore[attr-defined]
                return

        async def _docker_call(callable_obj, timeout_sec: float):
            return await asyncio.wait_for(
                loop.run_in_executor(None, callable_obj),
                timeout=timeout_sec,
            )

        # Fast-fail if the remote host is unreachable (avoids 30 s timeout)
        # Fast-fail if the remote host is unreachable (avoids 30 s Docker timeout).
        # Retry once after 1 s to tolerate brief network blips.
        if is_remote:
            _reach_ok = False
            for _ping_attempt in range(2):
                _reach_ok = await loop.run_in_executor(
                    None,
                    lambda: subprocess.call(
                        ["ping", "-c", "1", "-W", "3", host_ip],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    ) == 0,
                )
                if _reach_ok:
                    break
                # Brief pause before retry
                await asyncio.sleep(1)
            if not _reach_ok:
                log.warning("TERMINAL host %s unreachable (ping failed)", host_ip)
                await _send_error(
                    websocket,
                    f"GPU host {host_ip} is offline — check Headscale mesh connectivity",
                    4410,
                )
                await websocket.close(code=4410)
                _conn_errors.labels(reason="host_unreachable").inc()  # type: ignore[attr-defined]
                return

        for attempt in range(_CONTAINER_POLL_MAX_ATTEMPTS):
            exists = await loop.run_in_executor(
                None,
                _container_exists,
                container_ref,
                host_ip if is_remote else None,
            )
            if exists:
                break
            await _send_status(
                websocket,
                f"Container starting… ({attempt + 1}/{_CONTAINER_POLL_MAX_ATTEMPTS})",
                retry=True,
            )
            await asyncio.sleep(_CONTAINER_POLL_INTERVAL_SEC)
        else:
            await _send_error(
                websocket,
                "Container did not start within 30 s — please retry in a moment",
                4410,
            )
            await websocket.close(code=4410)
            _conn_errors.labels(reason="container_not_found").inc()  # type: ignore[attr-defined]
            return

        has_tmux = await loop.run_in_executor(
            None,
            _tmux_available,
            container_ref,
            host_ip if is_remote else None,
        )

        tmux_session = f"xcl-{instance_id}"
        exec_socket = None
        docker_cl = None
        exec_id = None
        raw_sock = None

        try:
            docker_cl = await _docker_call(
                lambda: _docker_client(host_ip if is_remote else None),
                _DOCKER_CLIENT_TIMEOUT_SEC,
            )
            container = await _docker_call(
                lambda: docker_cl.containers.get(container_ref),
                _DOCKER_CONTAINER_GET_TIMEOUT_SEC,
            )
            if not _container_identity_matches(
                container,
                instance_id=instance_id,
                expected_name=container_ref,
                expected_container_id=str(instance.get("container_id", "") or ""),
            ):
                raise DockerException("Resolved container identity does not match the instance record")

            env_vars = {"TERM": "xterm-256color", "PS1": "\\u@xcelsior:\\w\\$ "}
            if has_tmux:
                env_vars["TMUX_SESSION"] = tmux_session
                exec_cmd = ["tmux", "new-session", "-A", "-s", tmux_session, shell]
            else:
                exec_cmd = [shell]

            exec_id = await _docker_call(
                lambda: docker_cl.api.exec_create(
                    container.id,
                    exec_cmd,
                    stdin=True,
                    tty=True,
                    environment=env_vars,
                )["Id"],
                _DOCKER_EXEC_CREATE_TIMEOUT_SEC,
            )
            exec_socket = await _docker_call(
                lambda: docker_cl.api.exec_start(
                    exec_id,
                    socket=True,
                    tty=True,
                ),
                _DOCKER_EXEC_START_TIMEOUT_SEC,
            )
        except (asyncio.TimeoutError, NotFound, APIError, DockerException, OSError) as exc:
            if docker_cl is not None:
                try:
                    docker_cl.close()
                except Exception:
                    pass
            await _send_error(websocket, f"Failed to spawn terminal: {exc}", 4500)
            await websocket.close(code=4500)
            _conn_errors.labels(reason="spawn_failed").inc()  # type: ignore[attr-defined]
            return

        raw_sock = exec_socket._sock
        raw_sock.setblocking(False)

        log.info(
            "terminal.session.open user=%s ip=%s instance=%s container=%s tmux=%s remote=%s",
            user.get("email", "?"),
            client_ip,
            instance_id,
            container_ref,
            has_tmux,
            is_remote,
        )
        _active_sessions.inc()  # type: ignore[attr-defined]
        session_opened = True

        session_start = time.monotonic()
        last_input_ts = time.monotonic()
        closed = False
        idle_warned = False
        bytes_this_second = 0
        rate_window_start = time.monotonic()
        malformed_control_frames = 0

        await _send_status(
            websocket,
            f"Connected to {instance.get('name', instance_id)}",
            retry=False,
        )

        def _resize_exec(cols: int, rows: int) -> None:
            try:
                docker_cl.api.exec_resize(exec_id, height=rows, width=cols)
            except (APIError, DockerException, OSError):
                pass

        async def _stdout_relay() -> None:
            nonlocal closed, bytes_this_second, idle_warned, rate_window_start

            while not closed:
                session_age = time.monotonic() - session_start
                if session_age >= _MAX_SESSION_LIFETIME_SEC:
                    await _send_error(websocket, "Session reached maximum lifetime", 4408)
                    _disconnections.labels(cause="max_lifetime").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

                idle_time = time.monotonic() - last_input_ts
                if idle_time >= _SESSION_TIMEOUT_SEC:
                    await _send_error(websocket, "Session timed out due to inactivity", 4408)
                    _disconnections.labels(cause="idle_timeout").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

                if not idle_warned and idle_time >= _IDLE_WARN_THRESHOLD_SEC:
                    idle_warned = True
                    await _send_status(websocket, "Session closing in 5 min due to inactivity")

                try:
                    chunk = await asyncio.wait_for(
                        loop.sock_recv(raw_sock, _EXEC_READ_CHUNK),
                        timeout=_EXEC_READ_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    continue
                except (OSError, ConnectionError):
                    _disconnections.labels(cause="process_exit").inc()  # type: ignore[attr-defined]
                    break

                if not chunk:
                    _disconnections.labels(cause="eof").inc()  # type: ignore[attr-defined]
                    break

                now = time.monotonic()
                if now - rate_window_start >= 1.0:
                    bytes_this_second = 0
                    rate_window_start = now
                bytes_this_second += len(chunk)
                if bytes_this_second > _RATE_LIMIT_BYTES_PER_SEC:
                    await asyncio.sleep(0.05)

                try:
                    await websocket.send_bytes(chunk)
                    _bytes_sent.inc(len(chunk))  # type: ignore[attr-defined]
                except (WebSocketDisconnect, RuntimeError):
                    _disconnections.labels(cause="client_disconnect").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

        async def _stdin_relay() -> None:
            nonlocal closed, idle_warned, last_input_ts, malformed_control_frames

            while not closed:
                session_age = time.monotonic() - session_start
                if session_age >= _MAX_SESSION_LIFETIME_SEC:
                    await _send_error(websocket, "Session reached maximum lifetime", 4408)
                    _disconnections.labels(cause="max_lifetime").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

                if time.monotonic() - last_input_ts > _SESSION_TIMEOUT_SEC:
                    await _send_error(websocket, "Session timed out due to inactivity", 4408)
                    _disconnections.labels(cause="idle_timeout").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

                try:
                    raw = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=_STDIN_RECV_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_json({"type": "pong", "ts": time.time()})
                    except (WebSocketDisconnect, RuntimeError):
                        _disconnections.labels(cause="client_disconnect").inc()  # type: ignore[attr-defined]
                        closed = True
                    continue
                except WebSocketDisconnect:
                    _disconnections.labels(cause="client_disconnect").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

                if raw.get("type") == "websocket.disconnect":
                    _disconnections.labels(cause="client_disconnect").inc()  # type: ignore[attr-defined]
                    closed = True
                    break

                if raw.get("type") != "websocket.receive":
                    continue

                payload_bytes: Optional[bytes] = raw.get("bytes")
                if payload_bytes is not None:
                    if _frame_size_exceeded(payload_bytes):
                        await _send_error(websocket, "Input frame too large", 1009)
                        await websocket.close(code=1009)
                        _conn_errors.labels(reason="frame_too_large").inc()  # type: ignore[attr-defined]
                        closed = True
                        break
                    if payload_bytes:
                        try:
                            await loop.sock_sendall(raw_sock, payload_bytes)
                            _bytes_recv.inc(len(payload_bytes))  # type: ignore[attr-defined]
                            last_input_ts = time.monotonic()
                            idle_warned = False
                        except (OSError, ConnectionError):
                            closed = True
                    continue

                payload_text: Optional[str] = raw.get("text")
                if payload_text is None:
                    continue
                if _frame_size_exceeded(payload_text):
                    await _send_error(websocket, "Input frame too large", 1009)
                    await websocket.close(code=1009)
                    _conn_errors.labels(reason="frame_too_large").inc()  # type: ignore[attr-defined]
                    closed = True
                    break
                if not payload_text:
                    continue

                try:
                    msg: dict = json.loads(payload_text)
                except (json.JSONDecodeError, ValueError):
                    malformed_control_frames += 1
                    if malformed_control_frames >= _MAX_MALFORMED_CONTROL_FRAMES:
                        await _send_error(websocket, "Too many malformed control frames", 1008)
                        await websocket.close(code=1008)
                        _conn_errors.labels(reason="malformed_control_frames").inc()  # type: ignore[attr-defined]
                        closed = True
                        break
                    continue

                if not isinstance(msg, dict):
                    malformed_control_frames += 1
                    if malformed_control_frames >= _MAX_MALFORMED_CONTROL_FRAMES:
                        await _send_error(websocket, "Too many malformed control frames", 1008)
                        await websocket.close(code=1008)
                        _conn_errors.labels(reason="malformed_control_frames").inc()  # type: ignore[attr-defined]
                        closed = True
                        break
                    continue

                msg_type = msg.get("type", "")
                if msg_type == "input":
                    data = msg.get("data", "")
                    if not isinstance(data, str):
                        malformed_control_frames += 1
                        if malformed_control_frames >= _MAX_MALFORMED_CONTROL_FRAMES:
                            await _send_error(websocket, "Too many malformed control frames", 1008)
                            await websocket.close(code=1008)
                            _conn_errors.labels(reason="malformed_control_frames").inc()  # type: ignore[attr-defined]
                            closed = True
                            break
                        continue
                    if _frame_size_exceeded(data):
                        await _send_error(websocket, "Input frame too large", 1009)
                        await websocket.close(code=1009)
                        _conn_errors.labels(reason="frame_too_large").inc()  # type: ignore[attr-defined]
                        closed = True
                        break
                    malformed_control_frames = 0
                    if data:
                        try:
                            encoded = data.encode("utf-8")
                            await loop.sock_sendall(raw_sock, encoded)
                            _bytes_recv.inc(len(encoded))  # type: ignore[attr-defined]
                            last_input_ts = time.monotonic()
                            idle_warned = False
                        except (OSError, ConnectionError):
                            closed = True

                elif msg_type == "resize":
                    try:
                        cols = min(_MAX_RESIZE_COLS, max(1, int(msg.get("cols", 80))))
                        rows = min(_MAX_RESIZE_ROWS, max(1, int(msg.get("rows", 24))))
                        malformed_control_frames = 0
                        await loop.run_in_executor(None, _resize_exec, cols, rows)
                    except (TypeError, ValueError):
                        malformed_control_frames += 1
                        if malformed_control_frames >= _MAX_MALFORMED_CONTROL_FRAMES:
                            await _send_error(websocket, "Too many malformed control frames", 1008)
                            await websocket.close(code=1008)
                            _conn_errors.labels(reason="malformed_control_frames").inc()  # type: ignore[attr-defined]
                            closed = True
                            break

                elif msg_type == "ping":
                    try:
                        malformed_control_frames = 0
                        await websocket.send_json({"type": "pong", "ts": time.time()})
                    except (WebSocketDisconnect, RuntimeError):
                        closed = True
                else:
                    malformed_control_frames += 1
                    if malformed_control_frames >= _MAX_MALFORMED_CONTROL_FRAMES:
                        await _send_error(websocket, "Too many malformed control frames", 1008)
                        await websocket.close(code=1008)
                        _conn_errors.labels(reason="malformed_control_frames").inc()  # type: ignore[attr-defined]
                        closed = True
                        break

        async def _session_slot_keepalive() -> None:
            if not _session_slot_is_shared(session_slot) or _TERMINAL_SESSION_LEASE_REFRESH_SEC <= 0:
                return
            while not closed:
                await asyncio.sleep(_TERMINAL_SESSION_LEASE_REFRESH_SEC)
                if closed:
                    break
                await loop.run_in_executor(
                    None,
                    _refresh_terminal_session_slot,
                    user,
                    session_slot,
                )

        exit_code = 0
        keepalive_task = None
        try:
            if _session_slot_is_shared(session_slot) and _TERMINAL_SESSION_LEASE_REFRESH_SEC > 0:
                keepalive_task = asyncio.ensure_future(_session_slot_keepalive())
            stdout_task = asyncio.ensure_future(_stdout_relay())
            stdin_task = asyncio.ensure_future(_stdin_relay())
            _done, pending = await asyncio.wait(
                {stdout_task, stdin_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        finally:
            closed = True
            if keepalive_task is not None:
                keepalive_task.cancel()
                await asyncio.gather(keepalive_task, return_exceptions=True)

            if not has_tmux and docker_cl is not None and exec_id is not None:
                try:
                    inspect = await asyncio.wait_for(
                        loop.run_in_executor(None, docker_cl.api.exec_inspect, exec_id),
                        timeout=_EXEC_INSPECT_TIMEOUT_SEC,
                    )
                    exit_code = inspect.get("ExitCode") or 0
                except (asyncio.TimeoutError, APIError, DockerException, OSError, Exception):
                    pass

            if has_tmux and raw_sock is not None:
                try:
                    await loop.sock_sendall(raw_sock, b"\x02d")
                    await asyncio.sleep(0.1)
                except (OSError, ConnectionError):
                    pass

            if exec_socket is not None:
                try:
                    exec_socket.close()
                except Exception:
                    pass

            if docker_cl is not None:
                try:
                    docker_cl.close()
                except Exception:
                    pass

            if session_opened:
                _active_sessions.dec()  # type: ignore[attr-defined]
                log.info(
                    "terminal.session.close user=%s ip=%s instance=%s duration=%.1fs",
                    user.get("email", "?"),
                    client_ip,
                    instance_id,
                    time.monotonic() - session_start,
                )

            try:
                await websocket.send_json({"type": "exit", "code": exit_code})
                await websocket.close()
            except Exception:
                pass
    finally:
        _release_terminal_session_slot(user, session_slot)
