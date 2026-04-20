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
import socket
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

# -- Prometheus metrics --------------------------------------------------------

from prometheus_client import Counter, Gauge

_active_sessions: object = Gauge(
    "xcelsior_terminal_active_sessions",
    "Number of active terminal WebSocket sessions",
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
_host_probe_failure_total: object = Counter(
    "xcelsior_terminal_host_probe_failure_total",
    "Failed async TCP probes to GPU host SSH port by reason",
    ["host_id", "reason"],
)


router = APIRouter()

# -- Constants -----------------------------------------------------------------

_SESSION_TIMEOUT_SEC: int = 14_400
_IDLE_WARN_THRESHOLD_SEC: int = _SESSION_TIMEOUT_SEC - 300
_RATE_LIMIT_BYTES_PER_SEC: int = 524_288
_EXEC_READ_CHUNK: int = 4096
_CONTAINER_POLL_INTERVAL_SEC: float = 1.0
_CONTAINER_POLL_MAX_ATTEMPTS: int = 15  # kept for backwards compat (unused after refactor)
_CONTAINER_POLL_BUDGET_SEC: float = 60.0  # total wait for container to appear
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
def _resolve_known_hosts_path() -> str:
    """Find or create a writable known_hosts file. Tries in order:
    1. Explicit env override
    2. /data/.ssh/known_hosts (persistent volume)
    3. ~/.ssh/known_hosts (home dir)
    4. /tmp/.ssh/known_hosts (always writable fallback)
    """
    explicit = os.environ.get("XCELSIOR_TERMINAL_KNOWN_HOSTS_PATH")
    if explicit:
        candidates = [explicit]
    else:
        candidates = [
            "/data/.ssh/known_hosts",
            os.path.expanduser("~/.ssh/known_hosts"),
            "/tmp/.ssh/known_hosts",
        ]
    for path in candidates:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Test write access
            with open(path, "a") as f:
                f.flush()
            return path
        except OSError:
            continue
    # Last resort: /tmp is always writable
    fallback = "/tmp/.ssh/known_hosts"
    os.makedirs("/tmp/.ssh", exist_ok=True)
    open(fallback, "a").close()
    return fallback

_KNOWN_HOSTS_PATH: str = _resolve_known_hosts_path()

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

_ssh_identity_configured = False
_ssh_identity_lock = threading.Lock()


def _ensure_ssh_identity(ssh_key_path: str) -> None:
    """Make *ssh_key_path* discoverable by the system `ssh` client.

    The Docker SDK's ssh:// transport (with use_ssh_client=True) spawns
    /usr/bin/ssh but does NOT pass `-i <keyfile>`. Without this helper, ssh
    falls back to ~/.ssh/id_* which doesn't exist in our API container,
    causing `BrokenPipeError` during the initial API version handshake.

    We write a minimal `~/.ssh/config` that binds the given IdentityFile
    to any host — idempotent, thread-safe, done once per process.
    """
    global _ssh_identity_configured
    if _ssh_identity_configured:
        return
    with _ssh_identity_lock:
        if _ssh_identity_configured:
            return
        try:
            if not ssh_key_path or not os.path.isfile(ssh_key_path):
                return
            ssh_dir = os.path.expanduser("~/.ssh")
            os.makedirs(ssh_dir, mode=0o700, exist_ok=True)
            cfg_path = os.path.join(ssh_dir, "config")
            marker = "# xcelsior-terminal-identity"
            existing = ""
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        existing = f.read()
                except OSError:
                    existing = ""
            if marker not in existing:
                block = (
                    f"\n{marker}\n"
                    "Host *\n"
                    f"    IdentityFile {ssh_key_path}\n"
                    "    IdentitiesOnly yes\n"
                    "    BatchMode yes\n"
                    "    ConnectTimeout 10\n"
                )
                try:
                    with open(cfg_path, "a", encoding="utf-8") as f:
                        f.write(block)
                    os.chmod(cfg_path, 0o600)
                except OSError as e:
                    log.warning("TERMINAL could not write ~/.ssh/config: %s", e)
                    return
            _ssh_identity_configured = True
        except Exception as e:  # pragma: no cover - defensive
            log.warning("TERMINAL _ensure_ssh_identity failed: %s", e)


_paramiko_host_keys_patched = False


def _patch_paramiko_host_keys() -> None:
    """Ensure paramiko's SSHClient auto-loads ~/.ssh/known_hosts on connect.

    The Docker SDK (use_ssh_client=False) instantiates `paramiko.SSHClient`
    without calling `load_system_host_keys()`, which causes
    `SSHException: Server 'X' not found in known_hosts` even when we've
    already pinned the host key via ssh-keyscan. We patch `connect` once
    per process so every SDK-created client picks up the system file.
    """
    global _paramiko_host_keys_patched
    if _paramiko_host_keys_patched:
        return
    try:
        import paramiko  # type: ignore
    except Exception:
        return
    _orig_connect = paramiko.SSHClient.connect

    def _patched_connect(self, *args, **kwargs):  # type: ignore[override]
        try:
            self.load_system_host_keys()
        except Exception:
            pass
        try:
            kh = os.path.expanduser("~/.ssh/known_hosts")
            if os.path.isfile(kh):
                self.load_host_keys(kh)
        except Exception:
            pass
        return _orig_connect(self, *args, **kwargs)

    paramiko.SSHClient.connect = _patched_connect  # type: ignore[assignment]
    _paramiko_host_keys_patched = True


# -- Docker client pool --------------------------------------------------------
# One persistent Docker SDK client per (ssh_user, host_ip). This eliminates the
# ~60 SSH handshakes per terminal open that the previous "new client per call"
# model produced (container probe loop × 60, plus probe, exec attach, tmux
# check, etc.). The SSH transport stays up with paramiko keepalive(20), and is
# evicted + rebuilt on any exception so a dead tunnel can't get stuck.
_docker_client_cache: dict[str, "docker.DockerClient"] = {}
_docker_client_cache_lock = threading.RLock()
_PARAMIKO_KEEPALIVE_INTERVAL_SEC: int = int(
    os.environ.get("XCELSIOR_TERMINAL_PARAMIKO_KEEPALIVE_SEC", "20")
)

# -- Probe / capability result cache ------------------------------------------
# Caches *positive* container-probe and tmux-availability results for a short
# TTL so that back-to-back calls within a WS session (probe → tmux_check →
# exec_create) don't each pay a full SSH+exec round trip, and so a quick
# client reconnect inside the TTL skips re-verification.
#
# NotFound and unreachable results are NEVER cached — those must always be
# re-probed so a freshly-created container is picked up without delay.
# Entries are invalidated whenever the underlying Docker client is evicted
# (e.g. the SSH transport died), so a rebuilt daemon reference always
# re-verifies its containers.
_PROBE_CACHE_TTL_SEC: float = float(
    os.environ.get("XCELSIOR_TERMINAL_PROBE_CACHE_TTL_SEC", "30")
)
# Key: (cache_key-or-"local", container_ref)  →  (expires_at, result_dict)
_probe_cache: dict[tuple[str, str], tuple[float, dict]] = {}
# Key: (cache_key-or-"local", container_ref)  →  (expires_at, tmux_present)
_tmux_cache: dict[tuple[str, str], tuple[float, bool]] = {}
_probe_cache_lock = threading.RLock()


def _probe_cache_scope(host_ip: str | None) -> str:
    return _docker_client_cache_key(host_ip, SSH_USER) or "local"


def _invalidate_probe_cache(scope: str) -> None:
    """Drop every cached probe/tmux entry whose scope matches *scope*.

    Called from ``_evict_docker_client`` so a dead SSH tunnel never leaves
    stale positive results behind."""
    with _probe_cache_lock:
        for key in [k for k in _probe_cache if k[0] == scope]:
            _probe_cache.pop(key, None)
        for key in [k for k in _tmux_cache if k[0] == scope]:
            _tmux_cache.pop(key, None)


def _paramiko_transport_from_client(cl: "docker.DockerClient"):
    """Best-effort: reach into the docker SDK to get the paramiko Transport.

    Returns None if the SDK internals change, which is fine — keepalive/health
    is a nice-to-have, never required for correctness.
    """
    try:
        # SSHHTTPAdapter instance; docker-py registers it for ssh:// base URLs.
        adapter = getattr(cl.api, "_custom_adapter", None)
        if adapter is None:
            return None
        ssh_client = getattr(adapter, "ssh_client", None)
        if ssh_client is None:
            return None
        return ssh_client.get_transport()
    except Exception:
        return None


def _docker_client_alive(cl: "docker.DockerClient") -> bool:
    """Cheap liveness probe: just check the paramiko transport is still up.

    We deliberately DO NOT issue a Docker API call here — that would cost an
    HTTP round-trip over SSH, negating the pool. Transport.is_active() is a
    local flag check. If we're wrong and the transport looks active but the
    next API call fails, callers will evict + rebuild.
    """
    transport = _paramiko_transport_from_client(cl)
    if transport is None:
        # Can't inspect — assume alive; real failures will cause eviction.
        return True
    try:
        return bool(transport.is_active())
    except Exception:
        return False


def _evict_docker_client(cache_key: str) -> None:
    """Remove and close a cached Docker client. Safe to call repeatedly.

    Also invalidates every cached probe/tmux result scoped to this client
    so a stale positive can never survive a tunnel rebuild.
    """
    with _docker_client_cache_lock:
        cl = _docker_client_cache.pop(cache_key, None)
    _invalidate_probe_cache(cache_key)
    if cl is not None:
        try:
            cl.close()
        except Exception:
            pass


def _docker_client_cache_key(host_ip: str | None, ssh_user: str) -> str | None:
    if not host_ip or host_ip in ("127.0.0.1", "localhost", "0.0.0.0"):
        return None
    return f"{ssh_user}@{host_ip}"


def _docker_client(
    host_ip: str | None = None,
    ssh_user: str = SSH_USER,
    ssh_key_path: str = SSH_KEY_PATH,
) -> docker.DockerClient:
    """Return a (pooled) Docker SDK client for the given host.

    - For LOCAL (no host_ip), returns a fresh docker.from_env() each call —
      caller owns closing it. These clients are cheap (unix socket).
    - For REMOTE hosts, returns a pooled paramiko-backed ssh:// client that
      callers MUST NOT close — use `_evict_docker_client(cache_key)` to drop
      it on error. Cache is keyed by (ssh_user, host_ip).

    The remote client is built lazily on first use or after eviction. We set
    paramiko keepalive to keep NAT/firewall state warm.
    """
    cache_key = _docker_client_cache_key(host_ip, ssh_user)
    if cache_key is None:
        # Local daemon — unix socket is cheap, keep the simple per-call pattern.
        return docker.from_env()

    with _docker_client_cache_lock:
        cached = _docker_client_cache.get(cache_key)
        if cached is not None and _docker_client_alive(cached):
            return cached
        if cached is not None:
            # Dead transport — drop it and build fresh below.
            try:
                cached.close()
            except Exception:
                pass
            _docker_client_cache.pop(cache_key, None)

        # Build a fresh client. Prerequisites live on disk / global paramiko.
        _ensure_remote_host_key_pinned(host_ip)
        _ensure_ssh_identity(ssh_key_path)
        _patch_paramiko_host_keys()
        # paramiko-native transport (use_ssh_client=False). The subprocess-ssh
        # path breaks with BrokenPipeError during `docker system dial-stdio`.
        cl = docker.DockerClient(
            base_url=f"ssh://{ssh_user}@{host_ip}",
            use_ssh_client=False,
            timeout=_REMOTE_DOCKER_TIMEOUT_SEC,
        )
        # Enable TCP/SSH keepalive on the underlying paramiko transport so
        # long-idle tunnels don't silently die behind NAT/firewalls.
        transport = _paramiko_transport_from_client(cl)
        if transport is not None:
            try:
                transport.set_keepalive(_PARAMIKO_KEEPALIVE_INTERVAL_SEC)
            except Exception:
                pass
        _docker_client_cache[cache_key] = cl
        return cl


# -- Preflight helpers ---------------------------------------------------------

def _container_exists(container_ref: str, host_ip: str | None = None) -> bool:
    """Return True when *container_ref* exists on the Docker daemon."""
    return _container_probe(container_ref, host_ip)["found"]


def _container_probe(container_ref: str, host_ip: str | None = None) -> dict:
    """Probe the Docker daemon for *container_ref*.

    Returns a dict with keys:
        found  (bool)          — True iff the container exists
        reason (str|None)      — "ok" | "not_found" | "unreachable"
        error  (str|None)      — human-readable error detail (if any)

    Uses the pooled remote client for remote hosts — does NOT close it. On
    unreachable errors we evict the pooled entry so the next call rebuilds
    the SSH tunnel.

    Positive results are cached for ``_PROBE_CACHE_TTL_SEC`` seconds so the
    probe→tmux_available→exec_create sequence inside a single WS session
    pays at most one SSH round trip per step. Negative and error results
    are never cached.
    """
    scope = _probe_cache_scope(host_ip)
    cache_key = _docker_client_cache_key(host_ip, SSH_USER)
    now = time.monotonic()
    with _probe_cache_lock:
        entry = _probe_cache.get((scope, container_ref))
        if entry is not None and entry[0] > now:
            return entry[1]
    cl = None
    try:
        cl = _docker_client(host_ip)
        cl.containers.get(container_ref)
        result = {"found": True, "reason": "ok", "error": None}
        with _probe_cache_lock:
            _probe_cache[(scope, container_ref)] = (now + _PROBE_CACHE_TTL_SEC, result)
        return result
    except NotFound:
        return {"found": False, "reason": "not_found", "error": None}
    except Exception as e:
        log.warning(
            "TERMINAL _container_probe(%s, host=%s) unreachable: %s: %s",
            container_ref, host_ip, type(e).__name__, e,
        )
        if cache_key is not None:
            _evict_docker_client(cache_key)
        return {
            "found": False,
            "reason": "unreachable",
            "error": f"{type(e).__name__}: {e}",
        }
    finally:
        # Close only unpooled (local) clients. Pooled remote clients stay up.
        if cl is not None and cache_key is None:
            try:
                cl.close()
            except Exception:
                pass


def _tmux_available(container_ref: str, host_ip: str | None = None) -> bool:
    """Return True when tmux is on PATH inside *container_ref*.

    Result is cached for ``_PROBE_CACHE_TTL_SEC`` — tmux presence doesn't
    change for the lifetime of a container image, so this is safe and
    eliminates an SSH round-trip on every reconnect. Cache scope is tied
    to the Docker client; ``_evict_docker_client`` invalidates it.
    """
    scope = _probe_cache_scope(host_ip)
    cache_key = _docker_client_cache_key(host_ip, SSH_USER)
    now = time.monotonic()
    with _probe_cache_lock:
        entry = _tmux_cache.get((scope, container_ref))
        if entry is not None and entry[0] > now:
            return entry[1]
    cl = None
    try:
        cl = _docker_client(host_ip)
        container = cl.containers.get(container_ref)
        result = container.exec_run("which tmux")
        has_tmux = result.exit_code == 0
        with _probe_cache_lock:
            _tmux_cache[(scope, container_ref)] = (now + _PROBE_CACHE_TTL_SEC, has_tmux)
        return has_tmux
    except NotFound:
        return False
    except Exception:
        if cache_key is not None:
            _evict_docker_client(cache_key)
        return False
    finally:
        if cl is not None and cache_key is None:
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
    # Paramiko (used by docker SDK with use_ssh_client=False) only reads the
    # user's ~/.ssh/known_hosts via load_system_host_keys(). Maintain BOTH the
    # primary managed path and the home-dir path so subprocess-ssh and
    # paramiko transports stay in sync.
    targets = [_KNOWN_HOSTS_PATH]
    home_kh = os.path.expanduser("~/.ssh/known_hosts")
    if home_kh not in targets:
        targets.append(home_kh)
    for path in targets:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                open(path, "a").close()
        except OSError:
            continue
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

    pinned_key_block: str | None = None
    if result.returncode == 0 and result.stdout.strip():
        pinned_key_block = result.stdout
    else:
        # TOFU: auto-scan and pin the key on first encounter (Tailscale already
        # authenticates both endpoints via WireGuard, so this is safe).
        try:
            scan = subprocess.run(
                ["ssh-keyscan", "-H", host_ip],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if scan.returncode == 0 and scan.stdout.strip():
                pinned_key_block = scan.stdout
                log.info("TERMINAL: auto-pinned host key for %s", host_ip)
            else:
                raise DockerException(
                    f"ssh-keyscan failed for {host_ip}: {scan.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            raise DockerException(f"Timed out scanning SSH host key for {host_ip}")
        except OSError as exc:
            raise DockerException(f"Failed to pin host key for {host_ip}: {exc}")

    # Mirror the pinned key into every target that doesn't already have it.
    if pinned_key_block:
        for path in targets:
            try:
                # Skip if already present
                check = subprocess.run(
                    ["ssh-keygen", "-F", host_ip, "-f", path],
                    capture_output=True, text=True, timeout=5,
                )
                if check.returncode == 0 and check.stdout.strip():
                    continue
                with open(path, "a") as f:
                    f.write(pinned_key_block)
            except (OSError, subprocess.TimeoutExpired) as exc:
                log.warning("TERMINAL: could not mirror host key to %s: %s", path, exc)


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

        # Allow WS connections for any active (non-terminal) status. For
        # queued/assigned/starting we stream lifecycle log lines while polling
        # for the running transition — the UX is "you can open a terminal as
        # soon as you click launch; it'll auto-connect when the container is
        # up" rather than "connection refused until fully booted". Only
        # terminal states (failed/completed/cancelled/terminated) get rejected.
        if instance.get("status") not in ("running", "starting", "queued", "assigned"):
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
        if instance.get("status") in ("queued", "assigned", "starting"):
            # Budget: 30 min (600 polls × 3s) for large image pulls (e.g. 20GB
            # pytorch). Workers advance queued→assigned→starting→running; if we
            # reach 30 min total without running, we drop. The reaper will also
            # clean up stale jobs on its cadence.
            status_poll_max = 600
            status_poll_sec = 3.0
            prev_status = instance.get("status")
            for s_attempt in range(status_poll_max):
                await _send_status(
                    websocket,
                    f"{prev_status.capitalize()}… ({s_attempt + 1})",
                    retry=True,
                )
                await asyncio.sleep(status_poll_sec)
                instance = next((j for j in list_jobs() if j["job_id"] == instance_id), None)
                if not instance:
                    await _send_error(websocket, "Instance disappeared", 4004)
                    await websocket.close(code=4004)
                    return
                cur = instance.get("status", "")
                if cur != prev_status:
                    prev_status = cur
                if cur == "running":
                    break
                if cur in ("failed", "cancelled", "terminated"):
                    await _send_error(websocket, f"Instance {cur} during startup", 4003)
                    await websocket.close(code=4003)
                    return
            else:
                await _send_error(
                    websocket,
                    "Instance did not reach running state in time — please retry",
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

        # Fast-fail if the remote host is unreachable (avoids 30 s Docker timeout).
        # Use async TCP probe to the SSH port instead of ICMP ping: Tailscale
        # meshes sometimes drop ICMP while TCP flows fine, and the worker host
        # needs SSH open for Docker-over-SSH anyway — so port 22 is the exact
        # thing we care about. Three retries with 1 s spacing tolerate brief
        # scheduler/network blips.
        if is_remote:
            _reach_ok = False
            _probe_reason = "timeout"
            for _probe_attempt in range(3):
                try:
                    _reader, _writer = await asyncio.wait_for(
                        asyncio.open_connection(host_ip, 22),
                        timeout=3.0,
                    )
                    _writer.close()
                    try:
                        await _writer.wait_closed()
                    except Exception:
                        pass
                    _reach_ok = True
                    break
                except asyncio.TimeoutError:
                    _probe_reason = "timeout"
                except ConnectionRefusedError:
                    _probe_reason = "refused"
                except OSError as _oe:
                    _probe_reason = f"oserror:{_oe.errno}"
                except Exception as _e:  # pragma: no cover
                    _probe_reason = f"error:{type(_e).__name__}"
                if _probe_attempt < 2:
                    await asyncio.sleep(1)
            if not _reach_ok:
                log.warning(
                    "TERMINAL host %s unreachable via tcp/22 (reason=%s) host_id=%s",
                    host_ip, _probe_reason, host_id,
                )
                _host_probe_failure_total.labels(
                    host_id=host_id or "unknown",
                    reason=_probe_reason,
                ).inc()
                await _send_error(
                    websocket,
                    f"GPU host {host_ip} is not accepting SSH connections "
                    f"(tcp/22 {_probe_reason}). If you just launched the instance, "
                    "retry in ~30 s; otherwise the host may be offline.",
                    4410,
                )
                await websocket.close(code=4410)
                _conn_errors.labels(reason="host_unreachable").inc()  # type: ignore[attr-defined]
                return

        # Attempt to attach to the container. First call is immediate — if the
        # docker daemon is reachable and the container is present we proceed
        # right away (no artificial 2 s delay). We only keep polling while the
        # container is genuinely not yet created; any non-NotFound error
        # (ssh/paramiko/timeout/auth) surfaces immediately with the real reason.
        attach_deadline = time.monotonic() + _CONTAINER_POLL_BUDGET_SEC
        attempt = 0
        last_probe_error: str | None = None
        while True:
            attempt += 1
            probe_result = await loop.run_in_executor(
                None,
                _container_probe,
                container_ref,
                host_ip if is_remote else None,
            )
            if probe_result["found"]:
                break
            last_probe_error = probe_result.get("error")
            # Non-NotFound failure → fail immediately with actionable detail.
            if last_probe_error and probe_result.get("reason") != "not_found":
                await _send_error(
                    websocket,
                    f"Cannot reach docker daemon on host: {last_probe_error}",
                    4410,
                )
                await websocket.close(code=4410)
                _conn_errors.labels(reason="docker_unreachable").inc()  # type: ignore[attr-defined]
                return
            # Genuine "container not yet created" — wait briefly and retry until budget.
            if time.monotonic() >= attach_deadline:
                await _send_error(
                    websocket,
                    f"Container {container_ref} was not created within "
                    f"{_CONTAINER_POLL_BUDGET_SEC:.0f} s — the instance may have "
                    "failed to start. Check the instance logs for details.",
                    4410,
                )
                await websocket.close(code=4410)
                _conn_errors.labels(reason="container_not_found").inc()  # type: ignore[attr-defined]
                return
            await _send_status(
                websocket,
                f"Waiting for container {container_ref}…",
                retry=True,
            )
            await asyncio.sleep(_CONTAINER_POLL_INTERVAL_SEC)

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
            # Pooled remote clients: evict so the next session rebuilds the SSH
            # tunnel. Local clients: close outright.
            if docker_cl is not None:
                if is_remote:
                    cache_key = _docker_client_cache_key(host_ip, SSH_USER)
                    if cache_key is not None:
                        _evict_docker_client(cache_key)
                else:
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

        # --- Phase 1.2: exec socket hardening + initial prompt render -------
        # TCP keepalive so long-idle sessions don't silently die behind NAT.
        try:
            raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
            if hasattr(socket, "TCP_KEEPINTVL"):
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
            if hasattr(socket, "TCP_KEEPCNT"):
                raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 6)
        except (OSError, AttributeError):
            pass

        # Seed PTY with sensible default dims so bash has a WINSIZE before it
        # renders PS1; client's real dims arrive on the first resize frame.
        try:
            await _docker_call(
                lambda: docker_cl.api.exec_resize(exec_id, height=24, width=80),
                _DOCKER_EXEC_CREATE_TIMEOUT_SEC,
            )
        except (asyncio.TimeoutError, APIError, DockerException, OSError, Exception):
            pass
        # Force an immediate prompt render so the user sees PS1 right away
        # instead of a blank screen (root cause of "no initial $ prompt").
        try:
            await loop.sock_sendall(raw_sock, b"\r")
        except (OSError, ConnectionError):
            pass
        # --- end Phase 1.2 --------------------------------------------------

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

            # Pool-aware cleanup: remote clients stay in the pool (keep SSH
            # tunnel alive for subsequent sessions). Local clients close.
            if docker_cl is not None and not is_remote:
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
