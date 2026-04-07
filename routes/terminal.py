"""Routes: Interactive web terminal.

Architecture
------------
Browser (xterm.js) ← binary WebSocket → FastAPI (/ws/terminal/{instance_id})
    → ssh -tt {host_ip} → docker exec -it {container} → tmux / bare shell

Note: SSH subprocess is used rather than docker-py over TCP because GPU hosts
do not expose the Docker TCP daemon.  If that changes, docker-py with the
``ssh://`` transport (paramiko) is a drop-in replacement for the subprocess
calls in ``_container_exists_remote``, ``_tmux_available_remote``, and the
exec command builder.

Protocol
--------
Client → server (text JSON):
    {"type": "input",  "data": "<chars>"}
    {"type": "resize", "cols": N, "rows": N}
    {"type": "ping"}

Server → client (binary): raw PTY bytes  (xterm.js writes directly — no JSON wrap)

Server → client (text JSON, control only):
    {"type": "status", "message": "...", "retry": true|false}
    {"type": "error",  "message": "...", "code": N}
    {"type": "exit",   "code": N}
    {"type": "pong",   "ts": <float>}

Security
--------
- Auth validated before accept() via cookie or ?token= query param.
- Instance ownership enforced (admin bypass permitted).
- docker exec inherits the container's user namespace / gVisor/Kata sandbox.
- Host IP validated against an allowlist regex before use in subprocess args.
- No Docker socket exposed to the browser.
"""

import asyncio
import fcntl
import json
import os
import pty as _pty
import re as _re
import shlex
import struct
import subprocess
import termios
import time
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from routes._deps import (
    AUTH_REQUIRED,
    _validate_ws_auth,
    log,
)
from scheduler import (
    SSH_KEY_PATH,
    SSH_USER,
    list_hosts,
    list_jobs,
)

# ── Optional Prometheus metrics (graceful no-op when not installed) ───────────

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
    _conn_errors: object = Counter(
        "xcelsior_terminal_errors_total",
        "Terminal WebSocket connection errors by reason",
        ["reason"],
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
    _conn_errors = _Noop()      # type: ignore[assignment]


router = APIRouter()

# ── Constants ──────────────────────────────────────────────────────────────────

_SESSION_TIMEOUT_SEC: int = 14_400          # 4 hours hard max
_IDLE_WARN_THRESHOLD_SEC: int = _SESSION_TIMEOUT_SEC - 300  # warn 5 min before
_RATE_LIMIT_BYTES_PER_SEC: int = 524_288    # 512 KB/s output throttle
_PTY_READ_CHUNK: int = 4096
_CONTAINER_POLL_INTERVAL_SEC: float = 2.0
_CONTAINER_POLL_MAX_ATTEMPTS: int = 15      # 30 s total
_TMUX_CHECK_TIMEOUT_SEC: float = 5.0
_STDIN_RECV_TIMEOUT_SEC: float = 10.0
_PTY_READ_TIMEOUT_SEC: float = 30.0

# Strict allowlist for host_ip to prevent any command injection.
_HOST_IP_RE: _re.Pattern[str] = _re.compile(r"\A[a-zA-Z0-9.\-]+\Z")


# ── SSH option bundle (no allocate-pty; added per-use) ────────────────────────

def _build_ssh_opts(key_path: str) -> list[str]:
    return [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        "-i", key_path,
    ]


# ── Preflight helpers ─────────────────────────────────────────────────────────

def _container_exists_remote(ssh_opts: list[str], ssh_target: str, container_ref: str) -> bool:
    """Return True when *container_ref* exists on the remote Docker daemon."""
    remote_cmd = shlex.join(["docker", "inspect", "--format=.", container_ref])
    cmd = ["ssh", *ssh_opts, ssh_target, remote_cmd]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=8)
        return r.returncode == 0
    except Exception:
        return False


def _container_exists_local(container_ref: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "inspect", "--format=.", container_ref],
            capture_output=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def _tmux_available_remote(ssh_opts: list[str], ssh_target: str, container_ref: str) -> bool:
    """Return True when tmux is on PATH inside *container_ref*."""
    remote_cmd = shlex.join(["docker", "exec", container_ref, "which", "tmux"])
    cmd = ["ssh", *ssh_opts, ssh_target, remote_cmd]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=_TMUX_CHECK_TIMEOUT_SEC)
        return r.returncode == 0
    except Exception:
        return False


def _tmux_available_local(container_ref: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "exec", container_ref, "which", "tmux"],
            capture_output=True, timeout=_TMUX_CHECK_TIMEOUT_SEC,
        )
        return r.returncode == 0
    except Exception:
        return False


# ── PTY event-driven read ─────────────────────────────────────────────────────

async def _read_pty(
    master_fd: int,
    loop: asyncio.AbstractEventLoop,
    timeout: float = _PTY_READ_TIMEOUT_SEC,
) -> bytes:
    """Read one chunk from *master_fd* using asyncio event-driven I/O.

    Raises ``asyncio.TimeoutError`` if no data arrives within *timeout*
    seconds (used for keepalive / session-timeout checks).
    Raises ``OSError`` (errno EBADF / EIO) when the PTY is closed.
    """
    fut: asyncio.Future[bytes] = loop.create_future()

    def _on_readable() -> None:
        loop.remove_reader(master_fd)
        if fut.done():
            return
        try:
            data = os.read(master_fd, _PTY_READ_CHUNK)
            if data:
                fut.set_result(data)
            else:
                # EOF on PTY — process has exited
                fut.set_exception(OSError("PTY EOF"))
        except BlockingIOError:
            # Spurious wakeup; re-register the reader and wait again
            loop.add_reader(master_fd, _on_readable)
        except OSError as exc:
            fut.set_exception(exc)

    loop.add_reader(master_fd, _on_readable)
    try:
        return await asyncio.wait_for(fut, timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        loop.remove_reader(master_fd)
        raise
    except BaseException:
        loop.remove_reader(master_fd)
        raise


# ── Control-frame helpers ─────────────────────────────────────────────────────

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


# ── WebSocket endpoint ────────────────────────────────────────────────────────

@router.websocket("/ws/terminal/{instance_id}")
async def ws_terminal(websocket: WebSocket, instance_id: str) -> None:
    """Interactive terminal session for a running GPU instance."""

    # ── 1. Auth (before accept — avoids half-open zombies) ───────────
    user = _validate_ws_auth(websocket)
    if not user:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    await websocket.accept()

    # ── 2. Resolve instance ──────────────────────────────────────────
    instance = next((j for j in list_jobs() if j["job_id"] == instance_id), None)
    if not instance:
        await _send_error(websocket, "Instance not found", 4004)
        await websocket.close(code=4004)
        _conn_errors.labels(reason="not_found").inc()  # type: ignore[attr-defined]
        return

    # ── 3. Ownership check ───────────────────────────────────────────
    is_admin = user.get("role") == "admin" or bool(user.get("is_admin"))
    if not is_admin:
        job_owner = instance.get("owner", "")
        caller_id = user.get("customer_id", user.get("user_id", ""))
        if job_owner and job_owner != caller_id:
            await _send_error(websocket, "Not authorized", 4003)
            await websocket.close(code=4003)
            _conn_errors.labels(reason="unauthorized").inc()  # type: ignore[attr-defined]
            return

    if instance.get("status") != "running":
        await _send_error(
            websocket,
            f"Instance is '{instance.get('status', 'unknown')}', not running",
            4003,
        )
        await websocket.close(code=4003)
        _conn_errors.labels(reason="not_running").inc()  # type: ignore[attr-defined]
        return

    # ── 4. Resolve host address & container reference ────────────────
    host_id = instance.get("host_id", "")
    container_ref: str = instance.get("container_name") or f"xcl-{instance_id}"

    host_ip: str = instance.get("host_ip", "")
    if not host_ip and host_id:
        hmap = {h["host_id"]: h for h in list_hosts(active_only=False)}
        hr = hmap.get(host_id)
        host_ip = hr.get("ip", "") if hr else ""

    # Strict allowlist: only hostname/IP characters (prevents shell injection)
    if host_ip and not _HOST_IP_RE.fullmatch(host_ip):
        await _send_error(websocket, "Invalid host address", 4003)
        await websocket.close(code=4003)
        return

    shell: str = instance.get("shell") or "/bin/bash"
    is_remote: bool = bool(host_ip) and host_ip not in ("127.0.0.1", "localhost", "0.0.0.0")

    ssh_opts = _build_ssh_opts(SSH_KEY_PATH)
    ssh_target = f"{SSH_USER}@{host_ip}"

    # ── 5. Container readiness polling ───────────────────────────────
    loop = asyncio.get_running_loop()

    for attempt in range(_CONTAINER_POLL_MAX_ATTEMPTS):
        exists = await loop.run_in_executor(
            None,
            _container_exists_remote if is_remote else _container_exists_local,
            *(([ssh_opts, ssh_target, container_ref]) if is_remote else ([container_ref])),
        )
        if exists:
            break
        await _send_status(
            websocket,
            f"Container starting\u2026 ({attempt + 1}/{_CONTAINER_POLL_MAX_ATTEMPTS})",
            retry=True,
        )
        await asyncio.sleep(_CONTAINER_POLL_INTERVAL_SEC)
    else:
        await _send_error(
            websocket,
            "Container did not start within 30 s \u2014 please retry in a moment",
            4410,
        )
        await websocket.close(code=4410)
        _conn_errors.labels(reason="container_not_found").inc()  # type: ignore[attr-defined]
        return

    # ── 6. Tmux availability check ───────────────────────────────────
    has_tmux: bool = await loop.run_in_executor(
        None,
        _tmux_available_remote if is_remote else _tmux_available_local,
        *(([ssh_opts, ssh_target, container_ref]) if is_remote else ([container_ref])),
    )

    # ── 7. Build exec command ────────────────────────────────────────
    tmux_session = f"xcl-{instance_id}"

    if has_tmux:
        # tmux new-session -A  →  attach-or-create; survives reconnects
        inner_cmd = shlex.join([
            "docker", "exec",
            "-e", "TERM=xterm-256color",
            "-e", f"TMUX_SESSION={tmux_session}",
            "-it", container_ref,
            "tmux", "new-session", "-A", "-s", tmux_session, shell,
        ])
    else:
        inner_cmd = shlex.join([
            "docker", "exec",
            "-e", "TERM=xterm-256color",
            "-it", container_ref,
            shell,
        ])

    if is_remote:
        exec_cmd: list[str] = ["ssh", *ssh_opts, "-tt", ssh_target, inner_cmd]
    else:
        if has_tmux:
            exec_cmd = [
                "docker", "exec",
                "-e", "TERM=xterm-256color",
                "-e", f"TMUX_SESSION={tmux_session}",
                "-it", container_ref,
                "tmux", "new-session", "-A", "-s", tmux_session, shell,
            ]
        else:
            exec_cmd = [
                "docker", "exec", "-e", "TERM=xterm-256color",
                "-it", container_ref, shell,
            ]

    # ── 8. Spawn PTY process ─────────────────────────────────────────
    master_fd: Optional[int] = None
    slave_fd: Optional[int] = None
    process: Optional[asyncio.subprocess.Process] = None

    try:
        master_fd, slave_fd = _pty.openpty()
        os.set_blocking(master_fd, False)  # event-driven reads via add_reader
        process = await asyncio.create_subprocess_exec(
            *exec_cmd,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
        )
        os.close(slave_fd)
        slave_fd = None
    except (OSError, FileNotFoundError) as exc:
        for fd in (master_fd, slave_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        await _send_error(websocket, f"Failed to spawn terminal: {exc}", 4500)
        await websocket.close(code=4500)
        _conn_errors.labels(reason="spawn_failed").inc()  # type: ignore[attr-defined]
        return

    # ── 9. Session state ─────────────────────────────────────────────
    log.info(
        "terminal.session.open user=%s instance=%s container=%s tmux=%s",
        user.get("email", "?"), instance_id, container_ref, has_tmux,
    )
    _active_sessions.inc()  # type: ignore[attr-defined]

    session_start = time.monotonic()
    last_input_ts = time.monotonic()
    closed = False
    idle_warned = False  # fire idle warning at most once per session

    # Rate-limit state
    bytes_this_second: int = 0
    rate_window_start: float = time.monotonic()

    # Notify frontend: connected, spinner can stop
    await _send_status(
        websocket,
        f"Connected to {instance.get('name', instance_id)}",
        retry=False,
    )

    # ── PTY resize ───────────────────────────────────────────────────
    def _resize_pty(cols: int, rows: int) -> None:
        if master_fd is None:
            return
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(master_fd, termios.TIOCSWINSZ, winsize)
        except (OSError, ValueError):
            pass

    # ── Stdout relay: PTY → WebSocket (binary) ───────────────────────
    async def _stdout_relay() -> None:
        nonlocal closed, bytes_this_second, rate_window_start, idle_warned

        while not closed:
            # ── Idle timeout (resets on user input) ──────────────────
            idle_time = time.monotonic() - last_input_ts
            if idle_time >= _SESSION_TIMEOUT_SEC:
                await _send_error(websocket, "Session timed out due to inactivity", 4408)
                closed = True
                break

            # ── Idle warning (resets when user types) ────────────────
            if not idle_warned and idle_time >= _IDLE_WARN_THRESHOLD_SEC:
                idle_warned = True
                await _send_status(
                    websocket,
                    "Session closing in 5 min due to inactivity",
                )

            # ── Read PTY ─────────────────────────────────────────────
            try:
                chunk = await _read_pty(master_fd, loop, timeout=_PTY_READ_TIMEOUT_SEC)
            except asyncio.TimeoutError:
                # No data for 30 s — normal for idle sessions; loop back
                continue
            except OSError:
                # PTY closed (process exited)
                break

            if not chunk:
                break

            # ── Rate limiting ─────────────────────────────────────────
            now = time.monotonic()
            if now - rate_window_start >= 1.0:
                bytes_this_second = 0
                rate_window_start = now
            bytes_this_second += len(chunk)
            if bytes_this_second > _RATE_LIMIT_BYTES_PER_SEC:
                await asyncio.sleep(0.05)

            # ── Send raw bytes to browser (no JSON wrapping) ──────────
            try:
                await websocket.send_bytes(chunk)
                _bytes_sent.inc(len(chunk))  # type: ignore[attr-defined]
            except (WebSocketDisconnect, RuntimeError):
                closed = True
                break

    # ── Stdin relay: WebSocket → PTY ─────────────────────────────────
    async def _stdin_relay() -> None:
        nonlocal closed, last_input_ts, idle_warned

        while not closed:
            # Idle timeout guard (resets on user input)
            if time.monotonic() - last_input_ts > _SESSION_TIMEOUT_SEC:
                await _send_error(websocket, "Session timed out due to inactivity", 4408)
                closed = True
                break

            # Receive with a timeout so we can run keepalive checks
            try:
                raw = await asyncio.wait_for(
                    websocket.receive(), timeout=_STDIN_RECV_TIMEOUT_SEC
                )
            except asyncio.TimeoutError:
                # Keepalive: send a pong so dead peers are detected
                try:
                    await websocket.send_json({"type": "pong", "ts": time.time()})
                except (WebSocketDisconnect, RuntimeError):
                    closed = True
                continue
            except WebSocketDisconnect:
                closed = True
                break

            if raw.get("type") == "websocket.disconnect":
                closed = True
                break

            if raw.get("type") != "websocket.receive":
                continue

            # Binary path: raw keystroke bytes forwarded directly (faster)
            payload_bytes: Optional[bytes] = raw.get("bytes")
            if payload_bytes:
                try:
                    os.write(master_fd, payload_bytes)
                    last_input_ts = time.monotonic()
                    idle_warned = False
                except OSError:
                    closed = True
                continue

            # Text path: JSON control frame
            payload_text: Optional[str] = raw.get("text")
            if not payload_text:
                continue

            try:
                msg: dict = json.loads(payload_text)
            except (json.JSONDecodeError, ValueError):
                continue

            msg_type = msg.get("type", "")

            if msg_type == "input":
                data: str = msg.get("data", "")
                if data:
                    try:
                        os.write(master_fd, data.encode("utf-8"))
                        last_input_ts = time.monotonic()
                        idle_warned = False
                    except OSError:
                        closed = True

            elif msg_type == "resize":
                try:
                    cols = max(1, int(msg.get("cols", 80)))
                    rows = max(1, int(msg.get("rows", 24)))
                    _resize_pty(cols, rows)
                except (TypeError, ValueError):
                    pass

            elif msg_type == "ping":
                try:
                    await websocket.send_json({"type": "pong", "ts": time.time()})
                except (WebSocketDisconnect, RuntimeError):
                    closed = True

    # ── 10. Run both relays; stop on whichever exits first ───────────
    try:
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

        # ── Graceful tmux detach: keeps session alive for reconnect ──
        if has_tmux and master_fd is not None:
            try:
                os.write(master_fd, b"\x02d")   # Ctrl+B then d  → tmux detach
                await asyncio.sleep(0.1)
            except OSError:
                pass

        # ── Close PTY master fd ───────────────────────────────────────
        if master_fd is not None:
            try:
                os.close(master_fd)
            except OSError:
                pass

        # ── Process reaping ───────────────────────────────────────────
        if process is not None:
            if not has_tmux:
                # No tmux: kill the bare shell
                try:
                    process.kill()
                except Exception:
                    pass
            try:
                await asyncio.wait_for(process.wait(), timeout=3.0)
            except Exception:
                pass

        # ── Metrics + audit log ───────────────────────────────────────
        _active_sessions.dec()  # type: ignore[attr-defined]
        log.info(
            "terminal.session.close user=%s instance=%s duration=%.1fs",
            user.get("email", "?"), instance_id, time.monotonic() - session_start,
        )

        # ── Exit notification (best-effort) ───────────────────────────
        try:
            exit_code = process.returncode if (process and not has_tmux) else 0
            await websocket.send_json({"type": "exit", "code": exit_code or 0})
            await websocket.close()
        except Exception:
            pass
