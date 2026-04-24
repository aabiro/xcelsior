"""
Xcelsior SSH Gateway — Production TCP relay for GPU instance SSH access.

Architecture:
  - Runs on the VPS alongside the API as a docker-compose service (network_mode: host)
  - Queries the DB for all running instances → opens TCP listeners on their ssh_port
  - Subscribes to Postgres NOTIFY events for real-time instance lifecycle updates
  - When a client connects to VPS_IP:ssh_port, relays TCP bidirectionally to GPU_HOST_IP:ssh_port via Headscale
  - Handles connection limits, idle timeouts, graceful drain, health endpoint

Port range: 10000–64999 (deterministic from job_id hash, set by worker_agent.py)
GPU hosts are reachable via Headscale mesh (100.64.x.x)
"""

import asyncio
import json
import logging
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

# ── Logging ──────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s %(levelname)-5s [ssh-gw] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("ssh-gateway")

# ── Configuration ────────────────────────────────────────────────────────────

POSTGRES_DSN = os.environ.get(
    "XCELSIOR_POSTGRES_DSN",
    "postgresql://xcelsior:xcelsior@127.0.0.1:5432/xcelsior",
)
HEALTH_PORT = int(os.environ.get("SSH_GW_HEALTH_PORT", "9510"))
MAX_CONNS_PER_INSTANCE = int(os.environ.get("SSH_GW_MAX_CONNS", "10"))
IDLE_TIMEOUT_SEC = int(os.environ.get("SSH_GW_IDLE_TIMEOUT", "3600"))  # 1 hour
RELAY_BUFFER_SIZE = 65536  # 64 KiB — optimal for SSH traffic
SYNC_INTERVAL_SEC = int(os.environ.get("SSH_GW_SYNC_INTERVAL", "10"))
BIND_ADDR = os.environ.get("SSH_GW_BIND", "0.0.0.0")

# TCP keepalive — without these, NAT/firewalls drop idle SSH sessions in ~60s
TCP_KEEPIDLE_SEC = int(os.environ.get("SSH_GW_KEEPIDLE", "30"))  # send first probe after 30s idle
TCP_KEEPINTVL_SEC = int(os.environ.get("SSH_GW_KEEPINTVL", "10"))  # then every 10s
TCP_KEEPCNT = int(os.environ.get("SSH_GW_KEEPCNT", "6"))  # drop after 6 missed probes (~90s total)


def _enable_keepalive(writer: "asyncio.StreamWriter", label: str = "") -> None:
    """Enable TCP keepalive on a stream writer's underlying socket.

    Critical for long-lived SSH connections: many NAT routers and stateful
    firewalls drop idle TCP flows after 60-300 seconds. Keepalive packets
    keep the flow entry warm and detect dead peers within ~90 seconds.
    """
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        # Linux-specific tunables (Headscale + VPS are Linux)
        if hasattr(socket, "TCP_KEEPIDLE"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, TCP_KEEPIDLE_SEC)
        if hasattr(socket, "TCP_KEEPINTVL"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, TCP_KEEPINTVL_SEC)
        if hasattr(socket, "TCP_KEEPCNT"):
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, TCP_KEEPCNT)
        # Disable Nagle to reduce keystroke latency in interactive SSH sessions
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as e:
        log.debug("Could not set keepalive on %s: %s", label, e)


# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class InstanceRoute:
    """Represents a routable SSH instance."""

    job_id: str
    ssh_port: int  # Port on the VPS we listen on (same as host-side mapped port)
    host_ip: str  # Headscale IP of the GPU host (e.g. 100.64.0.2)
    host_id: str
    status: str
    instance_name: str = ""


@dataclass
class ListenerState:
    """Tracks state for an active TCP listener."""

    route: InstanceRoute
    server: Optional[asyncio.AbstractServer] = None
    active_connections: int = 0
    total_connections: int = 0
    started_at: float = field(default_factory=time.time)


# ── Database Layer (direct psycopg — no dependency on app modules) ───────────


async def _get_pg_conn():
    """Create an async psycopg connection."""
    import psycopg

    return await psycopg.AsyncConnection.connect(POSTGRES_DSN, autocommit=True)


async def query_running_instances() -> list[InstanceRoute]:
    """Query DB for all instances that should have SSH access."""
    conn = await _get_pg_conn()
    try:
        # Prefer `public_ssh_port` (gateway-side public port written by the
        # control plane when the worker reports its host-side mapping). Fall
        # back to legacy `ssh_port` for rows predating the split (see the
        # self-poisoning incident notes in routes/instances.py).
        query = """
            SELECT j.job_id, j.status,
                   COALESCE(j.payload->>'public_ssh_port',
                            j.payload->>'ssh_port')    AS ssh_port,
                   j.payload->>'host_id'               AS host_id,
                   j.payload->>'name'                  AS instance_name,
                   h.payload->>'ip'                    AS host_ip
            FROM jobs j
            LEFT JOIN hosts h ON j.payload->>'host_id' = h.host_id
            WHERE j.status IN ('running', 'starting')
              AND COALESCE(j.payload->>'public_ssh_port',
                           j.payload->>'ssh_port') IS NOT NULL
              AND j.payload->>'host_id'  IS NOT NULL
        """
        async with conn.cursor() as cur:
            await cur.execute(query)
            rows = await cur.fetchall()
            routes = []
            for row in rows:
                job_id, status, ssh_port_str, host_id, name, host_ip = row
                if not ssh_port_str or not host_ip:
                    continue
                try:
                    ssh_port = int(ssh_port_str)
                except (ValueError, TypeError):
                    continue
                routes.append(
                    InstanceRoute(
                        job_id=job_id,
                        ssh_port=ssh_port,
                        host_ip=host_ip,
                        host_id=host_id,
                        status=status,
                        instance_name=name or "",
                    )
                )
            return routes
    finally:
        await conn.close()


# ── TCP Relay Core ───────────────────────────────────────────────────────────


class SSHGateway:
    """
    Async TCP relay gateway. Manages per-instance listeners, relays SSH
    traffic bidirectionally with backpressure, idle timeouts, and conn limits.
    """

    def __init__(self):
        self.listeners: dict[int, ListenerState] = {}  # port → state
        self._shutting_down = False
        self._stats = {
            "total_connections": 0,
            "total_bytes_relayed": 0,
            "started_at": time.time(),
        }
        # Health-tracking — /health flips to "degraded" when these deviate
        # from "recently ok". Without this the process can look healthy
        # (listeners present, accepting connections) while the LISTEN
        # channel has gone silently stale — which is the exact failure
        # mode that made the gateway "not work properly".
        self._last_pg_notify_ok_at = 0.0
        self._last_db_query_ok_at = 0.0
        self._pg_reconnects = 0
        self._failed_binds: dict[int, float] = {}  # port → last-failed-ts

    # ── Relay logic ──────────────────────────────────────────────────────

    async def _relay_half(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        label: str,
        stats: dict,
    ):
        """Relay data from reader → writer with backpressure. One direction."""
        try:
            while True:
                data = await asyncio.wait_for(
                    reader.read(RELAY_BUFFER_SIZE),
                    timeout=IDLE_TIMEOUT_SEC,
                )
                if not data:
                    break
                writer.write(data)
                await writer.drain()
                stats["bytes"] += len(data)
        except asyncio.TimeoutError:
            log.info("Idle timeout on %s", label)
        except (ConnectionResetError, BrokenPipeError, OSError):
            pass

    async def _handle_connection(
        self,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        state: ListenerState,
    ):
        """Handle a single inbound SSH connection — connect to backend, relay."""
        route = state.route
        peer = client_writer.get_extra_info("peername")
        peer_str = f"{peer[0]}:{peer[1]}" if peer else "unknown"

        # Connection limit check
        if state.active_connections >= MAX_CONNS_PER_INSTANCE:
            log.warning(
                "Connection limit reached for %s (port %d, %d active) — rejecting %s",
                route.job_id[:8],
                route.ssh_port,
                state.active_connections,
                peer_str,
            )
            client_writer.close()
            await client_writer.wait_closed()
            return

        state.active_connections += 1
        state.total_connections += 1
        self._stats["total_connections"] += 1

        log.info(
            "New connection: %s → %s:%d (instance %s %s) [%d active]",
            peer_str,
            route.host_ip,
            route.ssh_port,
            route.job_id[:8],
            route.instance_name,
            state.active_connections,
        )

        backend_reader = None
        backend_writer = None
        relay_stats = {"bytes": 0}

        try:
            # Connect to GPU host via Headscale
            backend_reader, backend_writer = await asyncio.wait_for(
                asyncio.open_connection(route.host_ip, route.ssh_port),
                timeout=10.0,
            )

            # Enable TCP keepalive on BOTH sockets so NAT/firewalls don't kill
            # idle SSH sessions every 60s (the most common cause of unexpected
            # disconnects in interactive shells). Also disables Nagle so
            # keystrokes aren't batched.
            _enable_keepalive(client_writer, f"client {peer_str}")
            _enable_keepalive(backend_writer, f"backend {route.host_ip}:{route.ssh_port}")

            # Bidirectional relay with backpressure
            await asyncio.gather(
                self._relay_half(client_reader, backend_writer, f"{peer_str}→backend", relay_stats),
                self._relay_half(backend_reader, client_writer, f"backend→{peer_str}", relay_stats),
            )
        except asyncio.TimeoutError:
            log.warning(
                "Backend connect timeout: %s:%d (instance %s)",
                route.host_ip,
                route.ssh_port,
                route.job_id[:8],
            )
        except ConnectionRefusedError:
            log.warning(
                "Backend refused: %s:%d (instance %s) — container may not be ready",
                route.host_ip,
                route.ssh_port,
                route.job_id[:8],
            )
        except OSError as e:
            log.warning(
                "Backend connect error: %s:%d (instance %s) — %s",
                route.host_ip,
                route.ssh_port,
                route.job_id[:8],
                e,
            )
        finally:
            self._stats["total_bytes_relayed"] += relay_stats["bytes"]
            state.active_connections -= 1

            # Clean up both sides
            for w in (client_writer, backend_writer):
                if w is not None:
                    try:
                        w.close()
                        await w.wait_closed()
                    except OSError:
                        pass

            log.info(
                "Connection closed: %s → instance %s (%s bytes relayed) [%d active]",
                peer_str,
                route.job_id[:8],
                f"{relay_stats['bytes']:,}",
                state.active_connections,
            )

    # ── Listener management ──────────────────────────────────────────────

    async def open_listener(self, route: InstanceRoute):
        """Start listening on a port for an instance."""
        port = route.ssh_port
        if port in self.listeners:
            # Already listening — update route info (host IP might change)
            self.listeners[port].route = route
            return

        state = ListenerState(route=route)

        async def on_connect(reader, writer):
            asyncio.create_task(self._handle_connection(reader, writer, state))

        try:
            server = await asyncio.start_server(
                on_connect,
                BIND_ADDR,
                port,
                reuse_address=True,
                reuse_port=True,
            )
            state.server = server
            self.listeners[port] = state
            # Clear any prior bind failure — the port is alive again.
            self._failed_binds.pop(port, None)
            log.info(
                "Listening on :%d → %s:%d (instance %s %s)",
                port,
                route.host_ip,
                port,
                route.job_id[:8],
                route.instance_name,
            )
        except OSError as e:
            # Record the failure so /health can surface it and the next
            # sync pass can retry explicitly rather than silently
            # dropping the port. Most common cause: previous listener
            # still in TIME_WAIT (fixed by SO_REUSEADDR/SO_REUSEPORT but
            # some kernels are strict) or another process squatting on
            # the port.
            self._failed_binds[port] = time.time()
            log.error(
                "Failed to bind port %d for instance %s: %s "
                "(will retry on next sync)",
                port, route.job_id[:8], e,
            )

    async def close_listener(self, port: int, drain: bool = True):
        """Stop listening on a port and optionally drain active connections."""
        state = self.listeners.pop(port, None)
        if not state or not state.server:
            return

        state.server.close()
        await state.server.wait_closed()

        if drain and state.active_connections > 0:
            log.info(
                "Draining %d active connections on port %d (instance %s)...",
                state.active_connections,
                port,
                state.route.job_id[:8],
            )
            # Give active connections 10s to finish gracefully
            for _ in range(100):
                if state.active_connections <= 0:
                    break
                await asyncio.sleep(0.1)

        log.info(
            "Closed listener :%d (instance %s, %d total connections served)",
            port,
            state.route.job_id[:8],
            state.total_connections,
        )

    # ── Sync loop — reconcile listeners with DB state ────────────────────

    async def sync_routes(self):
        """Reconcile active listeners with the current DB state."""
        try:
            routes = await query_running_instances()
            self._last_db_query_ok_at = time.time()
        except Exception as e:
            log.error("Failed to query instances: %s", e)
            return

        desired_ports = {}
        for r in routes:
            desired_ports[r.ssh_port] = r

        # Open new listeners (or retry previously-failed binds)
        for port, route in desired_ports.items():
            if port not in self.listeners:
                await self.open_listener(route)
            else:
                # Update route (host IP might change on migration)
                self.listeners[port].route = route

        # Close stale listeners (instances that stopped/terminated)
        stale = [p for p in self.listeners if p not in desired_ports]
        for port in stale:
            await self.close_listener(port)

        if desired_ports:
            log.debug(
                "Synced: %d listeners active, %d opened, %d closed",
                len(self.listeners),
                len(desired_ports) - len(self.listeners) + len(stale),
                len(stale),
            )

    async def _sync_loop(self):
        """Periodically reconcile listeners with DB."""
        while not self._shutting_down:
            await self.sync_routes()
            await asyncio.sleep(SYNC_INTERVAL_SEC)

    # ── Postgres LISTEN for real-time events ─────────────────────────────

    async def _pg_listen_loop(self):
        """
        Subscribe to Postgres NOTIFY on 'xcelsior_events' channel.
        Triggers immediate sync when instance status changes.
        """
        import psycopg

        while not self._shutting_down:
            conn = None
            try:
                # TCP keepalive on the LISTEN connection. Without this
                # a stateful NAT/firewall between this gateway and the
                # Postgres VPS can silently drop the idle TCP stream
                # after ~2h; psycopg's `async for notifies()` then
                # blocks forever on a half-dead socket. With keepalive
                # probes every 60s the kernel kills the socket inside
                # ~3min and we reconnect via the outer retry loop.
                conn = await psycopg.AsyncConnection.connect(
                    POSTGRES_DSN,
                    autocommit=True,
                    keepalives=1,
                    keepalives_idle=60,
                    keepalives_interval=30,
                    keepalives_count=3,
                )
                await conn.execute("LISTEN xcelsior_events")
                self._last_pg_notify_ok_at = time.time()
                log.info("Postgres LISTEN active on 'xcelsior_events'")

                async for notify in conn.notifies():
                    if self._shutting_down:
                        break
                    self._last_pg_notify_ok_at = time.time()
                    try:
                        payload = json.loads(notify.payload)
                        event_type = payload.get("type", "")
                        # React to every lifecycle transition that
                        # affects routability. New additions vs. the
                        # original set: job_starting / job_created so
                        # a fresh instance gets a listener within ~1s
                        # instead of waiting up to SYNC_INTERVAL_SEC;
                        # job_paused so we close the tunnel promptly
                        # when billing pauses a pod (otherwise an
                        # attacker holding the pre-pause SSH session
                        # can keep using the container for 30s).
                        if event_type in (
                            "job_status",
                            "job_created",
                            "job_starting",
                            "job_running",
                            "job_paused",
                            "job_resumed",
                            "job_stopped",
                            "job_terminated",
                            "job_failed",
                            "job_cancelled",
                        ):
                            log.info("Event: %s — triggering sync", event_type)
                            await self.sync_routes()
                    except (json.JSONDecodeError, KeyError):
                        pass

            except Exception as e:
                self._pg_reconnects += 1
                log.warning(
                    "Postgres LISTEN connection lost (#%d): %s — reconnecting in 5s",
                    self._pg_reconnects, e,
                )
                if conn:
                    try:
                        await conn.close()
                    except Exception:
                        pass
                await asyncio.sleep(5)

    # ── Health endpoint ──────────────────────────────────────────────────

    async def _health_handler(self, reader, writer):
        """Minimal HTTP health check endpoint for monitoring."""
        try:
            await asyncio.wait_for(reader.read(4096), timeout=5)
        except (asyncio.TimeoutError, Exception):
            pass

        uptime = int(time.time() - self._stats["started_at"])
        active = sum(s.active_connections for s in self.listeners.values())

        # Degraded criteria — anything that indicates the gateway may
        # not react correctly to lifecycle events. We keep returning
        # HTTP 200 (so the TCP listener probes still succeed and k8s
        # doesn't thrash us) but surface the condition in the body so
        # alerting / dashboards catch it.
        now = time.time()
        issues: list[str] = []
        # LISTEN connection should see traffic OR be freshly connected;
        # 10 min of complete silence with no reconnect means the async
        # loop is likely stuck.
        if self._last_pg_notify_ok_at and now - self._last_pg_notify_ok_at > 600:
            issues.append("pg_listen_stale")
        # DB polling should succeed at least every 2× sync interval.
        if self._last_db_query_ok_at and now - self._last_db_query_ok_at > max(60, SYNC_INTERVAL_SEC * 3):
            issues.append("db_query_stale")
        if self._failed_binds:
            issues.append(f"failed_binds:{len(self._failed_binds)}")

        body = json.dumps(
            {
                "status": "ok" if not issues else "degraded",
                "issues": issues,
                "listeners": len(self.listeners),
                "active_connections": active,
                "total_connections": self._stats["total_connections"],
                "total_bytes_relayed": self._stats["total_bytes_relayed"],
                "uptime_sec": uptime,
                "pg_reconnects": self._pg_reconnects,
                "last_pg_notify_age_sec": (
                    int(now - self._last_pg_notify_ok_at)
                    if self._last_pg_notify_ok_at else None
                ),
                "last_db_query_age_sec": (
                    int(now - self._last_db_query_ok_at)
                    if self._last_db_query_ok_at else None
                ),
                "failed_binds": sorted(self._failed_binds.keys()),
                "ports": sorted(self.listeners.keys()),
            }
        )

        response = (
            f"HTTP/1.1 200 OK\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
            f"{body}"
        )
        writer.write(response.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def _start_health_server(self):
        server = await asyncio.start_server(
            self._health_handler,
            "127.0.0.1",
            HEALTH_PORT,
        )
        log.info("Health endpoint on http://127.0.0.1:%d/", HEALTH_PORT)
        async with server:
            await server.serve_forever()

    # ── Main entry ───────────────────────────────────────────────────────

    async def run(self):
        """Main entry point — start all loops."""
        log.info("Xcelsior SSH Gateway starting...")
        log.info(
            "Config: bind=%s, max_conns=%d, idle_timeout=%ds, sync=%ds",
            BIND_ADDR,
            MAX_CONNS_PER_INSTANCE,
            IDLE_TIMEOUT_SEC,
            SYNC_INTERVAL_SEC,
        )

        # Initial sync
        await self.sync_routes()
        log.info("Initial sync complete: %d listeners active", len(self.listeners))

        # Run all loops concurrently
        await asyncio.gather(
            self._sync_loop(),
            self._pg_listen_loop(),
            self._start_health_server(),
        )

    async def shutdown(self):
        """Graceful shutdown — close all listeners with drain."""
        log.info("Shutting down — draining %d listeners...", len(self.listeners))
        self._shutting_down = True
        ports = list(self.listeners.keys())
        for port in ports:
            await self.close_listener(port, drain=True)
        log.info("Shutdown complete")


# ── Entry Point ──────────────────────────────────────────────────────────────


def main():
    gateway = SSHGateway()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Graceful shutdown on SIGTERM/SIGINT
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda: asyncio.ensure_future(gateway.shutdown()))

    try:
        loop.run_until_complete(gateway.run())
    except KeyboardInterrupt:
        loop.run_until_complete(gateway.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
