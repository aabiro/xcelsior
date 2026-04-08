"""
Xcelsior SSH Gateway — Production TCP relay for GPU instance SSH access.

Architecture:
  - Runs on the VPS alongside the API as a docker-compose service (network_mode: host)
  - Queries the DB for all running instances → opens TCP listeners on their ssh_port
  - Subscribes to Postgres NOTIFY events for real-time instance lifecycle updates
  - When a client connects to VPS_IP:ssh_port, relays TCP bidirectionally to GPU_HOST_IP:ssh_port via Tailscale
  - Handles connection limits, idle timeouts, graceful drain, health endpoint

Port range: 10000–64999 (deterministic from job_id hash, set by worker_agent.py)
GPU hosts are reachable via Tailscale mesh (100.64.x.x)
"""

import asyncio
import json
import logging
import os
import signal
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
SYNC_INTERVAL_SEC = int(os.environ.get("SSH_GW_SYNC_INTERVAL", "30"))
BIND_ADDR = os.environ.get("SSH_GW_BIND", "0.0.0.0")

# ── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class InstanceRoute:
    """Represents a routable SSH instance."""
    job_id: str
    ssh_port: int           # Port on the VPS we listen on (same as host-side mapped port)
    host_ip: str            # Tailscale IP of the GPU host (e.g. 100.64.0.2)
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
        query = """
            SELECT j.job_id, j.status,
                   j.payload->>'ssh_port'   AS ssh_port,
                   j.payload->>'host_id'    AS host_id,
                   j.payload->>'name'       AS instance_name,
                   h.payload->>'ip'         AS host_ip
            FROM jobs j
            LEFT JOIN hosts h ON j.payload->>'host_id' = h.host_id
            WHERE j.status IN ('running', 'starting')
              AND j.payload->>'ssh_port' IS NOT NULL
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
                routes.append(InstanceRoute(
                    job_id=job_id,
                    ssh_port=ssh_port,
                    host_ip=host_ip,
                    host_id=host_id,
                    status=status,
                    instance_name=name or "",
                ))
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
                route.job_id[:8], route.ssh_port, state.active_connections, peer_str,
            )
            client_writer.close()
            await client_writer.wait_closed()
            return

        state.active_connections += 1
        state.total_connections += 1
        self._stats["total_connections"] += 1

        log.info(
            "New connection: %s → %s:%d (instance %s %s) [%d active]",
            peer_str, route.host_ip, route.ssh_port,
            route.job_id[:8], route.instance_name, state.active_connections,
        )

        backend_reader = None
        backend_writer = None
        relay_stats = {"bytes": 0}

        try:
            # Connect to GPU host via Tailscale
            backend_reader, backend_writer = await asyncio.wait_for(
                asyncio.open_connection(route.host_ip, route.ssh_port),
                timeout=10.0,
            )

            # Bidirectional relay with backpressure
            await asyncio.gather(
                self._relay_half(client_reader, backend_writer, f"{peer_str}→backend", relay_stats),
                self._relay_half(backend_reader, client_writer, f"backend→{peer_str}", relay_stats),
            )
        except asyncio.TimeoutError:
            log.warning(
                "Backend connect timeout: %s:%d (instance %s)",
                route.host_ip, route.ssh_port, route.job_id[:8],
            )
        except ConnectionRefusedError:
            log.warning(
                "Backend refused: %s:%d (instance %s) — container may not be ready",
                route.host_ip, route.ssh_port, route.job_id[:8],
            )
        except OSError as e:
            log.warning(
                "Backend connect error: %s:%d (instance %s) — %s",
                route.host_ip, route.ssh_port, route.job_id[:8], e,
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
                peer_str, route.job_id[:8],
                f"{relay_stats['bytes']:,}", state.active_connections,
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
                on_connect, BIND_ADDR, port,
                reuse_address=True,
                reuse_port=True,
            )
            state.server = server
            self.listeners[port] = state
            log.info(
                "Listening on :%d → %s:%d (instance %s %s)",
                port, route.host_ip, port, route.job_id[:8], route.instance_name,
            )
        except OSError as e:
            log.error("Failed to bind port %d for instance %s: %s", port, route.job_id[:8], e)

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
                state.active_connections, port, state.route.job_id[:8],
            )
            # Give active connections 10s to finish gracefully
            for _ in range(100):
                if state.active_connections <= 0:
                    break
                await asyncio.sleep(0.1)

        log.info(
            "Closed listener :%d (instance %s, %d total connections served)",
            port, state.route.job_id[:8], state.total_connections,
        )

    # ── Sync loop — reconcile listeners with DB state ────────────────────

    async def sync_routes(self):
        """Reconcile active listeners with the current DB state."""
        try:
            routes = await query_running_instances()
        except Exception as e:
            log.error("Failed to query instances: %s", e)
            return

        desired_ports = {}
        for r in routes:
            desired_ports[r.ssh_port] = r

        # Open new listeners
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
                len(self.listeners), len(desired_ports) - len(self.listeners) + len(stale), len(stale),
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
                conn = await psycopg.AsyncConnection.connect(
                    POSTGRES_DSN, autocommit=True,
                )
                await conn.execute("LISTEN xcelsior_events")
                log.info("Postgres LISTEN active on 'xcelsior_events'")

                async for notify in conn.notifies():
                    if self._shutting_down:
                        break
                    try:
                        payload = json.loads(notify.payload)
                        event_type = payload.get("type", "")
                        # React to job lifecycle events
                        if event_type in (
                            "job_status", "job_running", "job_stopped",
                            "job_terminated", "job_failed", "job_cancelled",
                        ):
                            log.info("Event: %s — triggering sync", event_type)
                            await self.sync_routes()
                    except (json.JSONDecodeError, KeyError):
                        pass

            except Exception as e:
                log.warning("Postgres LISTEN connection lost: %s — reconnecting in 5s", e)
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
        body = json.dumps({
            "status": "ok",
            "listeners": len(self.listeners),
            "active_connections": active,
            "total_connections": self._stats["total_connections"],
            "total_bytes_relayed": self._stats["total_bytes_relayed"],
            "uptime_sec": uptime,
            "ports": sorted(self.listeners.keys()),
        })

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
            self._health_handler, "127.0.0.1", HEALTH_PORT,
        )
        log.info("Health endpoint on http://127.0.0.1:%d/", HEALTH_PORT)
        async with server:
            await server.serve_forever()

    # ── Main entry ───────────────────────────────────────────────────────

    async def run(self):
        """Main entry point — start all loops."""
        log.info("Xcelsior SSH Gateway starting...")
        log.info("Config: bind=%s, max_conns=%d, idle_timeout=%ds, sync=%ds",
                 BIND_ADDR, MAX_CONNS_PER_INSTANCE, IDLE_TIMEOUT_SEC, SYNC_INTERVAL_SEC)

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
