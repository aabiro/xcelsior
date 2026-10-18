"""Bounded HTTP for untrusted public URLs (webhooks and OAuth metadata).

Resolve once per request, reject the entire DNS answer if any address is not
public, and connect to a numeric sockaddr. HTTP Host and TLS SNI/certificate
verification retain the requested hostname. No redirects, proxies, or cookies.
"""

from __future__ import annotations

import http.client
import ipaddress
import math
import queue
import socket
import ssl
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit


class PublicHTTPError(ValueError):
    """Safe to expose: messages never contain URLs, credentials, or DNS data."""


@dataclass(frozen=True)
class Destination:
    url: str
    host: str
    port: int
    tls: bool
    target: str
    addresses: tuple[tuple[int, tuple], ...]


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str]
    body: bytes


# libc DNS has no per-call timeout. Bound caller latency AND outstanding work.
# Daemon threads prevent a stuck resolver from holding process shutdown open.
_resolver_slots = threading.BoundedSemaphore(4)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise PublicHTTPError("public HTTP request timed out")
    return remaining


def _resolve(host: str, port: int, deadline: float) -> list:
    slots = _resolver_slots
    if not slots.acquire(blocking=False):
        raise PublicHTTPError("public HTTP resolver is busy")
    result: queue.Queue = queue.Queue(maxsize=1)

    def run():
        try:
            result.put(socket.getaddrinfo(host, port, type=socket.SOCK_STREAM))
        except Exception:
            result.put(None)
        finally:
            slots.release()

    try:
        threading.Thread(target=run, daemon=True, name="public-http-dns").start()
    except Exception:
        slots.release()
        raise PublicHTTPError("public HTTP resolver is unavailable") from None
    try:
        addresses = result.get(timeout=_remaining(deadline))
    except queue.Empty:
        raise PublicHTTPError("public HTTP request timed out") from None
    if not addresses:
        raise PublicHTTPError("public HTTP destination cannot be resolved")
    return addresses


def _public_address(raw: str) -> bool:
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return _public_address(str(addr.ipv4_mapped))
        # Transition mechanisms can translate a seemingly public destination
        # into private IPv4. Do not allow them as public callback addresses.
        if addr.sixtofour is not None or addr.teredo is not None:
            return False
        if addr in ipaddress.ip_network("64:ff9b::/96"):
            return False
    return addr.is_global and not (addr.is_multicast or addr.is_reserved)


def resolve_destination(url: str, *, deadline: float) -> Destination:
    u = url.strip()
    if not u or len(u) > 2048 or any(ord(c) <= 32 or ord(c) == 127 for c in u):
        raise PublicHTTPError("invalid public HTTP URL")
    try:
        parsed = urlsplit(u)
        host = (parsed.hostname or "").encode("idna").decode("ascii").lower()
        if (
            parsed.scheme not in ("http", "https")
            or not host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.fragment
            or "\\" in u
            or "%" in host
            or parsed.netloc.endswith(":")
        ):
            raise ValueError
        port = parsed.port if parsed.port is not None else (443 if parsed.scheme == "https" else 80)
        if not 1 <= port <= 65535:
            raise ValueError
        target = parsed.path or "/"
        if parsed.query:
            target += "?" + parsed.query
        target.encode("ascii")  # callers must percent-encode non-ASCII paths
    except (ValueError, UnicodeError):
        raise PublicHTTPError("invalid public HTTP URL") from None

    addresses = []
    for family, _, _, _, sockaddr in _resolve(host, port, deadline):
        if family not in (socket.AF_INET, socket.AF_INET6) or not _public_address(sockaddr[0]):
            raise PublicHTTPError("blocked public HTTP destination")
        candidate = (family, sockaddr)
        if candidate not in addresses:
            addresses.append(candidate)
    if not addresses:
        raise PublicHTTPError("public HTTP destination cannot be resolved")
    return Destination(u, host, port, parsed.scheme == "https", target, tuple(addresses))


class _PinnedConnection(http.client.HTTPConnection):
    def __init__(self, destination: Destination, deadline: float):
        super().__init__(destination.host, destination.port, timeout=_remaining(deadline))
        self.destination = destination
        self.deadline = deadline
        self.expired = threading.Event()
        self.transport_socket: socket.socket | None = None

    def abort(self):
        self.expired.set()
        # getresponse() detaches self.sock for Connection: close responses;
        # the response file still owns the descriptor until its body is read.
        sock = self.transport_socket
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()

    def connect(self):
        for family, sockaddr in self.destination.addresses:
            if self.expired.is_set():
                raise PublicHTTPError("public HTTP request timed out")
            sock = socket.socket(family, socket.SOCK_STREAM)
            self.sock = sock
            self.transport_socket = sock
            try:
                sock.settimeout(_remaining(self.deadline))
                sock.connect(sockaddr)  # numeric address: never resolve again
                if self.destination.tls:
                    # Do the handshake AFTER assigning the SSLSocket so the
                    # absolute deadline can interrupt it as well as HTTP IO.
                    sock = ssl.create_default_context().wrap_socket(
                        sock,
                        server_hostname=self.destination.host,
                        do_handshake_on_connect=False,
                    )
                    self.sock = sock
                    self.transport_socket = sock
                    sock.settimeout(_remaining(self.deadline))
                    sock.do_handshake()
                if self.expired.is_set():
                    raise PublicHTTPError("public HTTP request timed out")
                return
            except (OSError, PublicHTTPError):
                sock.close()
                self.sock = None
        raise PublicHTTPError("public HTTP connection failed")


def request(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10,
    max_response_bytes: int = 65536,
) -> Response:
    """One attempt including DNS, TLS, headers, and bounded response body.

    A zero response limit means discard the body without reading it (webhooks).
    Non-2xx statuses are returned without reading a body or following Location.
    """
    if (
        not math.isfinite(timeout)
        or timeout <= 0
        or max_response_bytes < 0
        or method not in {"GET", "POST"}
    ):
        raise PublicHTTPError("invalid public HTTP request settings")
    deadline = time.monotonic() + timeout
    try:
        destination = resolve_destination(url, deadline=deadline)
    except PublicHTTPError:
        from metrics_catalog import PUBLIC_HTTP_REFUSALS

        PUBLIC_HTTP_REFUSALS.inc()
        raise
    conn = _PinnedConnection(destination, deadline)
    timer = threading.Timer(_remaining(deadline), conn.abort)
    timer.daemon = True
    timer.start()
    try:
        conn.request(method, destination.target, body=body, headers=headers or {})
        response = conn.getresponse()
        try:
            data = b""
            if max_response_bytes and 200 <= response.status < 300:
                data = response.read(max_response_bytes + 1)
                if len(data) > max_response_bytes:
                    raise PublicHTTPError("public HTTP response is too large")
            _remaining(deadline)
            return Response(
                response.status, dict((k.lower(), v) for k, v in response.getheaders()), data
            )
        finally:
            response.close()
    except PublicHTTPError:
        raise
    except Exception:
        raise PublicHTTPError("public HTTP request failed") from None
    finally:
        timer.cancel()
        conn.close()
