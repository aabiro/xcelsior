"""Gunicorn configuration for Xcelsior API — zero-downtime deployments.

Uses uvicorn workers for async support. The PORT env var allows
blue-green deployments on different ports (9500 / 9501).
"""

import os

# ---------- Server socket ----------
_port = os.getenv("XCELSIOR_API_PORT", "9500")

# `0.0.0.0` is right for production, where nginx terminates TLS and proxies to
# this socket. It is wrong for a staging stack on a developer machine: with
# `network_mode: host` the port is then reachable from the LAN and the tailnet,
# so a local test environment exposes the worker protocol to every peer.
#
# Configurable rather than changed, and defaulted to the existing value, so
# production's posture is untouched and staging opts into loopback. The
# alternative — flipping staging's agent ingress to `allow` — would have made
# the exposure worse, not better.
# **A list, not a wildcard.** Reaching a staging stack from another host used to
# mean widening this to `0.0.0.0`, which is precisely the exposure above — the
# choice was "loopback only" or "every interface", and wanting one peer meant
# accepting all of them. A comma-separated allowlist removes that trade:
# `XCELSIOR_API_BIND=127.0.0.1,100.64.0.6` serves loopback (so the compose
# healthcheck's `curl localhost` still works) and the tailnet address (so a
# named peer can reach it), while the LAN interface stays unbound.
#
# The default is unchanged and single-valued, so production — where nginx
# terminates TLS and proxies to this socket — is untouched.
_hosts = [h.strip() for h in os.getenv("XCELSIOR_API_BIND", "0.0.0.0").split(",") if h.strip()]
bind = [f"{_host}:{_port}" for _host in _hosts]

# ---------- Worker processes ----------
workers = int(os.getenv("GUNICORN_WORKERS", "2"))

# In-memory auth and more than one worker are silently incompatible, and the
# symptom is the worst kind: authentication that works about half the time.
#
# `XCELSIOR_PERSISTENT_AUTH=false` makes `routes._deps._users_db` — a plain dict
# in the worker's own memory — the *user store*, not a cache. Registration lands
# on whichever worker served that request; the next login round-robins, and a
# worker that never saw the registration answers 401. Measured on this repo's
# test stack with the default two workers:
#
#     401 401 401 401 200 200 401 200 401 401
#
# Nothing reported an error. Each worker was behaving correctly, the user row
# existed in no database because it had never been written to one, and the
# register response carried a complete user object because the worker that
# served it genuinely had created one. It reads as a flaky password.
#
# Refusing to boot rather than quietly setting `workers = 1`: a developer who
# asked for four workers and got one would debug the wrong thing later, and the
# combination is never what anyone means.
#: Flags whose "off" position replaces a shared store with a per-process one.
#: Each is fine on a single worker and silently broken on several, in the same
#: way and for the same reason.
_PER_PROCESS_WHEN_OFF = {
    "XCELSIOR_PERSISTENT_AUTH": (
        "users live in `routes._deps._users_db`, a dict in the worker's own "
        "memory, so a login served by a worker that did not handle the "
        "registration answers 401"
    ),
    # `_shared_state_update` returns False without logging when this is off, and
    # WebSocket tickets then fall back to `routes._deps._WS_TICKETS` — also
    # per-process. A terminal ticket issued by one worker cannot be redeemed by
    # another, which is the same intermittent failure one layer along, in the
    # connection path rather than the login path.
    "XCELSIOR_SHARED_RUNTIME_LIMITS": (
        "WebSocket tickets and rate-limit buckets live in per-process dicts, so "
        "a ticket issued by one worker cannot be redeemed by another and rate "
        "limits are effectively multiplied by the worker count"
    ),
}

_off = {
    name: why
    for name, why in _PER_PROCESS_WHEN_OFF.items()
    if os.getenv(name, "true").strip().lower() in {"0", "false", "no", "off"}
}
if _off and workers > 1:
    _detail = "; ".join(f"{name}=false — {why}" for name, why in sorted(_off.items()))
    raise RuntimeError(
        f"{', '.join(sorted(_off))} disabled with GUNICORN_WORKERS={workers}: "
        f"{_detail}. Requests round-robin, so this fails roughly "
        f"{100 - int(100 / workers)}% of the time with nothing logged. Re-enable "
        "the shared store (what production does), or set GUNICORN_WORKERS=1 if "
        "the per-process behaviour is genuinely what you want."
    )
worker_class = "uvicorn.workers.UvicornWorker"

# ---------- Which proxies may name the client ----------
# `routes._deps._get_real_client_ip` reads only `request.client.host`, which
# uvicorn resolves from X-Forwarded-For by walking it from the right and
# stopping at the first hop it does not trust. That closed header spoofing
# (fa37079), but with only loopback trusted it stopped one hop too early:
# every request reaches nginx *from Cloudflare*, so nginx appends a Cloudflare
# edge address, and that edge became "the client" for everyone behind it.
# Measured on production on 2026-10-06: a request from 207.219.90.237 was
# logged as 172.69.130.143. Every per-IP control was then shared by all
# visitors on an edge — the login brute-force limit, the flood limit, the
# WebSocket ticket's IP pin, audit addresses, the demo-account gate.
#
# Trusting Cloudflare's published edge ranges as well moves the walk one hop
# further, to the address Cloudflare saw — the visitor. It does not reopen
# spoofing: a client that bypasses Cloudflare arrives with its own address
# appended by nginx, that address is not in these ranges, and the walk stops
# there; anything it wrote further left is never read. Source:
# https://www.cloudflare.com/ips-v4 and /ips-v6, fetched 2026-10-06.
CLOUDFLARE_EDGE_RANGES = (
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
    "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
    "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
    "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32",
    "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32",
)
forwarded_allow_ips = ",".join(("127.0.0.1", "::1", *CLOUDFLARE_EDGE_RANGES))
worker_tmp_dir = "/dev/shm"  # faster heartbeat on Linux

# ---------- Graceful lifecycle ----------
graceful_timeout = int(os.getenv("GUNICORN_GRACEFUL_TIMEOUT", "120"))
timeout = 120  # hard kill after this
keepalive = 5  # keep-alive between nginx ↔ gunicorn

# ---------- Preload for faster worker spawns ----------
preload_app = True


# ---------- Post-fork: reset shared state that doesn't survive fork ----------
def post_fork(server, worker):
    """Reset PostgreSQL connection pool after fork.

    With preload_app=True the pool may be created in the master process during
    module import.  Forked workers inherit the pool object but the underlying
    TCP sockets are shared/corrupted across processes.  Resetting the global
    forces each worker to create its own pool on first use.
    """
    import threading
    import db

    db._pg_pool = None
    db._pg_pool_lock = threading.Lock()


# ---------- Logging ----------
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
