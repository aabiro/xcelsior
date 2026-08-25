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
worker_class = "uvicorn.workers.UvicornWorker"
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
