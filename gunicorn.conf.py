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
_persistent_auth = os.getenv("XCELSIOR_PERSISTENT_AUTH", "true").strip().lower() != "false"
if not _persistent_auth and workers > 1:
    raise RuntimeError(
        f"XCELSIOR_PERSISTENT_AUTH=false with GUNICORN_WORKERS={workers}: "
        "in-memory auth is per-process, so logins would fail on every worker "
        "that did not serve the registration — roughly "
        f"{100 - int(100 / workers)}% of the time, with no error logged. "
        "Set XCELSIOR_PERSISTENT_AUTH=true to use the database (what production "
        "does), or GUNICORN_WORKERS=1 if you genuinely want the in-memory store."
    )
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
