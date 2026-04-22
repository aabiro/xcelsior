"""Gunicorn configuration for Xcelsior API — zero-downtime deployments.

Uses uvicorn workers for async support. The PORT env var allows
blue-green deployments on different ports (9500 / 9501).
"""

import os

# ---------- Server socket ----------
_port = os.getenv("XCELSIOR_API_PORT", "9500")
bind = f"0.0.0.0:{_port}"

# ---------- Worker processes ----------
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
worker_class = "uvicorn.workers.UvicornWorker"
worker_tmp_dir = "/dev/shm"  # faster heartbeat on Linux

# ---------- Graceful lifecycle ----------
graceful_timeout = 30  # seconds to finish in-flight requests
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
