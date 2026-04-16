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
worker_tmp_dir = "/dev/shm"          # faster heartbeat on Linux

# ---------- Graceful lifecycle ----------
graceful_timeout = 30                 # seconds to finish in-flight requests
timeout = 120                         # hard kill after this
keepalive = 5                         # keep-alive between nginx ↔ gunicorn

# ---------- Preload for faster worker spawns ----------
preload_app = True

# ---------- Logging ----------
accesslog = "-"
errorlog = "-"
loglevel = os.getenv("GUNICORN_LOG_LEVEL", "info")
