"""Stuck-job reaper tests."""

from __future__ import annotations

import logging
import os
import tempfile
import time

import scheduler
from db import DB_BACKEND
from reaper import reaper_tick


def _db_backend() -> str:
    return "postgres" if DB_BACKEND in ("postgres", "dual") else "sqlite"

_tmp_ctx = tempfile.TemporaryDirectory(prefix="xcelsior_reaper_test_")
_tmpdir = _tmp_ctx.name

os.environ["XCELSIOR_API_TOKEN"] = ""
os.environ["XCELSIOR_DB_PATH"] = os.path.join(_tmpdir, "xcelsior.db")
os.environ["XCELSIOR_ENV"] = "test"

scheduler.HOSTS_FILE = os.path.join(_tmpdir, "hosts.json")
scheduler.JOBS_FILE = os.path.join(_tmpdir, "jobs.json")
scheduler.BILLING_FILE = os.path.join(_tmpdir, "billing.json")
scheduler.MARKETPLACE_FILE = os.path.join(_tmpdir, "marketplace.json")
scheduler.AUTOSCALE_POOL_FILE = os.path.join(_tmpdir, "autoscale_pool.json")
scheduler.SPOT_PRICES_FILE = os.path.join(_tmpdir, "spot_prices.json")
scheduler.COMPUTE_SCORES_FILE = os.path.join(_tmpdir, "compute_scores.json")
scheduler.LOG_FILE = os.path.join(_tmpdir, "xcelsior.log")

for h in scheduler.log.handlers[:]:
    if isinstance(h, logging.FileHandler):
        scheduler.log.removeHandler(h)
        h.close()
_fh = logging.FileHandler(scheduler.LOG_FILE)
_fh.setLevel(logging.INFO)
_fh.setFormatter(
    logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
)
scheduler.log.addHandler(_fh)


def _reset_jobs():
    with scheduler._atomic_mutation() as conn:
        conn.execute("DELETE FROM jobs")


class TestReaper:
    def test_reaper_sql_syncs_payload_status(self):
        import inspect

        source = inspect.getsource(reaper_tick)
        assert "'{status}'" in source
        assert '"failed"' in source

    def test_requeue_resets_submitted_at(self):
        _reset_jobs()
        job = scheduler.submit_job("requeue-clock", 0)
        job_id = job["job_id"]
        backend = _db_backend()
        with scheduler._atomic_mutation() as conn:
            row = scheduler.DatabaseOps.get_job(conn, job_id, backend=backend)
            row["submitted_at"] = time.time() - 10_000
            row["status"] = "failed"
            scheduler.DatabaseOps.upsert_job(conn, row, backend=backend)

        result = scheduler.requeue_job(job_id, user_initiated=True)
        assert result is not None
        refreshed = scheduler.get_job(job_id)
        assert refreshed["status"] == "queued"
        assert refreshed["submitted_at"] > time.time() - 60

    def test_requeue_cancelled_job(self):
        _reset_jobs()
        job = scheduler.submit_job("cancel-relaunch", 0)
        job_id = job["job_id"]
        scheduler.update_job_status(job_id, "assigned", host_id="host-1")
        scheduler.update_job_status(job_id, "cancelled")

        result = scheduler.requeue_job(job_id)
        assert result is not None
        assert result["status"] == "queued"
        assert result.get("retries", 0) == 0
        assert result.get("host_id") is None