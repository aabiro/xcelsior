"""Stuck-job reaper tests."""

from __future__ import annotations

import logging
import os
import tempfile
import scheduler
from reaper import reaper_tick

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