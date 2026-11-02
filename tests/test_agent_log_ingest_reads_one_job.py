"""The agent log endpoint looks up the one job it is writing for.

Every agent posts a log batch about once a second per running job. The guard
that drops logs for unknown and long-finished jobs used to load the whole jobs
table to read one row, on the event loop: 0.2 s per batch on prod and 9 s on a
137k-job database, during which no other request on that process moved.
"""

import os
import time
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import scheduler  # noqa: E402
from api import app  # noqa: E402

client = TestClient(app)


@pytest.fixture
def jobs(monkeypatch):
    rows: dict[str, dict] = {}
    monkeypatch.setattr(scheduler, "get_job", rows.get)

    def whole_table():
        raise AssertionError("log ingest must not load every job")

    monkeypatch.setattr(scheduler, "load_jobs", whole_table)
    return rows


def post(job_id):
    return client.post(
        f"/agent/logs/{job_id}", json={"lines": [{"message": "output", "level": "info"}]}
    ).json()


def test_unknown_job_is_dropped(jobs):
    assert post("job-" + uuid.uuid4().hex[:12])["dropped"] == "unknown_job"


def test_job_finished_over_an_hour_ago_is_dropped(jobs):
    job_id = "job-" + uuid.uuid4().hex[:12]
    jobs[job_id] = {"job_id": job_id, "status": "completed", "completed_at": time.time() - 7200}
    assert post(job_id)["dropped"] == "job_terminal"


def test_running_and_just_finished_jobs_are_accepted(jobs):
    for status, completed_at in (("running", None), ("completed", time.time() - 5)):
        job_id = "job-" + uuid.uuid4().hex[:12]
        jobs[job_id] = {"job_id": job_id, "status": status, "completed_at": completed_at}
        assert post(job_id)["accepted"] == 1
