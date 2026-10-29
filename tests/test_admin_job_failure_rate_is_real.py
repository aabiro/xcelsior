"""The admin dashboard's job failure rate must come from real job outcomes.

`routes/admin.py` computed it as

    SELECT COUNT(*) FILTER (WHERE status = 'failed') ... FROM usage_meters

and `usage_meters` has no `status` column. The query raised `UndefinedColumn`
on every request, `except Exception: log.debug(...)` swallowed it, and
`job_failure_rate` kept its initial `0.0` — so the admin overview has always
reported a 0% failure rate, which is also exactly what a perfectly healthy
fleet looks like.

`usage_meters` was the wrong table regardless: it records *metered* runs, so a
job that failed before metering never reaches it. The rate now comes from
`jobs.phase`, which a BEFORE INSERT/UPDATE trigger (`control_plane_project_job`)
derives from `status`, normalising a legacy vocabulary that a direct count would
have to reproduce by hand: `completed` becomes `succeeded`, and `cancelled`,
`stopped`, `paused` and `terminated` all become `stopped`. Counting
`status = 'failed'` instead would work for the numerator and quietly get the
denominator wrong.

The denominator is terminal jobs only. Counting queued and running jobs would
make the rate fall whenever the fleet got busy, which is the opposite of what
the number is for.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("XCELSIOR_ENV", "test")

from api import app
from db import _get_pg_pool

client = TestClient(app)


def _admin_headers() -> dict:
    token = os.environ.get("XCELSIOR_API_TOKEN") or "test-token-not-for-production"
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def jobs_with_known_outcomes():
    """Four jobs: one failed, two other terminal, one still running.

    Written as `status` values, because `phase` is derived from them by the
    trigger and cannot be set directly — inserting `status='succeeded'` (not in
    the vocabulary) lands `phase = NULL` and the row silently leaves the
    denominator.
    """
    now = time.time()
    made = []
    pool = _get_pg_pool()
    with pool.connection() as conn:
        for status in ("completed", "failed", "cancelled", "running"):
            job_id = f"j-fr-{uuid.uuid4().hex[:8]}"
            conn.execute(
                """INSERT INTO jobs (job_id, status, priority, submitted_at, payload)
                   VALUES (%s, %s, 0, %s, %s)""",
                (job_id, status, now - 60, json.dumps({"job_id": job_id})),
            )
            made.append(job_id)
        conn.commit()
        phases = dict(
            conn.execute(
                "SELECT job_id, phase FROM jobs WHERE job_id = ANY(%s)", (made,)
            ).fetchall()
        )
    assert sorted(p for p in phases.values() if p) == [
        "failed",
        "running",
        "stopped",
        "succeeded",
    ], f"the trigger did not derive the phases this test depends on: {phases}"
    yield made
    with pool.connection() as conn:
        conn.execute("DELETE FROM jobs WHERE job_id = ANY(%s)", (made,))
        conn.commit()


def _authoritative_rate() -> float:
    """The same figure counted directly, so pre-existing rows cannot skew this."""
    with _get_pg_pool().connection() as conn:
        failed, total = conn.execute(
            """SELECT COUNT(*) FILTER (WHERE phase = 'failed'), COUNT(*)
                 FROM jobs
                WHERE submitted_at >= %s
                  AND phase IN ('succeeded', 'failed', 'stopped')""",
            (time.time() - 30 * 86400,),
        ).fetchone()
    return round(failed / total * 100, 1) if total else 0.0


def _failure_rate() -> float:
    r = client.get("/api/admin/overview", headers=_admin_headers())
    assert r.status_code == 200, r.text[:300]
    body = r.json()
    # The KPI lives under whichever key the overview nests it in.
    found = []

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "job_failure_rate":
                    found.append(v)
                else:
                    walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(body)
    assert found, f"no job_failure_rate in the overview response: {list(body)}"
    return float(found[0])


def test_a_failed_job_moves_the_rate_off_zero(jobs_with_known_outcomes):
    rate = _failure_rate()
    assert rate > 0, (
        "the rate is still 0.0 with a failed job on the books — the query behind "
        "it is failing and the handler is swallowing it"
    )


def test_the_reported_rate_matches_a_direct_count(jobs_with_known_outcomes):
    assert _failure_rate() == pytest.approx(_authoritative_rate(), abs=0.1)


def test_running_jobs_are_not_in_the_denominator(jobs_with_known_outcomes):
    """A job that has not finished cannot have failed, so it is not a trial."""
    with _get_pg_pool().connection() as conn:
        counted = conn.execute(
            """SELECT COUNT(*) FROM jobs
                WHERE submitted_at >= %s
                  AND phase IN ('succeeded', 'failed', 'stopped')
                  AND phase = 'running'""",
            (time.time() - 30 * 86400,),
        ).fetchone()[0]
    assert counted == 0

    # And the rate is strictly above what it would be with running jobs added in.
    with _get_pg_pool().connection() as conn:
        failed, terminal, everything = conn.execute(
            """SELECT COUNT(*) FILTER (WHERE phase = 'failed'),
                      COUNT(*) FILTER (WHERE phase IN ('succeeded','failed','stopped')),
                      COUNT(*) FILTER (WHERE phase IS NOT NULL)
                 FROM jobs WHERE submitted_at >= %s""",
            (time.time() - 30 * 86400,),
        ).fetchone()
    assert everything > terminal, "fixture must leave a non-terminal job to compare against"
    assert _failure_rate() == pytest.approx(round(failed / terminal * 100, 1), abs=0.1)
