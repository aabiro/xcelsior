"""CVP-01 §3, control-plane side: the record, the reply, the decision."""

import uuid

import pytest

import mining_review as mrv
from db import DatabaseOps, get_engine


@pytest.fixture
def job():
    """A job row on host h-1 for a fresh owner; the state it leaves is removed."""
    owner = "mining-test-" + uuid.uuid4().hex[:12]
    job_id = "job-" + uuid.uuid4().hex[:12]
    with get_engine().transaction() as (conn, backend):
        DatabaseOps.upsert_job(
            conn,
            {
                "job_id": job_id,
                "name": "probe",
                "owner": owner,
                "owner_id": owner,
                "status": "running",
                "host_id": "h-1",
                "egress": "restricted",
            },
            backend=backend,
        )
    yield {"job_id": job_id, "owner": owner}
    with get_engine().transaction() as (conn, backend):
        ph = "%s" if backend == "postgres" else "?"
        conn.execute(f"DELETE FROM state WHERE namespace LIKE {ph}", (f"security.%{owner}%",))
        conn.execute(f"DELETE FROM state WHERE namespace LIKE {ph}", (f"security.%{job_id}%",))
        conn.execute(f"DELETE FROM jobs WHERE job_id = {ph}", (job_id,))


@pytest.fixture
def quiet(monkeypatch):
    """Capture the alert and the event instead of sending them."""
    sent = []
    monkeypatch.setattr(
        mrv, "_emit", lambda job_id, owner, host_id, record: sent.append(dict(record))
    )
    return sent


def report(job, rung, confidence=0.6, host_id="h-1", **evidence):
    return mrv.record_episode(
        host_id=host_id,
        job_id=job["job_id"],
        container_name="xcl-1",
        rung=rung,
        confidence=confidence,
        evidence={"signals": [{"family": "S1", "detail": "x"}], **evidence},
    )


def test_a_host_may_only_report_its_own_containers(job, quiet):
    assert report(job, "flag", host_id="h-2") == {"ok": False, "error": "job is not on this host"}
    assert mrv.record_episode(
        host_id="h-1", job_id="nope", container_name="c", rung="flag", confidence=0.5, evidence={}
    ) == {"ok": False, "error": "unknown job"}
    assert quiet == []


def test_first_flag_records_reviews_alerts_and_gives_the_unfunded_prior(job, quiet):
    reply = report(job, "flag")
    assert reply == {
        "ok": True,
        "received": True,
        "prior": mrv.PRIOR,
        "repeat_offender": False,
        "decision": None,
    }
    assert [r["rung"] for r in quiet] == ["flag"]
    with get_engine().connection() as (conn, backend):
        record = DatabaseOps.get_state(conn, mrv._job_ns(job["job_id"]), backend=backend)
        review = DatabaseOps.get_state(
            conn, f"security.review.{job['owner']}.{job['job_id']}", backend=backend
        )
    assert (
        record["host_id"] == "h-1" and record["rung"] == "flag" and record["owner"] == job["owner"]
    )
    assert review["status"] == "needs_review" and review["reason"] == mrv.REASON
    # The same rung again is not a new alert; a higher one is.
    report(job, "flag")
    report(job, "throttle", 0.8)
    assert [r["rung"] for r in quiet] == ["flag", "throttle"]


def test_a_prior_suspension_on_the_account_makes_a_repeat_offender(job, quiet):
    other = "job-" + uuid.uuid4().hex[:12]
    with get_engine().transaction() as (conn, backend):
        DatabaseOps.upsert_state(
            conn,
            mrv._account_ns(job["owner"]),
            {"jobs": {other: {"rung": "suspend", "updated_at": 1.0, "closed": None}}},
            backend=backend,
        )
    reply = report(job, "flag")
    assert reply["repeat_offender"] is True and reply["prior"] == 0.0


def test_decision_reaches_the_next_report_and_settles_the_review(job, quiet):
    report(job, "suspend", 0.95, suspended_at=123.0)
    with pytest.raises(KeyError):
        mrv.decide("job-unknown", "clear", actor="a", reason="r")
    with pytest.raises(ValueError):
        mrv.decide(job["job_id"], "pause", actor="a", reason="r")
    with pytest.raises(ValueError):
        mrv.decide(job["job_id"], "clear", actor="a", reason="")
    record = mrv.decide(
        job["job_id"], "clear", actor="admin@example.test", reason="their own training run"
    )
    assert record["decision"] == "clear" and record["decided_by"] == "admin@example.test"
    assert report(job, "suspend", 0.95, suspended_at=123.0)["decision"] == "clear"
    with get_engine().connection() as (conn, backend):
        review = DatabaseOps.get_state(
            conn, f"security.review.{job['owner']}.{job['job_id']}", backend=backend
        )
    assert review["status"] == "cleared"


def test_open_episodes_are_listed_and_closed_ones_are_not(job, quiet):
    report(job, "throttle", 0.8)
    assert any(r["job_id"] == job["job_id"] for r in mrv.list_episodes())
    report(job, None, 0.0, closed="signals gone for 30 min")
    assert not any(r["job_id"] == job["job_id"] for r in mrv.list_episodes())
    assert any(r["job_id"] == job["job_id"] for r in mrv.list_episodes(include_closed=True))


def test_evidence_is_bounded():
    big = {"signals": [{"detail": "x" * 1000} for _ in range(100)]}
    trimmed = mrv._trim_evidence(big)
    assert len(trimmed["signals"]) < 100
    assert mrv._trim_evidence({"blob": "y" * 20_000}) == {"truncated": True}
