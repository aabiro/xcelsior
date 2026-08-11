"""C3's executor: the order, and what "resumed" is allowed to mean.

Gate P5 clause 1 says a migrated job resumes *"proven by comparing state before
and after — not by the absence of an error."* So the assertion that matters here
is not that `migrate_job` returns ok; it is that `resumed` is **never `True`
without a comparison having happened**.

The CRIU halves are stubbed. What is under test is the orchestration: admission
before anything is frozen, no teardown before the target is up, and an honest
verdict on whether it resumed. The live version of this is
`tests/live/test_migration_resumes_live.py`, which needs a fleet.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

from control_plane.scheduler import migration_executor as mx  # noqa: E402
from control_plane.scheduler.preference import (  # noqa: E402
    PlacementChoice,
    PlacementRefused,
)

JOB = {"job_id": "j-mig", "status": "running", "host_id": "host-a", "vram_needed_gb": 8}


@pytest.fixture
def wiring(monkeypatch):
    """Stub the CRIU halves and the gate; record the call order."""
    calls: list[str] = []

    monkeypatch.setattr("scheduler.get_job", lambda job_id: dict(JOB), raising=False)

    def _checkpoint(host_id, job_id, container_name=""):
        calls.append("checkpoint")
        return {"checkpoint_name": "ckpt-1", "host_ip": "10.0.0.1"}

    def _resume(job_id, target_host_id, meta):
        calls.append("resume")
        return True

    monkeypatch.setattr("scheduler.checkpoint_container", _checkpoint, raising=False)
    monkeypatch.setattr("scheduler.resume_from_checkpoint", _resume, raising=False)

    def _admit(conn, job, target, *, preference=None, now=None):
        calls.append("admission")
        return PlacementChoice(
            host={"host_id": target}, baseline_price=10.0, chosen_price=10.0,
            premium_pct=0.0, evidence={},
        )

    monkeypatch.setattr(mx, "evaluate_migration_target", _admit)
    return calls


# ── The clause ────────────────────────────────────────────────────────


def test_a_matching_probe_is_the_only_thing_that_proves_resumption(wiring):
    out = mx.migrate_job("j-mig", "host-b", state_probe=lambda job_id: "step=42")
    assert out.ok is True
    assert out.resumed is True
    assert out.state_before == "step=42" and out.state_after == "step=42"


def test_no_probe_means_unknown_never_true(wiring):
    """**The clause's actual demand.** Absence of an error is not proof.

    `resumed=None` has to stay distinguishable from `resumed=True`, or "nobody
    checked" renders identically to "verified".
    """
    out = mx.migrate_job("j-mig", "host-b")
    assert out.ok is True
    assert out.resumed is None, "an unverified migration reported itself verified"
    assert "not verified" in out.detail


def test_a_mismatched_probe_is_a_failure_to_resume(wiring):
    readings = iter(["step=42", "step=0"])
    out = mx.migrate_job("j-mig", "host-b", state_probe=lambda job_id: next(readings))
    assert out.resumed is False
    assert out.state_before != out.state_after


def test_a_probe_that_cannot_read_before_stops_the_migration(wiring):
    """Nothing is frozen if the comparison could never happen."""
    def _boom(job_id):
        raise RuntimeError("no telemetry")

    out = mx.migrate_job("j-mig", "host-b", state_probe=_boom)
    assert out.ok is False
    assert out.failure_code == "state_probe_failed"
    assert "checkpoint" not in wiring, "the source was frozen for an unverifiable move"


# ── The order ─────────────────────────────────────────────────────────


def test_admission_runs_before_the_container_is_frozen(wiring):
    mx.migrate_job("j-mig", "host-b")
    assert wiring == ["admission", "checkpoint", "resume"]


def test_a_refused_target_never_touches_the_running_job(monkeypatch, wiring):
    """A job stopped for a migration that is then refused is lost for nothing."""
    monkeypatch.setattr(
        mx,
        "evaluate_migration_target",
        lambda *a, **k: PlacementRefused(code="target_not_admissible", detail="no"),
    )
    out = mx.migrate_job("j-mig", "host-b")
    assert out.ok is False
    assert out.failure_code == "target_not_admissible"
    assert wiring == [], "the source was checkpointed for a migration that was refused"


def test_a_failed_resume_says_the_source_was_not_destroyed(monkeypatch, wiring):
    """The operator needs to know whether they still have a job."""
    monkeypatch.setattr("scheduler.resume_from_checkpoint", lambda *a: False, raising=False)
    out = mx.migrate_job("j-mig", "host-b")
    assert out.ok is False
    assert out.failure_code == "resume_failed"
    assert "frozen rather than removed" in out.detail


# ── Preconditions ─────────────────────────────────────────────────────


def test_a_job_that_is_not_running_is_refused(monkeypatch, wiring):
    monkeypatch.setattr(
        "scheduler.get_job", lambda job_id: dict(JOB, status="queued"), raising=False
    )
    out = mx.migrate_job("j-mig", "host-b")
    assert out.failure_code == "job_not_running"
    assert wiring == []


def test_migrating_to_the_current_host_is_refused(wiring):
    out = mx.migrate_job("j-mig", "host-a")
    assert out.failure_code == "already_on_target"
    assert wiring == []


def test_an_unknown_job_is_refused(monkeypatch, wiring):
    monkeypatch.setattr("scheduler.get_job", lambda job_id: None, raising=False)
    out = mx.migrate_job("j-mig", "host-b")
    assert out.failure_code == "job_not_found"
