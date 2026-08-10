"""A migration target must pass the gate a launch passes. C3.

*"Migrated to cheaper" must never become a path onto a host that would have
failed the gate at launch* — and cheaper capacity is precisely the incentive
that would make someone want it to.

These drive the gate against real `hosts` rows, because the thing being asserted
is that a host's **current** admission state decides the answer, and that state
is computed by a database trigger from `admission_state`. A dict fixture would
be asserting my own arithmetic.

Gate P5 clause 1 — that a migrated job *resumes from its checkpoint* — is not
here and is not claimed. It needs two live instances that can share a volume.
"""

from __future__ import annotations

import json
import os
import time
import uuid

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

try:
    from control_plane.db import control_plane_transaction as pg_transaction

    with pg_transaction() as _c:
        _c.execute("SELECT 1")
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")

from control_plane.scheduler.migration_gate import (  # noqa: E402
    MigrationRefused,
    assert_target_admissible,
    evaluate_migration_target,
    migration_candidates,
)
from control_plane.scheduler.preference import (  # noqa: E402
    PlacementPreference,
    PlacementRefused,
)

DAY = 86400.0
JOB = {"gpu_model": None, "num_gpus": 1, "vram_needed_gb": 8, "region": None}


def _insert_host(conn, host_id, *, admission="admitted", status="active", vram=24.0,
                 cost=0.20, last_seen=None):
    # `admission_state`, not `administrative_state`: a BEFORE trigger derives
    # the latter, so writing it directly is silently overwritten.
    conn.execute(
        """INSERT INTO hosts (host_id, status, registered_at, payload,
                              admission_state, admitted_at)
           VALUES (%s, %s, %s, %s, %s, clock_timestamp())""",
        (
            host_id,
            status,
            time.time(),
            json.dumps(
                {
                    "host_id": host_id,
                    "status": status,
                    "admitted": admission == "admitted",
                    "gpu_model": "RTX 4090",
                    "gpu_count": 1,
                    "compute_score": 8.3,
                    "total_vram_gb": vram,
                    "free_vram_gb": vram,
                    "cost_per_hour": cost,
                    "last_seen": time.time() if last_seen is None else last_seen,
                }
            ),
            admission,
        ),
    )


@pytest.fixture
def hosts():
    tag = uuid.uuid4().hex[:10]
    ids = {
        "source": f"h-{tag}-source",
        "target": f"h-{tag}-target",
        "pending": f"h-{tag}-pending",
        "small": f"h-{tag}-small",
    }
    with pg_transaction() as conn:
        _insert_host(conn, ids["source"], cost=0.30)
        _insert_host(conn, ids["target"], cost=0.20)
        _insert_host(conn, ids["pending"], admission="pending", cost=0.10)
        _insert_host(conn, ids["small"], vram=4.0, cost=0.05)
    yield ids
    with pg_transaction() as conn:
        conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", (list(ids.values()),))


# ── Calibration ───────────────────────────────────────────────────────


def test_an_admissible_target_is_accepted(hosts):
    """If everything refused, every refusal below would prove nothing."""
    with pg_transaction() as conn:
        decision = evaluate_migration_target(conn, JOB, hosts["target"])
    assert not isinstance(decision, PlacementRefused), getattr(decision, "detail", "")
    assert decision.host["host_id"] == hosts["target"]


# ── The rule ──────────────────────────────────────────────────────────


def test_a_never_admitted_host_cannot_be_reached_by_migrating_to_it(hosts):
    """The whole point. It is also the **cheapest** host in the fixture."""
    with pg_transaction() as conn:
        decision = evaluate_migration_target(conn, JOB, hosts["pending"])

    assert isinstance(decision, PlacementRefused)
    assert decision.code == "target_not_admissible"
    assert "host_not_admitted" in decision.detail


def test_a_host_that_lost_admission_after_launch_is_refused(hosts):
    """The check is against **current** state, not the state at launch.

    This is why re-running the gate is not decorative: `administrative_state` is
    recomputed by a trigger, so a host drained or disabled since the job started
    genuinely answers differently now.
    """
    with pg_transaction() as conn:
        assert not isinstance(
            evaluate_migration_target(conn, JOB, hosts["target"]), PlacementRefused
        )
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE hosts SET admission_state = 'revoked' WHERE host_id = %s",
            (hosts["target"],),
        )
    with pg_transaction() as conn:
        decision = evaluate_migration_target(conn, JOB, hosts["target"])

    assert isinstance(decision, PlacementRefused)
    assert decision.code == "target_not_admissible"


def test_a_migration_cannot_relax_the_jobs_own_requirements(hosts):
    """Migrating is not an opportunity to land on a smaller card."""
    with pg_transaction() as conn:
        decision = evaluate_migration_target(
            conn, {**JOB, "vram_needed_gb": 16}, hosts["small"]
        )
    assert isinstance(decision, PlacementRefused)
    assert decision.code == "target_not_admissible"


def test_an_unknown_target_is_refused_rather_than_assumed_fine(hosts):
    with pg_transaction() as conn:
        decision = evaluate_migration_target(conn, JOB, "no-such-host")
    assert isinstance(decision, PlacementRefused)
    assert decision.code == "unknown_migration_target"


def test_a_missing_target_is_refused(hosts):
    with pg_transaction() as conn:
        decision = evaluate_migration_target(conn, JOB, "")
    assert isinstance(decision, PlacementRefused)
    assert decision.code == "no_migration_target"


# ── The preference is re-evaluated, not inherited ─────────────────────


def test_a_preference_the_target_cannot_satisfy_refuses_the_migration(hosts):
    """The original placement satisfying a constraint says nothing about the target.

    Inheriting the earlier decision is how a job "migrated to cheaper" onto a
    host that never met the constraint it was placed under.
    """
    with pg_transaction() as conn:
        decision = evaluate_migration_target(
            conn, JOB, hosts["target"], preference=PlacementPreference(require_verified=True)
        )
    assert isinstance(decision, PlacementRefused)
    assert decision.code in ("verification_unsatisfiable", "verification_stale")


def test_a_preference_the_target_does_satisfy_allows_it(hosts):
    """Setup and evaluation take **separate** transactions.

    `evaluate_migration_target` calls `take_snapshot`, which issues
    `SET TRANSACTION ISOLATION LEVEL REPEATABLE READ` — rejected once anything
    has run on that connection. Writing the fixture row on the same connection
    is what makes this fail, and it is the trap every caller will meet.
    """
    now = time.time()
    with pg_transaction() as conn:
        conn.execute(
            """INSERT INTO host_verifications
                 (host_id, verification_id, state, verified_at, last_check_at,
                  next_check_at)
               VALUES (%s, %s, 'verified', %s, %s, %s)""",
            (hosts["target"], f"v-{hosts['target']}", now - DAY, now - 3600, now + 82800),
        )
    try:
        with pg_transaction() as conn:
            decision = evaluate_migration_target(
                conn,
                JOB,
                hosts["target"],
                preference=PlacementPreference(require_verified=True),
            )
    finally:
        with pg_transaction() as conn:
            conn.execute(
                "DELETE FROM host_verifications WHERE host_id = %s", (hosts["target"],)
            )

    assert not isinstance(decision, PlacementRefused), getattr(decision, "detail", "")
    assert decision.host["host_id"] == hosts["target"]


# ── The raising form, for executors ───────────────────────────────────


def test_the_raising_form_carries_the_typed_refusal(hosts):
    """A migration executor has nothing sensible to do with "no" except stop.

    The refusal travels on the exception so the caller renders it rather than
    reformatting a string.
    """
    with pg_transaction() as conn:
        with pytest.raises(MigrationRefused) as caught:
            assert_target_admissible(conn, JOB, hosts["pending"])

    assert caught.value.as_dict()["code"] == "target_not_admissible"
    assert caught.value.as_dict()["refused"] is True


def test_the_raising_form_returns_the_choice_when_admissible(hosts):
    with pg_transaction() as conn:
        choice = assert_target_admissible(conn, JOB, hosts["target"])
    assert choice.host["host_id"] == hosts["target"]


# ── Listing legal targets ─────────────────────────────────────────────


def test_candidates_exclude_the_source_and_every_inadmissible_host(hosts):
    """A target that appears here is one launch would have admitted."""
    with pg_transaction() as conn:
        candidates = migration_candidates(
            conn, {**JOB, "vram_needed_gb": 16}, exclude_host_ids=[hosts["source"]]
        )

    ids = {h["host_id"] for h in candidates}
    assert hosts["target"] in ids
    assert hosts["source"] not in ids, "a job cannot be migrated onto itself"
    assert hosts["pending"] not in ids, "the cheapest host was never admitted"
    assert hosts["small"] not in ids, "a host too small for the job is not a target"
