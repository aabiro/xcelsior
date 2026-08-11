"""Gate P5 clause 2 and clause 3, through the route rather than the module.

`tests/test_placement_preference_refuses.py` drives `choose_host` over dicts and
proves the refusal semantics. It cannot prove the clause, because the clause is
about what *the system* does — and for four commits the honest answer was "the
module refuses correctly and nothing calls it".

This calls `POST /api/v1/placements/evaluate` against real hosts in the database
and asserts what a caller is actually told, and what is left behind in the
trail.

| Clause | Asserted here |
|---|---|
| 2 — an unsatisfiable preference refuses clearly rather than falling back | by asking for something the fleet cannot give and checking no host comes back |
| 3 — reputation and SLA at time of placement are recorded | by reading the row afterwards and comparing it to what was true then |

The fixture hosts are deliberately *placeable*: an empty fleet refuses with
`no_eligible_hosts` for reasons that have nothing to do with the preference, and
a test that passes for the wrong reason is worse than none.
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
        _has = (
            _c.execute("SELECT to_regclass('placement_decisions')").fetchone()[0]
            is not None
        )
except Exception as _e:  # pragma: no cover
    pytestmark = pytest.mark.skip(f"no control-plane db: {_e}")
else:
    if not _has:  # pragma: no cover
        pytestmark = pytest.mark.skip("test database is behind migration 106")

DAY = 86400.0
YEAR = 365 * DAY


def _past_month(now: float, offset: int) -> tuple[str, float]:
    """(`YYYY-MM`, seconds in it) for the month `offset` months before `now`.

    Full past months only, so the projection's elapsed-time clamp is a no-op
    here and the fixture's uptime is the number the test asserts.
    """
    import calendar
    import datetime as dt

    ref = dt.datetime.fromtimestamp(now)
    year, month = ref.year, ref.month - offset
    while month <= 0:
        month += 12
        year -= 1
    return f"{year:04d}-{month:02d}", calendar.monthrange(year, month)[1] * DAY


@pytest.fixture
def fleet():
    """Two placeable hosts, and the evidence that separates them.

    `cheap` is unverified at 97%; `good` is verified at 99.95% and costs 10%
    more. Both carry three full past months of SLA rows — comfortably over the
    30-day observation floor, so `min_uptime_pct` exercises the constraint and
    not the floor. Every assertion below turns on that difference being real in
    the database rather than in a dict.
    """
    tag = uuid.uuid4().hex[:10]
    cheap, good = f"h-{tag}-cheap", f"h-{tag}-good"
    now = time.time()
    # **A GPU model no other test can produce.** `filter_hosts` matches
    # `gpu_model` exactly, so naming it here confines every assertion below to
    # this fixture's two hosts. Without it the tests assert counts and premiums
    # over *the whole fleet*, and any other admitted host in the database at
    # that moment becomes a third candidate — which is exactly how
    # `candidate_count == 2` became `3` in a full-suite run and passed in
    # isolation.
    model = f"TESTGPU-{tag}"

    with pg_transaction() as conn:
        for host_id, cost, gpu in ((cheap, 0.20, model), (good, 0.22, model)):
            # `admission_state`, **not** `administrative_state`: a BEFORE
            # trigger (`control_plane_project_host`) derives the latter from
            # the former, so setting the projected column directly is silently
            # overwritten and the host arrives at the hard filter as `pending`.
            conn.execute(
                """INSERT INTO hosts (host_id, status, registered_at, payload,
                                      admission_state, admitted_at)
                   VALUES (%s, 'active', %s, %s, 'admitted', clock_timestamp())""",
                (
                    host_id,
                    now,
                    json.dumps(
                        {
                            "host_id": host_id,
                            "status": "active",
                            "admitted": True,
                            "gpu_model": gpu,
                            "gpu_count": 1,
                            "compute_score": 8.3,
                            "total_vram_gb": 24.0,
                            "free_vram_gb": 24.0,
                            "cost_per_hour": cost,
                            "last_seen": now,
                        }
                    ),
                ),
            )
            # **Real monthly rows, not a year in one of them.** `sla_monthly` is
            # per calendar month, so `total_seconds = YEAR` in a single row is
            # data production cannot produce — and the projection clamps each
            # month to elapsed time, which turned a fixture meant to read 99.95%
            # into 99.55%. The same impossible-fixture class the C0 builder
            # refuses outright.
            #
            # `good` is 99.95% and not 100%: a host with literally zero downtime
            # satisfies *every* uptime constraint, so the "unsatisfiable" test
            # below would have had nothing to refuse.
            downtime_fraction = 0.0005 if host_id == good else 0.03
            for offset in (1, 2, 3):
                month_str, month_seconds = _past_month(now, offset)
                conn.execute(
                    """INSERT INTO sla_monthly (host_id, month, tier,
                                                total_seconds, downtime_seconds)
                       VALUES (%s, %s, 'community', %s, %s)""",
                    (
                        host_id,
                        month_str,
                        month_seconds,
                        month_seconds * downtime_fraction,
                    ),
                )

        conn.execute(
            """INSERT INTO host_verifications
                 (host_id, verification_id, state, verified_at, last_check_at,
                  next_check_at)
               VALUES (%s, %s, 'verified', %s, %s, %s)""",
            (good, f"v-{good}", now - 2 * DAY, now - 3600, now + 82800),
        )
        conn.execute(
            """INSERT INTO host_verifications (host_id, verification_id, state)
               VALUES (%s, %s, 'unverified')""",
            (cheap, f"v-{cheap}"),
        )
        for host_id, score in ((cheap, 50.0), (good, 500.0)):
            conn.execute(
                """INSERT INTO reputation_scores
                     (entity_id, entity_type, tier, raw_score, final_score)
                   VALUES (%s, 'host', 'new_user', %s, %s)""",
                (host_id, score, score),
            )

    yield {"cheap": cheap, "good": good, "gpu_model": model}

    ids = [cheap, good]
    with pg_transaction() as conn:
        conn.execute("DELETE FROM hosts WHERE host_id = ANY(%s)", (ids,))
        conn.execute("DELETE FROM sla_monthly WHERE host_id = ANY(%s)", (ids,))
        conn.execute("DELETE FROM host_verifications WHERE host_id = ANY(%s)", (ids,))
        conn.execute("DELETE FROM reputation_scores WHERE entity_id = ANY(%s)", (ids,))


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import api as api_mod
    import routes.action_plans as ap
    from routes import _deps

    tenant = f"tenant-{uuid.uuid4().hex[:10]}"
    principal = {
        "email": "demo@xcelsior.ca",
        "user_id": f"user-{tenant}",
        "role": "user",
        "auth_type": "oauth_access_token",
        "session_type": "browser",
        "client_id": "xcelsior-web",
        "customer_id": tenant,
        "scopes": ["instances:read", "instances:write"],
    }
    monkeypatch.setattr(_deps, "_require_auth", lambda request: dict(principal))
    monkeypatch.setattr(ap, "_require_auth", lambda request: dict(principal))
    monkeypatch.setattr(ap, "_effective_billing_customer_id", lambda user: tenant)
    monkeypatch.setattr(ap, "_canonical_owner_id", lambda user: principal["user_id"])
    monkeypatch.setattr(ap, "_user_team_id", lambda user: None)
    return TestClient(api_mod.app), tenant


def _evaluate(client, fleet, preference, *, vram=8):
    """Always asks for the fixture's own GPU model — see the `fleet` fixture."""
    c, tenant = client
    response = c.post(
        "/api/v1/placements/evaluate",
        json={
            "spec": {
                "name": "p5-gate",
                "vram_needed_gb": vram,
                "num_gpus": 1,
                "gpu_model": fleet["gpu_model"],
            },
            "preference": preference,
        },
    )
    assert response.status_code == 200, response.text
    return response.json(), tenant


# ── Calibration: the fleet is placeable ───────────────────────────────


def test_an_unconstrained_preference_places(client, fleet):
    """If everything refused, every refusal below would prove nothing."""
    body, _ = _evaluate(client, fleet, {})
    assert body["preference"]["refused"] is False, body["preference"]
    assert body["preference"]["host_id"] in (fleet["cheap"], fleet["good"])


# ── Clause 2: it refuses, it does not settle ──────────────────────────


def test_an_unsatisfiable_uptime_constraint_refuses_with_the_number(client, fleet):
    """The clause, end to end.

    Silently returning `cheap` at 97% would be the failure the plan calls *"the
    failure mode that would quietly destroy trust"*.
    """
    body, _ = _evaluate(client, fleet, {"min_uptime_pct": 99.99})
    pref = body["preference"]

    assert pref["refused"] is True, "an unsatisfiable preference placed a host anyway"
    assert pref["code"] == "uptime_unsatisfiable"
    assert pref["asked"] == 99.99
    assert pref["best_available"] == pytest.approx(99.95, abs=0.01), (
        "the refusal does not say what was actually available, so the caller "
        "cannot decide whether to relax it"
    )
    assert "host_id" not in pref


def test_a_satisfiable_constraint_gets_the_host_that_meets_it(client, fleet):
    """The other half. A gate that always refuses is indistinguishable from broken."""
    body, _ = _evaluate(client, fleet, {"min_uptime_pct": 99.5, "require_verified": True})
    pref = body["preference"]
    assert pref["refused"] is False, pref
    assert pref["host_id"] == fleet["good"]
    assert pref["premium_pct"] == pytest.approx(10.0), (
        "the premium is not measured against the cheapest eligible host"
    )


def test_a_premium_bound_below_the_cost_of_the_constraint_refuses(client, fleet):
    """"Even at 15% more" is a bound. A bound not enforced is a hint."""
    body, _ = _evaluate(
        client, fleet, {"require_verified": True, "max_premium_pct": 5}
    )
    pref = body["preference"]
    assert pref["refused"] is True
    assert pref["code"] == "premium_exceeded"
    assert pref["best_available"] == pytest.approx(10.0)


def test_the_verified_constraint_does_not_fall_back_to_unverified(client, fleet):
    """§5.4 — the reconciliation, asserted rather than described.

    `scheduler.allocate_best_host` prefers verified hosts and **falls back to
    all of them** when none qualifies. That is right for the unconstrained path
    and is exactly wrong here: a request that asked for verification must not be
    answered with an unverified host.
    """
    with pg_transaction() as conn:
        conn.execute(
            "UPDATE host_verifications SET state = 'unverified' WHERE host_id = %s",
            (fleet["good"],),
        )

    body, _ = _evaluate(client, fleet, {"require_verified": True})
    pref = body["preference"]
    assert pref["refused"] is True, (
        "with no verified host in the fleet, the constrained path fell back to "
        "an unverified one — the silent fallback this gate names"
    )
    assert pref["code"] == "verification_unsatisfiable"


# ── Clause 3: the trail says what was true then ───────────────────────


def test_a_placement_is_recorded_with_the_evidence_it_was_made_on(client, fleet):
    c, _ = client
    body, tenant = _evaluate(client, fleet, {"min_uptime_pct": 99.5, "require_verified": True})
    decision_id = body["decision_id"]
    assert decision_id, "the decision was not recorded"

    from control_plane.scheduler.placement_record import read_placement

    with pg_transaction() as conn:
        stored = read_placement(conn, decision_id, tenant_id=tenant)

    assert stored["outcome"] == "placed"
    assert stored["host_id"] == fleet["good"]
    assert stored["asked"]["min_uptime_pct"] == 99.5
    assert stored["asked"]["require_verified"] is True
    assert stored["evidence"]["verification_status"] == "verified"
    assert stored["evidence"]["uptime_pct"] == pytest.approx(99.95, abs=0.01)
    assert stored["evidence"]["reputation_score"] == pytest.approx(500.0)
    assert stored["candidate_count"] == 2, (
        "a decision is only interpretable against the field it was made over"
    )


def test_the_record_survives_the_host_changing_afterwards(client, fleet):
    """Copied, not referenced — the whole point of clause 3.

    Re-reading the host later answers what its reputation is *now*, which is a
    different question during an incident review six weeks on.
    """
    body, tenant = _evaluate(client, fleet, {"require_verified": True})
    decision_id = body["decision_id"]

    with pg_transaction() as conn:
        conn.execute(
            "UPDATE reputation_scores SET final_score = 1.0 WHERE entity_id = %s",
            (fleet["good"],),
        )
        conn.execute(
            """UPDATE host_verifications
                  SET state = 'deverified', deverified_at = %s
                WHERE host_id = %s""",
            (time.time(), fleet["good"]),
        )

    from control_plane.scheduler.placement_record import read_placement

    with pg_transaction() as conn:
        stored = read_placement(conn, decision_id, tenant_id=tenant)

    assert stored["evidence"]["reputation_score"] == pytest.approx(500.0), (
        "the record moved with the host; it is a reference, not a copy"
    )
    assert stored["evidence"]["verification_state"] == "verified"


def test_a_refusal_is_recorded_too(client, fleet):
    """The half a successes-only trail would drop.

    "Why did nothing launch last Tuesday" is the question an operator actually
    arrives with, and only this row can answer it.
    """
    body, tenant = _evaluate(client, fleet, {"min_uptime_pct": 99.99})
    decision_id = body["decision_id"]
    assert decision_id, "a refusal left no trace"

    from control_plane.scheduler.placement_record import read_placement

    with pg_transaction() as conn:
        stored = read_placement(conn, decision_id, tenant_id=tenant)

    assert stored["outcome"] == "refused"
    assert stored["refusal_code"] == "uptime_unsatisfiable"
    assert stored["host_id"] is None
    assert stored["candidate_count"] == 2


def test_the_recorded_decision_is_scoped_to_the_asking_tenant(client, fleet):
    body, _ = _evaluate(client, fleet, {})
    from control_plane.scheduler.placement_record import read_placement

    with pg_transaction() as conn:
        assert read_placement(
            conn, body["decision_id"], tenant_id="someone-else"
        ) is None
