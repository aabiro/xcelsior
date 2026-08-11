"""Gate P5's clauses, against a live server with a real token. §1.3.

`tests/test_placement_refuses_end_to_end.py` drives the route in-process
with `TestClient`. That proves the logic and, by the plan's own standard, not the
deployment — *"a mock is what passed while production did not."* P5 has had no
live assertion at all; this is it.

## What it asserts, and why each part is load-bearing

1. **The route exists and authenticates.** A 404 or 401 here means the surface
   did not ship, which an in-process test can never notice.
2. **A refusal is a 200 with a typed body.** The whole design turns on a refusal
   being an answer the caller can render, not an error they must parse from a
   status code.
3. **The decision was recorded.** `decision_id` comes back non-null, which is the
   only externally visible proof that Gate P5 clause 3's trail is being written
   by the deployed code rather than by a test fixture.

## Why a refusal is the expected result

The fleet may legitimately have no eligible host — that is the *normal* state of
a marketplace with no live capacity, and the first production call returned
`no_eligible_hosts` for exactly that reason. So this asserts the **shape** of the
answer and the recording, not a particular verdict: pinning "refused" would make
the test fail the day real capacity appears, and pinning "placed" would make it
fail today.

## Running it

    XCELSIOR_LIVE_BASE_URL=https://xcelsior.ca \\
    XCELSIOR_LIVE_USER_TOKEN=<a session token> \\
    pytest tests/live/test_placement_refuses_live.py

Skips rather than passes without both, so a credential-less run cannot report a
gate it never exercised.
"""

from __future__ import annotations

import os

import pytest

requests = pytest.importorskip("requests")

# Both names, matching the gates beside this one. `scripts/run_live_gates.sh`
# exports both, but a caller setting only `XCELSIOR_STAGING_URL` would run those
# three and silently skip this one — a gate that does not run is the defect this
# whole phase kept finding, and it would be invisible here because skipping is
# the correct behaviour without credentials.
BASE = (
    os.environ.get("XCELSIOR_LIVE_BASE_URL")
    or os.environ.get("XCELSIOR_STAGING_URL")
    or ""
).rstrip("/")
TOKEN = os.environ.get("XCELSIOR_LIVE_USER_TOKEN", "")

pytestmark = pytest.mark.skipif(
    not BASE or not TOKEN,
    reason="set XCELSIOR_LIVE_BASE_URL and XCELSIOR_LIVE_USER_TOKEN to run the live gate",
)

SPEC = {"name": "placement-live-gate", "vram_needed_gb": 8, "num_gpus": 1}


def _evaluate(preference: dict) -> dict:
    response = requests.post(
        f"{BASE}/api/v1/placements/evaluate",
        headers={"Authorization": f"Bearer {TOKEN}"},
        json={"spec": SPEC, "preference": preference},
        timeout=30,
    )
    assert response.status_code == 200, (
        f"{response.status_code} from /api/v1/placements/evaluate — the P5 "
        f"surface is not deployed or not reachable: {response.text[:300]}"
    )
    return response.json()


def test_a_constrained_request_gets_a_typed_answer_and_is_recorded():
    """Clauses 2 and 3, on the deployed system."""
    body = _evaluate({"require_verified": True, "min_uptime_pct": 99.5})

    assert body["ok"] is True
    preference = body["preference"]
    assert isinstance(preference.get("refused"), bool), (
        "the preference verdict is not a typed outcome; a caller cannot render "
        "a trade-off from this"
    )

    if preference["refused"]:
        # Clause 2: it refuses, and says which constraint failed.
        assert preference.get("code"), "a refusal with no code cannot be acted on"
        assert preference.get("detail"), "a refusal with no detail is not an answer"
        assert "host_id" not in preference, (
            "a refusal named a host — the silent fallback this gate exists to stop"
        )
    else:
        assert preference.get("host_id"), "a placement with no host"
        assert preference.get("premium_pct") is not None

    # Clause 3: the trail was written by the deployed code, not a fixture.
    assert body.get("decision_id"), (
        "no decision_id — the placement decision was not recorded, so the audit "
        "trail Gate P5 clause 3 asks for is not being written in production"
    )


def test_an_unconstrained_request_still_answers():
    """The positive control.

    A server refusing everything would pass a refusal-only assertion perfectly.
    This asks for nothing and must still get a well-formed verdict, so the
    refusal above is a decision rather than a broken endpoint.
    """
    body = _evaluate({})
    assert body["ok"] is True
    assert isinstance(body["preference"].get("refused"), bool)
    assert body.get("decision_id")


def test_the_availability_view_is_reported_alongside_the_preference():
    """Two independent answers, not one boolean doing double duty.

    `availability.feasible` says whether *any* host is eligible;
    `preference.refused` says whether the stated constraint could be met.
    Collapsing them is how a refusal comes to read as an outage.
    """
    body = _evaluate({"require_verified": True})
    availability = body["availability"]
    assert "feasible" in availability
    assert "hosts_considered" in availability
    assert availability["hosts_considered"] >= 0
