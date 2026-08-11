"""§1.3: one live-credential assertion per phase, for P1, P2, P3 and P4.

The clause asks each phase to prove *something* against a real server with a
real token. P0 has `test_named_scopes_refuse_live.py` and P5 has
`test_placement_refuses_live.py`; these are the four that had none. P4's only
other live coverage is `test_access_journey_live.py`, which needs a fleet and
therefore skips — so until now P4 had no live assertion that ever ran.

Each asserts the narrowest true thing that a mock cannot establish — that the
deployed surface exists, authenticates, and refuses correctly. They are not
substitutes for the phases' own clauses; they are the live-credential proof
those clauses were supposed to carry.

Positive controls throughout: a server refusing everything passes a refusal-only
suite perfectly, which is how a Cloudflare interstitial once reported a fix that
had not happened.
"""

from __future__ import annotations

import pytest

requests = pytest.importorskip("requests")

from tests.live._fleet import (  # noqa: E402
    BASE,
    MISSING_CREDENTIALS,
    TOKEN,
    auth,
)

pytestmark = pytest.mark.skipif(not BASE or not TOKEN, reason=MISSING_CREDENTIALS)


def _get(path: str):
    return requests.get(f"{BASE}{path}", headers=auth(), timeout=30)


def _post(path: str, body: dict):
    return requests.post(f"{BASE}{path}", headers=auth(), json=body, timeout=30)


def _assert_routed(path: str) -> None:
    """Fail unless `path` is a route on this deployment.

    Every refusal below is written as "not a 200". A path that does not exist
    also is not a 200, so without this the whole file passes against a server
    that serves none of these surfaces — which is exactly what happened: the
    P1 assertion pointed at `/api/billing/topup`, a route that has never
    existed, and reported a refusal for eight commits.

    The probe sends a method no route here implements. FastAPI answers **405**
    when the path matches a route and the method does not, and **404** when
    nothing matches the path at all — so the distinction is structural rather
    than a guess about error-body shape. It needs the real token, because the
    auth middleware answers 401 before routing is ever consulted.
    """
    probe = requests.request("DELETE", f"{BASE}{path}", headers=auth(), timeout=30)
    assert probe.status_code != 404, (
        f"{path} is not a route on this deployment; a refusal from it asserts "
        "nothing. Fix the path rather than the assertion."
    )


# ── The positive control ──────────────────────────────────────────────


def test_the_token_reaches_the_api_at_all():
    """Without this, every refusal below could be an edge blocking the origin."""
    response = _get("/instances")
    assert response.status_code == 200, (
        f"the credential does not reach the API ({response.status_code}); no "
        "refusal in this file means anything until it does"
    )


# ── P1: the money levers ──────────────────────────────────────────────


def test_p1_the_webhook_refuses_an_event_it_cannot_verify():
    """Gate P1 clause 4, which the gate itself calls its second headline.

    The webhook is the only completion signal P1 has left, so a forged event
    must not be actionable. The clause asks for this to be asserted by posting
    a body signed with the wrong secret — which needs no card, no funded
    account and no fleet, and is therefore the one P1 clause that is provable
    from here.

    **400 rather than any other refusal.** Stripe retries a 400 and leaves the
    attempt visible in `delivery_success=false`; a silently-swallowed 200 would
    look identical to a delivered event.
    """
    forged = requests.post(
        f"{BASE}/api/connect/webhooks",
        headers={
            "Content-Type": "application/json",
            # Well-formed shape, signed with nothing.
            "Stripe-Signature": "t=1700000000,v1=" + "0" * 64,
        },
        json={"id": "evt_forged", "type": "v2.core.account.updated"},
        timeout=30,
    )
    assert forged.status_code != 503, (
        "the webhook answered 503 — it has no secret configured, so it is "
        "refusing everything including real events. That is not the clause: it "
        "cannot verify, rather than having verified and refused."
    )
    assert forged.status_code == 400, (
        f"a forged event got {forged.status_code}; the clause requires 400 so "
        "Stripe retries and the attempt stays visible"
    )


def test_p1_the_webhook_refuses_an_event_carrying_no_signature_at_all():
    """The absent-header case, which is a different code path from a bad one."""
    unsigned = requests.post(
        f"{BASE}/api/connect/webhooks",
        headers={"Content-Type": "application/json"},
        json={"id": "evt_unsigned", "type": "v2.core.account.updated"},
        timeout=30,
    )
    assert unsigned.status_code == 400, unsigned.status_code


def test_p1_the_billing_surface_is_reachable_with_this_token():
    """Positive control for the pair above — P1's surface answers this token.

    Previously this accepted a 404, which made it a control that passes when
    the surface is missing. It asserts a real read now.
    """
    response = _get("/api/billing/attestation")
    assert response.status_code in (200, 403), response.status_code


# ── P2: access ────────────────────────────────────────────────────────


def test_p2_an_ssh_key_endpoint_requires_the_scope():
    """P2's surface authenticates rather than being open."""
    response = _get("/api/ssh/keys")
    assert response.status_code in (200, 403), response.status_code


def test_p2_a_terminal_ticket_is_not_issued_for_an_unknown_instance():
    """A credential with a clock on it must not be minted for nothing."""
    _assert_routed("/api/terminal/ticket")
    response = _post("/api/terminal/ticket", {"instance_id": "does-not-exist"})
    assert response.status_code != 200, "a terminal ticket was issued for no instance"


# ── P3: volumes and promotion ─────────────────────────────────────────


def test_p3_the_volume_surface_is_reachable_and_scoped():
    response = _get("/api/v2/volumes")
    assert response.status_code in (200, 403), response.status_code


def test_p3_a_promotion_onto_an_unknown_volume_is_refused():
    """Promotion is a copy onto a user's volume; it must not accept a stranger.

    The path is volume-scoped — `/api/v2/volumes/{id}/promotions` — which is
    the point: there is no route that takes a volume id in the body, so a
    promotion cannot be aimed at a volume the caller does not own by naming it.
    """
    path = "/api/v2/volumes/does-not-exist/promotions"
    _assert_routed(path)
    response = _post(path, {"job_id": "does-not-exist"})
    assert response.status_code != 200, "a promotion was accepted onto a volume that does not exist"
    assert response.status_code in (400, 403, 404, 422), response.status_code


# ── P4: the pipeline ──────────────────────────────────────────────────


def test_p4_an_unapproved_pipeline_will_not_run():
    """Gate P4's server-bound property, in the form that needs no fleet.

    Execution is refused before a single stage is materialised, so this asserts
    the binding without launching anything. The graph lives in the plan, and a
    caller can only name a plan — so "approve it first" is the whole of the
    authority check at this point.

    The create call is this test's positive control: a server that refused
    everything would produce the same 409 below while proving nothing.
    """
    created = _post(
        "/api/v1/pipelines",
        {
            "name": "live-gate-unapproved",
            # The plan's own worked example, and deliberately two *different*
            # action types: the union-of-scopes rule only has anything to say
            # when the stages do not require the same scope.
            "stages": [
                {"name": "train", "action_type": "create_instance", "estimate_micros": 0},
                {
                    "name": "serve",
                    "action_type": "create_serverless_endpoint",
                    "estimate_micros": 0,
                },
            ],
        },
    )
    if created.status_code == 403:
        pytest.skip("this credential lacks the scope to quote a pipeline")
    assert created.status_code == 200, (
        f"could not create a pipeline to test against ({created.status_code}): {created.text[:200]}"
    )
    plan_id = (created.json() or {}).get("plan_id")
    assert plan_id, f"the pipeline was created with no plan id: {created.text[:200]}"

    ran = _post(f"/api/v1/pipelines/{plan_id}/execute", {})
    assert ran.status_code == 409, (
        f"an unapproved pipeline returned {ran.status_code}; the approval is not binding"
    )
    # `problem_response` emits RFC 7807, where the code is top-level — not the
    # `{"ok": false, "error": {...}}` envelope the auth layer uses. Reading the
    # wrong one gives `None`, which compares unequal to everything and would
    # have read as "refused for the wrong reason" against correct behaviour.
    assert ran.json().get("code") == "approval_required", (
        f"refused, but not for the reason the clause names: {ran.text[:200]}"
    )


def test_p4_a_pipeline_belonging_to_nobody_is_not_found():
    """Tenant scoping, asserted at the same surface: naming a plan is not owning it."""
    ran = _post("/api/v1/pipelines/00000000-0000-0000-0000-000000000000/execute", {})
    assert ran.status_code == 404, ran.status_code
