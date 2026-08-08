"""Raising a spend cap requires approval; lowering one does not.

Gate P1 clause 6, and the one clause the implementation had decided against.
`configure_auto_topup` gated on `billing:write` alone, and
`tests/test_auto_topup_change_is_recorded.py` argued in its docstring that this
was right: the caller already holds a deliberately granted scope, so asking again
"re-decides their decision".

That argument was ruled against in `docs/gate-truth-table.md`. The short version:
`top_up_wallet` charges **a stated amount, once, while the user is watching**;
`configure_auto_topup` installs **standing unattended authority that fires
repeatedly with nobody present**. The smaller lever being ungated is not a reason
to leave the larger one ungated.

## What "approval" means, and what it does not

Approval is the `action_plans` substrate — the same one Gate P1 clause 7 depends
on when it asks for a charge "traceable to its approving plan". It is stronger
than a two-step call: `confirm:true` is accepted for symmetry and deliberately
ignored, execute refuses any plan not `approved`, the plan is bound to its
canonical argument hash, and `approval_mode: "human"` refuses a machine principal.

`approval_mode` here is hard-coded `"human"` rather than taken from the spend
policy, and that is deliberate. A standing policy lets a client pre-authorise
spending *inside ceilings*; letting it approve a change *to those ceilings* would
be circular.

## Who is exempt, and why that is not a loophole

A human at the dashboard is not asked to approve their own click — the click is
the approval. `_is_interactive_human` draws that line, and it is deliberately not
`routes.action_plans._is_human`, which tests only `auth_type !=
"client_credentials"` and therefore counts a third-party connector token as a
person. Using the weaker predicate here would have left the gate open to exactly
the caller it exists to stop.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("XCELSIOR_ENV", "test")

PREVIOUS = {"enabled": True, "amount_cad": 20.0, "threshold_cad": 5.0}


def _config(**kw):
    from routes.billing import AutoTopupConfig

    base = {
        "enabled": True,
        "amount_cad": 20.0,
        "threshold_cad": 5.0,
        "stripe_payment_method_id": "pm_test",
    }
    base.update(kw)
    return AutoTopupConfig(**base)


def _agent(scopes=("billing:write",)) -> dict:
    """A client-credentials principal — no human present."""
    return {
        "email": "demo@xcelsior.ca",
        "user_id": "demo-user",
        "auth_type": "client_credentials",
        "grant_type": "client_credentials",
        "client_id": "agent-client",
        "scopes": list(scopes),
    }


def _dashboard_human() -> dict:
    return {
        "email": "demo@xcelsior.ca",
        "user_id": "demo-user",
        "auth_type": "oauth_access_token",
        "session_type": "browser",
        "client_id": "xcelsior-web",
        "scopes": ["profile", "email"],
    }


def _connector() -> dict:
    """A third-party token. Relays a human's intent; is not a human."""
    return {
        "email": "demo@xcelsior.ca",
        "user_id": "demo-user",
        "auth_type": "oauth_access_token",
        "session_type": "browser",
        "client_id": "third-party-agent",
        "scopes": ["billing:write"],
    }


# --------------------------------------------------------------------------
# Which changes widen
# --------------------------------------------------------------------------


def test_raising_the_amount_widens():
    from routes.billing import _auto_topup_widens

    assert _auto_topup_widens(PREVIOUS, _config(amount_cad=50.0)) is True


def test_raising_the_threshold_widens():
    """The one that gets missed.

    A higher threshold does not raise any single charge — it makes the charge
    fire sooner, and therefore more often. That is more unattended spending.
    """
    from routes.billing import _auto_topup_widens

    assert _auto_topup_widens(PREVIOUS, _config(threshold_cad=500.0)) is True


def test_enabling_a_disabled_lever_widens():
    from routes.billing import _auto_topup_widens

    off = {"enabled": False, "amount_cad": 0.0, "threshold_cad": 0.0}
    assert _auto_topup_widens(off, _config()) is True


def test_lowering_the_amount_does_not_widen():
    from routes.billing import _auto_topup_widens

    assert _auto_topup_widens(PREVIOUS, _config(amount_cad=5.0)) is False


def test_disabling_never_widens_whatever_the_amounts_say():
    """`enabled: false` with a huge amount must not read as a widening."""
    from routes.billing import _auto_topup_widens

    assert _auto_topup_widens(PREVIOUS, _config(enabled=False, amount_cad=9999.0)) is False


def test_an_unchanged_setting_does_not_widen():
    from routes.billing import _auto_topup_widens

    assert _auto_topup_widens(PREVIOUS, _config()) is False


# --------------------------------------------------------------------------
# Who must have a plan
# --------------------------------------------------------------------------


def test_a_dashboard_human_is_the_approval():
    from routes._deps import _is_interactive_human

    assert _is_interactive_human(_dashboard_human()) is True


def test_an_agent_is_not_a_human_approver():
    from routes._deps import _is_interactive_human

    assert _is_interactive_human(_agent()) is False


def test_a_connector_token_is_not_a_human_approver():
    """The loophole this gate would have had if it used the weaker predicate.

    `routes.action_plans._is_human` returns True here, because it only asks
    whether the grant was `client_credentials`. Asserting the two disagree is
    the point: if `_is_interactive_human` ever starts agreeing with it, an agent
    holding a connector token can approve its own spend-cap increase.
    """
    from routes._deps import _is_interactive_human
    from routes.action_plans import _is_human

    connector = _connector()
    assert _is_interactive_human(connector) is False
    assert _is_human(connector) is True, (
        "the weaker predicate no longer counts a connector token as human — if "
        "that was fixed, say so here; if it drifted, this gate depended on it"
    )


# --------------------------------------------------------------------------
# The refusal
# --------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import api as api_mod

    return TestClient(api_mod.app)


def _as(monkeypatch, principal: dict) -> None:
    from routes import _deps

    monkeypatch.setattr(_deps, "_get_current_user", lambda request: dict(principal))
    import routes.billing as billing_mod

    monkeypatch.setattr(billing_mod, "_get_current_user", lambda request: dict(principal))


def _wallet(monkeypatch, enabled=True, amount_cad=20.0, threshold_cad=5.0):
    """Pin the stored setting so `previous` is known."""
    import routes.billing as billing_mod
    from money import cad_to_micros

    class _Engine:
        def get_wallet(self, customer_id):
            return {
                "auto_topup_enabled": enabled,
                "auto_topup_amount_micros": cad_to_micros(amount_cad),
                "auto_topup_threshold_micros": cad_to_micros(threshold_cad),
            }

        def configure_auto_topup(self, **kw):
            self.configured = kw

    engine = _Engine()
    monkeypatch.setattr(billing_mod, "get_billing_engine", lambda: engine)
    return engine


def test_an_agent_widening_directly_is_refused(client, monkeypatch):
    """The headline behaviour."""
    _as(monkeypatch, _agent())
    _wallet(monkeypatch)
    r = client.post(
        "/api/v2/billing/auto-topup",
        json={"enabled": True, "amount_cad": 500.0, "threshold_cad": 5.0},
    )
    assert r.status_code == 409, (
        f"an agent raised the unattended charge amount and got {r.status_code}"
    )
    assert "auto-topup-plans" in r.text, "the refusal does not say how to proceed"


def test_an_agent_narrowing_directly_is_allowed(client, monkeypatch):
    """The other half of the asymmetry.

    Safety must never be harder to reach than the risk it undoes. If lowering
    needed a plan too, this would be a blanket gate rather than the asymmetry
    the clause asks for.
    """
    _as(monkeypatch, _agent())
    engine = _wallet(monkeypatch)
    r = client.post(
        "/api/v2/billing/auto-topup",
        json={"enabled": True, "amount_cad": 5.0, "threshold_cad": 5.0},
    )
    assert r.status_code == 200, f"lowering the amount was refused: {r.text}"
    assert engine.configured["amount_cad"] == 5.0


def test_an_agent_disabling_directly_is_allowed(client, monkeypatch):
    """Turning it off is the safest thing a caller can do; never gate it."""
    _as(monkeypatch, _agent())
    engine = _wallet(monkeypatch)
    r = client.post(
        "/api/v2/billing/auto-topup",
        json={"enabled": False, "amount_cad": 20.0, "threshold_cad": 5.0},
    )
    assert r.status_code == 200, f"disabling auto-top-up was refused: {r.text}"
    assert engine.configured["enabled"] is False


def test_a_dashboard_human_widening_directly_is_allowed(client, monkeypatch):
    """No ceremony in front of a person who is already there."""
    _as(monkeypatch, _dashboard_human())
    engine = _wallet(monkeypatch)
    r = client.post(
        "/api/v2/billing/auto-topup",
        json={"enabled": True, "amount_cad": 500.0, "threshold_cad": 5.0},
    )
    assert r.status_code == 200, f"the dashboard was refused its own click: {r.text}"
    assert engine.configured["amount_cad"] == 500.0


def test_a_connector_token_widening_directly_is_refused(client, monkeypatch):
    """The case the weaker predicate would have admitted."""
    _as(monkeypatch, _connector())
    _wallet(monkeypatch)
    r = client.post(
        "/api/v2/billing/auto-topup",
        json={"enabled": True, "amount_cad": 500.0, "threshold_cad": 5.0},
    )
    assert r.status_code == 409, (
        f"a third-party connector token widened the spend cap and got {r.status_code}"
    )


def test_a_plan_is_refused_for_a_change_that_does_not_widen(client, monkeypatch):
    """A plan for a narrowing would sit in the trail looking like an approval."""
    _as(monkeypatch, _agent())
    _wallet(monkeypatch)
    r = client.post(
        "/api/v2/billing/auto-topup-plans",
        json={"enabled": True, "amount_cad": 5.0, "threshold_cad": 5.0},
    )
    assert r.status_code == 409
    assert "does not widen" in r.text


def test_the_action_type_demands_billing_write():
    """The plan may add approval; it must never add authority."""
    from control_plane.launch.service import ACTION_REQUIRED_SCOPES

    assert ACTION_REQUIRED_SCOPES["configure_auto_topup"] == ["billing:write"]
