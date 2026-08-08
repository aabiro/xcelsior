"""A change to unattended spending must leave a trail that says what changed.

`configure_auto_topup` is the only lever that alters what gets charged with
nobody present. Until now the change was written to `log.info` and nowhere the
account holder could see, so an agent could raise the automatic charge from $20
to $500 and the only evidence was a server log line.

The plan asks for an asymmetry: *"raising a cap requires approval; lowering one
does not."*

**This file used to argue that the record was enough and the approval was not
needed** — that the caller already holds `billing:write`, so blocking a
capability the user explicitly granted "re-decides their decision". That argument
was ruled against in `docs/gate-truth-table.md`, and the ruling is worth keeping
here because this is where the losing case was made. It failed on the analogy:
`top_up_wallet` charges **a stated amount, once, while the user is watching**,
whereas `configure_auto_topup` installs **standing unattended authority that
fires repeatedly with nobody present**. The smaller lever being ungated was never
a reason to leave the larger one ungated.

Widening is now gated for any caller that is not an interactive human — see
`tests/test_widening_auto_topup_needs_approval.py`. What survives from the
original argument, unchanged and still correct, is everything below: a gate does
not remove the need for a trail, and the trail is what tells the account holder
*what* changed rather than merely that something did.

So both directions are audited, and the *direction is named in the event type*.
A widening is `user.billing.auto_topup_widened`; anything that narrows or
disables is `user.billing.auto_topup_changed`. That distinction is the whole
point — a trail you have to reconstruct arithmetic from is a trail nobody reads.

Widening means any of: the amount rose, the threshold rose, or auto top-up was
switched on when it had been off. Each increases what happens unattended.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")

import routes.billing as billing_routes  # noqa: E402


def _route_source() -> str:
    """Every function the configure path runs through, concatenated.

    The route was one function and is now three — `api_billing_configure_topup`
    decides, `_auto_topup_widens` classifies, `_apply_auto_topup` writes and
    records. Scanning only the first would have quietly stopped asserting most
    of what this file exists for the moment the code was split, which is the
    failure mode of every source-scanning test: it keeps passing while measuring
    less. Reading all three keeps the claims attached to the behaviour rather
    than to a function name.
    """
    return "\n".join(
        inspect.getsource(fn)
        for fn in (
            billing_routes.api_billing_configure_topup,
            billing_routes._auto_topup_widens,
            billing_routes._apply_auto_topup,
        )
    )


def test_the_scan_still_covers_the_whole_configure_path():
    """Prove the reach. An empty or truncated scan passes everything below."""
    source = _route_source()
    assert "def api_billing_configure_topup" in source
    assert "def _auto_topup_widens" in source
    assert "def _apply_auto_topup" in source


def test_the_change_is_written_to_the_user_audit_trail():
    """A `log.info` is not a record the account holder can read."""
    source = _route_source()
    assert "append_user_audit_event" in source, (
        "changing unattended spending is no longer recorded on the account; "
        "the only evidence would be a server log line"
    )


def test_the_previous_setting_is_captured_before_the_write():
    """"Something changed" is not a trail. What it changed from is."""
    source = _route_source()
    assert "get_wallet" in source, "the route no longer reads the previous setting"
    assert '"previous"' in source, (
        "the audit row no longer carries the previous values, so the trail "
        "cannot say what the setting changed from"
    )


def test_widening_and_narrowing_are_distinguishable_event_types():
    """The asymmetry, as something a reader can filter on.

    If both directions shared an event type, finding the widenings would mean
    reconstructing arithmetic across every row — which is the same as not
    having the distinction.
    """
    source = _route_source()
    assert "user.billing.auto_topup_widened" in source
    assert "user.billing.auto_topup_changed" in source


def test_every_way_of_widening_counts_as_one():
    """Three ways to increase unattended spending, and all three are widenings.

    Asserted against the route's own logic rather than restated, because a
    condition that drifts from the one in the code is worse than none: the trail
    would then say "narrowed" about a change that widened.
    """
    source = _route_source()
    # amount up, threshold up, or switched on from off.
    assert "body.amount_cad > previous" in source, "an amount increase is not counted as widening"
    assert "body.threshold_cad > previous" in source, (
        "a threshold increase is not counted as widening — a higher threshold "
        "fires the charge more often"
    )
    assert "not previous[\"enabled\"]" in source, (
        "turning auto top-up on from off is not counted as widening, and it is "
        "the largest widening there is: from never charging to charging"
    )


def test_the_response_lets_the_caller_report_what_changed():
    """The model should say "I raised it from $20 to $50", not echo its input.

    Without `previous` in the response the agent can only restate what it sent,
    which reads as confirmation and hides a mistake.
    """
    source = _route_source()
    assert 'result = {"ok": True, "auto_topup": body.model_dump(), "previous": previous}' in source


def test_configuring_auto_topup_still_requires_billing_write():
    """Unchanged by any of the above, and the reason none of it is a gate."""
    source = _route_source()
    assert '_require_scope(user, "billing:write")' in source
    assert 'raise HTTPException(401, "Not authenticated")' in source
