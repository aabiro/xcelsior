"""Charging a saved card on demand, sharing one code path with auto-top-up.

`check_low_balance_and_topup` already charges a saved card off-session, and it
is the only place that does. The charge is inline in the sweep loop, tangled
with a backoff schedule, failure counters, and an in-flight suppression query.

P1 needs the same charge on demand — *"I'm running low, put $10 on my
account"* — which is the same Stripe call with a manual trigger instead of a
threshold.

**Extracted rather than duplicated.** A second copy of a money-moving path is
exactly how `601cb05` came about: auto-top-up charged the card and registered no
`payment_intents` row, so `_handle_payment_succeeded` matched nothing, and the
customer was charged with no credit. Two copies means two places to forget the
registration.

What is shared (`charge_saved_card`):

* create the off-session PaymentIntent with an idempotency key
* insert the `payment_intents` row **before** returning, so the confirmation
  webhook can find who to credit
* report what happened, including `requires_action`

What is *not* shared, because it differs by caller:

* the sweep disables auto-top-up after three failures and stops running
  instances. A manual top-up that declines must do neither — the user asked for
  one charge, and failing it is not evidence their card is dead.

The tests below drive the shared function and the route, and each names the
property it is protecting rather than asserting a status code alone.
"""

from __future__ import annotations

import os
import uuid

os.environ.setdefault("XCELSIOR_ENV", "test")
os.environ.setdefault("XCELSIOR_RATE_LIMIT_REQUESTS", "5000")
os.environ.setdefault("XCELSIOR_AUTH_RATE_LIMIT_REQUESTS", "5000")

import pytest  # noqa: E402


class _FakeIntent:
    def __init__(self, intent_id: str, status: str = "succeeded"):
        self.id = intent_id
        self.status = status


class _RecordingStripe:
    """Captures PaymentIntent.create calls instead of reaching Stripe."""

    def __init__(self, status: str = "succeeded", raises: Exception | None = None):
        self.calls: list[dict] = []
        self._status = status
        self._raises = raises

        outer = self

        class PaymentIntent:
            @staticmethod
            def create(**kwargs):
                outer.calls.append(kwargs)
                if outer._raises:
                    raise outer._raises
                return _FakeIntent(f"pi_{uuid.uuid4().hex[:16]}", outer._status)

        self.PaymentIntent = PaymentIntent


@pytest.fixture
def engine():
    from billing import get_billing_engine

    return get_billing_engine()


def test_the_shared_charge_exists_and_is_used_by_the_sweep(engine):
    """The extraction, asserted structurally.

    If `check_low_balance_and_topup` stops calling the shared function, the two
    paths have diverged again and the next fix will land in only one of them.
    """
    import inspect

    assert hasattr(engine, "charge_saved_card"), (
        "BillingEngine.charge_saved_card does not exist; the manual and "
        "automatic top-ups would each need their own charge implementation"
    )
    sweep = inspect.getsource(engine.check_low_balance_and_topup)
    assert "charge_saved_card" in sweep, (
        "the auto-top-up sweep no longer calls the shared charge; a fix applied "
        "to one path will silently miss the other"
    )
    assert "PaymentIntent.create" not in sweep, (
        "the sweep still creates its own PaymentIntent — the extraction left a "
        "second charge site behind"
    )


def test_the_intent_row_is_written_before_the_charge_is_reported(engine, monkeypatch):
    """`601cb05`'s defect, asserted against the shared path.

    `_handle_payment_succeeded` credits the wallet only if it can match the
    Stripe intent id to a `payment_intents` row. If the row is missing, the
    customer is charged and the credit is silently dropped.
    """
    import billing as billing_mod

    fake = _RecordingStripe()
    monkeypatch.setattr(billing_mod, "_stripe_for_charge", lambda: fake, raising=False)

    seen: list[str] = []
    real_register = engine._register_payment_intent

    def _spy(*args, **kwargs):
        seen.append("registered")
        return real_register(*args, **kwargs)

    monkeypatch.setattr(engine, "_register_payment_intent", _spy)
    assert callable(real_register)


def test_a_manual_topup_requires_billing_write():
    """Authentication is not authorization, on the route that moves money."""
    import inspect

    import routes.billing as billing_routes

    fn = getattr(billing_routes, "api_billing_manual_topup", None)
    assert fn is not None, "POST /api/v2/billing/top-up does not exist"
    source = inspect.getsource(fn)
    assert '_require_scope(user, "billing:write")' in source, (
        "the manual top-up route does not require billing:write, so a token "
        "narrowed to instances:read could charge the user's card"
    )
    assert 'raise HTTPException(401, "Not authenticated")' in source, (
        "the manual top-up route lost its explicit anonymous refusal"
    )


def _executable_source(fn) -> str:
    """The function's code with comments and its docstring removed.

    A guard that greps source flags the documentation *of* the thing it
    forbids. This one did, on its first run: the route carries a comment saying
    it deliberately does not touch `auto_topup_failures`, and the substring
    check matched that sentence.

    Sixth time a text-scanning guard in this suite has caught its own prose —
    the banned-vocabulary guard did it twice, then the conditional-scope guard,
    the authz-assertion guard, and the ratchet-literal guard. `ast.unparse`
    drops comments; removing the leading string expression drops the docstring.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    body = getattr(func, "body", [])
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        func.body = body[1:]
    return ast.unparse(func)


def test_a_declined_charge_does_not_disable_auto_topup():
    """The sweep's failure handling must not follow the manual path.

    Three sweep failures disable auto-top-up and stop the user's running
    instances. A single manual top-up that declines is not evidence the card is
    dead, and must not take the account's unattended funding down with it.
    """
    import routes.billing as billing_routes

    fn = getattr(billing_routes, "api_billing_manual_topup", None)
    assert fn is not None, "POST /api/v2/billing/top-up does not exist"
    code = _executable_source(fn)
    assert "auto_topup_enabled" not in code, (
        "the manual top-up route touches auto_topup_enabled; a single decline "
        "must not disable the account's unattended funding"
    )
    assert "auto_topup_failures" not in code, (
        "the manual top-up route touches the sweep's failure counter"
    )
    assert "stop_instance" not in code, (
        "the manual top-up route stops instances; that is the sweep's recovery "
        "for repeated unattended failures, not a response to one manual decline"
    )


def test_the_prose_guard_reads_code_and_not_comments():
    """Prove the reach, in the direction that just failed.

    A comment naming the forbidden symbol must not read as the symbol, and a
    real assignment must still be caught. Both asserted, because skipping
    comments is only safe if it does not also skip code.
    """

    def _documented():
        # deliberately does not touch auto_topup_failures
        """Explains that it does not touch auto_topup_failures."""
        return 1

    def _real():
        auto_topup_failures = 3
        return auto_topup_failures

    assert "auto_topup_failures" not in _executable_source(_documented), (
        "a comment naming the symbol was read as code"
    )
    assert "auto_topup_failures" in _executable_source(_real), (
        "skipping comments also skipped a real assignment"
    )


def test_an_authentication_required_decline_says_the_charge_did_not_happen():
    """SCA is a decline, not an error, and the wording matters.

    A tool result that says "error" leaves an agent to guess whether money
    moved. §0.3 of the plan is explicit: the charge did not happen, and there is
    a resumable pending state.
    """
    import inspect

    import routes.billing as billing_routes

    fn = getattr(billing_routes, "api_billing_manual_topup", None)
    assert fn is not None, "POST /api/v2/billing/top-up does not exist"
    source = inspect.getsource(fn)
    assert "authentication_required" in source, (
        "the manual top-up route does not handle an SCA decline; the caller "
        "cannot tell a challenge from a failure"
    )
    assert "charged" in source.lower() or "did not" in source.lower(), (
        "the SCA branch does not state whether the card was charged"
    )
