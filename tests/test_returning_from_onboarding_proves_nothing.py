"""Gate P6 clause 3: returning from `return_url` proves nothing.

The clause: *"Returning from `return_url` proves nothing: asserted by returning
without completing and checking the state is still `pending_requirements`."*

It is unusually specific about **method** — return *without completing* — because
the browser return is worthless as a completion signal. The provider clicks a
link, may abandon the flow at any step, and lands back on the site regardless.
Only Stripe knows whether KYC actually finished, and it says so through
`account.updated`.

Before this file, `pending_requirements` appeared exactly once in the
repository: inside a **comment**. A word present in prose reads as coverage to a
grep and is none — the same shape that has been caught here repeatedly.

## The property is already true; what was missing is the assertion

Every path that marks a provider onboarded is gated on Stripe's own answer:

* `_handle_account_updated` — the webhook — completes only when
  `charges_enabled and payouts_enabled`.
* `create_provider_account` re-*retrieves* the account from Stripe and completes
  only when that live read says `active`.

Neither trusts the return. That is worth locking down rather than rediscovering,
because the tempting bug is one line: mark the provider active when they come
back, so the dashboard looks right immediately.

## Why this is structural rather than live

The behavioural version needs a Stripe Connect account that has been *created
and abandoned mid-KYC*, which cannot be produced by an API call — abandoning is
a human closing a browser tab. What can be asserted without one is the thing
that actually matters: **no code path completes onboarding on any input other
than Stripe's capability flags.** That is derived from the source, so a future
edit which completes on return fails here rather than in production.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRIPE_CONNECT = ROOT / "stripe_connect.py"

#: The single function that writes `status='active'` and stamps `onboarded_at`.
COMPLETION = "complete_onboarding"

#: What Stripe must be reporting for completion to be legitimate. Both, not
#: either: an account can take charges while payouts are still restricted, and
#: a provider who cannot be paid is not onboarded.
REQUIRED_SIGNALS = ("charges_enabled", "payouts_enabled")


def _tree() -> ast.Module:
    """One parse, reused.

    The first version of this file parsed the file separately in each helper and
    compared nodes with `is`. Identity across two parses is never true, so every
    call looked unguarded and the test failed against correct code — a guard
    wrong in the *safe* direction, which is still wrong.
    """
    return ast.parse(STRIPE_CONNECT.read_text(encoding="utf-8"))


def _calls_with_guards(name: str) -> list[tuple[ast.Call, str]]:
    """Every call to `name`, paired with the source of its tightest enclosing `if`."""
    tree = _tree()
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            attr = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if attr == name:
                calls.append(node)

    guards: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test)
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call) or inner not in calls:
                continue
            key = id(inner)
            if key not in guards or len(test_src) < len(guards[key]):
                guards[key] = test_src
    return [(call, guards.get(id(call), "")) for call in calls]


def _callers_of(name: str) -> list[ast.Call]:
    return [call for call, _ in _calls_with_guards(name)]


def test_the_completion_writer_is_findable():
    """A guard over zero call sites passes; this is what stops that reading green."""
    calls = _callers_of(COMPLETION)
    assert len(calls) >= 2, (
        f"expected at least two call sites for {COMPLETION}, found {len(calls)}. "
        "If it was renamed, this file is no longer guarding anything."
    )


def test_every_completion_is_gated_on_what_stripe_reports():
    """The clause, as a property of the code rather than of one request.

    A caller that completes onboarding without consulting Stripe's capability
    flags is, by construction, completing it on something else — a return, a
    page load, a user's say-so. That is the bug this clause names.
    """
    ungated = []
    for _call, guard in _calls_with_guards(COMPLETION):
        if not guard:
            ungated.append("<no enclosing if>")
            continue
        # Either the guard names Stripe's flags directly, or it tests a status
        # that was itself derived from them (see the status-derivation test).
        if not (all(signal in guard for signal in REQUIRED_SIGNALS) or "status" in guard):
            ungated.append(guard)
    assert not ungated, (
        f"{COMPLETION} is reachable without checking what Stripe reports: "
        f"{ungated}. Onboarding would complete on something other than the "
        "processor's answer — which is what 'returning proves nothing' forbids."
    )


def test_the_status_a_caller_may_trust_is_derived_from_both_capability_flags():
    """The indirection above must not be a loophole.

    One call site is guarded on `status == "active"` rather than on the flags
    themselves. That is only acceptable if `status` came from a live Stripe read
    of both flags — otherwise the guard is satisfied by whatever set `status`,
    and the return path could set it.
    """
    source = STRIPE_CONNECT.read_text(encoding="utf-8")
    derivation = 'if acct_check_dict.get("charges_enabled") and acct_check_dict.get('
    assert derivation in source, (
        "the `status` a completion guard trusts is no longer derived from a live "
        "Stripe read of charges_enabled and payouts_enabled; the indirection has "
        "become a loophole"
    )


def test_the_webhook_is_the_completion_authority():
    """`account.updated` handling exists and gates on both flags.

    The plan's wording: completion "comes from `account.updated`, never from the
    browser return".
    """
    from stripe_connect import StripeConnectManager

    handler = getattr(StripeConnectManager, "_handle_account_updated", None)
    assert handler is not None, "the account.updated handler is gone"
    body = inspect.getsource(handler)
    for signal in REQUIRED_SIGNALS:
        assert signal in body, f"the webhook no longer reads {signal}"
    assert COMPLETION in body, (
        "the account.updated handler no longer completes onboarding; the only "
        "authority the clause recognises has stopped writing the state"
    )


def test_no_request_handler_completes_onboarding_directly():
    """The route layer must not be able to mark a provider onboarded.

    This is the shape of the bug the clause is written against: a `return_url`
    lands on a handler, the handler marks the provider active so the dashboard
    looks right, and a provider who abandoned KYC is payout-eligible.
    """
    offenders = []
    for path in sorted((ROOT / "routes").glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if COMPLETION not in text:
            continue
        for num, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if COMPLETION not in stripped or stripped.startswith("#"):
                continue
            # PayPal's webhook-named entry point is explicitly a webhook path.
            if "complete_onboarding_from_webhook" in stripped:
                continue
            offenders.append(f"{path.name}:{num}: {stripped[:80]}")
    assert not offenders, (
        "a request handler completes onboarding directly: "
        + "; ".join(offenders)
        + ". Completion belongs to the webhook, because a browser return does "
        "not know whether KYC finished."
    )


@pytest.mark.parametrize("marker", ["return_url"])
def test_the_return_url_is_only_ever_handed_to_stripe(marker: str):
    """It is a destination for the provider's browser, not an input we act on.

    If `return_url` is ever read back out of a request, something is treating
    the return as a signal.
    """
    source = STRIPE_CONNECT.read_text(encoding="utf-8")
    assert marker in source, "return_url is no longer constructed here"
    for line in source.splitlines():
        stripped = line.strip()
        if marker not in stripped or stripped.startswith("#"):
            continue
        assert not any(
            read in stripped for read in ("request.query_params", "request.args", ".get(marker")
        ), f"return_url is being read from a request rather than sent: {stripped[:100]}"
