"""Resuming an SCA challenge, and the four ways that could go wrong.

The last backend piece of P1's recovery. `charge_saved_card` catches the
decline and registers the intent, `pending-verification` lists what is waiting,
and this returns the `client_secret` a browser needs for
`stripe.handleNextAction`. Without it the resume link points at a page that
cannot resume anything — which is what it did until now.

A `client_secret` confirms a payment. Anyone holding one can complete or cancel
that intent, which is why `2efd3e2` added it to the log scrubber. Handing one
out therefore has to be narrower than reading a balance, and the tests below are
mostly about the narrowing:

* **`billing:write`, not `billing:read`.** It does not move money by itself, but
  it hands over the only thing between a stopped charge and a completed one.
* **Only an intent that is actually waiting.** A succeeded or cancelled intent
  yields nothing: there is no challenge left, and a secret for one would be a
  credential with no purpose and a real blast radius.
* **404, not 403, for someone else's intent.** A 403 confirms the id exists,
  which turns the endpoint into a probe for other people's payments. Same
  reasoning as `_require_host_visible` in `eedb344`.
* **POST, not GET.** GET responses are cached by intermediaries, logged with
  their URLs, and retried by well-meaning clients. None of that suits a
  response containing something that can complete a charge.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("XCELSIOR_ENV", "test")


def _source() -> str:
    import routes.billing as billing_routes

    return inspect.getsource(billing_routes.api_billing_resume_verification)


def _executable_source() -> str:
    """The route's code with its docstring and comments removed.

    The ordering assertion below failed on its first run against the docstring,
    which explains what a `client_secret` is several paragraphs before the guard
    runs — so "the secret is reached before ownership is checked" was true of the
    prose and false of the code.

    Eighth time a text-scanning guard in this suite has flagged the
    documentation *of* the thing it checks. `ast.unparse` drops comments;
    removing the leading string expression drops the docstring.
    """
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(_source()))
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


def test_it_requires_the_write_scope_not_merely_read():
    """A token narrowed to reading balances must not mint payment credentials."""
    assert "billing_write=True" in _source(), (
        "the resume endpoint does not demand billing:write, so a read-scoped "
        "credential could obtain a client_secret and complete a charge"
    )


def test_it_checks_ownership_before_revealing_anything():
    """The guard must run before the response says anything about the intent."""
    source = _executable_source()
    guard = source.index("_require_customer_access")
    secret = source.index("client_secret")
    assert guard < secret, (
        "the client_secret is reached before ownership is checked"
    )


def test_a_foreign_intent_is_not_found_rather_than_forbidden():
    """403 on someone else's id confirms the id exists.

    That turns this into an oracle for other people's payment intents. Absence
    and denial must look identical from outside.
    """
    assert 'HTTPException(404, "No such pending payment")' in _source(), (
        "a foreign or unknown intent no longer answers 404; a 403 tells the "
        "caller the id is real"
    )


def test_only_an_intent_awaiting_verification_can_be_resumed():
    """A settled intent has no challenge left to satisfy."""
    source = _source()
    assert 'row["status"] != "requires_action"' in source, (
        "the endpoint no longer restricts itself to intents that are actually "
        "waiting, so it would hand out secrets for settled payments"
    )
    assert "HTTPException(\n            409," in source or "409," in source, (
        "a non-resumable intent should be refused, not silently handled"
    )


def test_it_is_a_post_because_the_response_carries_a_credential():
    """GET would put a payment-confirming response through every cache."""
    import routes.billing as billing_routes

    module_source = inspect.getsource(billing_routes)
    assert (
        '@router.post(\n    "/api/v2/billing/pending-verification/{stripe_intent_id}/resume"'
        in module_source
    ), "the resume endpoint is no longer a POST"


def test_the_processor_error_is_not_returned_to_the_caller():
    """Stripe's exception text can carry the intent and its secret.

    The scrubber redacts both from logs, but the message must not be echoed
    into an HTTP response either — that is a surface the scrubber never sees.
    """
    source = _executable_source()
    assert "Could not reach the payment processor" in source, (
        "the processor's raw error is being returned to the caller"
    )
    # The 502's detail must be the fixed string, not the exception. `raise ...
    # from exc` keeps the cause for the traceback without putting it on the wire.
    detail = source.split("HTTPException(502,")[1].split(")")[0]
    assert "exc" not in detail, (
        f"the 502 detail interpolates the processor's exception: {detail!r}"
    )


def test_the_listing_and_the_resume_are_separate_endpoints():
    """The reason the list returns no secret.

    Twenty rows of payment-confirming credentials in one response is a far
    wider surface than one, fetched deliberately, behind a stronger scope.
    """
    import routes.billing as billing_routes

    listing = inspect.getsource(billing_routes.api_billing_pending_verification)
    assert "client_secret" not in listing.replace("no `client_secret`", ""), (
        "the listing endpoint has started returning client secrets; that is "
        "what the single-intent resume exists to avoid"
    )
