"""Named, not fixed: one account can mint unlimited Stripe Connect accounts.

`POST /api/providers/register` pins `email` to the caller — *"You can only
register a provider for your own email"*, admins exempt. It does **not** pin
`provider_id`. That field is `provider_id: str` on the request model, taken
whole from the body, and `create_provider_account` looks for an existing row by
it. A caller sending a new one each time therefore creates a **new Stripe Connect
account each time**, and the re-registration then runs

    UserStore.update_user(register_email, {"provider_id": req.provider_id, ...})

so the previous account is orphaned: still live at Stripe, no longer linked to
anyone.

## This is the shape of a leak that already happened

The test-mode dashboard held **1,389** Connect accounts on 2026-08-16, purged
that day — created by fixtures registering as `prov-{uuid4}`, `idor-*`,
`provcov-*`. Those fixtures were fixed. The route was not, and the fixtures were
only the callers that happened to do it.

The dashboard never did. It sends `providerId || customerId` — a deterministic
id — which is why the UI path reuses one account and looks correct. The rule
lived in the frontend; the server has none. Any third caller, an agent tool
among them, arrives with no such convention.

## Why this file asserts the hazard instead of removing it

The obvious fix is to resolve `provider_id` server-side for non-admins, exactly
as `email` is resolved — the caller's existing `provider_id`, else their
canonical owner id. It was written, and it works. It also changes the semantics
of a **financial-account-creation** endpoint and breaks nine tests whose setup
assumes the caller names the id, including a cross-account isolation test that
starts returning 404 instead of 403 because the account it probes is no longer
the one it created.

That is a decision about a money route, not a cleanup, and it is not mine to
take unprompted. So the behaviour is pinned here with its consequence spelled
out, following `test_the_filter_still_fails_open_without_the_loader`: the
default is load-bearing, the next reader should know, and the fix should be
deliberate.

**This test passing is not good news.** It failing means someone constrained the
id — in which case delete this file and keep their constraint.

## The precondition it guards

`POST /api/providers/register` stays classified `gap` rather than becoming a
tool. A `register_provider` tool would hand an agent a loop that mints real
Connect accounts under a caller-chosen id, and a retry on a timeout would create
a second one. Constrain the id first; then the tool is safe to build.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
ROUTE = ROOT / "routes" / "providers.py"


def _register_handler() -> str:
    source = ROUTE.read_text(encoding="utf-8")
    body = source.split("def api_register_provider", 1)[1]
    return body.split("\n@router.", 1)[0]


def test_the_email_is_pinned_to_the_caller():
    """The half that *is* constrained, so the contrast is asserted rather than
    described. If this ever stops holding, the route lets one account register
    providers under someone else's email and that is a larger problem."""
    handler = _register_handler()
    assert "You can only register a provider for your own email" in handler
    assert "caller_email" in handler


def test_the_provider_id_is_not_pinned_to_the_caller():
    """The hazard. See the module docstring before changing this."""
    handler = _register_handler()

    # The id reaches `create_provider_account` straight from the request body.
    assert re.search(r"provider_id=req\.provider_id", handler), (
        "register no longer passes the caller's `provider_id` through — if it "
        "now resolves the id server-side, the hazard this file documents is "
        "gone and the file should be deleted along with the `gap` note on "
        "POST /api/providers/register in docs/endpoint-classification.json"
    )
    # And nothing derives it from the caller first.
    assert "_canonical_owner_id" not in handler, (
        "the handler now derives a provider identity from the caller, which is "
        "the fix this file exists to make visible — delete this file"
    )


def test_re_registering_moves_the_link_and_leaves_the_old_account():
    """Why a duplicate is not merely untidy: the previous account is orphaned."""
    handler = _register_handler()
    assert re.search(r'update_user\(\s*register_email,\s*\{"provider_id": req\.provider_id', handler), (
        "the link update changed shape; re-check whether re-registration still "
        "orphans the previously created Connect account"
    )


def test_the_route_is_still_classified_a_gap():
    """The precondition, pinned where it can be seen.

    `covered` would mean a tool calls it. It must not, until the id is
    constrained — `tests/test_classification_matches_the_tools.py` enforces the
    label against reality, and this asserts the label is the intended one.
    """
    import json

    classification = json.loads(
        (ROOT / "docs" / "endpoint-classification.json").read_text(encoding="utf-8")
    )
    entry = classification["POST /api/providers/register"]
    klass = entry["class"] if isinstance(entry, dict) else entry
    assert klass == "gap", (
        f"POST /api/providers/register is now `{klass}`. If a tool was built "
        "for it, the caller-controlled provider_id above means a retry mints a "
        "second real Stripe Connect account"
    )
