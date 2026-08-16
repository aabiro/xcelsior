"""Delete the Stripe Connect accounts the suite creates, after it creates them.

Fixtures that register a provider create a **real** Connect account in the test
account, and nothing removed them. On 2026-08-16 the test-mode dashboard held
more than a hundred, all `charges_enabled: false` and `details_submitted:
false`, created over four days — 42 from `idor-*` fixtures, 39 from `provcov-*`,
19 from `compliance-*`. Every full-suite run adds more, and the suite runs many
times a day.

Nothing breaks at a hundred. It is a leak rather than a fault, and the reason to
close it is that the count only goes up: a test account nobody can read is a
test account nobody notices a real problem in.

## Two refusals worth stating

**Live mode, never.** The guard is the *key prefix*, not a config flag, because
a flag can be wrong while a key cannot: `sk_live_` refuses outright. Deleting a
real connected account would destroy a provider's onboarding and their payout
destination, and no test is worth being one environment variable away from that.

**A cleanup failure is not a test failure.** Teardown swallows everything. A
network blip while removing a fixture's leftovers says nothing about the code
under test, and a suite that goes red for it teaches its reader to ignore red.
The cost of a swallowed error here is one undeleted account — the state we are
already in.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _is_test_mode_key(key: str) -> bool:
    """Only a key that says test, in the key itself."""
    return key.startswith(("sk_test_", "rk_test_"))


def delete_connected_account(account_id: str | None) -> bool:
    """Best-effort delete. Returns whether it went; never raises.

    Express accounts created by a platform can be deleted in test mode. A
    missing id, a live key, a missing SDK or any API error all return False
    quietly — the caller is a fixture tearing down, and the worst outcome of
    being wrong here is the leftover that already exists.
    """
    if not account_id or not account_id.startswith("acct_"):
        return False
    try:
        import stripe

        key = getattr(stripe, "api_key", "") or ""
        if not _is_test_mode_key(key):
            # Not an error and not worth a warning on every teardown: a suite
            # run without Stripe configured is ordinary.
            return False
        stripe.Account.delete(account_id)
        return True
    except Exception as exc:  # pragma: no cover - teardown is never fatal
        log.debug("stripe cleanup: could not delete %s: %s", account_id, exc)
        return False


def account_id_from_registration(response_json: dict) -> str | None:
    """Pull the account id out of `/api/providers/register`'s response.

    The route returns `{"ok": True, **result}` and `create_provider_account`
    puts `stripe_account_id` in `result`, so the id is at the top level. Read
    defensively anyway: a fixture that raises in teardown fails the test it
    just finished passing.
    """
    if not isinstance(response_json, dict):
        return None
    value = response_json.get("stripe_account_id")
    return value if isinstance(value, str) else None
