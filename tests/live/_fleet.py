"""Shared guard for the live gates. Skip, never pass, without a fleet.

Every gate in this directory needs a reachable server *and* a credential. A
gate that passes without them is worse than no gate: it reports a clause met by
a run that never contacted anything, which is precisely what the truth table
exists to refuse.

Both env names are honoured because `scripts/run_live_gates.sh` exports both,
and a caller setting only `XCELSIOR_STAGING_URL` would otherwise run some gates
here and silently skip others.
"""

from __future__ import annotations

import os

BASE = (
    os.environ.get("XCELSIOR_LIVE_BASE_URL")
    or os.environ.get("XCELSIOR_STAGING_URL")
    or ""
).rstrip("/")
TOKEN = os.environ.get("XCELSIOR_LIVE_USER_TOKEN", "")

#: Set when a fleet with at least one admitted, running host is expected. The
#: instance-level gates need somewhere to launch; without it they skip rather
#: than fail, because "no capacity" is not the defect they exist to catch.
FLEET_EXPECTED = os.environ.get("XCELSIOR_LIVE_FLEET", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

#: Set when Stripe webhooks are being delivered to this deployment — normally
#: `stripe listen --forward-to …/api/providers/webhook` with its `whsec_` in the
#: environment.
#:
#: The gates that spend money need it, and the reason is exact: a top-up only
#: *submits* a charge. The wallet is credited by `payment_intent.succeeded`,
#: because the processor is the sole authority on whether money moved. With no
#: forwarder the balance never rises and a launch is refused `402`.
#:
#: **The failure mode this guards is a pass, not a failure.** A wallet holding
#: balance from an earlier run would let the journey succeed while the thing it
#: claims to prove — that funding works — never happened. That is the
#: phantom-path defect again: an assertion that cannot fail because its
#: precondition was satisfied by accident.
WEBHOOKS_EXPECTED = os.environ.get("XCELSIOR_LIVE_WEBHOOKS", "").strip().lower() in (
    "1",
    "true",
    "yes",
)

MISSING_WEBHOOKS = (
    "set XCELSIOR_LIVE_WEBHOOKS=1 once Stripe events are being delivered here "
    "(stripe listen --forward-to <base>/api/providers/webhook). Without it a "
    "wallet is never credited, so this gate would either fail on 402 or pass on "
    "a balance left over from an earlier run."
)

MISSING_CREDENTIALS = "set XCELSIOR_LIVE_BASE_URL and XCELSIOR_LIVE_USER_TOKEN"
MISSING_FLEET = (
    "set XCELSIOR_LIVE_FLEET=1 once at least one host is admitted and running — "
    "these gates launch a real instance"
)


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}
