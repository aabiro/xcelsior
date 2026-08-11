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

MISSING_CREDENTIALS = "set XCELSIOR_LIVE_BASE_URL and XCELSIOR_LIVE_USER_TOKEN"
MISSING_FLEET = (
    "set XCELSIOR_LIVE_FLEET=1 once at least one host is admitted and running — "
    "these gates launch a real instance"
)


def auth() -> dict[str, str]:
    return {"Authorization": f"Bearer {TOKEN}"}
