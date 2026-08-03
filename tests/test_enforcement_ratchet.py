"""How many security controls the test suite runs with switched off.

Three. That is the number, and it may only go down.

The suite ran for a long time with `AUTH_REQUIRED = False`, which makes
`_require_auth` hand an anonymous caller a synthetic principal carrying
`is_admin: True`. Every authorization test that did not explicitly undo that was
measuring the fixture rather than the endpoint — it could not fail. One
"verified under enforced auth" run turned out to have executed with auth off,
because an autouse fixture silently reverted the fixture that was supposed to
turn it on.

`conftest.SUITE_RELAXATIONS` names what is currently off, and
`pytest_runtest_call` refuses any test that experiences a relaxation outside
that set. This file pins the size of the set.

**Why a ratchet and not a fix.** These relaxations are global — they come from
`_pin_test_auth_env` and `XCELSIOR_ENV=test`, so all ~4,700 tests experience all
three. Turning one off is not an edit to a test; it is a change to how the whole
suite authenticates, and it has to be paid for once, deliberately, with the
resulting failures understood rather than suppressed. A ratchet lets that
happen on its own schedule while making the debt visible and non-increasing.

**What each one costs to remove.**

* `auth_required` — the largest. Every test that calls an endpoint without
  presenting a credential currently succeeds as an admin. Removing it means
  auditing which of those genuinely test anonymous access and which were only
  ever relying on the synthetic principal.
* `asymmetric_signing` — needs `XCELSIOR_OAUTH_JWT_KEYS_JSON` provisioned for
  the test environment. Mechanical, and it is the same key material staging
  needs before Gate P2, so the two land together.
* `startup_gate` — follows `asymmetric_signing`: the gate's own findings
  include the missing signing key, so enforcing it first would just fail on
  that.

`secrets_key` is already enforcing: `.env.test` provides a real Fernet key.
It is listed here as the proof that removal is possible and that the probe
distinguishes on from off.
"""

from __future__ import annotations

import pytest

from tests.conftest import SUITE_RELAXATIONS, _RELAXATIONS

#: Never raise this. Lower it when a control is genuinely enforced, in the same
#: commit that removes the entry from SUITE_RELAXATIONS.
MAX_SUITE_RELAXATIONS = 3

#: Controls that must never appear in SUITE_RELAXATIONS again.
PERMANENTLY_ENFORCED = frozenset({"secrets_key"})


def test_the_number_of_relaxed_controls_does_not_increase():
    """The ratchet."""
    assert len(SUITE_RELAXATIONS) <= MAX_SUITE_RELAXATIONS, (
        f"the suite now runs with {len(SUITE_RELAXATIONS)} security controls "
        f"relaxed ({sorted(SUITE_RELAXATIONS)}), up from "
        f"{MAX_SUITE_RELAXATIONS}. A control may be switched off only by "
        "lowering this number deliberately, never by adding to the set."
    )


def test_a_recovered_control_is_never_given_back():
    """Once enforced, a control may not return to the relaxed set."""
    regressions = sorted(SUITE_RELAXATIONS & PERMANENTLY_ENFORCED)
    assert not regressions, (
        f"these controls were enforced and have been relaxed again: {regressions}"
    )


def test_every_relaxation_names_a_real_control():
    """A typo in the set would silently exempt nothing — or everything."""
    unknown = sorted(SUITE_RELAXATIONS - set(_RELAXATIONS))
    assert not unknown, (
        f"SUITE_RELAXATIONS names controls that do not exist: {unknown}; "
        f"valid names are {sorted(_RELAXATIONS)}"
    )


def test_the_secrets_key_control_is_actually_enforcing():
    """Proof the probes distinguish on from off.

    If every probe returned False, the ratchet would read as three relaxations
    forever and nobody could tell. `secrets_key` is the control known to be on,
    so it is the calibration.
    """
    assert _RELAXATIONS["secrets_key"](), (
        "XCELSIOR_SECRETS_KEY is not set for the test run, so the one control "
        "this file uses to prove the probes work is itself off"
    )


@pytest.mark.parametrize("name", sorted(SUITE_RELAXATIONS))
def test_each_declared_relaxation_is_genuinely_relaxed(name):
    """No stale entries.

    A control listed as relaxed but actually enforcing is debt that has already
    been paid and not booked — it hides real progress and lets the ratchet stall
    at a number that is no longer true.
    """
    assert not _RELAXATIONS[name](), (
        f"{name!r} is listed in SUITE_RELAXATIONS but is already enforcing. "
        f"Remove it and lower MAX_SUITE_RELAXATIONS to "
        f"{len(SUITE_RELAXATIONS) - 1}."
    )
