"""`XCELSIOR_STRIPE_MODE=sandbox` must not resolve to a live secret key.

The resolution read:

    if _STRIPE_MODE == "sandbox":
        STRIPE_SECRET_KEY = os.environ.get("XCELSIOR_STRIPE_SANDBOX_SECRET_KEY", "") \\
                            or os.environ.get("XCELSIOR_STRIPE_SECRET_KEY", "")

The `or` is a convenience for an environment that has only one key. A **live**
key is precisely the case where that convenience is a real charge on a real
card while the operator believes they are in a sandbox — the one place a
fallback must not be helpful.

This was not hypothetical. Staging was found running `XCELSIOR_STRIPE_MODE=live`
against an `sk_live` key while being used as the target for gate work that the
plan explicitly scoped to test mode. Flipping the mode to `sandbox` would, with
no sandbox key present, have kept the live key and read as safe.

Disabled-and-loud is the correct failure. A deployment that asks for sandbox and
cannot have one should stop taking payments, not take real ones.
"""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.fixture(autouse=True)
def _restore_stripe_connect():
    """Put the module back exactly as it was.

    These tests reimport `stripe_connect` under a patched environment, and the
    module resolves its keys **at import time**. `monkeypatch` reverts the env
    at teardown but cannot revert the module object, so without this the whole
    session keeps whichever fake key the last test happened to set — and a
    later test that reads `STRIPE_WEBHOOK_SECRET` fails for a reason that has
    nothing to do with it. That is exactly what happened:
    `test_stripe_webhook_refuses_unverified.py` went red in the full suite and
    green on its own.
    """
    original = sys.modules.get("stripe_connect")
    yield
    if original is not None:
        # Re-insert *before* reloading: `importlib.reload` requires the module
        # to be present in `sys.modules` and raises `ImportError` otherwise.
        sys.modules["stripe_connect"] = original
        importlib.reload(original)
    else:
        sys.modules.pop("stripe_connect", None)


def _reload_with(monkeypatch, **env):
    for key in (
        "XCELSIOR_STRIPE_MODE",
        "XCELSIOR_STRIPE_SECRET_KEY",
        "XCELSIOR_STRIPE_SANDBOX_SECRET_KEY",
        "XCELSIOR_STRIPE_WEBHOOK_SECRET",
        "XCELSIOR_STRIPE_SANDBOX_WEBHOOK_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    sys.modules.pop("stripe_connect", None)
    return importlib.import_module("stripe_connect")


def test_sandbox_without_a_sandbox_key_disables_stripe_rather_than_going_live(monkeypatch):
    """The defect, asserted directly."""
    mod = _reload_with(
        monkeypatch,
        XCELSIOR_STRIPE_MODE="sandbox",
        XCELSIOR_STRIPE_SECRET_KEY="sk_live_not_a_real_key_0000000000",
    )
    assert not mod.STRIPE_SECRET_KEY.startswith("sk_live"), (
        "sandbox mode resolved to a LIVE key; a top-up test would charge a real card"
    )
    assert mod.STRIPE_ENABLED is False, (
        "Stripe is still enabled with no usable sandbox key — it must fail "
        "closed, not fall back to live"
    )


def test_sandbox_with_a_sandbox_key_uses_it(monkeypatch):
    """Positive control: the fix must not break the case that works."""
    mod = _reload_with(
        monkeypatch,
        XCELSIOR_STRIPE_MODE="sandbox",
        XCELSIOR_STRIPE_SANDBOX_SECRET_KEY="sk_test_not_a_real_key_0000000000",
        XCELSIOR_STRIPE_SECRET_KEY="sk_live_not_a_real_key_0000000000",
    )
    assert mod.STRIPE_SECRET_KEY.startswith("sk_test")
    assert mod.STRIPE_ENABLED is True


def test_live_mode_is_untouched(monkeypatch):
    """Production's behaviour must not move; only sandbox's fallback does."""
    mod = _reload_with(
        monkeypatch,
        XCELSIOR_STRIPE_MODE="live",
        XCELSIOR_STRIPE_SECRET_KEY="sk_live_not_a_real_key_0000000000",
    )
    assert mod.STRIPE_SECRET_KEY.startswith("sk_live")
    assert mod.STRIPE_ENABLED is True


def test_the_default_mode_is_recorded_so_a_change_is_deliberate(monkeypatch):
    """An unset mode resolves to live — stated, not discovered.

    Left as-is because narrowing it silently would disable payments on any
    deployment relying on the default. It is asserted so that a future change
    is a decision someone made rather than a side effect.
    """
    mod = _reload_with(monkeypatch, XCELSIOR_STRIPE_SECRET_KEY="sk_live_not_a_real_key_0000000000")
    assert mod._STRIPE_MODE == "live", (
        "the default Stripe mode changed. That is a payments-wide change: "
        "every deployment that does not set XCELSIOR_STRIPE_MODE is affected."
    )
