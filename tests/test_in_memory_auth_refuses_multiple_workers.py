"""In-memory auth with more than one worker is never correct, so it must not boot.

`XCELSIOR_PERSISTENT_AUTH=false` makes `routes._deps._users_db` — a plain dict
in the worker's own memory — the user *store*, not a cache. With more than one
gunicorn worker, registration lands on whichever worker served that request and
the next login round-robins elsewhere, so a worker that never saw the
registration answers 401.

Measured on this repo's own test stack, default two workers:

    401 401 401 401 200 200 401 200 401 401

Nothing logged an error. Each worker was behaving correctly. The user existed in
no database because it had never been written to one, and the register response
carried a complete user object because the worker that served it really had
created one. It presents as a password that works about half the time, which is
close to the hardest thing to diagnose from the outside — the search goes to the
database, the cache, the load balancer, and the answer is in a module-level
dict.

Guarded at boot rather than documented, because a warning in a log nobody reads
is how this survived in the first place.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONF = ROOT / "gunicorn.conf.py"


def _load(env: dict[str, str]) -> subprocess.CompletedProcess:
    """Execute the gunicorn config the way gunicorn does: as a module."""
    return subprocess.run(
        [sys.executable, "-c", f"exec(open({str(CONF)!r}).read())"],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("workers", ["2", "4"])
def test_in_memory_auth_with_many_workers_refuses_to_boot(workers: str) -> None:
    result = _load({"XCELSIOR_PERSISTENT_AUTH": "false", "GUNICORN_WORKERS": workers})
    assert result.returncode != 0, (
        f"gunicorn accepted in-memory auth with {workers} workers. Logins would "
        "fail on every worker that did not serve the registration, with nothing "
        "logged."
    )
    _assert_message_is_actionable(result.stderr, "XCELSIOR_PERSISTENT_AUTH", workers)


#: Flags whose "off" position swaps a shared store for a per-process one. Each
#: is fine on one worker and silently broken on several, in the same way.
PER_PROCESS_FLAGS = ("XCELSIOR_PERSISTENT_AUTH", "XCELSIOR_SHARED_RUNTIME_LIMITS")


def _assert_message_is_actionable(stderr: str, flag: str, workers: str) -> None:
    """What the refusal must say, without pinning how it says it.

    This used to assert the literal sentence "in-memory auth is per-process".
    When the guard was generalised to cover a second flag the wording changed,
    the phrase vanished, and the test failed while the guard was working
    perfectly — a false alarm that costs exactly as much attention as a real
    one. Assert the three things an operator needs instead: which variable is
    at fault, how many workers provoked it, and what to do about it.
    """
    assert flag in stderr, f"the refusal does not name {flag}, so nobody knows what to change:\n{stderr[-500:]}"
    assert f"GUNICORN_WORKERS={workers}" in stderr, (
        f"the refusal does not say the worker count that provoked it:\n{stderr[-500:]}"
    )
    assert "GUNICORN_WORKERS=1" in stderr, (
        f"the refusal offers no remediation:\n{stderr[-500:]}"
    )


@pytest.mark.parametrize("workers", ["2", "4"])
def test_per_process_runtime_limits_with_many_workers_refuses_to_boot(workers: str) -> None:
    """The same defect one layer along, in the connection path.

    With `XCELSIOR_SHARED_RUNTIME_LIMITS` off, `_shared_state_update` returns
    False without logging and WebSocket tickets fall back to
    `routes._deps._WS_TICKETS` — also per-process. A terminal ticket issued by
    one worker cannot be redeemed by another, and rate-limit buckets are
    multiplied by the worker count. It presents the same way the auth bug did:
    intermittent, correct-looking, and unlogged.
    """
    result = _load(
        {"XCELSIOR_SHARED_RUNTIME_LIMITS": "false", "GUNICORN_WORKERS": workers}
    )
    assert result.returncode != 0, (
        f"gunicorn accepted per-process runtime limits with {workers} workers; "
        "WebSocket tickets would be redeemable only on the worker that issued them"
    )
    _assert_message_is_actionable(result.stderr, "XCELSIOR_SHARED_RUNTIME_LIMITS", workers)


def test_both_flags_off_names_both() -> None:
    """A refusal that mentions one of two faults sends the operator back twice."""
    result = _load(
        {
            "XCELSIOR_PERSISTENT_AUTH": "false",
            "XCELSIOR_SHARED_RUNTIME_LIMITS": "false",
            "GUNICORN_WORKERS": "2",
        }
    )
    assert result.returncode != 0
    for flag in PER_PROCESS_FLAGS:
        assert flag in result.stderr, (
            f"{flag} is also off and the refusal does not mention it; fixing the "
            f"one named would just produce the same refusal again:\n{result.stderr[-600:]}"
        )


@pytest.mark.parametrize("flag", PER_PROCESS_FLAGS)
@pytest.mark.parametrize("off", ["false", "FALSE", "0", "no", "off"])
def test_every_spelling_of_off_is_caught(flag: str, off: str) -> None:
    """`"0"` and `"no"` disable these as surely as `"false"` does.

    The original check was `!= "false"`, so `XCELSIOR_PERSISTENT_AUTH=0` — which
    every other flag in this codebase treats as off — sailed past the guard and
    booted the broken configuration.
    """
    result = _load({flag: off, "GUNICORN_WORKERS": "2"})
    assert result.returncode != 0, (
        f"{flag}={off!r} was not recognised as off, so the guard let a "
        "per-process store boot on 2 workers"
    )


@pytest.mark.parametrize("flag", PER_PROCESS_FLAGS)
def test_a_single_worker_is_allowed_for_either_flag(flag: str) -> None:
    result = _load({flag: "false", "GUNICORN_WORKERS": "1"})
    assert result.returncode == 0, (
        f"a single-worker setup with {flag} off was refused, which is "
        f"over-blocking: {result.stderr[-300:]}"
    )


def test_in_memory_auth_with_one_worker_is_allowed() -> None:
    """A single worker makes the in-memory store coherent — that is a real setup."""
    result = _load({"XCELSIOR_PERSISTENT_AUTH": "false", "GUNICORN_WORKERS": "1"})
    assert result.returncode == 0, (
        f"a single-worker in-memory setup was refused, which is over-blocking: "
        f"{result.stderr[-300:]}"
    )


@pytest.mark.parametrize("workers", ["1", "2", "8"])
def test_persistent_auth_is_never_blocked(workers: str) -> None:
    """The production posture must not be affected at any worker count."""
    for value in ("true", "TRUE", ""):
        result = _load({"XCELSIOR_PERSISTENT_AUTH": value, "GUNICORN_WORKERS": workers})
        assert result.returncode == 0, (
            f"persistent auth with {workers} workers was refused "
            f"(XCELSIOR_PERSISTENT_AUTH={value!r}): {result.stderr[-300:]}"
        )


def test_the_default_posture_boots() -> None:
    """No env at all is production's shape: persistent auth, two workers."""
    assert _load({}).returncode == 0
